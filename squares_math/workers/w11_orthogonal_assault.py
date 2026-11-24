#!/usr/bin/env python3
"""Wave 11 Orthogonal Whole-Packing Assault (80 Cores / 20 Shards, 100% pure-math open data).

Reconstructs the latest public-artifact frontier (~22.0897) by merging pub_base_v3 + Wave 6B/7B/8B/9B
wins (+ exact L-extend / delete-1 propagation) in ~5s, then runs 3 engines 100% ORTHOGONAL to 4D-DP:
  - Shards 0..7   (--mode stretch_all):    Multi-cut Arm-Stretch (+-1, +-2, +-3, all cuts) seeded from
                                           fresh Wave 6B..9B winners + all large packings n=300..1000!
  - Shards 8..13  (--mode bandflip_melt):  Bandflip + Anneal Melt -> fq + deep round-press hopping
                                           on top lagging puzzles n=120..1000!
  - Shards 14..19 (--mode insert_delete):  Whole-packing Insert-1 (n-1 -> n via --round-compress --round-press)
                                           + Smart Wall-Square Delete-1/Delete-2 (n+1/n+2 -> n) + Contact LP!
"""

import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import linprog, minimize_scalar
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
sys.path.insert(0, DS_DIR)
sys.path.insert(0, HERE)

import w5b_2d_faultdp_gadgetq as f2d
import stretch as st_mod

ANNEAL_BIN = os.environ.get("ANNEAL_BIN", "/tmp/sq_bins/anneal")


def bounds(p):
    half = (np.abs(np.cos(p[:, 2])) + np.abs(np.sin(p[:, 2]))) * 0.5
    lo = (p[:, :2] - half[:, None]).min(0)
    hi = (p[:, :2] + half[:, None]).max(0)
    return half, lo, hi


def span(p):
    _, lo, hi = bounds(p)
    return float((hi - lo).max())


def wall_slack(p, unit=1e-7):
    n = len(p)
    if n < 2:
        return p
    half, low, high = bounds(p)
    length = float((high - low).max())
    if length <= math.sqrt(n) + 1e-13:
        return p
    pairs = cKDTree(p[:, :2]).query_pairs(
        math.sqrt(2.0) + 8.0 * unit, output_type="ndarray"
    )
    if len(pairs) == 0:
        return p
    i, j = pairs.T
    c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
    axes = np.stack([np.stack([c, s], axis=1), np.stack([-s, c], axis=1)], axis=1)
    normals = np.concatenate([axes[i], axes[j]], axis=1)
    delta = p[j, :2] - p[i, :2]
    projections = np.einsum("pki,pi->pk", normals, delta)
    k = np.abs(projections).argmax(axis=1)
    ix = np.arange(len(i))
    v = projections[ix, k]
    direction = normals[ix, k] * np.sign(v)[:, None]
    angle = p[i, 2] - p[j, 2]
    gap = np.abs(v) - (1.0 + np.abs(np.cos(angle)) + np.abs(np.sin(angle))) * 0.5
    active = gap < 5.0 * unit
    i, j, direction, gap = i[active], j[active], direction[active], gap[active]
    row = np.arange(len(i))
    rows, cols = [row] * 4, [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
    data = [direction[:, 0], direction[:, 1], -direction[:, 0], -direction[:, 1]]
    slack = [gap / unit - 2e-6]
    offset = len(i)
    for dim in (0, 1):
        for sign in (-1, 1):
            room = (
                (p[:, dim] - half - low[dim])
                if sign == -1
                else (low[dim] + length - half - p[:, dim])
            )
            idx = np.flatnonzero(room < 6.0 * unit)
            if len(idx) == 0:
                continue
            r = np.arange(offset, offset + len(idx))
            offset += len(idx)
            rows.append(r)
            cols.append(2 * idx + dim)
            data.append(np.full(len(idx), float(sign)))
            slack.append(room[idx] / unit)
            if sign == 1:
                rows.append(r)
                cols.append(np.full(len(idx), 2 * n))
                data.append(np.ones(len(idx)))
    if offset == 0:
        return p
    matrix = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(offset, 2 * n + 1),
    ).tocsr()
    c_obj = np.zeros(2 * n + 1)
    c_obj[-1] = -1.0
    res = linprog(
        c_obj,
        A_ub=matrix,
        b_ub=np.concatenate(slack),
        bounds=[(-1.0, 1.0)] * (2 * n) + [(0.0, 2.0)],
        method="highs",
        options={"threads": 1},
    )
    if not res.success or res.x is None or res.x[-1] < 1e-4:
        return p
    q = p.copy()
    q[:, :2] += unit * res.x[:-1].reshape(n, 2)
    return q


def rotated(p, angle):
    q = p.copy()
    center = p[:, :2].mean(axis=0)
    c, s = math.cos(angle), math.sin(angle)
    q[:, :2] = (p[:, :2] - center) @ np.array([[c, s], [-s, c]]) + center
    q[:, 2] += angle
    return q


def lp_rot_polish(sq):
    cur = sq.copy()
    for unit in (1e-5, 1e-6, 1e-7, 1e-8):
        for _ in range(3):
            q = wall_slack(cur, unit)
            if span(q) < span(cur) - 5e-11:
                cur = q
            else:
                break

    def side_fn(ang):
        return span(rotated(cur, ang))

    steps = np.geomspace(1e-11, 5e-3, 19)
    angles = np.r_[-steps[::-1], 0.0, steps]
    sides = np.array([side_fn(float(a)) for a in angles])
    k = int(sides.argmin())
    ang = float(angles[k])
    if 0 < k < len(angles) - 1:
        fit = minimize_scalar(
            side_fn,
            method="bounded",
            bounds=(float(angles[k - 1]), float(angles[k + 1])),
            options={"xatol": 1e-15},
        )
        if fit.fun < sides[k]:
            ang = float(fit.x)
    q_rot = rotated(cur, ang)
    if span(q_rot) < span(cur) - 1e-11:
        cur = q_rot
    return cur


def load_frontier_with_wins(union_npz, win_globs):
    d = np.load(union_npz)
    X, Y, T, L = d["x"], d["y"], d["t"], d["L"]
    offs = [0]
    for n in range(1, 1001):
        offs.append(offs[-1] + n)
    cur = {}
    for n in range(1, 1001):
        s0, e0 = offs[n - 1], offs[n]
        cur[n] = (float(L[n - 1]), np.column_stack([X[s0:e0], Y[s0:e0], T[s0:e0]]))
    fresh = set()
    n_files = 0
    for pat in win_globs:
        for p in sorted(glob.glob(pat, recursive=True)):
            try:
                sd = np.load(p)
            except Exception:
                continue
            n_files += 1
            for k in sd.files:
                if k.startswith("x_"):
                    n = int(k[2:])
                    arr = np.column_stack(
                        [sd[f"x_{n}"], sd[f"y_{n}"], sd[f"t_{n}"]]
                    ).astype(np.float64)
                    if len(arr) != n:
                        continue
                    if float(np.max(np.abs(arr[:, 2]))) > 2.0:
                        arr[:, 2] = np.radians(arr[:, 2])
                    Ln = span(arr)
                    if Ln < cur[n][0] - 1e-8:
                        cur[n] = (Ln, arr)
                        fresh.add(n)
    # Quick exact L-extend propagation so frontier matches R9 closely!
    for _ in range(3):
        moved = 0
        for n in range(10, 1000):
            Ln, sq = cur[n]
            k = int(math.floor(Ln))
            n2 = n + 2 * k + 1
            if n2 <= 1000 and Ln + 1.0 + 1e-7 < cur[n2][0] - 1e-7:
                he = 0.5 * (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2])))
                xs = sq[:, 0] - np.min(sq[:, 0] - he)
                ys = sq[:, 1] - np.min(sq[:, 1] - he)
                L_act = max(float(np.max(xs + he)), float(np.max(ys + he)))
                L_new = L_act + 1.0 + 1e-7
                extra = []
                for i in range(k):
                    extra.append((L_new - 0.5, i + 0.5, 0.0))
                    extra.append((i + 0.5, L_new - 0.5, 0.0))
                extra.append((L_new - 0.5, L_new - 0.5, 0.0))
                sq2 = np.vstack(
                    [
                        np.column_stack([xs, ys, sq[:, 2]]),
                        np.array(extra, dtype=np.float64),
                    ]
                )
                cur[n2] = (span(sq2), sq2)
                fresh.add(n2)
                moved += 1
        if not moved:
            break
    tot = sum(cur[n][0] / math.sqrt(n) - 1.0 for n in range(1, 1001))
    return cur, fresh, tot, n_files


def pick_all_cuts(a, axis):
    P = [p for p in st_mod.patches(a) if p["horiz"] == (axis == 0)]
    if not P:
        return []
    p = max(P, key=lambda q: len(q["idx"]))
    lo, hi = (p["x0"], p["x1"]) if axis == 0 else (p["y0"], p["y1"])
    c = math.floor((lo + hi) / 2) + 0.0
    cuts = []
    for dc in (0, 1, -1, 2, -2, 3, -3, 4, -4):
        cc = c + dc
        others = [
            q
            for q in st_mod.patches(a)
            if q["horiz"] != (axis == 0)
            and (
                (q["x0"] - 0.8 < cc < q["x1"] + 0.8)
                if axis == 0
                else (q["y0"] - 0.8 < cc < q["y1"] + 0.8)
            )
        ]
        if not others and lo + 1.8 < cc < hi - 1.8:
            cuts.append(cc)
    return cuts


def stretch_multi(a, steps, cut_idx=0):
    b = np.array(a, dtype=np.float64)
    op = 1 if steps > 0 else -1
    for _ in range(abs(steps)):
        for axis in (0, 1):
            cuts = pick_all_cuts(b, axis)
            if not cuts:
                return None
            c = cuts[min(cut_idx, len(cuts) - 1)]
            b2, _ = st_mod.apply(b, axis, c, op)
            if b2 is None:
                return None
            b = b2
    return b


def quench_candidate(
    m, target_L, cand_sq, tmp, rng, allow_round_press=True, timeout=52
):
    best_s = target_L
    best_a = None
    s_in = span(cand_sq) + 1e-4
    local_best = None
    scheds = [
        ["--loosen", "1.0", "--flip-top", "32", "--pit", "220"],
        [
            "--loosen",
            "1.015",
            "--kick",
            "0.005",
            "--seed",
            str(rng.randrange(1 << 30)),
            "--flip-top",
            "32",
            "--pit",
            "240",
        ],
        [
            "--loosen",
            "1.032",
            "--kick",
            "0.014",
            "--seed",
            str(rng.randrange(1 << 30)),
            "--flip-top",
            "32",
            "--pit",
            "240",
        ],
    ]
    if allow_round_press:
        scheds.append(
            [
                "--loosen",
                "1.0",
                "--round-r0",
                "0.10",
                "--round-compress",
                "--round-press",
                "--round-steps",
                "6",
                "--round-press-outer",
                "10",
                "--flip-top",
                "24",
                "--pit",
                "130",
            ]
        )
    for sc in scheds:
        r = f2d.run_fq(s_in, cand_sq, tmp, sc, timeout=timeout)
        if r is not None:
            if local_best is None or r[0] < local_best[0]:
                local_best = r
            if r[0] < best_s - 1e-9:
                best_s, best_a = r[0], r[1]
    if local_best is not None and local_best[0] < best_s + 0.014:
        ls, la = local_best
        for _ in range(4):
            r2 = f2d.run_fq(
                ls,
                la,
                tmp,
                [
                    "--loosen",
                    rng.choice(("1.006", "1.014", "1.024")),
                    "--kick",
                    rng.choice(("0.002", "0.006", "0.012")),
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "48",
                    "--pit",
                    "300",
                ],
                timeout=timeout,
            )
            if r2 and r2[0] < ls:
                ls, la = r2
            if r2 and r2[0] < best_s - 1e-9:
                best_s, best_a = r2[0], r2[1]
    if best_a is not None and best_s < target_L - 1e-9:
        r_fin = f2d.run_fq(
            best_s,
            best_a,
            tmp,
            ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
            timeout=75,
        )
        if r_fin and r_fin[0] < best_s:
            best_s, best_a = r_fin[0], r_fin[1]
        best_a = lp_rot_polish(best_a)
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(m, best_a)
        if ok and Lf < target_L - 1e-9:
            return (Lf, xf, yf, tf)
    return None


def worker_stretch(job):
    m, target_L, cand_sq, tag, seed = job
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as tmp:
        res = quench_candidate(
            m, target_L, cand_sq, tmp, rng, allow_round_press=True, timeout=50
        )
    return m, tag, res


def worker_bandflip_melt(job):
    n, L0, sq, seed = job
    rng = random.Random(seed)
    best_s = L0
    best_a = None
    with tempfile.TemporaryDirectory() as tmp:
        # 1) Bandflip variants
        starts = []
        try:
            import bandflip as bf_mod

            pts_deg = np.column_stack(
                [sq[:, 0], sq[:, 1], np.degrees(sq[:, 2]) % 90.0]
            ).tolist()
            B = bf_mod.bands(pts_deg, min_size=12)
            for bi in range(min(3, len(B))):
                fl = bf_mod.flip(L0, pts_deg, bi, rng, B)
                if fl is not None:
                    arr_fl = np.array(fl, dtype=np.float64)
                    arr_fl[:, 2] = np.radians(arr_fl[:, 2])
                    starts.append(("bf", arr_fl))
        except Exception:
            pass

        for tag_s, st_sq in starts:
            for sc in (
                [
                    "--loosen",
                    "1.015",
                    "--kick",
                    "0.006",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "32",
                    "--pit",
                    "240",
                ],
                [
                    "--loosen",
                    "1.035",
                    "--kick",
                    "0.016",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "32",
                    "--pit",
                    "240",
                ],
            ):
                r = f2d.run_fq(span(st_sq) + 1e-4, st_sq, tmp, sc, timeout=50)
                if r is not None and r[0] < best_s - 1e-9:
                    best_s, best_a = r[0], r[1]

        # 2) Anneal melt (3 localized melts at tilted-band/wall squares -> fq)
        if os.path.exists(ANNEAL_BIN):
            degs = np.degrees(sq[:, 2]) % 90.0
            tilted_idx = np.flatnonzero(np.minimum(degs, 90.0 - degs) > 2.5)
            for _ in range(3):
                try:
                    pin = os.path.join(tmp, "min.txt")
                    pout = os.path.join(tmp, "mout.txt")
                    f2d.write_fq_cfg(pin, L0, sq)
                    if os.path.exists(pout):
                        os.remove(pout)
                    pick = (
                        int(rng.choice(tilted_idx))
                        if len(tilted_idx)
                        else rng.randrange(n)
                    )
                    cx, cy = float(sq[pick, 0]), float(sq[pick, 1])
                    subprocess.run(
                        [
                            ANNEAL_BIN,
                            "melt",
                            "--in",
                            pin,
                            "--out",
                            pout,
                            "--cx",
                            repr(cx),
                            "--cy",
                            repr(cy),
                            "--rad",
                            f"{rng.uniform(2.0, 4.8):.3f}",
                            "--rmax",
                            f"{rng.uniform(0.08, 0.20):.3f}",
                            "--sweeps",
                            "3200",
                            "--bp",
                            "2200.0",
                            "--seed",
                            str(rng.randrange(1 << 30)),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=40,
                    )
                    if os.path.exists(pout):
                        s_m, arr = f2d.read_fq_cfg_rad(pout)
                        if len(arr) == n:
                            r = f2d.run_fq(
                                s_m + 1e-4,
                                arr,
                                tmp,
                                [
                                    "--loosen",
                                    "1.012",
                                    "--kick",
                                    "0.005",
                                    "--seed",
                                    str(rng.randrange(1 << 30)),
                                    "--flip-top",
                                    "48",
                                    "--pit",
                                    "300",
                                ],
                                timeout=55,
                            )
                            if r is not None and r[0] < best_s - 1e-9:
                                best_s, best_a = r[0], r[1]
                except Exception:
                    pass

        # 3) Deep round-press perturbation on whole packing
        for r0 in ("0.09", "0.14"):
            r = f2d.run_fq(
                L0 + 0.005,
                sq,
                tmp,
                [
                    "--loosen",
                    "1.0",
                    "--round-r0",
                    r0,
                    "--round-compress",
                    "--round-press",
                    "--round-steps",
                    "7",
                    "--round-press-outer",
                    "11",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "32",
                    "--pit",
                    "180",
                ],
                timeout=55,
            )
            if r is not None and r[0] < best_s - 1e-9:
                best_s, best_a = r[0], r[1]

        if best_a is not None and best_s < L0 - 1e-9:
            r_fin = f2d.run_fq(
                best_s,
                best_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
                timeout=75,
            )
            if r_fin and r_fin[0] < best_s:
                best_s, best_a = r_fin[0], r_fin[1]
            best_a = lp_rot_polish(best_a)
            ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, best_a)
            if ok and Lf < L0 - 1e-9:
                return n, "bf_melt", (Lf, xf, yf, tf)
    return n, "bf_melt", None


def worker_insert_delete(job):
    n, L0, sq_prev, sq_next, sq_next2, seed = job
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    best_s = L0
    best_a = None
    best_tag = ""
    with tempfile.TemporaryDirectory() as tmp:
        # 1) Whole-packing Insert-1 (n-1 -> n via --round-compress --round-press!)
        if sq_prev is not None:
            Lp = span(sq_prev)
            if Lp < L0 - 0.001:
                he = 0.5 * (
                    np.abs(np.cos(sq_prev[:, 2])) + np.abs(np.sin(sq_prev[:, 2]))
                )
                xs = sq_prev[:, 0] - np.min(sq_prev[:, 0] - he)
                ys = sq_prev[:, 1] - np.min(sq_prev[:, 1] - he)
                sq_p = np.column_stack([xs, ys, sq_prev[:, 2]])
                pts = np.column_stack(
                    [
                        np_rng.uniform(0.5, max(0.6, Lp - 0.5), 600),
                        np_rng.uniform(0.5, max(0.6, Lp - 0.5), 600),
                    ]
                )
                dmin = np.min(
                    np.hypot(
                        pts[:, None, 0] - sq_p[None, :, 0],
                        pts[:, None, 1] - sq_p[None, :, 1],
                    ),
                    axis=1,
                )
                top3 = np.argsort(-dmin)[:3]
                degs = np.degrees(sq_p[:, 2]) % 90.0
                tilted = sq_p[np.minimum(degs, 90.0 - degs) > 3.0, 2]
                med_t = float(np.median(tilted)) if len(tilted) else math.radians(24.0)
                for idx_h, ang_try in (
                    (top3[0], med_t),
                    (top3[1], 0.0),
                    (top3[2], math.radians(45.0)),
                ):
                    sq_plus = np.vstack(
                        [
                            sq_p,
                            np.array(
                                [[float(pts[idx_h, 0]), float(pts[idx_h, 1]), ang_try]],
                                dtype=np.float64,
                            ),
                        ]
                    )
                    r = f2d.run_fq(
                        max(Lp, L0) + 0.04,
                        sq_plus,
                        tmp,
                        [
                            "--loosen",
                            "1.0",
                            "--round-r0",
                            "0.15",
                            "--round-compress",
                            "--round-press",
                            "--round-steps",
                            "7",
                            "--round-press-outer",
                            "12",
                            "--seed",
                            str(rng.randrange(1 << 30)),
                            "--flip-top",
                            "28",
                            "--pit",
                            "150",
                        ],
                        timeout=55,
                    )
                    if r is not None and r[0] < best_s - 1e-9:
                        best_s, best_a, best_tag = r[0], r[1], "ins1_rp"
                    elif r is not None and r[0] < best_s + 0.008:
                        r2 = f2d.run_fq(
                            r[0],
                            r[1],
                            tmp,
                            [
                                "--loosen",
                                "1.012",
                                "--kick",
                                "0.005",
                                "--seed",
                                str(rng.randrange(1 << 30)),
                                "--flip-top",
                                "48",
                                "--pit",
                                "280",
                            ],
                            timeout=55,
                        )
                        if r2 is not None and r2[0] < best_s - 1e-9:
                            best_s, best_a, best_tag = r2[0], r2[1], "ins1_rp_k"

        # 2) Smart Wall-Contact Delete-1 (n+1 -> n): drop the square with tightest wall/tilted contact!
        if sq_next is not None and span(sq_next) < L0 + 0.045:
            he = 0.5 * (np.abs(np.cos(sq_next[:, 2])) + np.abs(np.sin(sq_next[:, 2])))
            lo_x, hi_x = np.min(sq_next[:, 0] - he), np.max(sq_next[:, 0] + he)
            lo_y, hi_y = np.min(sq_next[:, 1] - he), np.max(sq_next[:, 1] + he)
            wall_dist = np.minimum(
                np.minimum(sq_next[:, 0] - he - lo_x, hi_x - (sq_next[:, 0] + he)),
                np.minimum(sq_next[:, 1] - he - lo_y, hi_y - (sq_next[:, 1] + he)),
            )
            degs = np.degrees(sq_next[:, 2]) % 90.0
            is_tilted = np.minimum(degs, 90.0 - degs) > 2.5
            # Try dropping (a) closest wall tilted square, (b) closest wall square overall, (c) random tilted square
            drop_ids = []
            tilted_wall = np.argsort(np.where(is_tilted, wall_dist, 999.0))
            if len(tilted_wall) and is_tilted[tilted_wall[0]]:
                drop_ids.append(int(tilted_wall[0]))
            drop_ids.append(int(np.argmin(wall_dist)))
            if np.any(is_tilted):
                drop_ids.append(int(rng.choice(np.flatnonzero(is_tilted))))
            for di in dict.fromkeys(drop_ids):
                sub = np.delete(sq_next, di, axis=0)
                for sc in (
                    [
                        "--loosen",
                        "1.012",
                        "--kick",
                        "0.005",
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "36",
                        "--pit",
                        "260",
                    ],
                    [
                        "--loosen",
                        "1.030",
                        "--kick",
                        "0.015",
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "36",
                        "--pit",
                        "240",
                    ],
                ):
                    r = f2d.run_fq(span(sub) + 1e-4, sub, tmp, sc, timeout=50)
                    if r is not None and r[0] < best_s - 1e-9:
                        best_s, best_a, best_tag = r[0], r[1], "del1_wall"

        # 3) Delete-2 (n+2 -> n) if n+2 is close!
        if sq_next2 is not None and span(sq_next2) < L0 + 0.055:
            he = 0.5 * (np.abs(np.cos(sq_next2[:, 2])) + np.abs(np.sin(sq_next2[:, 2])))
            lo_x, hi_x = np.min(sq_next2[:, 0] - he), np.max(sq_next2[:, 0] + he)
            lo_y, hi_y = np.min(sq_next2[:, 1] - he), np.max(sq_next2[:, 1] + he)
            wall_dist = np.minimum(
                np.minimum(sq_next2[:, 0] - he - lo_x, hi_x - (sq_next2[:, 0] + he)),
                np.minimum(sq_next2[:, 1] - he - lo_y, hi_y - (sq_next2[:, 1] + he)),
            )
            drop2 = np.argsort(wall_dist)[:2]
            sub = np.delete(sq_next2, drop2, axis=0)
            r = f2d.run_fq(
                span(sub) + 1e-4,
                sub,
                tmp,
                [
                    "--loosen",
                    "1.020",
                    "--kick",
                    "0.008",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "36",
                    "--pit",
                    "260",
                ],
                timeout=50,
            )
            if r is not None and r[0] < best_s - 1e-9:
                best_s, best_a, best_tag = r[0], r[1], "del2_wall"

        if best_a is not None and best_s < L0 - 1e-9:
            r_fin = f2d.run_fq(
                best_s,
                best_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
                timeout=75,
            )
            if r_fin and r_fin[0] < best_s:
                best_s, best_a = r_fin[0], r_fin[1]
            best_a = lp_rot_polish(best_a)
            ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, best_a)
            if ok and Lf < L0 - 1e-9:
                return n, best_tag, (Lf, xf, yf, tf)
    return n, best_tag, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--lags-json", required=True)
    ap.add_argument(
        "--mode",
        choices=("stretch_all", "bandflip_melt", "insert_delete"),
        required=True,
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=3500)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cur, fresh, tot_score, n_files = load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    lags = json.load(open(args.lags_json)) if os.path.exists(args.lags_json) else []
    lag_ns = [int(it["n"]) for it in lags if int(it["n"]) >= 120]
    print(
        f"[W11-{args.mode} s={args.shard}/{args.num_shards}] Frontier score={tot_score:.10f} (merged {len(fresh)} fresh wins from {n_files} files, {time.time() - t0:.1f}s)",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0

    if args.mode == "stretch_all":
        seed_ns = list(
            dict.fromkeys(
                sorted(fresh, reverse=True)
                + [
                    n
                    for n in range(1000, 299, -1)
                    if cur[n][0] < math.ceil(math.sqrt(n)) - 1e-6
                ]
            )
        )
        cands_by_key = {}
        for sn in seed_ns:
            sq = cur[sn][1]
            is_f = sn in fresh
            step_list = (-2, -1, 1, 2, -3, 3) if is_f else (-1, 1, -2, 2)
            cut_list = (0, 1, 2) if is_f else (0, 1)
            for st in step_list:
                for ci in cut_list:
                    b = stretch_multi(sq, st, cut_idx=ci)
                    if b is None:
                        continue
                    m = len(b)
                    if m < 100 or m > 1000:
                        continue
                    Km = math.isqrt(m)
                    if m == Km * Km:
                        continue
                    gap = span(b) - cur[m][0]
                    if gap < 0.085:
                        prio = (
                            gap
                            - (0.030 if is_f else 0.0)
                            - (0.015 if m >= 750 else 0.0)
                        )
                        k = (m, sn, st)
                        if k not in cands_by_key or prio < cands_by_key[k][0]:
                            cands_by_key[k] = (
                                prio,
                                gap,
                                m,
                                b,
                                f"{sn}->{m}(st={st:+d},c={ci})",
                            )
        per_m = {}
        for _, gap, m, b, tag in sorted(cands_by_key.values(), key=lambda x: x[0]):
            lst = per_m.setdefault(m, [])
            if len(lst) < 3:
                lst.append((gap, m, b, tag))
        all_cands = []
        for lst in per_m.values():
            all_cands.extend(lst)
        all_cands.sort(key=lambda x: x[0])
        my_slice = all_cands[args.shard :: args.num_shards]
        print(
            f"[W11-stretch s={args.shard}] Total stretch candidates={len(all_cands)} across {len(per_m)} targets, my_slice={len(my_slice)}",
            flush=True,
        )
        jobs = [
            (m, cur[m][0], b, tag, 1100001 + args.shard * 10000 + i * 31)
            for i, (gap, m, b, tag) in enumerate(my_slice)
        ]
        worker_fn = worker_stretch

    elif args.mode == "bandflip_melt":
        targets = list(
            dict.fromkeys(
                sorted([n for n in fresh if n >= 120], reverse=True) + lag_ns[:260]
            )
        )
        my_slice = targets[args.shard :: args.num_shards]
        print(
            f"[W11-bf_melt s={args.shard}] Targets={len(targets)}, my_slice={len(my_slice)}",
            flush=True,
        )
        jobs = [
            (n, cur[n][0], cur[n][1], 1120001 + args.shard * 10000 + i * 37)
            for i, n in enumerate(my_slice)
        ]
        worker_fn = worker_bandflip_melt

    else:  # insert_delete
        # Rank puzzles by max(jump from n-1, proximity to n+1) + lag weight!
        scored = []
        for n in range(120, 1001):
            K = math.isqrt(n)
            if n == K * K or cur[n][0] >= K + 1 - 1e-6:
                continue
            jump_prev = cur[n][0] - cur[n - 1][0] if (n - 1) in cur else 0.0
            prox_next = (
                max(0.0, 0.04 - (cur[n + 1][0] - cur[n][0])) if (n + 1) in cur else 0.0
            )
            boost = 0.015 if n in fresh else 0.0
            if jump_prev > 0.002 or prox_next > 0.001 or n in lag_ns[:180]:
                scored.append((jump_prev + prox_next + boost, n))
        scored.sort(reverse=True)
        targets = [n for _, n in scored[:320]]
        my_slice = targets[args.shard :: args.num_shards]
        print(
            f"[W11-ins_del s={args.shard}] Targets={len(targets)}, my_slice={len(my_slice)}",
            flush=True,
        )
        jobs = [
            (
                n,
                cur[n][0],
                cur[n - 1][1] if n > 1 else None,
                cur[n + 1][1] if n < 1000 else None,
                cur[n + 2][1] if n < 999 else None,
                1140001 + args.shard * 10000 + i * 41,
            )
            for i, n in enumerate(my_slice)
        ]
        worker_fn = worker_insert_delete

    done = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_fn, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            n, tag, res = fut.result()
            if res is not None:
                Lf, xf, yf, tf = res
                prev_L = wins[n][0] if n in wins else cur[n][0]
                if Lf < prev_L - 1e-10:
                    dL = Lf - cur[n][0]
                    dS = (Lf - prev_L) / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W11-{args.mode} WIN] n={n:4d} {tag:22s} {cur[n][0]:.7f} -> {Lf:.7f} dL={dL:.7f} dS={dL / math.sqrt(n):.8f}",
                        flush=True,
                    )
            if done % 15 == 0:
                print(
                    f"  [W11-{args.mode} s={args.shard}] {done}/{len(jobs)} done, wins={len(wins)}, dS={tot_ds:.8f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W11-{args.mode} s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    payload = {}
    for n, (Lf, xf, yf, tf) in sorted(wins.items()):
        payload[f"L_{n}"] = np.array([Lf], dtype=np.float64)
        payload[f"x_{n}"] = np.asarray(xf, dtype=np.float64)
        payload[f"y_{n}"] = np.asarray(yf, dtype=np.float64)
        payload[f"t_{n}"] = np.asarray(tf, dtype=np.float64)
    np.savez_compressed(args.out_npz, **payload)
    print(
        f"[W11-{args.mode} s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
