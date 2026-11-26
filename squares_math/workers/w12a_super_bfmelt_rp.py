#!/usr/bin/env python3
"""Wave 12A Super Bandflip/Relocate + Multi-Scale Anneal-Melt + Multi-Radius Round-Press Assault (80 Cores / 20 Shards).

Covers ALL ~560 non-grid packings n=60..1000 starting from the full ~22.0892 public-artifact frontier
(pub_base_v3 + Wave 6B/7B/8B/9B/10B/11 + L-extend propagation). Every packing gets ~2 minutes of deep
4-core orthogonal topology search (bandflip.flip + bandflip.relocate + 6x multi-scale anneal melt->fq
+ 4x multi-radius whole-packing round-compress/round-press + follow-up kick chains + Contact LP +
Rigid Rotation + Decimal64 >= 1.05e-9 certification).
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
    for _ in range(4):
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
        for n in range(999, 1, -1):
            if cur[n + 1][0] < cur[n][0] - 1e-8:
                sq_sub = cur[n + 1][1][:-1].copy()
                cur[n] = (span(sq_sub), sq_sub)
                fresh.add(n)
                moved += 1
        if not moved:
            break
    tot = sum(cur[n][0] / math.sqrt(n) - 1.0 for n in range(1, 1001))
    return cur, fresh, tot, n_files


def worker_super_bfmelt(job):
    n, L0, sq, seed = job
    rng = random.Random(seed)
    best_s = L0
    best_a = None
    best_tag = ""
    near_best = None
    with tempfile.TemporaryDirectory() as tmp:
        # 1) Bandflip.flip + Bandflip.relocate moves!
        starts = []
        try:
            import bandflip as bf_mod

            pts_deg = np.column_stack(
                [sq[:, 0], sq[:, 1], np.degrees(sq[:, 2]) % 90.0]
            ).tolist()
            B = bf_mod.bands(pts_deg, min_size=10)
            for bi in range(min(3, len(B))):
                fl = bf_mod.flip(L0, pts_deg, bi, rng, B)
                if fl is not None:
                    arr_fl = np.array(fl, dtype=np.float64)
                    arr_fl[:, 2] = np.radians(arr_fl[:, 2])
                    starts.append(("bf_flip", arr_fl))
                rel = bf_mod.relocate(L0, pts_deg, bi, rng, B)
                if rel is not None:
                    arr_rel = np.array(rel, dtype=np.float64)
                    arr_rel[:, 2] = np.radians(arr_rel[:, 2])
                    starts.append(("bf_reloc", arr_rel))
        except Exception:
            pass

        for tag_s, st_sq in starts[:5]:
            for sc in (
                [
                    "--loosen",
                    "1.015",
                    "--kick",
                    "0.006",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "36",
                    "--pit",
                    "240",
                ],
                [
                    "--loosen",
                    "1.038",
                    "--kick",
                    "0.018",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "36",
                    "--pit",
                    "220",
                ],
            ):
                r = f2d.run_fq(span(st_sq) + 1e-4, st_sq, tmp, sc, timeout=48)
                if r is not None:
                    if near_best is None or r[0] < near_best[0]:
                        near_best = (r[0], r[1], tag_s)
                    if r[0] < best_s - 1e-9:
                        best_s, best_a, best_tag = r[0], r[1], tag_s

        # 2) Multi-scale Anneal Melt (6 localized melts spanning small/medium/large radii on tilted & corner sites -> fq!)
        if os.path.exists(ANNEAL_BIN):
            degs = np.degrees(sq[:, 2]) % 90.0
            tilted_idx = np.flatnonzero(np.minimum(degs, 90.0 - degs) > 2.5)
            # Corner / wall tilted squares carry highest stress
            he = 0.5 * (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2])))
            lo_x, hi_x = np.min(sq[:, 0] - he), np.max(sq[:, 0] + he)
            lo_y, hi_y = np.min(sq[:, 1] - he), np.max(sq[:, 1] + he)
            wall_d = np.minimum(
                np.minimum(sq[:, 0] - he - lo_x, hi_x - (sq[:, 0] + he)),
                np.minimum(sq[:, 1] - he - lo_y, hi_y - (sq[:, 1] + he)),
            )
            stress_idx = np.argsort(
                np.where(np.minimum(degs, 90.0 - degs) > 2.5, wall_d, 999.0)
            )[:20]

            melt_cfgs = [
                (1.8, 3.0, 0.06, 0.14, 2800, "tilt"),
                (2.5, 4.2, 0.08, 0.18, 3400, "stress"),
                (3.5, 5.5, 0.10, 0.22, 3600, "tilt"),
                (1.6, 2.8, 0.07, 0.15, 2800, "stress"),
                (2.8, 4.8, 0.09, 0.20, 3400, "any"),
                (4.0, 6.2, 0.12, 0.24, 3800, "tilt"),
            ]
            for rmin, rmax_rad, m0, m1, swp, mode_p in melt_cfgs:
                try:
                    pin = os.path.join(tmp, "min.txt")
                    pout = os.path.join(tmp, "mout.txt")
                    src_sq = best_a if best_a is not None else sq
                    src_L = best_s if best_a is not None else L0
                    f2d.write_fq_cfg(pin, src_L, src_sq)
                    if os.path.exists(pout):
                        os.remove(pout)
                    if mode_p == "stress" and len(stress_idx):
                        pick = int(rng.choice(stress_idx))
                    elif mode_p == "tilt" and len(tilted_idx):
                        pick = int(rng.choice(tilted_idx))
                    else:
                        pick = rng.randrange(n)
                    cx, cy = float(src_sq[pick, 0]), float(src_sq[pick, 1])
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
                            f"{rng.uniform(rmin, rmax_rad):.3f}",
                            "--rmax",
                            f"{rng.uniform(m0, m1):.3f}",
                            "--sweeps",
                            str(swp),
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
                                    "1.010",
                                    "--kick",
                                    "0.004",
                                    "--seed",
                                    str(rng.randrange(1 << 30)),
                                    "--flip-top",
                                    "48",
                                    "--pit",
                                    "300",
                                ],
                                timeout=52,
                            )
                            if r is not None:
                                if near_best is None or r[0] < near_best[0]:
                                    near_best = (r[0], r[1], "melt")
                                if r[0] < best_s - 1e-9:
                                    best_s, best_a, best_tag = r[0], r[1], "melt"
                except Exception:
                    pass

        # 3) Multi-Radius Whole-Packing Round-Compress + Round-Press!
        for r0 in ("0.06", "0.10", "0.14", "0.19"):
            src_sq = best_a if best_a is not None else sq
            src_L = best_s if best_a is not None else L0
            r = f2d.run_fq(
                src_L + 0.006,
                src_sq,
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
                    "12",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "32",
                    "--pit",
                    "180",
                ],
                timeout=52,
            )
            if r is not None:
                if near_best is None or r[0] < near_best[0]:
                    near_best = (r[0], r[1], f"rp_{r0}")
                if r[0] < best_s - 1e-9:
                    best_s, best_a, best_tag = r[0], r[1], f"rp_{r0}"

        # 4) Iterative Follow-Up Kick Chain on Winner OR Tight Near-Miss (+0.004)!
        seed_chain = (
            (best_s, best_a, best_tag)
            if best_a is not None
            else (
                near_best
                if (near_best is not None and near_best[0] < L0 + 0.004)
                else None
            )
        )
        if seed_chain is not None:
            ls, la, ltag = seed_chain
            for _ in range(4):
                r2 = f2d.run_fq(
                    ls,
                    la,
                    tmp,
                    [
                        "--loosen",
                        rng.choice(("1.004", "1.010", "1.020")),
                        "--kick",
                        rng.choice(("0.0015", "0.004", "0.009")),
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "48",
                        "--pit",
                        "320",
                    ],
                    timeout=55,
                )
                if r2 is not None and r2[0] < ls:
                    ls, la = r2
                if r2 is not None and r2[0] < best_s - 1e-9:
                    best_s, best_a, best_tag = r2[0], r2[1], ltag + "+kc"

        if best_a is not None and best_s < L0 - 1e-9:
            r_fin = f2d.run_fq(
                best_s,
                best_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "650"],
                timeout=80,
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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=3800)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cur, fresh, tot_score, n_files = load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    lags = json.load(open(args.lags_json)) if os.path.exists(args.lags_json) else []
    lag_rank = {int(it["n"]): float(it.get("dS", 0.0)) for it in lags}

    # Include ALL non-grid packings n=60..1000, prioritized by (lag_dS + fresh boost + high-f boost)!
    all_ng = []
    for n in range(60, 1001):
        K = math.isqrt(n)
        if n == K * K:
            continue
        Ln = cur[n][0]
        if Ln >= K + 1 - 1e-6:
            continue
        f = Ln - K
        # Prioritize high-lag puzzles, freshly modified puzzles, and mid/high-f n>=400!
        prio = (
            lag_rank.get(n, 0.0)
            + (0.00015 if n in fresh else 0.0)
            + (0.00005 if (n >= 400 and f >= 0.65) else 0.0)
        )
        all_ng.append((prio, n))
    all_ng.sort(reverse=True)
    ordered_ns = [n for _, n in all_ng]
    my_slice = ordered_ns[args.shard :: args.num_shards]
    print(
        f"[W12A-SuperBFMelt s={args.shard}/{args.num_shards}] Frontier={tot_score:.10f} ({len(fresh)} fresh from {n_files} files), total_ng={len(ordered_ns)}, my_slice={len(my_slice)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0
    jobs = [
        (n, cur[n][0], cur[n][1], 1200001 + args.shard * 10000 + i * 53)
        for i, n in enumerate(my_slice)
    ]
    done = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_super_bfmelt, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            n, tag, res = fut.result()
            if res is not None:
                Lf, xf, yf, tf = res
                dL = Lf - cur[n][0]
                if dL < -1e-10:
                    dS = dL / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W12A WIN] n={n:4d} {tag:14s} {cur[n][0]:.7f} -> {Lf:.7f} dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
            if done % 8 == 0:
                print(
                    f"  [W12A s={args.shard}] {done}/{len(jobs)} done, wins={len(wins)}, dS={tot_ds:.8f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W12A s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
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
        f"[W12A s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
