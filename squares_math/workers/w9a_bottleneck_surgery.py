#!/usr/bin/env python3
"""Wave 9A Bottleneck-Targeted Gadget Surgery (80 Cores / 20 Shards, pure-math open data).

Instead of quenching random gadgets, Wave 9A solves the 3-wave pooled 10-bucket 4D Staircase DP
(pub_base_v3 + Wave 6A + Wave 7A + Wave 8A), identifies the EXACT BOTTLENECK GADGETS that set
L = K + max_i f(g_i) across all 366 lagging puzzles (weighted by downstream dS gap), and runs
3 precision surgery modes with graceful 60-min budget checkpoints (0% timeouts):
  - Shards 0..7   (--mode bottleneck_shrink): Deep multi-seed padded-fq + round-press + anneal melt->fq
  - Shards 8..15  (--mode bottleneck_grow):   Targeted (w, h, dm-1 -> dm) +1 square grow directly into bottleneck slots
  - Shards 16..19 (--mode bottleneck_pair):   Adjacent bottleneck-pair seam fusion + padded-fq quench
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

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
sys.path.insert(0, DS_DIR)
sys.path.insert(0, HERE)

import w5b_2d_faultdp_gadgetq as f2d

# Match Wave 35 upgraded 18-bucket refined THR (0.03-0.04 step)
f2d.THR = np.array(
    [
        0.0,
        0.25,
        0.40,
        0.48,
        0.52,
        0.56,
        0.60,
        0.64,
        0.68,
        0.72,
        0.76,
        0.80,
        0.83,
        0.86,
        0.89,
        0.92,
        0.95,
        0.97,
    ]
)
f2d.NB = len(f2d.THR)
ANNEAL_BIN = os.environ.get("ANNEAL_BIN", "/tmp/sq_bins/anneal")


def pad_to_square(sq_crop, w, h, f_box):
    M = max(w, h)
    pad = []
    if w < h:
        for ci in range(h - w):
            for rj in range(h):
                pad.append((w + f_box + 2e-5 + ci + 0.5, rj + 0.5, 0.0))
    elif h < w:
        for ci in range(w):
            for rj in range(w - h):
                pad.append((ci + 0.5, h + f_box + 2e-5 + rj + 0.5, 0.0))
    return (
        np.vstack([sq_crop, np.array(pad, dtype=np.float64)]) if pad else sq_crop.copy()
    ), M


def load_pooled_donors(union_npz, prev_globs, win_globs=None):
    d = np.load(union_npz)
    x, y, t, L = d["x"], d["y"], d["t"], d["L"]
    offs = [0]
    for n in range(1, 1001):
        offs.append(offs[-1] + n)
    cur_sq = {}
    cur_L = {}
    for n in range(1, 1001):
        s0, e0 = offs[n - 1], offs[n]
        cur_L[n] = float(L[n - 1])
        cur_sq[n] = np.column_stack([x[s0:e0], y[s0:e0], t[s0:e0]])
    n_win_upd = 0
    for pat in win_globs or []:
        for wf in sorted(glob.glob(pat, recursive=True)):
            try:
                wd = np.load(wf)
            except Exception:
                continue
            for k in wd.files:
                if k.startswith("x_"):
                    n = int(k[2:])
                    arr = np.column_stack(
                        [wd[f"x_{n}"], wd[f"y_{n}"], wd[f"t_{n}"]]
                    ).astype(np.float64)
                    if len(arr) != n:
                        continue
                    if float(np.max(np.abs(arr[:, 2]))) > 2.0:
                        arr[:, 2] = np.radians(arr[:, 2])
                    Ln = f2d.side(arr)
                    if Ln < cur_L[n] - 1e-8:
                        cur_L[n] = Ln
                        cur_sq[n] = arr
                        n_win_upd += 1
    donors = []
    seen_sig = set()
    for n in range(1, 1001):
        if cur_L[n] < math.ceil(math.sqrt(n)) - 1e-7:
            donors.append(cur_sq[n])
            seen_sig.add((n, round(cur_L[n], 6)))
    n_base = len(donors)
    n_prev = 0
    for pat in prev_globs:
        for gf in sorted(glob.glob(pat, recursive=True)):
            try:
                gd = np.load(gf)
            except Exception:
                continue
            for k in gd.files:
                arr = gd[k]
                if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) >= 5:
                    arr64 = arr.astype(np.float64)
                    sig = (len(arr64), round(f2d.side(arr64), 6))
                    if sig not in seen_sig:
                        seen_sig.add(sig)
                        donors.append(arr64)
                        n_prev += 1
    return donors, cur_L, n_base, n_prev


def extract_top_chains(dp, PG, PM, PQ, G, K, D, max_chains=3):
    sub = dp[: K + 1, : K + 1, D, :]
    if not np.isfinite(sub).any():
        return []
    flat_f = sub.min(axis=2)
    flat_q = sub.argmin(axis=2)
    ys, xs = np.nonzero(np.isfinite(flat_f))
    order = np.argsort(flat_f[ys, xs])
    best_f = float(flat_f[ys[order[0]], xs[order[0]]])
    picked = []
    used = []
    for idx in order:
        cw, ch = int(ys[idx]), int(xs[idx])
        f = float(flat_f[cw, ch])
        if f > best_f + 0.008 and len(picked) >= 1:
            break
        if any(abs(cw - pw) + abs(ch - ph) < 2 for pw, ph in used):
            continue
        q = int(flat_q[cw, ch])
        cw0, ch0, d0, q0 = cw, ch, D, q
        links = []
        while cw0 or ch0 or d0:
            g = int(PG[cw0, ch0, d0, q0])
            mode = int(PM[cw0, ch0, d0, q0])
            qp = int(PQ[cw0, ch0, d0, q0])
            links.append((g, mode))
            w, h, dd = int(G[g, 0]), int(G[g, 1]), int(G[g, 2])
            cw0, ch0, d0, q0 = cw0 - w - (mode == 1), ch0 - h - (mode == 2), d0 - dd, qp
        picked.append((f, links[::-1]))
        used.append((cw, ch))
        if len(picked) >= max_chains:
            break
    return picked


def compose_pair_rect(donors, G, g1, g2, mode):
    w1, h1, d1, f1, _, _, di1, si1 = G[g1]
    w2, h2, d2, f2, _, _, di2, si2 = G[g2]
    w1, h1, d1, di1, si1 = int(w1), int(h1), int(d1), int(di1), int(si1)
    w2, h2, d2, di2, si2 = int(w2), int(h2), int(d2), int(di2), int(si2)
    W = w1 + w2 + (1 if mode == 1 else 0)
    H = h1 + h2 + (1 if mode == 2 else 0)
    if max(W, H) > 28:
        return None
    F = max(float(f1), float(f2)) + 4 * f2d.E
    sq1, _ = f2d.crop_gadget(donors[di1], si1, w1, h1, d1)
    sq2, _ = f2d.crop_gadget(donors[di2], si2, w2, h2, d2)
    sq1 = sq1.copy()
    sq2 = sq2.copy()
    sq1[:, :2] *= 1.0 + 1e-11
    sq2[:, :2] *= 1.0 + 1e-11
    # Place g1 at top-left (x=0, y=H-h1) and g2 at bottom-right (x=w1+(mode==1), y=0)
    ox1, oy1 = 0, H - h1
    ox2, oy2 = w1 + (1 if mode == 1 else 0), 0
    sq1[:, 0] += ox1 + f2d.E
    sq1[:, 1] += oy1 + 2 * f2d.E
    sq2[:, 0] += ox2 + 2 * f2d.E
    sq2[:, 1] += oy2 + f2d.E
    # Fill remaining integer cells inside W x H rectangle cleanly
    cells = []
    for i in range(W):
        if i < w1:
            # below g1: j in [0, H-h1)
            for j in range(0, H - h1):
                cells.append((i + 0.5, j + 0.5, 0.0))
        elif mode == 1 and i == w1:
            # buffer col: j in [0, H-h1) below and [H-h1, H) above shifted by F
            for j in range(0, H - h1):
                cells.append((i + 0.5, j + 0.5, 0.0))
            for j in range(H - h1, H):
                cells.append((i + 0.5 + F, j + 0.5 + F, 0.0))
        else:
            # above g2: j in [h2, H) shifted by F
            for j in range(h2, H):
                cells.append((i + 0.5 + F, j + 0.5 + F, 0.0))
    parts = [sq1, sq2]
    if cells:
        parts.append(np.array(cells, dtype=np.float64))
    sq = np.vstack(parts)
    if len(sq) != W * H + d1 + d2:
        return None
    return W, H, d1 + d2, max(float(f1), float(f2)), sq


def worker_one(job):
    mode = job[0]
    rng = random.Random(20261125 + os.getpid() * 131 + job[-1])
    np_rng = np.random.default_rng(20261125 + os.getpid() * 137 + job[-1])
    with tempfile.TemporaryDirectory() as tmp:
        if mode == "bottleneck_shrink":
            _, w, h, dm, f0, sq_crop, _ = job
            full, M = pad_to_square(sq_crop, w, h, f0)
            s_in = M + f0 + 1e-4
            to = 32 if M <= 22 else 45
            best = None
            scheds = [
                [
                    "--loosen",
                    "1.006",
                    "--kick",
                    "0.002",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "32",
                    "--pit",
                    "220",
                ],
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
                    "220",
                ],
                [
                    "--loosen",
                    "1.032",
                    "--kick",
                    "0.015",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "32",
                    "--pit",
                    "220",
                ],
                [
                    "--loosen",
                    "1.055",
                    "--kick",
                    "0.028",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "32",
                    "--pit",
                    "200",
                ],
                [
                    "--loosen",
                    "1.0",
                    "--round-r0",
                    "0.08",
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
                ],
                [
                    "--loosen",
                    "1.0",
                    "--round-r0",
                    "0.13",
                    "--round-compress",
                    "--round-press",
                    "--round-steps",
                    "7",
                    "--round-press-outer",
                    "12",
                    "--flip-top",
                    "24",
                    "--pit",
                    "130",
                ],
            ]
            for sc in scheds:
                r = f2d.run_fq(s_in, full, tmp, sc, timeout=to)
                if r is not None and r[0] < M + f0 - 5e-5:
                    if best is None or r[0] < best[0]:
                        best = r
            # Anneal melt -> fq on stubborn bottleneck gadgets (M <= 24)
            if M <= 24 and os.path.exists(ANNEAL_BIN):
                for _ in range(2):
                    try:
                        pin = os.path.join(tmp, "min.txt")
                        pout = os.path.join(tmp, "mout.txt")
                        f2d.write_fq_cfg(pin, s_in, full)
                        if os.path.exists(pout):
                            os.remove(pout)
                        cx = rng.uniform(0.5, max(0.6, w + f0 - 0.5))
                        cy = rng.uniform(0.5, max(0.6, h + f0 - 0.5))
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
                                f"{rng.uniform(1.6, 3.8):.3f}",
                                "--rmax",
                                f"{rng.uniform(0.06, 0.16):.3f}",
                                "--sweeps",
                                "2400",
                                "--bp",
                                "2200.0",
                                "--seed",
                                str(rng.randrange(1 << 30)),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=28,
                        )
                        if os.path.exists(pout):
                            s_m, arr = f2d.read_fq_cfg_rad(pout)
                            if len(arr) == len(full):
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
                                        "32",
                                        "--pit",
                                        "220",
                                    ],
                                    timeout=to,
                                )
                                if r is not None and r[0] < M + f0 - 5e-5:
                                    if best is None or r[0] < best[0]:
                                        best = r
                    except Exception:
                        pass
            if best is not None:
                if (M + f0) - best[0] > 0.001:
                    r2 = f2d.run_fq(
                        best[0],
                        best[1],
                        tmp,
                        [
                            "--loosen",
                            "1.005",
                            "--kick",
                            "0.002",
                            "--seed",
                            str(rng.randrange(1 << 30)),
                            "--flip-top",
                            "48",
                            "--pit",
                            "300",
                        ],
                        timeout=to,
                    )
                    if r2 is not None and r2[0] < best[0]:
                        best = r2
                return (M + f0 - best[0], best[1])
            return None

        elif mode == "bottleneck_grow":
            _, w, h, dm_prev, f0, f_target, sq_crop, _ = job
            W_box = w + f0 + 0.05
            H_box = h + f0 + 0.05
            pts = np.column_stack(
                [
                    np_rng.uniform(0.5, max(0.6, W_box - 0.5), 480),
                    np_rng.uniform(0.5, max(0.6, H_box - 0.5), 480),
                ]
            )
            dmin = np.min(
                np.hypot(
                    pts[:, None, 0] - sq_crop[None, :, 0],
                    pts[:, None, 1] - sq_crop[None, :, 1],
                ),
                axis=1,
            )
            top4 = np.argsort(-dmin)[:4]
            degs = np.degrees(sq_crop[:, 2]) % 90.0
            tilted = sq_crop[np.minimum(degs, 90.0 - degs) > 3.0, 2]
            med_t = float(np.median(tilted)) if len(tilted) else math.radians(24.0)
            f_lim = min(0.968, f_target - 8e-5)
            best = None
            for idx_h, ang_try in (
                (top4[0], med_t),
                (top4[1], 0.0),
                (top4[2], math.radians(45.0)),
                (top4[3], -med_t),
            ):
                hx, hy = float(pts[idx_h, 0]), float(pts[idx_h, 1])
                sq_plus = np.vstack(
                    [sq_crop, np.array([[hx, hy, ang_try]], dtype=np.float64)]
                )
                f_init = min(0.985, max(f0, f_target) + 0.08)
                full, M = pad_to_square(sq_plus, w, h, f_init)
                s_in = M + f_init + 0.01
                r = f2d.run_fq(
                    s_in,
                    full,
                    tmp,
                    [
                        "--loosen",
                        "1.0",
                        "--round-r0",
                        "0.14",
                        "--round-compress",
                        "--round-press",
                        "--round-steps",
                        "7",
                        "--round-press-outer",
                        "12",
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "24",
                        "--pit",
                        "140",
                    ],
                    timeout=32,
                )
                if r is not None and r[0] < M + f_lim:
                    if best is None or r[0] < best[2]:
                        best = (M + f_target - r[0], r[1], r[0])
            if best is not None:
                r2 = f2d.run_fq(
                    best[2],
                    best[1],
                    tmp,
                    [
                        "--loosen",
                        "1.008",
                        "--kick",
                        "0.003",
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "36",
                        "--pit",
                        "240",
                    ],
                    timeout=30,
                )
                if r2 is not None and r2[0] < best[2]:
                    best = (max(w, h) + f_target - r2[0], r2[1], r2[0])
                return (best[0], best[1])
            return None

        else:  # bottleneck_pair
            _, W, H, dm_sum, f0, sq_pair, _ = job
            full, M = pad_to_square(sq_pair, W, H, f0)
            s_in = M + f0 + 1e-4
            to = 38 if M <= 22 else 48
            best = None
            for sc in (
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
                    "220",
                ],
                [
                    "--loosen",
                    "1.030",
                    "--kick",
                    "0.014",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "32",
                    "--pit",
                    "200",
                ],
                [
                    "--loosen",
                    "1.0",
                    "--round-r0",
                    "0.08",
                    "--round-compress",
                    "--round-press",
                    "--round-steps",
                    "5",
                    "--round-press-outer",
                    "9",
                    "--flip-top",
                    "24",
                    "--pit",
                    "120",
                ],
            ):
                r = f2d.run_fq(s_in, full, tmp, sc, timeout=to)
                if r is not None and r[0] < M + f0 - 5e-5:
                    if best is None or r[0] < best[0]:
                        best = r
            if best is not None:
                return (M + f0 - best[0], best[1])
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--prev-glob", action="append", default=[])
    ap.add_argument("--lags-json", required=True)
    ap.add_argument(
        "--mode",
        choices=("bottleneck_shrink", "bottleneck_grow", "bottleneck_pair"),
        required=True,
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=3600)
    ap.add_argument("--seed-off", type=int, default=0)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    donors, cur_L, n_base, n_prev = load_pooled_donors(
        args.union_npz, args.prev_glob, args.win_glob
    )
    G = f2d.extract_2d_gadgets(donors)
    dp, PG, PM, PQ = f2d.run_2d_dp(G)
    print(
        f"[W9A-{args.mode} s={args.shard}/{args.num_shards}] donors={len(donors)} (prev={n_prev}) -> G={len(G)}, 18-bucket 4D-DP solved ({time.time() - t0:.1f}s)",
        flush=True,
    )

    lags = json.load(open(args.lags_json)) if os.path.exists(args.lags_json) else []
    # Dynamic LIVE gap weights computed against current cur_L[n] (not stale initial Lu!)
    lag_wt = {}
    for it in lags:
        nn = int(it["n"])
        lb = float(it.get("Lb", cur_L.get(nn, 0.0)))
        live_ds = max(0.0, (cur_L.get(nn, lb) - lb) / math.sqrt(nn))
        if live_ds > 1e-7:
            lag_wt[nn] = live_ds
    grid_breaks = {605, 762, 819, 877, 1000}

    # Map best gadget index for each (w, h, dm) so bottleneck_grow can look up exact predecessors!
    best_gi = {}
    for gi in range(len(G)):
        k = (int(G[gi, 0]), int(G[gi, 1]), int(G[gi, 2]))
        if k not in best_gi or float(G[gi, 3]) < float(G[best_gi[k], 3]):
            best_gi[k] = gi

    # Extract bottleneck gadgets & pairs weighted by downstream LIVE lag dS across ALL n=80..1000!
    bn_weight = {}
    pair_weight = {}
    for n in range(80, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D == 0 or D > f2d.DMAX:
            continue
        if cur_L[n] >= K + 1 - 1e-7 and n not in grid_breaks:
            continue
        wt = lag_wt.get(n, 2e-6)
        chains = extract_top_chains(
            dp,
            PG,
            PM,
            PQ,
            G,
            K,
            D,
            max_chains=3 if (n in lag_wt or n in grid_breaks) else 1,
        )
        for f_ch, links in chains:
            if (K + f_ch) > cur_L[n] + 0.015:
                continue
            for idx_l, (g, mode) in enumerate(links):
                fg = float(G[g, 3])
                if fg >= f_ch - 0.009:
                    bn_weight[g] = bn_weight.get(g, 0.0) + wt
                if idx_l + 1 < len(links):
                    g2, mode2 = links[idx_l + 1]
                    if max(fg, float(G[g2, 3])) >= f_ch - 0.009:
                        pk = (g, g2, mode2)
                        pair_weight[pk] = pair_weight.get(pk, 0.0) + wt

    jobs = []
    if args.mode == "bottleneck_shrink":
        ranked = sorted(bn_weight.items(), key=lambda x: -x[1])
        # Also append top high-density M<=28 Pareto gadgets if fewer than 1200 unique bottlenecks
        seen_g = {g for g, _ in ranked}
        for gi in range(len(G)):
            if gi not in seen_g and 4 <= max(int(G[gi, 0]), int(G[gi, 1])) <= 28:
                ranked.append((gi, 1e-9))
        my_slice = ranked[args.shard :: args.num_shards][:160]
        for i, (gi, wt) in enumerate(my_slice):
            w, h, dm, f, _, _, di, si = G[gi]
            w, h, dm, di, si = int(w), int(h), int(dm), int(di), int(si)
            if max(w, h) > 30:
                continue
            sq_crop, _ = f2d.crop_gadget(donors[di], si, w, h, dm)
            jobs.append(
                (
                    "bottleneck_shrink",
                    w,
                    h,
                    dm,
                    float(f),
                    sq_crop,
                    args.seed_off + i * 19 + args.shard,
                )
            )

    elif args.mode == "bottleneck_grow":
        # For every bottleneck (w, h, dm), target growing (w, h, dm-1) -> (w, h, dm)!
        grow_targets = {}
        for gi, wt in bn_weight.items():
            w, h, dm, f_tgt = (
                int(G[gi, 0]),
                int(G[gi, 1]),
                int(G[gi, 2]),
                float(G[gi, 3]),
            )
            if max(w, h) > 24 or dm < 2:
                continue
            prev_gi = best_gi.get((w, h, dm - 1))
            if prev_gi is not None:
                k = (w, h, dm - 1)
                if k not in grow_targets or wt > grow_targets[k][0]:
                    grow_targets[k] = (wt, prev_gi, f_tgt)
        # Also include (w, h, dm) -> (w, h, dm+1) for high-weight bottlenecks to leapfrog D+1!
        for gi, wt in sorted(bn_weight.items(), key=lambda x: -x[1])[:400]:
            w, h, dm, f0 = int(G[gi, 0]), int(G[gi, 1]), int(G[gi, 2]), float(G[gi, 3])
            if max(w, h) > 22:
                continue
            f_next = (
                float(G[best_gi[(w, h, dm + 1)], 3])
                if (w, h, dm + 1) in best_gi
                else 0.968
            )
            if f_next > f0 + 0.008:
                k = (w, h, dm)
                if k not in grow_targets:
                    grow_targets[k] = (wt * 0.5, gi, f_next)
        ranked = sorted(grow_targets.values(), key=lambda x: -x[0])
        my_slice = ranked[args.shard :: args.num_shards][:220]
        for i, (wt, gi, f_tgt) in enumerate(my_slice):
            w, h, dm, f0, _, _, di, si = G[gi]
            w, h, dm, di, si = int(w), int(h), int(dm), int(di), int(si)
            sq_crop, _ = f2d.crop_gadget(donors[di], si, w, h, dm)
            jobs.append(
                (
                    "bottleneck_grow",
                    w,
                    h,
                    dm,
                    float(f0),
                    float(f_tgt),
                    sq_crop,
                    args.seed_off + i * 23 + args.shard,
                )
            )

    else:  # bottleneck_pair
        ranked = sorted(pair_weight.items(), key=lambda x: -x[1])
        my_slice = ranked[args.shard :: args.num_shards][:140]
        for i, ((g1, g2, mode2), wt) in enumerate(my_slice):
            comp = compose_pair_rect(donors, G, g1, g2, mode2)
            if comp is not None:
                W, H, dm_sum, f0, sq_pair = comp
                jobs.append(
                    (
                        "bottleneck_pair",
                        W,
                        H,
                        dm_sum,
                        f0,
                        sq_pair,
                        args.seed_off + i * 29 + args.shard,
                    )
                )

    print(
        f"[W9A-{args.mode} s={args.shard}] Unique bottlenecks={len(bn_weight)}, pairs={len(pair_weight)}, queued={len(jobs)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    res_all = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_one, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            if r is not None:
                res_all.append(r)
            if done % 20 == 0:
                print(
                    f"  [W9A-{args.mode} s={args.shard}] {done}/{len(jobs)} done, hits={len(res_all)} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W9A-{args.mode} s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining futures.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    res_all.sort(key=lambda x: -x[0])
    payload = {f"g_{i}": arr for i, (_, arr) in enumerate(res_all)}
    np.savez_compressed(args.out_npz, **payload)
    print(
        f"[W9A-{args.mode} s={args.shard} DONE] Saved {len(res_all)}/{done} bottleneck breakers -> {args.out_npz} ({time.time() - t0:.1f}s)"
        + (
            f", max_df={res_all[0][0]:.5f}, avg_df={sum(x[0] for x in res_all) / len(res_all):.5f}"
            if res_all
            else ""
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
