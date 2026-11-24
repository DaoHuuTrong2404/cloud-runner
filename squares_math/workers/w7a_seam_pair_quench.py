#!/usr/bin/env python3
"""Wave 7A Engine 2 (16 Cores / 4 Shards): JoeGun's `seg` Adjacent-Gadget Pair Seam Quench!
Solves 4D Staircase DP, extracts EVERY adjacent 2-gadget pair (g1, g2) actually used along
competitive staircases across n=150..1000, glues (g1, g2) with their exact staircase offset + buffer,
pads to square, and runs Rust fq directly on the seam between g1 and g2 (82% shrink rate in JoeGun)!
"""

import argparse
import math
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
sys.path.insert(0, DS_DIR)
sys.path.insert(0, HERE)

import w5b_2d_faultdp_gadgetq as f2d


def build_pair_box(g1_row, g2_row, mode, donors):
    w1, h1, dm1, f1, _, _, di1, si1 = g1_row
    w2, h2, dm2, f2, _, _, di2, si2 = g2_row
    w1, h1, dm1, di1, si1 = int(w1), int(h1), int(dm1), int(di1), int(si1)
    w2, h2, dm2, di2, si2 = int(w2), int(h2), int(dm2), int(di2), int(si2)
    f = max(float(f1), float(f2))
    W = w1 + w2 + (1 if mode == 1 else 0)
    H = h1 + h2 + (1 if mode == 2 else 0)
    if max(W, H) > 22:
        return None
    F = f + 4e-9
    sq1, _ = f2d.crop_gadget(donors[di1], si1, w1, h1, dm1)
    sq2, _ = f2d.crop_gadget(donors[di2], si2, w2, h2, dm2)
    sq1 = sq1.copy()
    sq1[:, :2] *= 1.0 + 1e-11
    sq2 = sq2.copy()
    sq2[:, :2] *= 1.0 + 1e-11
    # Place g1 top-left at (0, H - h1) and g2 bottom-right at (w1 + (mode==1), 0)
    ox2 = w1 + (1 if mode == 1 else 0)
    oy1 = H - h1
    sq1[:, 0] += 1e-9
    sq1[:, 1] += oy1 + 2e-9
    sq2[:, 0] += ox2 + 2e-9
    sq2[:, 1] += 1e-9

    # Fill lower-left upright region under g1 and upper-right F-shifted upright region above g2
    cells = []
    for i in range(w1):
        for j in range(oy1):
            cells.append((i + 0.5, j + 0.5, 0.0))
    if mode == 1:
        for j in range(H):
            cells.append((w1 + 0.5, j + 0.5, 0.0))
    for i in range(ox2, W):
        for j in range(h2, H):
            cells.append((i + 0.5 + F, j + 0.5 + F, 0.0))
    if mode == 2:
        # Extra upright row between g1 and g2 on left already covered by oy1 = h2 + 1!
        pass

    parts = [sq1, sq2]
    if cells:
        parts.append(np.array(cells, dtype=np.float64))
    comb = np.vstack(parts)
    # Pad short side to make [0, M + f]^2 square for Rust fq!
    M = max(W, H)
    pad = []
    if W < H:
        for ci in range(H - W):
            for rj in range(H):
                pad.append((W + F + 1e-4 + ci + 0.5, rj + 0.5, 0.0))
    elif H < W:
        for ci in range(W):
            for rj in range(W - H):
                pad.append((ci + 0.5, H + F + 1e-4 + rj + 0.5, 0.0))
    if pad:
        comb = np.vstack([comb, np.array(pad, dtype=np.float64)])
    return W, H, M, f, comb


def quench_pairs_worker(jobs):
    out = []
    rng = random.Random(20261130 + os.getpid() * 61)
    with tempfile.TemporaryDirectory() as tmp:
        for W, H, M, f0, comb in jobs:
            s_in = M + f0 + 2e-4
            best = None
            for args_fq in (
                [
                    "--loosen",
                    "1.012",
                    "--kick",
                    "0.004",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "24",
                    "--pit",
                    "150",
                ],
                [
                    "--loosen",
                    "1.025",
                    "--kick",
                    "0.012",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "24",
                    "--pit",
                    "150",
                ],
            ):
                r = f2d.run_fq(s_in, comb, tmp, args_fq, timeout=28)
                if r is not None and r[0] < M + f0 - 1e-4:
                    if best is None or r[0] < best[0]:
                        best = (M + f0 - r[0], r[1])
            if best is not None:
                out.append(best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    d = np.load(args.union_npz)
    x, y, t, L = d["x"], d["y"], d["t"], d["L"]
    offs = [0]
    for n in range(1, 1001):
        offs.append(offs[-1] + n)
    donors = []
    cur_L = {}
    for n in range(1, 1001):
        s0, e0 = offs[n - 1], offs[n]
        cur_L[n] = float(L[n - 1])
        if cur_L[n] < math.ceil(math.sqrt(n)) - 1e-7:
            donors.append(np.column_stack([x[s0:e0], y[s0:e0], t[s0:e0]]))

    G = f2d.extract_2d_gadgets(donors)
    dp, PG, PM, PQ = f2d.run_2d_dp(G)

    # Collect unique adjacent gadget pairs (g1, g2, mode) from all competitive DP chains across n=120..1000!
    pair_set = {}
    for n in range(120, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D == 0 or K > f2d.WMAX or D > f2d.DMAX:
            continue
        f_dp, links = f2d.extract_chain(dp, PG, PM, PQ, G, K, D)
        if f_dp is None or K + f_dp > cur_L[n] + 0.04:
            continue
        for idx in range(len(links) - 1):
            g1, _ = links[idx]
            g2, m2 = links[idx + 1]
            key = (g1, g2, m2)
            if key not in pair_set:
                pair_set[key] = (max(float(G[g1, 3]), float(G[g2, 3])), g1, g2, m2)

    pairs = sorted(pair_set.values())
    my_pairs = [
        p for idx, p in enumerate(pairs) if idx % args.num_shards == args.shard
    ][:360]
    jobs = []
    for _, g1, g2, m2 in my_pairs:
        res = build_pair_box(G[g1], G[g2], m2, donors)
        if res is not None:
            jobs.append(res)

    print(
        f"[W7A-PairSeam Shard {args.shard}/{args.num_shards}] Found {len(pairs)} unique staircase pairs; quenching {len(jobs)} on {args.procs} CPUs ({time.time() - t0:.1f}s)...",
        flush=True,
    )
    chunks = [jobs[i :: args.procs] for i in range(args.procs)]
    shrunk = []
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        for res in ex.map(quench_pairs_worker, chunks):
            shrunk.extend(res)

    shrunk.sort(key=lambda z: -z[0])
    payload = {f"g_{i}": arr.astype(np.float64) for i, (_, arr) in enumerate(shrunk)}
    np.savez_compressed(args.out_npz, **payload)
    avg_df = float(np.mean([z[0] for z in shrunk])) if shrunk else 0.0
    max_df = float(shrunk[0][0]) if shrunk else 0.0
    print(
        f"DONE W7A-PairSeam Shard {args.shard}: {len(shrunk)}/{len(jobs)} pair seams shrank ({100.0 * len(shrunk) / max(1, len(jobs)):.1f}%), avg_df={avg_df:.5f}, max_df={max_df:.5f} ({time.time() - t0:.1f}s) -> {args.out_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
