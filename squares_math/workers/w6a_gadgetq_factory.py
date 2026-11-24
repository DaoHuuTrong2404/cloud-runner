#!/usr/bin/env python3
"""Wave 6A Engine 1 (64 Cores / 16 Shards): Stratified 2D Pareto Gadget-Quench Factory (gadgetq).
Stratifies across all 4 overhang bands f in [0.45,0.62), [0.62,0.74), [0.74,0.84), [0.84,0.96]
and all sizes 2 <= max(w,h) <= 22, shrinks gadgets via padded-square Rust fq, and saves ALL
tighter gadget-donors for unified pooled 4D Staircase DP in Wave 6B!
"""

import argparse
import json
import math
import os
import random
import subprocess
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

FQ_BIN = os.environ.get("FQ_BIN", "/tmp/sq_bins/fq")


def quench_batch(jobs):
    out = []
    rng = random.Random(20261124 + os.getpid() * 31)
    with tempfile.TemporaryDirectory() as tmp:
        for w, h, dm, f0, sq_crop in jobs:
            M = max(w, h)
            pad = []
            if w < h:
                for ci in range(h - w):
                    for rj in range(h):
                        pad.append((w + f0 + 1e-5 + ci + 0.5, rj + 0.5, 0.0))
            elif h < w:
                for ci in range(w):
                    for rj in range(w - h):
                        pad.append((ci + 0.5, h + f0 + 1e-5 + rj + 0.5, 0.0))
            full = (
                np.vstack([sq_crop, np.array(pad, dtype=np.float64)])
                if pad
                else sq_crop.copy()
            )
            s_in = M + f0 + 1e-4
            best = None
            scheds = [
                [
                    "--loosen",
                    "1.015",
                    "--kick",
                    "0.005",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "24",
                    "--pit",
                    "140",
                ],
                [
                    "--loosen",
                    "1.03",
                    "--kick",
                    "0.015",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "24",
                    "--pit",
                    "140",
                ],
            ]
            if M <= 12:
                scheds.append(
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
                        "8",
                        "--flip-top",
                        "16",
                        "--pit",
                        "80",
                    ]
                )
            for args_fq in scheds:
                r = f2d.run_fq(s_in, full, tmp, args_fq, timeout=22)
                if r is not None and r[0] < M + f0 - 1e-4:
                    if best is None or r[0] < best[0]:
                        best = r
            if best is not None:
                # Extra quick follow-up polish on big shrinks (> 0.002)
                if (M + f0) - best[0] > 0.002:
                    r2 = f2d.run_fq(
                        best[0],
                        best[1],
                        tmp,
                        [
                            "--loosen",
                            "1.008",
                            "--kick",
                            "0.002",
                            "--seed",
                            str(rng.randrange(1 << 30)),
                            "--flip-top",
                            "32",
                            "--pit",
                            "200",
                        ],
                        timeout=22,
                    )
                    if r2 is not None and r2[0] < best[0]:
                        best = r2
                out.append((M + f0 - best[0], best[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--per-shard", type=int, default=420)
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
    for n in range(1, 1001):
        s0, e0 = offs[n - 1], offs[n]
        if float(L[n - 1]) < math.ceil(math.sqrt(n)) - 1e-7:
            donors.append(np.column_stack([x[s0:e0], y[s0:e0], t[s0:e0]]))

    G = f2d.extract_2d_gadgets(donors)
    print(
        f"[W6A-GQ Shard {args.shard}/{args.num_shards}] Extracted {len(G)} Pareto 2D gadgets ({time.time() - t0:.1f}s)",
        flush=True,
    )

    # Stratify evenly across 4 overhang f-bands so low-f (K^2+1..6) AND medium-f AND high-f ALL get quenched!
    bands = [(0.44, 0.62), (0.62, 0.74), (0.74, 0.84), (0.84, 0.965)]
    buckets = [[] for _ in bands]
    for gi in range(len(G)):
        w, h, dm, f, ybr, xtl, di, si = G[gi]
        w, h, dm, di, si = int(w), int(h), int(dm), int(di), int(si)
        if not (2 <= max(w, h) <= 22 and dm >= 1):
            continue
        for bi, (flo, fhi) in enumerate(bands):
            if flo <= f < fhi:
                # Prioritize high surplus density and larger leverage gadgets
                score = -(dm / math.sqrt(w * h)) - 0.15 * (min(w, h) >= 4)
                buckets[bi].append((score, f, w, h, dm, di, si))
                break
    for b in buckets:
        b.sort()

    # Round-robin interleave across the 4 f-bands!
    interleaved = []
    max_len = max((len(b) for b in buckets), default=0)
    for i in range(max_len):
        for b in buckets:
            if i < len(b):
                interleaved.append(b[i])

    total_take = args.per_shard * args.num_shards
    my_g = [
        g
        for idx, g in enumerate(interleaved[:total_take])
        if idx % args.num_shards == args.shard
    ]
    q_jobs = []
    for _, f, w, h, dm, di, si in my_g:
        sq_c, _ = f2d.crop_gadget(donors[di], si, w, h, dm)
        q_jobs.append((w, h, dm, f, sq_c))

    print(
        f"[W6A-GQ Shard {args.shard}] Quenching {len(q_jobs)} stratified gadgets (band sizes {[len(b) for b in buckets]}) on {args.procs} CPUs...",
        flush=True,
    )
    chunks = [q_jobs[i :: args.procs] for i in range(args.procs)]
    shrunk = []
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        for res in ex.map(quench_batch, chunks):
            shrunk.extend(res)

    shrunk.sort(key=lambda z: -z[0])
    payload = {}
    for idx, (df, arr) in enumerate(shrunk):
        payload[f"g_{idx}"] = arr.astype(np.float64)
    np.savez_compressed(args.out_npz, **payload)
    avg_df = float(np.mean([z[0] for z in shrunk])) if shrunk else 0.0
    max_df = float(shrunk[0][0]) if shrunk else 0.0
    print(
        f"DONE W6A-GQ Shard {args.shard}: {len(shrunk)}/{len(q_jobs)} shrank ({100.0 * len(shrunk) / max(1, len(q_jobs)):.1f}%), avg_df={avg_df:.5f}, max_df={max_df:.5f} ({time.time() - t0:.1f}s) -> {args.out_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
