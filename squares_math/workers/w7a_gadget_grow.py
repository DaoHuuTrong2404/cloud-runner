#!/usr/bin/env python3
"""Wave 7A Engine 1 (40 Cores / 10 Shards): True `faultdp-grow` (dm -> dm + 1) + Deep Remaining 2D GadgetQ!
Takes Pareto 2D gadgets (w, h, dm, f), inserts +1 square at max-clearance hole (upright OR tilted),
pads to square [0, M + f + 0.08]^2, and runs Rust fq round-press hardening to synthesize brand-new
super-dense (w, h, dm+1) gadgets that never existed in any public packing!
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


def pad_to_square(sq_crop, w, h, f_box):
    M = max(w, h)
    pad = []
    if w < h:
        for ci in range(h - w):
            for rj in range(h):
                pad.append((w + f_box + 1e-4 + ci + 0.5, rj + 0.5, 0.0))
    elif h < w:
        for ci in range(w):
            for rj in range(w - h):
                pad.append((ci + 0.5, h + f_box + 1e-4 + rj + 0.5, 0.0))
    return (
        np.vstack([sq_crop, np.array(pad, dtype=np.float64)]) if pad else sq_crop.copy()
    ), M


def grow_batch(jobs):
    """Mode 'grow': insert +1 square at max-clearance hole and round-press harden with Rust fq!
    Mode 'deepq': deep round-press + kick quench of untouched/larger Pareto gadgets (M up to 26)!
    """
    out = []
    rng = random.Random(20261125 + os.getpid() * 47)
    np_rng = np.random.default_rng(20261125 + os.getpid() * 53)
    with tempfile.TemporaryDirectory() as tmp:
        for mode, w, h, dm, f0, f_target_next, sq_crop in jobs:
            if mode == "grow":
                # Find max-clearance hole inside [0.5, w+f0-0.5] x [0.5, h+f0-0.5]
                W_box = w + f0 + 0.04
                H_box = h + f0 + 0.04
                pts = np.column_stack(
                    [
                        np_rng.uniform(0.5, max(0.6, W_box - 0.5), 360),
                        np_rng.uniform(0.5, max(0.6, H_box - 0.5), 360),
                    ]
                )
                dmin = np.min(
                    np.hypot(
                        pts[:, None, 0] - sq_crop[None, :, 0],
                        pts[:, None, 1] - sq_crop[None, :, 1],
                    ),
                    axis=1,
                )
                top2 = np.argsort(-dmin)[:2]
                degs = np.degrees(sq_crop[:, 2]) % 90.0
                tilted = sq_crop[np.minimum(degs, 90.0 - degs) > 3.0, 2]
                med_t = float(np.median(tilted)) if len(tilted) else math.radians(24.0)

                best = None
                f_lim = min(0.965, f_target_next - 1e-4)
                for idx_h, ang_try in ((top2[0], med_t), (top2[1], 0.0)):
                    hx, hy = float(pts[idx_h, 0]), float(pts[idx_h, 1])
                    sq_plus = np.vstack(
                        [sq_crop, np.array([[hx, hy, ang_try]], dtype=np.float64)]
                    )
                    f_init = min(0.98, f0 + 0.09)
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
                            "120",
                        ],
                        timeout=26,
                    )
                    if r is not None and r[0] < M + f_lim:
                        if best is None or r[0] < best[0]:
                            best = (M + f_target_next - r[0], r[1], r[0])
                if best is not None:
                    # Quick polish on newly grown super-dense gadget!
                    r2 = f2d.run_fq(
                        best[2],
                        best[1],
                        tmp,
                        [
                            "--loosen",
                            "1.01",
                            "--kick",
                            "0.003",
                            "--seed",
                            str(rng.randrange(1 << 30)),
                            "--flip-top",
                            "32",
                            "--pit",
                            "180",
                        ],
                        timeout=24,
                    )
                    if r2 is not None and r2[0] < best[2]:
                        best = (best[0] + (best[2] - r2[0]), r2[1], r2[0])
                    out.append((best[0], best[1]))
            else:
                # Deep round-press + multi-kick quench on untouched/large Pareto gadgets!
                full, M = pad_to_square(sq_crop, w, h, f0)
                s_in = M + f0 + 1e-4
                best = None
                for args_fq in (
                    [
                        "--loosen",
                        "1.0",
                        "--round-r0",
                        "0.12",
                        "--round-compress",
                        "--round-press",
                        "--round-steps",
                        "6",
                        "--round-press-outer",
                        "10",
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "24",
                        "--pit",
                        "120",
                    ],
                    [
                        "--loosen",
                        "1.02",
                        "--kick",
                        "0.008",
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "24",
                        "--pit",
                        "140",
                    ],
                ):
                    r = f2d.run_fq(s_in, full, tmp, args_fq, timeout=26)
                    if r is not None and r[0] < M + f0 - 1e-4:
                        if best is None or r[0] < best[0]:
                            best = (M + f0 - r[0], r[1])
                if best is not None:
                    out.append(best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--mode", choices=["grow", "deepq"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--per-shard", type=int, default=360)
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
    # Build lookup of best existing f per (w, h, dm) so `grow` only keeps genuine frontier breakers!
    best_f = {}
    for gi in range(len(G)):
        w, h, dm, f = int(G[gi, 0]), int(G[gi, 1]), int(G[gi, 2]), float(G[gi, 3])
        k = (w, h, dm)
        if k not in best_f or f < best_f[k]:
            best_f[k] = f

    bands = [(0.44, 0.62), (0.62, 0.74), (0.74, 0.84), (0.84, 0.965)]
    buckets = [[] for _ in bands]
    for gi in range(len(G)):
        w, h, dm, f, ybr, xtl, di, si = G[gi]
        w, h, dm, di, si = int(w), int(h), int(dm), int(di), int(si)
        if args.mode == "grow":
            if not (3 <= max(w, h) <= 18 and dm >= 1 and f <= 0.88):
                continue
            f_next = best_f.get((w, h, dm + 1), 0.965)
            # High leverage when (w, h, dm+1) has large f_next - f!
            score = -(f_next - f) - 0.1 * (dm / math.sqrt(w * h))
        else:
            if not (4 <= max(w, h) <= 26 and dm >= 1):
                continue
            f_next = f
            # Prioritize LARGER gadgets (M >= 13) and lower-f gadgets missed by Wave 6A!
            score = -(0.5 * (max(w, h) >= 13)) - (dm / math.sqrt(w * h))
        for bi, (flo, fhi) in enumerate(bands):
            if flo <= f < fhi:
                buckets[bi].append((score, f, f_next, w, h, dm, di, si))
                break
    for b in buckets:
        b.sort()

    interleaved = []
    max_len = max((len(b) for b in buckets), default=0)
    for i in range(max_len):
        for b in buckets:
            if i < len(b):
                interleaved.append(b[i])

    # For deepq, skip first 3200 already hit in Wave 6A if smaller, or take top stratified large ones directly
    total_take = args.per_shard * args.num_shards
    my_g = [
        g
        for idx, g in enumerate(interleaved[:total_take])
        if idx % args.num_shards == args.shard
    ]
    jobs = []
    for _, f, f_next, w, h, dm, di, si in my_g:
        sq_c, _ = f2d.crop_gadget(donors[di], si, w, h, dm)
        jobs.append((args.mode, w, h, dm, f, f_next, sq_c))

    print(
        f"[W7A-{args.mode} Shard {args.shard}/{args.num_shards}] Running {len(jobs)} gadgets (from {len(G)} Pareto) on {args.procs} CPUs...",
        flush=True,
    )
    chunks = [jobs[i :: args.procs] for i in range(args.procs)]
    shrunk = []
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        for res in ex.map(grow_batch, chunks):
            shrunk.extend(res)

    shrunk.sort(key=lambda z: -z[0])
    payload = {f"g_{i}": arr.astype(np.float64) for i, (_, arr) in enumerate(shrunk)}
    np.savez_compressed(args.out_npz, **payload)
    avg_df = float(np.mean([z[0] for z in shrunk])) if shrunk else 0.0
    max_df = float(shrunk[0][0]) if shrunk else 0.0
    print(
        f"DONE W7A-{args.mode} Shard {args.shard}: {len(shrunk)}/{len(jobs)} frontier breakers ({100.0 * len(shrunk) / max(1, len(jobs)):.1f}%), avg_df={avg_df:.5f}, max_df={max_df:.5f} ({time.time() - t0:.1f}s) -> {args.out_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
