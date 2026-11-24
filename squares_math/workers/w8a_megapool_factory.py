#!/usr/bin/env python3
"""Wave 8A Mega-Pool Gadget Factory (80 Cores / 20 Shards on public pure-math data):
Pools pub_base_v3 + ALL Wave 6A (4,952 shrunk) + ALL Wave 7A (grow+deepq+pair) gadgets, then runs:
  - Shards 0..9   (--mode wide_reach):      Untouched large-reach Pareto gadgets 23 <= M <= 31 (n=530..1000!)
  - Shards 10..15 (--mode round2_requench): Round-2 iterative deep re-quench of top pooled Pareto frontier
  - Shards 16..19 (--mode grow_v2):         faultdp-grow (dm -> dm+1) on newly shrunk Wave 6A+7A frontier
"""

import argparse
import glob
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
                pad.append((w + f_box + 2e-5 + ci + 0.5, rj + 0.5, 0.0))
    elif h < w:
        for ci in range(w):
            for rj in range(w - h):
                pad.append((ci + 0.5, h + f_box + 2e-5 + rj + 0.5, 0.0))
    return (
        np.vstack([sq_crop, np.array(pad, dtype=np.float64)]) if pad else sq_crop.copy()
    ), M


def load_pooled_donors(union_npz, prev_globs):
    d = np.load(union_npz)
    x, y, t, L = d["x"], d["y"], d["t"], d["L"]
    offs = [0]
    for n in range(1, 1001):
        offs.append(offs[-1] + n)
    donors = []
    for n in range(1, 1001):
        s0, e0 = offs[n - 1], offs[n]
        if float(L[n - 1]) < math.ceil(math.sqrt(n)) - 1e-7:
            donors.append(np.column_stack([x[s0:e0], y[s0:e0], t[s0:e0]]))
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
                    donors.append(arr.astype(np.float64))
                    n_prev += 1
    return donors, n_base, n_prev


def worker_batch(jobs):
    out = []
    rng = random.Random(20261125 + os.getpid() * 83)
    np_rng = np.random.default_rng(20261125 + os.getpid() * 97)
    with tempfile.TemporaryDirectory() as tmp:
        for mode, w, h, dm, f0, f_target_next, sq_crop in jobs:
            M = max(w, h)
            if mode == "grow_v2":
                W_box = w + f0 + 0.04
                H_box = h + f0 + 0.04
                pts = np.column_stack(
                    [
                        np_rng.uniform(0.5, max(0.6, W_box - 0.5), 420),
                        np_rng.uniform(0.5, max(0.6, H_box - 0.5), 420),
                    ]
                )
                dmin = np.min(
                    np.hypot(
                        pts[:, None, 0] - sq_crop[None, :, 0],
                        pts[:, None, 1] - sq_crop[None, :, 1],
                    ),
                    axis=1,
                )
                top3 = np.argsort(-dmin)[:3]
                degs = np.degrees(sq_crop[:, 2]) % 90.0
                tilted = sq_crop[np.minimum(degs, 90.0 - degs) > 3.0, 2]
                med_t = float(np.median(tilted)) if len(tilted) else math.radians(24.0)
                f_lim = min(0.965, f_target_next - 1e-4)
                best = None
                for idx_h, ang_try in (
                    (top3[0], med_t),
                    (top3[1], 0.0),
                    (top3[2], math.radians(45.0)),
                ):
                    hx, hy = float(pts[idx_h, 0]), float(pts[idx_h, 1])
                    sq_plus = np.vstack(
                        [sq_crop, np.array([[hx, hy, ang_try]], dtype=np.float64)]
                    )
                    f_init = min(0.985, f0 + 0.09)
                    full, M2 = pad_to_square(sq_plus, w, h, f_init)
                    s_in = M2 + f_init + 0.01
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
                            "130",
                        ],
                        timeout=26,
                    )
                    if r is not None and r[0] < M2 + f_lim:
                        if best is None or r[0] < best[2]:
                            best = (M2 + f_target_next - r[0], r[1], r[0])
                if best is not None:
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
                            "200",
                        ],
                        timeout=24,
                    )
                    if r2 is not None and r2[0] < best[2]:
                        best = (M + f_target_next - r2[0], r2[1], r2[0])
                    out.append((best[0], best[1]))
            else:
                # wide_reach or round2_requench
                full, M2 = pad_to_square(sq_crop, w, h, f0)
                s_in = M2 + f0 + 1e-4
                best = None
                if mode == "wide_reach":
                    to = 34 if M2 <= 26 else 44
                    scheds = [
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
                            "1.028",
                            "--kick",
                            "0.014",
                            "--seed",
                            str(rng.randrange(1 << 30)),
                            "--flip-top",
                            "24",
                            "--pit",
                            "150",
                        ],
                        [
                            "--loosen",
                            "1.0",
                            "--round-r0",
                            "0.07",
                            "--round-compress",
                            "--round-press",
                            "--round-steps",
                            "5",
                            "--round-press-outer",
                            "8",
                            "--flip-top",
                            "20",
                            "--pit",
                            "100",
                        ],
                    ]
                else:
                    # round2_requench: deeper kick + round-press on already-shrunk Pareto winners!
                    to = 30
                    scheds = [
                        [
                            "--loosen",
                            "1.008",
                            "--kick",
                            "0.003",
                            "--seed",
                            str(rng.randrange(1 << 30)),
                            "--flip-top",
                            "32",
                            "--pit",
                            "220",
                        ],
                        [
                            "--loosen",
                            "1.022",
                            "--kick",
                            "0.010",
                            "--seed",
                            str(rng.randrange(1 << 30)),
                            "--flip-top",
                            "32",
                            "--pit",
                            "220",
                        ],
                        [
                            "--loosen",
                            "1.045",
                            "--kick",
                            "0.022",
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
                            "0.09",
                            "--round-compress",
                            "--round-press",
                            "--round-steps",
                            "6",
                            "--round-press-outer",
                            "10",
                            "--flip-top",
                            "24",
                            "--pit",
                            "120",
                        ],
                    ]
                for sc in scheds:
                    r = f2d.run_fq(s_in, full, tmp, sc, timeout=to)
                    if r is not None and r[0] < M2 + f0 - 8e-5:
                        if best is None or r[0] < best[0]:
                            best = r
                if best is not None:
                    if (M2 + f0) - best[0] > 0.0015:
                        r2 = f2d.run_fq(
                            best[0],
                            best[1],
                            tmp,
                            [
                                "--loosen",
                                "1.006",
                                "--kick",
                                "0.002",
                                "--seed",
                                str(rng.randrange(1 << 30)),
                                "--flip-top",
                                "40",
                                "--pit",
                                "260",
                            ],
                            timeout=to,
                        )
                        if r2 is not None and r2[0] < best[0]:
                            best = r2
                    out.append((M2 + f0 - best[0], best[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--prev-glob", action="append", default=[])
    ap.add_argument(
        "--mode", choices=("wide_reach", "round2_requench", "grow_v2"), required=True
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--per-shard", type=int, default=260)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    donors, n_base, n_prev = load_pooled_donors(args.union_npz, args.prev_glob)
    G = f2d.extract_2d_gadgets(donors)
    print(
        f"[W8A-{args.mode} s={args.shard}/{args.num_shards}] donors={len(donors)} (base={n_base}, prev={n_prev}) -> {len(G)} Pareto 2D gadgets ({time.time() - t0:.1f}s)",
        flush=True,
    )

    # Map best f for each (w, h, dm) so grow_v2 knows target f_next for (w, h, dm+1)
    best_f = {}
    for gi in range(len(G)):
        w, h, dm, f = int(G[gi, 0]), int(G[gi, 1]), int(G[gi, 2]), float(G[gi, 3])
        k = (w, h, dm)
        if k not in best_f or f < best_f[k]:
            best_f[k] = f

    bands = [(0.44, 0.62), (0.62, 0.74), (0.74, 0.84), (0.84, 0.965)]
    buckets = [[] for _ in bands]
    seen = set()
    for gi in range(len(G)):
        w, h, dm, f, _, _, di, si = G[gi]
        w, h, dm, di, si = int(w), int(h), int(dm), int(di), int(si)
        f = float(f)
        M = max(w, h)
        if f < 0.44 or f > 0.965:
            continue
        k = (w, h, dm, round(f, 5))
        if k in seen:
            continue
        if args.mode == "wide_reach":
            if not (23 <= M <= 31):
                continue
            f_next = 0.97
        elif args.mode == "round2_requench":
            # Prioritize gadgets that came from Wave 6A/7A (di >= n_base) OR tight M<=24 frontier!
            if M < 3 or M > 24:
                continue
            if di < n_base and M > 16:
                continue
            f_next = 0.97
        else:  # grow_v2
            if M < 3 or M > 22 or di < n_base:
                continue
            f_next = best_f.get((w, h, dm + 1), 0.97)
            if f_next <= f + 0.008:
                continue
        for bi, (lo, hi) in enumerate(bands):
            if lo <= f < hi:
                seen.add(k)
                buckets[bi].append((w, h, dm, f, f_next, di, si))
                break

    rng = random.Random(20261125 + hash(args.mode) % 10000)
    selected = []
    quota = max(1, args.per_shard // len(bands))
    for bi in range(len(bands)):
        rng.shuffle(buckets[bi])
        my_slice = buckets[bi][args.shard :: args.num_shards][:quota]
        for w, h, dm, f, f_next, di, si in my_slice:
            sq_crop, _ = f2d.crop_gadget(donors[di], si, w, h, dm)
            selected.append((args.mode, w, h, dm, f, f_next, sq_crop))

    print(
        f"[W8A-{args.mode} s={args.shard}] Queued {len(selected)} gadgets from pool sizes {[len(b) for b in buckets]}",
        flush=True,
    )

    batches = [selected[i::16] for i in range(16) if selected[i::16]]
    res_all = []
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        for out in ex.map(worker_batch, batches):
            res_all.extend(out)
    res_all.sort(key=lambda x: -x[0])

    payload = {f"g_{i}": arr for i, (_, arr) in enumerate(res_all)}
    np.savez_compressed(args.out_npz, **payload)
    print(
        f"[W8A-{args.mode} s={args.shard} DONE] Saved {len(res_all)}/{len(selected)} tightened gadgets -> {args.out_npz} ({time.time() - t0:.1f}s)"
        + (
            f", max_df={res_all[0][0]:.5f}, avg_df={sum(x[0] for x in res_all) / len(res_all):.5f}"
            if res_all
            else ""
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
