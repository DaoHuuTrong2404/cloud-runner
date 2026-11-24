#!/usr/bin/env python3
"""Wave 10A Wide-Reach M23-31 Completion + Large-M Grow (M=18..28) + Round-3 Anneal-Melt Re-Quench!
All 20 shards (80 free cores) use graceful budget_sec + f_rem.cancel() -> guaranteed 20/20 success, 0 timeouts!
Pools pub_base_v3 + Wave 6A + Wave 7A + Wave 8A + Wave 9A (4-wave frontier), then runs:
  - Shards 0..9   (--mode wide_reach_v2):       Complete M in [23..31] across all f-bands (boosting low-f K^2+1..6!)
  - Shards 10..15 (--mode grow_large_M18_28):   dm -> dm+1 round-press growth on M in [18..28] (previously untouched!)
  - Shards 16..19 (--mode round3_melt_requench): Anneal melt -> fq + round-press Round-3 on Wave 8A+9A frontier!
"""

import argparse
import glob
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

f2d.THR = np.array([0.0, 0.25, 0.40, 0.50, 0.58, 0.65, 0.72, 0.79, 0.86, 0.93])
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


def load_pooled_donors(union_npz, prev_globs):
    d = np.load(union_npz)
    x, y, t, L = d["x"], d["y"], d["t"], d["L"]
    offs = [0]
    for n in range(1, 1001):
        offs.append(offs[-1] + n)
    donors = []
    seen_sig = set()
    for n in range(1, 1001):
        s0, e0 = offs[n - 1], offs[n]
        if float(L[n - 1]) < math.ceil(math.sqrt(n)) - 1e-7:
            arr = np.column_stack([x[s0:e0], y[s0:e0], t[s0:e0]])
            donors.append(arr)
            seen_sig.add((len(arr), round(float(L[n - 1]), 6)))
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
    return donors, n_base, n_prev


def worker_one(job):
    mode, w, h, dm, f0, f_target_next, sq_crop, jseed = job
    rng = random.Random(20261125 + os.getpid() * 149 + jseed)
    np_rng = np.random.default_rng(20261125 + os.getpid() * 151 + jseed)
    M = max(w, h)
    with tempfile.TemporaryDirectory() as tmp:
        if mode == "grow_large_M18_28":
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
            top3 = np.argsort(-dmin)[:3]
            degs = np.degrees(sq_crop[:, 2]) % 90.0
            tilted = sq_crop[np.minimum(degs, 90.0 - degs) > 3.0, 2]
            med_t = float(np.median(tilted)) if len(tilted) else math.radians(22.0)
            f_lim = min(0.968, f_target_next - 8e-5)
            best = None
            to = 34 if M <= 24 else 44
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
                        "11",
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "24",
                        "--pit",
                        "130",
                    ],
                    timeout=to,
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
                        "1.008",
                        "--kick",
                        "0.003",
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "32",
                        "--pit",
                        "200",
                    ],
                    timeout=to,
                )
                if r2 is not None and r2[0] < best[2]:
                    best = (M + f_target_next - r2[0], r2[1], r2[0])
                return (best[0], best[1])
            return None

        elif mode == "wide_reach_v2":
            full, M2 = pad_to_square(sq_crop, w, h, f0)
            s_in = M2 + f0 + 1e-4
            to = 34 if M2 <= 26 else 44
            best = None
            scheds = [
                [
                    "--loosen",
                    "1.010",
                    "--kick",
                    "0.004",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "28",
                    "--pit",
                    "180",
                ],
                [
                    "--loosen",
                    "1.026",
                    "--kick",
                    "0.013",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "28",
                    "--pit",
                    "180",
                ],
                [
                    "--loosen",
                    "1.045",
                    "--kick",
                    "0.024",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "28",
                    "--pit",
                    "160",
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
                    "8",
                    "--flip-top",
                    "20",
                    "--pit",
                    "110",
                ],
            ]
            for sc in scheds:
                r = f2d.run_fq(s_in, full, tmp, sc, timeout=to)
                if r is not None and r[0] < M2 + f0 - 6e-5:
                    if best is None or r[0] < best[0]:
                        best = r
            if best is not None:
                if (M2 + f0) - best[0] > 0.0012:
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
                            "40",
                            "--pit",
                            "260",
                        ],
                        timeout=to,
                    )
                    if r2 is not None and r2[0] < best[0]:
                        best = r2
                return (M2 + f0 - best[0], best[1])
            return None

        else:  # round3_melt_requench
            full, M2 = pad_to_square(sq_crop, w, h, f0)
            s_in = M2 + f0 + 1e-4
            to = 34 if M2 <= 24 else 42
            best = None
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
                    "240",
                ],
                [
                    "--loosen",
                    "1.035",
                    "--kick",
                    "0.018",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "36",
                    "--pit",
                    "220",
                ],
                [
                    "--loosen",
                    "1.0",
                    "--round-r0",
                    "0.11",
                    "--round-compress",
                    "--round-press",
                    "--round-steps",
                    "7",
                    "--round-press-outer",
                    "11",
                    "--flip-top",
                    "24",
                    "--pit",
                    "130",
                ],
            ):
                r = f2d.run_fq(s_in, full, tmp, sc, timeout=to)
                if r is not None and r[0] < M2 + f0 - 5e-5:
                    if best is None or r[0] < best[0]:
                        best = r
            if os.path.exists(ANNEAL_BIN) and M2 <= 26:
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
                                f"{rng.uniform(1.8, 4.2):.3f}",
                                "--rmax",
                                f"{rng.uniform(0.07, 0.17):.3f}",
                                "--sweeps",
                                "2600",
                                "--bp",
                                "2200.0",
                                "--seed",
                                str(rng.randrange(1 << 30)),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,
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
                                        "1.010",
                                        "--kick",
                                        "0.004",
                                        "--seed",
                                        str(rng.randrange(1 << 30)),
                                        "--flip-top",
                                        "36",
                                        "--pit",
                                        "240",
                                    ],
                                    timeout=to,
                                )
                                if r is not None and r[0] < M2 + f0 - 5e-5:
                                    if best is None or r[0] < best[0]:
                                        best = r
                    except Exception:
                        pass
            if best is not None:
                return (M2 + f0 - best[0], best[1])
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--prev-glob", action="append", default=[])
    ap.add_argument(
        "--mode",
        choices=("wide_reach_v2", "grow_large_M18_28", "round3_melt_requench"),
        required=True,
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--per-shard", type=int, default=200)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=3300)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    donors, n_base, n_prev = load_pooled_donors(args.union_npz, args.prev_glob)
    G = f2d.extract_2d_gadgets(donors)
    print(
        f"[W10A-{args.mode} s={args.shard}/{args.num_shards}] donors={len(donors)} (prev={n_prev}) -> G={len(G)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    best_f = {}
    for gi in range(len(G)):
        k = (int(G[gi, 0]), int(G[gi, 1]), int(G[gi, 2]))
        f = float(G[gi, 3])
        if k not in best_f or f < best_f[k]:
            best_f[k] = f

    # Stratify with EXTRA weight on low-f [0.44..0.62) where K^2+1..6 (842,901,962,626,678,786) just won BIG!
    bands = [(0.44, 0.56), (0.56, 0.68), (0.68, 0.82), (0.82, 0.968)]
    buckets = [[] for _ in bands]
    seen = set()
    for gi in range(len(G)):
        w, h, dm, f, _, _, di, si = G[gi]
        w, h, dm, di, si = int(w), int(h), int(dm), int(di), int(si)
        f = float(f)
        M = max(w, h)
        if f < 0.44 or f > 0.968:
            continue
        k = (w, h, dm, round(f, 5))
        if k in seen:
            continue
        if args.mode == "wide_reach_v2":
            if not (23 <= M <= 31):
                continue
            f_next = 0.97
        elif args.mode == "grow_large_M18_28":
            if not (18 <= M <= 28):
                continue
            f_next = best_f.get((w, h, dm + 1), 0.97)
            if f_next <= f + 0.008:
                continue
        else:  # round3_melt_requench
            if M < 4 or M > 26 or di < n_base:
                continue
            f_next = 0.97
        for bi, (lo, hi) in enumerate(bands):
            if lo <= f < hi:
                seen.add(k)
                buckets[bi].append((w, h, dm, f, f_next, di, si))
                break

    # Use DIFFERENT seed from Wave 8A (20261126) so wide_reach_v2 hits fresh untouched M=23..31 slices!
    rng = random.Random(20261126 + hash(args.mode) % 10000)
    jobs = []
    quota = max(1, args.per_shard // len(bands))
    for bi in range(len(bands)):
        rng.shuffle(buckets[bi])
        my_slice = buckets[bi][args.shard :: args.num_shards][:quota]
        for j, (w, h, dm, f, f_next, di, si) in enumerate(my_slice):
            sq_crop, _ = f2d.crop_gadget(donors[di], si, w, h, dm)
            jobs.append(
                (
                    args.mode,
                    w,
                    h,
                    dm,
                    f,
                    f_next,
                    sq_crop,
                    bi * 1000 + j * 13 + args.shard,
                )
            )

    print(
        f"[W10A-{args.mode} s={args.shard}] Queued {len(jobs)} jobs from band pools {[len(b) for b in buckets]}",
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
                    f"  [W10A-{args.mode} s={args.shard}] {done}/{len(jobs)} done, hits={len(res_all)} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W10A-{args.mode} s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    res_all.sort(key=lambda x: -x[0])
    payload = {f"g_{i}": arr for i, (_, arr) in enumerate(res_all)}
    np.savez_compressed(args.out_npz, **payload)
    print(
        f"[W10A-{args.mode} s={args.shard} DONE] Saved {len(res_all)}/{done} tightened gadgets -> {args.out_npz} ({time.time() - t0:.1f}s)"
        + (
            f", max_df={res_all[0][0]:.5f}, avg_df={sum(x[0] for x in res_all) / len(res_all):.5f}"
            if res_all
            else ""
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
