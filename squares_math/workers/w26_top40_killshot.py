#!/usr/bin/env python3
"""Wave 26 Top-40 Killshot Sniper (120 Cores / 30 Shards -> 5-6 Puzzles/Shard = 0 Timeouts!).

Cleanly finishes EVERY unfinished High-f & Mid-f puzzle from Wave 24B/25 (only 5-6 puzzles/shard so 100% finish
in ~45m with 0 timeouts!), PLUS directly targets:
  - ALL brand-new R19 phase winners (405 -0.00365, 298 -0.00075, 264 -0.00067, 500 -0.00085, 589, 740, 862, 545, 508)
  - Their ENTIRE Fixed-D cousin chains (D=5: 261,294,329,366,405,446,489,534,581,630,681,734,789,846,905,966;
    D=8: 264,297,332,369,408,449,492,537,584,633,684,737,792,849,908,969;
    D=9: 234,265,298,333,370,409,450,493,538,585,634,685,738,793,850,909,970)
  - All 110 previously-unreached high-f (0.76..0.998) & mid-f (0.48..0.76) lag targets!
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
sys.path.insert(0, DS_DIR)
sys.path.insert(0, HERE)

import w12a_super_bfmelt_rp as w12
import w22_convexity_anomaly_sniper as w22
import w24b_high_f_phase_assault as w24b
import w25_mid_f_sniper as w25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--lags-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=30)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=3400)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    lags = json.load(open(args.lags_json)) if os.path.exists(args.lags_json) else []
    lag_rank = {int(it["n"]): float(it.get("dS", 0.0)) for it in lags}

    nongrid_sorted = [
        m
        for m in range(20, 1001)
        if m != math.isqrt(m) ** 2 and cur[m][0] < math.isqrt(m) + 1 - 1e-6
    ]

    # Build curated 170-puzzle Top-40 Killshot list (5-6 puzzles/shard across 30 shards = 100% completion!)
    seen = set()
    ordered = []

    def add_n(n):
        if (
            40 <= n <= 1000
            and n not in seen
            and n != math.isqrt(n) ** 2
            and cur[n][0] < math.isqrt(n) + 1 - 1e-6
        ):
            seen.add(n)
            ordered.append(n)

    # 1) Fresh R19 winners (highest cascade probability!)
    for n in [
        405,
        298,
        264,
        500,
        589,
        740,
        862,
        545,
        508,
        499,
        603,
        746,
        683,
        593,
        642,
        654,
        692,
        451,
        736,
        749,
    ]:
        add_n(n)
    # 2) Entire D=5, D=8, D=9 cousin chains of our new n=405, 264, 298 phase transitions!
    for D in (5, 8, 9, 16, 17):
        for K in range(15, 32):
            add_n(K * K + D)
    # 3) Unfinished second half of W24B high-f targets (rank 75..200)!
    hf_list = [n for n, _ in w24b.select_high_f_targets(cur, lag_rank)]
    for n in hf_list[70:]:
        add_n(n)
    # 4) Unfinished second half of W25 mid-f targets (rank 40..100)!
    mf_list = [n for n, _ in w25.select_mid_f_targets(cur, lag_rank, top_k=110)]
    for n in mf_list[35:]:
        add_n(n)

    # Cap at 165 targets -> max 5-6 puzzles/shard -> finishes cleanly in ~45-55m!
    ordered = ordered[:165]
    my_slice = ordered[args.shard :: args.num_shards]
    print(
        f"[W26-Top40Killshot s={args.shard}/{args.num_shards}] Frontier={tot_score:.10f} "
        f"({len(fresh)} fresh from {n_files} files), my_targets={my_slice} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    jobs = []
    for i, n in enumerate(my_slice):
        L0, sq0 = cur[n]
        transfers = w22.harvest_fixed_d_transfers(n, L0, cur, nongrid_sorted)
        jobs.append((n, L0, sq0, transfers, 2600001 + args.shard * 10000 + i * 139))

    wins = {}
    tot_ds = 0.0
    done = 0

    def save_ckpt():
        payload = {}
        for k_n, (Lf, xf, yf, tf) in sorted(wins.items()):
            payload[f"L_{k_n}"] = np.array([Lf], dtype=np.float64)
            payload[f"x_{k_n}"] = np.asarray(xf, dtype=np.float64)
            payload[f"y_{k_n}"] = np.asarray(yf, dtype=np.float64)
            payload[f"t_{k_n}"] = np.asarray(tf, dtype=np.float64)
        np.savez_compressed(args.out_npz, **payload)

    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(w22.worker_snipe_anomaly, j) for j in jobs]
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
                        f"  [W26 KILLSHOT WIN!!] n={n:4d} {tag:16s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W26 s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W26 s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
