#!/usr/bin/env python3
"""Wave 25 Mid-f (0.50 <= f < 0.76) Top-80 Lag Triple-Engine Sniper (40 Cores / 10 Shards -> 8 Puzzles/Shard!).

Complements Wave 24B (which covers f >= 0.76) by hitting the Top-80 biggest-gap Mid-f packings (0.50 <= f < 0.76,
n >= 200 — the exact regime of n=736 -0.00198, n=539 -0.00187, n=451 -0.00214, n=785 -0.00132, n=582 -0.00131,
n=491 -0.00094, n=683 -0.00077!) with the complete Wave 22 Triple-Engine Sniper:
  1) Dense 14-Point Round-Press Sweep (r0=0.03..0.34, L0+0.008..0.032) + 6-Step Kick Chain
  2) Fixed-D Neighbor Stretch/Peel + Global Sharp Peels + Wall Delete
  3) 4-Round Box-Expanded Single/Dual/Triple-Center Melt + Round-Press Re-Crystallization + Contact LP + Decimal64!
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


def select_mid_f_targets(cur, lag_rank, top_k=80):
    scored = []
    for n in range(180, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D == 0:
            continue
        Ln = cur[n][0]
        fn = Ln - K
        if 0.48 <= fn < 0.76:
            # Combine lag_rank + Fixed-K bump + Fixed-D bump!
            bump_k = 0.0
            if (
                D > 1
                and n + 1 < (K + 1) * (K + 1)
                and cur[n - 1][0] < K + 1 - 1e-6
                and cur[n + 1][0] < K + 1 - 1e-6
            ):
                bump_k = max(
                    0.0, Ln - 0.5 * (cur[n - 1][0] + cur[n + 1][0])
                ) / math.sqrt(n)
            prio = (
                lag_rank.get(n, 0.0) + 1.5 * bump_k + (0.00010 if fn >= 0.60 else 0.0)
            )
            scored.append((prio, n, fn))
    scored.sort(reverse=True)
    return [(n, fn) for _, n, fn in scored[:top_k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--lags-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=3450)
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
    targets = select_mid_f_targets(cur, lag_rank, top_k=80)
    my_slice = targets[args.shard :: args.num_shards]
    print(
        f"[W25-MidFSniper s={args.shard}/{args.num_shards}] Frontier={tot_score:.10f} "
        f"({len(fresh)} fresh from {n_files} files), my_targets={[(n, round(f, 3)) for n, f in my_slice]} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    jobs = []
    for i, (n, fn) in enumerate(my_slice):
        L0, sq0 = cur[n]
        transfers = w22.harvest_fixed_d_transfers(n, L0, cur, nongrid_sorted)
        jobs.append((n, L0, sq0, transfers, 2500001 + args.shard * 10000 + i * 131))

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
                        f"  [W25 MID-F WIN!!] n={n:4d} {tag:16s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W25 s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W25 s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
