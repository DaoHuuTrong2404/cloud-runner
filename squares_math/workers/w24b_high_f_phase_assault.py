#!/usr/bin/env python3
"""Wave 24B High-f (0.76 <= f <= 0.998) Phase-Transition Sniper (72 Cores / 18 Shards -> ~10 Puzzles/Shard!).

Empirical law across Waves 15-22:
  Nearly EVERY massive -0.002 to -0.007 phase transition (n=642 -0.00666, n=593 -0.00610, n=500 -0.00339,
  n=545 -0.00304, n=547 -0.00291, n=499 -0.00270, n=654 -0.00269, n=603 -0.00209, n=595 -0.00203,
  n=371 -0.00198, n=692 -0.00120, n=746 -0.00110, n=932 -0.00104) lives in f = L - K in [0.76, 0.998]!
  Moreover, packings near the ceiling (f in [0.90, 0.998]) were skipped by Fixed-D bump_D because (K-1)^2+D > K^2!

Wave 24B sweeps EVERY non-grid puzzle n >= 220 with f in [0.76, 0.999] (~190 puzzles -> ~10 puzzles/shard!)
plus top-30 Fixed-K convexity bumps, applying the complete Wave 22 winning arsenal (Dense 14-point Round-Press
r0=0.03..0.34 + Fixed-D/Global Peels & Stretches & Wall-Deletes + 4-Round Box-Expanded Multi-Center Melt +
Contact LP + Decimal64 >= 1.05e-9)!
"""

import argparse
import glob
import json
import math
import os
import random
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


def select_high_f_targets(cur, lag_rank):
    """Select ALL non-grid packings n=220..1000 with f in [0.76, 0.999] plus highest-bump Fixed-K ceilings."""
    scored = []
    for n in range(220, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D == 0:
            continue
        Ln = cur[n][0]
        fn = Ln - K
        if fn >= 1.0 - 1e-6:
            # Near-grid ceiling breakthrough candidates: if n-1 has f < 0.985, n might break below K+1!
            if n - 1 >= K * K + 1 and (cur[n - 1][0] - K) < 0.980:
                scored.append((0.00030 + lag_rank.get(n, 0.0), n, fn))
            continue
        if fn >= 0.76:
            # Prioritize ceiling band [0.90, 0.998] (skipped by bump_D!) and classic phase band [0.76, 0.90]
            ceil_bonus = 0.00045 if fn >= 0.90 else 0.00025
            # Fixed-K bump bonus
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
            prio = ceil_bonus + lag_rank.get(n, 0.0) + 1.5 * bump_k
            scored.append((prio, n, fn))
    scored.sort(reverse=True)
    return [(n, fn) for _, n, fn in scored[:200]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--lags-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=18)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=3500)
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
    targets = select_high_f_targets(cur, lag_rank)
    my_slice = targets[args.shard :: args.num_shards]
    print(
        f"[W24B-HighFPhase s={args.shard}/{args.num_shards}] Frontier={tot_score:.10f} "
        f"({len(fresh)} fresh from {n_files} files), my_targets={[(n, round(f, 3)) for n, f in my_slice]} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    jobs = []
    for i, (n, fn) in enumerate(my_slice):
        L0, sq0 = cur[n]
        transfers = w22.harvest_fixed_d_transfers(n, L0, cur, nongrid_sorted)
        jobs.append((n, L0, sq0, transfers, 2400001 + args.shard * 10000 + i * 127))

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
                        f"  [W24B HIGH-F WIN!!] n={n:4d} {tag:16s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W24B s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W24B s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
