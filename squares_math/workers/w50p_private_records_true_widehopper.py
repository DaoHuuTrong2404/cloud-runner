#!/usr/bin/env python3
"""Wave 50P: 4-Seed True `eta<=0.28` Wider-Move Hopper + Interleaved SLP on Private/Recent Frontier Packings!

Runs 4 parallel seeds x 220s -> `SLP.polish(iters=115)` -> 4 parallel seeds x 180s on our Top-80
private/recent two-band & high-gap packings (`f in [0.54, 0.985]` and `gap_to_WR > 1e-5`), which
have NEVER been run with `eta <= 0.28` (>0.2071 45-deg orientation barrier!).
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
TOOLS_DIR = os.path.abspath(os.path.join(HERE, "..", "tools"))
for _d in (DS_DIR, TOOLS_DIR, HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import w12a_super_bfmelt_rp as w12
import w32a_fresh_basin_fast_hopper as w32a
import w49_fault_open_sa_engine as w49a
import w50d_exhaustive_private_del1_widehopper as w50d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--r1-sec", type=int, default=220)
    ap.add_argument("--r2-sec", type=int, default=180)
    ap.add_argument("--max-targets", type=int, default=72)
    ap.add_argument("--seed-off", type=int, default=508000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    base_only, _, _, _ = w12.load_frontier_with_wins(args.union_npz, [])
    cur, fresh, tot_score, _ = w49a.load_closed_frontier(args.union_npz, args.win_glob)
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    ranked = []
    for n in range(200, 1001):
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        f_part = cur[n][0] - K
        fl = front_L.get(n, cur[n][0])
        gds = (cur[n][0] - fl) / math.sqrt(n)
        if gds < 2e-6:
            continue
        priv_adv = max(0.0, base_only[n][0] - cur[n][0])
        # Focus on two-band sweet spot f in [0.54, 0.985] + private/recent advances!
        in_sweet = 0.54 <= f_part <= 0.985
        prio = (
            gds
            + (0.00045 if priv_adv > 1e-4 else 0.0)
            + (0.00025 if in_sweet else -0.00010)
        )
        ranked.append((prio, n))

    ranked.sort(reverse=True)
    targets = [n for _, n in ranked[: args.max_targets]]
    my_slice = targets[args.shard :: args.num_shards]
    deadline = t0 + args.budget_sec

    print(
        f"[W50P-PrivateWide28 s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_slice({len(my_slice)})={my_slice} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0

    def save_ckpt():
        pl = {}
        for kn, (Lf, xf, yf, tf) in sorted(wins.items()):
            pl[f"L_{kn}"] = np.array([Lf], dtype=np.float64)
            pl[f"x_{kn}"] = np.asarray(xf, dtype=np.float64)
            pl[f"y_{kn}"] = np.asarray(yf, dtype=np.float64)
            pl[f"t_{kn}"] = np.asarray(tf, dtype=np.float64)
        np.savez_compressed(args.out_npz, **pl)

    ex = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, n in enumerate(my_slice):
            rem = deadline - time.time()
            if rem < 40.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            r1 = min(args.r1_sec, max(30, int(0.42 * (rem - 15.0))))
            r2 = min(args.r2_sec, max(20, int(0.35 * (rem - 15.0))))
            bseed = args.seed_off + args.shard * 10007 + i * 401
            Qb, Sb, (tr, ft, wn) = w50d.run_4seed_2round_widehopper(
                ex, cur[n][1], L0_cur, bseed, r1, r2, deadline
            )
            if Sb < L0_cur - 1e-9:
                res = w50d.certify_winner(n, Qb, L0_cur, deadline)
                if res is not None:
                    Lf = res[0]
                    dL = Lf - cur[n][0]
                    tot_ds += (Lf - L0_cur) / math.sqrt(n)
                    wins[n] = res
                    print(
                        f"  [W50P WIN!!] n={n:4d} [priv_wh28] tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{i + 1}/{len(my_slice)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W50P done] n={n:4d} dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{i + 1}/{len(my_slice)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W50P s={args.shard} EXC] {exc}", flush=True)

    save_ckpt()
    print(
        f"[W50P s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    for proc in list(getattr(ex, "_processes", {}).values()):
        try:
            proc.terminate()
        except Exception:
            pass
    os._exit(0)


if __name__ == "__main__":
    main()
