#!/usr/bin/env python3
"""Wave 55K: Full-Frontier Exact KKT Gauss-Newton `newton_finish` + SQP Hessian Sweep (`N=1..1000`)
+ Rust Iterative Shrink-Hopping (`--shrink --hops 6`) on Top Residual Gaps!

Applies two second-order capabilities discovered in `fq.rs` across ALL 969 non-grid packings:
  1. Disables default `--stag-w 15` (`2e-9` early stagnation exit) via `--stag-w 0 --rmin 1e-12`,
     and runs BOTH active-set SOC + dual-ranked `--flip-top 64` + `newton_finish` (`--finish`)
     AND exact analytical Lagrangian Hessian SQP (`--sqp --finish`) down to `1e-13`!
  2. Spends remaining shard time running compiled Rust Iterative Shrink-Hopping (`--shrink`)
     with KKT `newton_finish` on the shard's highest-gap tilted packings.
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

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

import rust_engines as RE
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r


def eval_kkt_polish_one(args_tuple):
    n, P0, L0 = args_tuple
    try:
        Sb, Qb = RE.kkt_sqp_polish_fq(P0, timeout_each=11.0)
        if Sb < L0 - 5e-10:
            try:
                Qr, Sr = w51r.residual_continuation_polish(Qb, budget_s=10.0)
                if Sr < Sb - 1e-12 and E.min_gap(Qr) >= -1e-12:
                    Sb, Qb = float(Sr), Qr
            except Exception:
                pass
            ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, w32a.safe_open(Qb))
            if ok and Lf < L0 - 1e-9:
                return n, float(Lf), xf, yf, tf
    except Exception:
        pass
    return n, None, None, None, None


def eval_shrink_hop_one(args_tuple):
    n, P0, L0, seed = args_tuple
    try:
        rq = RE.run_fq_quench_args(
            P0,
            [
                "--loosen",
                "1.0008",
                "--shrink",
                "--d0",
                "1.4e-3",
                "--dmin",
                "4e-8",
                "--hops",
                "6",
                "--hop-sigma",
                "0.011",
                "--hop-k",
                "4",
                "--seed",
                str(int(seed)),
                "--finish",
                "--stag-w",
                "0",
                "--rmin",
                "1e-12",
                "--flip-top",
                "48",
                "--pit",
                "700",
                "--deadline",
                "26.0",
            ],
            s_ref=L0,
            timeout_s=28.5,
        )
        if rq is None:
            return n, None, None, None, None
        S_s, Q_s, _ = rq
        if S_s < L0 - 5e-10:
            Sb, Qb = RE.kkt_sqp_polish_fq(Q_s, timeout_each=10.0)
            if Sb < S_s:
                S_s, Q_s = Sb, Qb
            ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, w32a.safe_open(Q_s))
            if ok and Lf < L0 - 1e-9:
                return n, float(Lf), xf, yf, tf
    except Exception:
        pass
    return n, None, None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2220)
    ap.add_argument("--seed-off", type=int, default=650000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    all_non_grid = [
        n for n in range(5, 1001) if math.isqrt(n) ** 2 != n and cur[n][1] is not None
    ]
    my_puzzles = all_non_grid[args.shard :: args.num_shards]
    print(
        f"[W55K-KKTFinishSweep s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"sweeping {len(my_puzzles)} puzzles ({time.time() - t0:.1f}s)",
        flush=True,
    )

    deadline = t0 + args.budget_sec
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

    # Stage 1: Full KKT + SQP `newton_finish` sweep across all ~242 puzzles in shard
    ex = ProcessPoolExecutor(max_workers=args.procs)
    try:
        batch_sz = 24
        for bi in range(0, len(my_puzzles), batch_sz):
            if time.time() > deadline - 120.0:
                break
            chunk = my_puzzles[bi : bi + batch_sz]
            jobs = [(n, cur[n][1], cur[n][0]) for n in chunk]
            futs = [ex.submit(eval_kkt_polish_one, jb) for jb in jobs]
            try:
                for f in as_completed(futs, timeout=220.0):
                    n, Lf, xf, yf, tf = f.result()
                    if Lf is not None:
                        L_prev = wins[n][0] if n in wins else cur[n][0]
                        if Lf < L_prev - 1e-9:
                            ds = (Lf - L_prev) / math.sqrt(n)
                            tot_ds += ds
                            wins[n] = (Lf, xf, yf, tf)
                            print(
                                f"  [W55K KKT WIN!!] n={n:4d} {L_prev:.8f} -> {Lf:.8f} (dL={Lf - L_prev:+.8f}, dS={ds:+.8f})",
                                flush=True,
                            )
                            save_ckpt()
            except TimeoutError:
                w51d.terminate_pool(ex)
                ex = ProcessPoolExecutor(max_workers=args.procs)
            print(
                f"  [W55K sweep] done {min(bi + batch_sz, len(my_puzzles))}/{len(my_puzzles)} wins={len(wins)} dS={tot_ds:+.8f} ({time.time() - t0:.1f}s)",
                flush=True,
            )

        # Stage 2: Rust Iterative Shrink-Hopping (`--shrink --hops 6`) on top gap puzzles in remaining time
        rem = deadline - time.time()
        if rem > 90.0:
            ranked = []
            for n in my_puzzles:
                Lc = wins[n][0] if n in wins else cur[n][0]
                gap_l = max(0.0, Lc - front_L.get(n, Lc))
                ranked.append((gap_l / math.sqrt(n), n))
            ranked.sort(reverse=True)
            top_sh = [n for _, n in ranked[:32]]
            print(
                f"[W55K s={args.shard}] Stage 2 Shrink-Hopping on top {len(top_sh)} puzzles (rem={rem:.0f}s)",
                flush=True,
            )
            round_id = 0
            while deadline - time.time() > 65.0 and round_id < 4:
                sjobs = []
                for idx, n in enumerate(top_sh):
                    if n in wins:
                        _, xf, yf, tf = wins[n]
                        Pn = np.column_stack([xf, yf, tf])
                        Ln = wins[n][0]
                    else:
                        Pn = cur[n][1]
                        Ln = cur[n][0]
                    sjobs.append(
                        (
                            n,
                            Pn,
                            Ln,
                            args.seed_off + args.shard * 907 + round_id * 101 + idx,
                        )
                    )
                sfuts = [ex.submit(eval_shrink_hop_one, jb) for jb in sjobs]
                try:
                    for f in as_completed(
                        sfuts, timeout=max(45.0, deadline - time.time() - 20.0)
                    ):
                        n, Lf, xf, yf, tf = f.result()
                        if Lf is not None:
                            L_prev = wins[n][0] if n in wins else cur[n][0]
                            if Lf < L_prev - 1e-9:
                                ds = (Lf - L_prev) / math.sqrt(n)
                                tot_ds += ds
                                wins[n] = (Lf, xf, yf, tf)
                                print(
                                    f"  [W55K SHRINK-HOP WIN!!] n={n:4d} {L_prev:.8f} -> {Lf:.8f} (dS={ds:+.8f})",
                                    flush=True,
                                )
                                save_ckpt()
                except TimeoutError:
                    break
                round_id += 1
    finally:
        w51d.terminate_pool(ex)

    save_ckpt()
    print(
        f"[W55K s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
