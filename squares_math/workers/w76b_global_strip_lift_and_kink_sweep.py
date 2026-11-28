#!/usr/bin/env python3
"""Wave 76B (`w76b_global_strip_lift_and_kink_sweep.py`):
48-Core (`12` Shards) ALL-`1000` Global Sweep of:
  1) Exact Convex-Hull Support-Interval Global Box Rotation (`hull_rotate_exact`),
  2) Bidirectional `sq_engine` Single-Strip (`n +- K`, `n +- (K+1)`) & L-Band (`n +- (2K+1)`) Lifts
     across our entire private `R39 + W75` bank (`sweep_strip_and_l_lifts_for_n`),
  3) Kink-Aware Free-Angle Sequential Sparse LP (`kink_contact_lp_polish`, `0/90 deg` cusp subgradients!)
     on EVERY non-grid puzzle `n in [20..1000][shard :: 12]` (~46 non-grid puzzles per shard in ~550s!),
  4) Remaining ~1,250s per shard: Breadth-First `continuous_momentum_widehopper` across Tier-3
     untouched Two-Band puzzles (`Ranks #151 .. #270` of `n >= 300, f in [0.58, 0.997]`)!
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

import kink_polish_and_strip_lifts_w76 as kps
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w63k_1puzzle_per_shard_killshot as w63k
import w76a_twoband_breadth_hopper_kink_polish as w76a


def eval_kink_and_lifts_job(job):
    n, L0, P0, lifts_raw, seed = job
    best_S, best_Q, best_tag = L0, P0, "none"
    try:
        for seed_off in (0, 29):
            sk, qk = kps.kink_contact_lp_polish(
                best_Q, rounds=38, seed=seed + seed_off, max_seconds=11.0
            )
            if sk < best_S - 1e-10:
                best_S, best_Q, best_tag = sk, qk, f"kink_s{seed_off}"
        for s_lf, tg_lf, q_lf in lifts_raw[:3]:
            if s_lf < best_S + 0.0045:
                sk2, qk2 = kps.kink_contact_lp_polish(
                    q_lf, rounds=36, seed=seed + 71, max_seconds=10.0
                )
                if sk2 < best_S - 1e-10:
                    best_S, best_Q, best_tag = sk2, qk2, f"{tg_lf}+kink"
    except Exception:
        pass
    return n, best_S, best_Q, best_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=4250000)
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

    my_slice = [
        n
        for n in range(18 + args.shard, 1001, args.num_shards)
        if cur[n][1] is not None and cur[n][0] < math.ceil(math.sqrt(n)) - 1e-7
    ]

    print(
        f"[W76B-GlobalKink+StripSweep s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"non_grid_slice={len(my_slice)} ({time.time() - t0:.1f}s)",
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

    def record_win_n(n_w, Q_cand, S_cand, src_tag, w_cnt):
        nonlocal tot_ds
        L_ref = wins[n_w][0] if n_w in wins else cur[n_w][0]
        if S_cand >= L_ref - 1e-9:
            return
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n_w, Q_cand)
        if ok and Lf < L_ref - 1e-9:
            ds = (Lf - L_ref) / math.sqrt(n_w)
            tot_ds += ds
            wins[n_w] = (float(Lf), xf, yf, tf)
            cur[n_w] = (float(Lf), np.column_stack([xf, yf, tf]))
            print(
                f"  [W76B GLOBAL-SWEEP WIN!!] n={n_w:4d} [{src_tag}] -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    # Phase 1 (~450s): Parallel Kink-Aware LP + Hull-Rotate + Strip/L-Lift Sweep on all ~46 non-grid puzzles in slice!
    t_p1 = time.time()
    bseed = args.seed_off + args.shard * 9973
    sweep_jobs = []
    for idx_s, n in enumerate(my_slice):
        lifts = kps.sweep_strip_and_l_lifts_for_n(n, cur)
        sweep_jobs.append((n, cur[n][0], cur[n][1], lifts, bseed + idx_s * 37))

    moved_puzzles = []
    ex = ProcessPoolExecutor(max_workers=args.procs)
    try:
        futs = [ex.submit(eval_kink_and_lifts_job, jb) for jb in sweep_jobs]
        try:
            for f in as_completed(futs, timeout=540.0):
                n_j, best_S, best_Q, best_tag = f.result()
                if best_S < cur[n_j][0] - 1e-9:
                    Qp, Sp = w51d.deep_contact_polish(
                        n_j, best_Q, deadline, slp_sec=32.0, fq_sec=18
                    )
                    record_win_n(n_j, Qp, Sp, best_tag, 0)
                    moved_puzzles.append(n_j)
        except TimeoutError:
            w51d.terminate_pool(ex)
            ex = ProcessPoolExecutor(max_workers=args.procs)

        print(
            f"  [W76B phase1-done] swept={len(sweep_jobs)} wins_so_far={len(wins)} ({time.time() - t_p1:.1f}s)",
            flush=True,
        )

        # Phase 2 (~1,250s): Breadth-First WideHopper on moved puzzles + Tier-3 Two-Band (`Ranks #141..#260`)!
        OVERMINED = {234, 367, 491, 738, 456, 730, 261, 904, 962, 414}
        twoband = []
        for n in range(300, 1000):
            K = math.isqrt(n)
            if K * K == n or cur[n][1] is None or n in OVERMINED:
                continue
            f_val = cur[n][0] - K
            if 0.58 <= f_val <= 0.997:
                gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
                twoband.append(((gl + 2.0e-4) / math.sqrt(n), n))
        twoband.sort(reverse=True)
        # Skip the first 125 covered by W76A so W76A + W76B cover 200+ distinct Two-Band puzzles in a single wave!
        tier3 = [x[1] for x in twoband[125:245]][args.shard :: args.num_shards]
        hop_targets = list(dict.fromkeys(moved_puzzles + tier3))[:5]

        for hi, n_h in enumerate(hop_targets):
            rem = deadline - time.time()
            if rem < 90.0:
                break
            hop_b = max(80.0, min(320.0, (rem - 14.0) / max(1, len(hop_targets) - hi)))
            L0_h = cur[n_h][0]
            P0_h = cur[n_h][1]
            Qb, Sb, (_, _, wt, nr) = w63k.continuous_momentum_widehopper(
                ex,
                n_h,
                P0_h,
                L0_h,
                bseed + 60000 + hi * 1009,
                hop_b,
                deadline,
                lambda Qx, Sx, stg, wc, _n=n_h: record_win_n(
                    _n, Qx, Sx, f"tier3->{stg}", wc
                ),
            )
            S_fin, Q_fin = kps.kink_contact_lp_polish(
                Qb, rounds=36, seed=bseed + 901 + hi, max_seconds=11.0
            )
            if S_fin < Sb:
                Qb, Sb = Q_fin, S_fin
            record_win_n(n_h, Qb, Sb, "tier3->wh->kink", wt)
            print(
                f"  [W76B hop-done] n={n_h:4d} dEnd={Sb - L0_h:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex)

    save_ckpt()
    print(
        f"[W76B s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
