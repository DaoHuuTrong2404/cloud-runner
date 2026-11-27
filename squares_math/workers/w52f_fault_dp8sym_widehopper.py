#!/usr/bin/env python3
"""Wave 52F: Full 8-Symmetry `fault_dp_8sym` Multi-Gadget Composer over 558 Private Donors
-> Numba `squeeze2` Seam Quench -> 3-Round Deep-Polish `eta<=0.28` Wider-Hopper!

Extracts ALL non-dominated `(w, h, dm)` cropped gadgets across ALL 8 dihedral symmetries of our
entire 558-donor closed frontier (including our 103 exclusive private R32 packings!), solves the
3-mode (`touch`, `buffer-X`, `buffer-Y`) 4D fault-line DP, composes fresh multi-gadget topologies
for high-gap puzzles (`234, 261, 295, 332, 333, 367, 401, 414, 456, 491, 530, 632, 730, 738, 785, 904`),
heals gadget seams via `squeeze2` + `Compactor`, and attacks with 3-Round Deep-Polish `eta<=0.28` Wider-Hopper!
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

import fault_dp_8sym as fdp8
import squeeze2 as sq2
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w52s_staggered_band_quench_transplanter as w52s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--r1-sec", type=int, default=180)
    ap.add_argument("--r2-sec", type=int, default=165)
    ap.add_argument("--r3-sec", type=int, default=175)
    ap.add_argument("--seed-off", type=int, default=540000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    _dm = np.array(
        [[0.5, 0.5, 0.0], [1.6, 0.5, 0.1], [0.5, 1.6, 0.0]], dtype=np.float64
    )
    sq2.squeeze(_dm[:, 0], _dm[:, 1], _dm[:, 2], max_iter=20, max_tries=3)

    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    donors = []
    for n in range(5, 1001):
        K = math.isqrt(n)
        if K * K != n and cur[n][0] < K + 1 - 1e-6 and cur[n][1] is not None:
            donors.append(cur[n][1])

    G = fdp8.extract_gadgets(donors)
    dp, PG, PM, PQ = fdp8.solve_dp(G)
    print(
        f"[W52F-FaultDP8Sym s={args.shard}/{args.num_shards}] extracted {len(G)} non-dominated 8-sym gadgets from {len(donors)} donors in {time.time() - t0:.1f}s!",
        flush=True,
    )

    # Collect all target puzzles for this shard & generate quenched multi-gadget DP starts
    all_targets = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D <= 0 or D > fdp8.DMAX or K > fdp8.WMAX:
            continue
        gap_ldr = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gap_ldr > 1e-4 or cur[n][0] >= K + 1 - 1e-6:
            all_targets.append((gap_ldr / math.sqrt(n), n, K, D))
    all_targets.sort(reverse=True)
    my_targets = all_targets[args.shard :: args.num_shards]

    cands = []
    for _, n, K, D in my_targets:
        if time.time() - t0 > 320.0:
            break
        L0 = cur[n][0]
        dp_chains = fdp8.trace_and_compose(dp, PG, PM, PQ, G, donors, K, D, top_m=2)
        for L_est, dtag, P_comp in dp_chains:
            if L_est < L0 + 0.028:
                rq = w52s.fast_quench_candidate(P_comp, L0)
                if rq is not None:
                    Qc, Sc = rq
                    if Sc < L0 + 0.015:
                        cands.append((Sc - L0, n, Sc, dtag, Qc))

    cands.sort(key=lambda z: z[0])
    queued = cands[:10]
    print(
        f"[W52F s={args.shard}] DP cands(<+0.015)={len(cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
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

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, S_init, tag, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 45.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            if S_init < L0_cur + 0.00030:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    n, Q_init, deadline, slp_sec=38.0, fq_sec=26
                )
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        tot_ds += (Lf0 - L0_cur) / math.sqrt(n)
                        wins[n] = (float(Lf0), xf0, yf0, tf0)
                        L0_cur = float(Lf0)
                        print(
                            f"  [W52F DIRECT DP WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf0:.7f}!",
                            flush=True,
                        )
                        save_ckpt()

            r1 = min(args.r1_sec, max(30, int(0.35 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(25, int(0.30 * (rem - 20.0))))
            r3 = min(args.r3_sec, max(25, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + args.shard * 10007 + i * 601
            Qb, Sb, (tr, ft, wn) = w51d.deep_3round_widehopper(
                ex_hop, n, Q_init, L0_cur, bseed, r1, r2, r3, deadline
            )
            if Sb < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, Qb)
                if ok and Lf < L0_cur - 1e-9:
                    dL = Lf - cur[n][0]
                    tot_ds += (Lf - L0_cur) / math.sqrt(n)
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W52F WIN!!] n={n:4d} [{tag}_wh28] gap0={gap0:+.5f} tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W52F done] n={n:4d} [{tag}] gap0={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W52F s={args.shard} EXC] {exc}", flush=True)
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W52F s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
