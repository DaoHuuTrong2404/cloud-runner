#!/usr/bin/env python3
"""Wave 56G (`w56g_unified_global_faultdp.py`): Unified Global 640-Gadget 4D 8-Sym Fault-DP
+ Full-Frontier Instant DP Sweep + KKT-Polished 3-Round Wider-Hopper!

Solves the exact `max(f(g_1), f(g_2), f(g_3))` strip bottleneck discovered in Wave 54G:
  - In Wave 54G, 10 isolated shards upgraded 73 rectangle gadgets (up to `df = -0.01687` on `20x10+6`!),
    winning immediately on `n = 842` (+ L-ext `901, 962`), while 14 multi-gadget puzzles (`449, 544, 582,
    590, 685, 789, 850, 903, 906, ...`) landed at `gap0 = +0.00000` with brand-new topologies (`dA = 1.4..113`)
    ONLY because each shard saw 1/10 of the upgraded gadgets!
  - Here, Stage 1+2 (`w56_gadget_forge` + `w56_merge_gadgets`) already forged ONE unified
    `global_rect_gadgets.npz` from 640 active gadgets across 64 cores!
  - Every W56G shard loads 100% of `global_rect_gadgets.npz`, solves `fault_dp_8sym.solve_dp` globally,
    sweeps its puzzle slice for immediate KKT DP wins, and runs `r1=240, r2=220, r3=220` KKT Wider-Hopper
    on genuinely novel multi-gadget basins!
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
import rust_engines as RE
import squeeze2 as sq2
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53g_rect_quench_8sym_faultdp as w53g
import w53h_nakajima_reharden_and_softinsert as w53h


def quench_composed_chain(P_comp, L0):
    """Relax & polish a 4D Fault-DP composed chain using fast Rust `fq` + `squeeze2` fallback."""
    rq = RE.run_fq_quench_args(
        P_comp,
        [
            "--loosen",
            "1.002",
            "--outer",
            "38",
            "--pit",
            "240",
            "--flip-top",
            "20",
            "--finish",
            "--stag-w",
            "0",
            "--rmin",
            "1e-12",
            "--deadline",
            "10.5",
        ],
        s_ref=max(RE.side_exact(P_comp), L0),
        timeout_s=12.5,
    )
    if rq is not None:
        return rq[1], float(rq[0])
    return w53g.fast_quench_dp(P_comp, L0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--global-gadgets-npz", required=True)
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=22)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1880)
    ap.add_argument("--r1-sec", type=int, default=235)
    ap.add_argument("--r2-sec", type=int, default=215)
    ap.add_argument("--r3-sec", type=int, default=215)
    ap.add_argument("--seed-off", type=int, default=720000)
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

    n_forged = 0
    if os.path.exists(args.global_gadgets_npz):
        zg = np.load(args.global_gadgets_npz)
        if "keys" in zg.files:
            for i in range(len(zg["keys"])):
                donors.append(np.asarray(zg[f"sq_{i}"], dtype=np.float64))
                n_forged += 1

    G = fdp8.extract_gadgets(donors)
    dp0, PG0, PM0, PQ0 = fdp8.solve_dp(G)

    print(
        f"[W56G-UnifiedGlobalDP s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"forged_gadgets={n_forged} total_pareto_G={len(G)} ({time.time() - t0:.1f}s)",
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

    all_targets = []
    for n in range(60, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D <= 0 or D > fdp8.DMAX or K > fdp8.WMAX:
            continue
        gap_ldr = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        all_targets.append((gap_ldr / math.sqrt(n), n, K, D))
    all_targets.sort(reverse=True)

    my_targets = all_targets[args.shard :: args.num_shards]
    cands = []
    for _, n, K, D in my_targets:
        if time.time() - t0 > 340.0:
            break
        L0 = wins[n][0] if n in wins else cur[n][0]
        dp_chains = fdp8.trace_and_compose(dp0, PG0, PM0, PQ0, G, donors, K, D, top_m=4)
        for L_est, dtag, P_comp in dp_chains:
            if L_est < L0 + 0.022:
                rh = quench_composed_chain(P_comp, L0)
                if rh is not None:
                    Qc, Sc = rh
                    if Sc < L0 + 0.00025:
                        Qp, Sp = w51d.deep_contact_polish(
                            n, Qc, deadline, slp_sec=32.0, fq_sec=22
                        )
                        if Sp < L0 - 1e-9:
                            ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp)
                            if ok0 and Lf0 < L0 - 1e-9:
                                ds = (Lf0 - L0) / math.sqrt(n)
                                tot_ds += ds
                                wins[n] = (float(Lf0), xf0, yf0, tf0)
                                L0 = float(Lf0)
                                print(
                                    f"  [W56G DIRECT GLOBAL DP WIN!!] n={n:4d} [{dtag}] {cur[n][0]:.7f} -> {Lf0:.7f} (dS={ds:+.8f})!",
                                    flush=True,
                                )
                                save_ckpt()
                        Qc, Sc = Qp, Sp
                    adist = (
                        w53h.topology_angle_dist(Qc, cur[n][1])
                        if cur[n][1] is not None
                        else 1.0
                    )
                    if Sc < L0 - 1e-9 or (Sc < L0 + 0.0135 and adist > 0.045):
                        cands.append((Sc - L0, n, Sc, adist, dtag, Qc))

    cands.sort(key=lambda z: z[0])
    seen_n = {}
    queued = []
    for item in cands:
        n = item[1]
        if seen_n.get(n, 0) >= 2:
            continue
        seen_n[n] = seen_n.get(n, 0) + 1
        queued.append(item)
        if len(queued) >= 7:
            break

    print(
        f"[W56G s={args.shard}] novel global-DP cands={len(cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, S_init, adist, dtag, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 50.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            r1 = min(args.r1_sec, max(35, int(0.35 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(30, int(0.30 * (rem - 20.0))))
            r3 = min(args.r3_sec, max(30, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + 20000 + args.shard * 10009 + i * 733
            Qb, Sb, (tr, ft, wn) = w51d.deep_3round_widehopper(
                ex_hop, n, Q_init, L0_cur, bseed, r1, r2, r3, deadline
            )
            if Sb < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, Qb)
                if ok and Lf < L0_cur - 1e-9:
                    ds = (Lf - L0_cur) / math.sqrt(n)
                    tot_ds += ds
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W56G HOP WIN!!] n={n:4d} [{dtag}] dA={adist:.2f} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"(dL={Lf - cur[n][0]:+.7f}, dS={ds:+.8f}, w={wn})",
                        flush=True,
                    )
                    save_ckpt()
                else:
                    print(
                        f"  [W56G miss] n={n:4d} [{dtag}] dA={adist:.2f} gap0={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f}",
                        flush=True,
                    )
            else:
                print(
                    f"  [W56G done] n={n:4d} [{dtag}] dA={adist:.2f} gap0={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} w={wn}",
                    flush=True,
                )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W56G s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
