#!/usr/bin/env python3
"""Wave 60G (`w60g_bandlns_globaldp_rust_hop.py`): Unified Global 640-Gadget 4D 8-Sym Fault-DP
+ Macro Band-LNS (`lns_band.lns_burst`: `band_angles`, `reseed`, `PIN` angle jumps)
+ Compiled Rust `packer hhop` / `packer hop` Escalation!

Upgrades Wave 56G (`w56g_unified_global_faultdp.py`) where Python `w32a` stalled on novel multi-gadget DP seams:
  1. Loads `global_rect_gadgets.npz` (`w56-global-gadgets` from `46076151884`) + full closed frontier (incl. W59 wins!).
  2. Sweeps Global 8-Sym Fault-DP chains (`top_m=4`) across the shard's high-gap puzzles AND runs `lns_band.lns_burst`
     (`band_angles` smooth angle waves + `reseed` tilted lattice strips + `PIN` jumps) across all 4 cores!
  3. Escalates the sharpest novel basins (`< L0 + 0.012`) through compiled Rust `packer hhop` & `packer hop`
     (both unfrozen and far-grid frozen) + KKT `deep_contact_polish` + `Decimal64`!
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

import fault_dp_8sym as fdp8
import lns_band as LB
import rust_engines as RE
import squeeze2 as sq2
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w56g_unified_global_faultdp as w56g
import w59l_faithful_lcut_and_rust_hop as w59l


def run_lns_screen_job(job_tup):
    n, P0, L0, budget_s, seed = job_tup
    try:
        best_P, Lbest, near_cands, trials = LB.lns_burst(
            P0, budget_s=budget_s, seed=seed, near_tol=0.0095
        )
        out = []
        if Lbest < L0 - 1e-9:
            out.append((Lbest - L0, f"lns_win", best_P))
        for gap_c, tg_c, Q_c in near_cands[:3]:
            out.append((float(gap_c), tg_c, Q_c))
        return n, out, trials
    except Exception:
        return n, [], 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--global-gadgets-npz", required=True)
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1820)
    ap.add_argument("--seed-off", type=int, default=1020000)
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
        with np.load(args.global_gadgets_npz) as zg:
            if "keys" in zg.files:
                for i in range(len(zg["keys"])):
                    donors.append(np.asarray(zg[f"sq_{i}"], dtype=np.float64))
                    n_forged += 1

    G = fdp8.extract_gadgets(donors)
    dp0, PG0, PM0, PQ0 = fdp8.solve_dp(G)

    print(
        f"[W60G-BandLNS+GlobalDP s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"forged_gadgets={n_forged} pareto_G={len(G)} ({time.time() - t0:.1f}s)",
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
        if D <= 0:
            continue
        gap_ldr = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gap_ldr > 6.0e-5:
            all_targets.append((gap_ldr / math.sqrt(n), n, K, D))
    all_targets.sort(reverse=True)
    my_targets = all_targets[args.shard :: args.num_shards]

    cands = []
    # 1) Global 4D 8-Sym Fault-DP sweep (~180s cap)
    for _, n, K, D in my_targets:
        if time.time() - t0 > 220.0:
            break
        if D > fdp8.DMAX or K > fdp8.WMAX:
            continue
        L0 = wins[n][0] if n in wins else cur[n][0]
        dp_chains = fdp8.trace_and_compose(dp0, PG0, PM0, PQ0, G, donors, K, D, top_m=4)
        for L_est, dtag, P_comp in dp_chains:
            if L_est < L0 + 0.020:
                rh = w56g.quench_composed_chain(P_comp, L0)
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
                                    f"  [W60G DIRECT DP WIN!!] n={n:4d} [{dtag}] {cur[n][0]:.7f} -> {Lf0:.7f} (dS={ds:+.8f})!",
                                    flush=True,
                                )
                                save_ckpt()
                        Qc, Sc = Qp, Sp
                    adist = (
                        w53h.topology_angle_dist(Qc, cur[n][1])
                        if cur[n][1] is not None
                        else 1.0
                    )
                    if Sc < L0 - 1e-9 or (Sc < L0 + 0.0125 and adist > 0.045):
                        cands.append((Sc - L0, n, Sc, adist, dtag, Qc))

    # 2) Macro Band-LNS (`lns_burst`) across top 8 puzzles of `my_targets` (~210s across 4 cores!)
    ex_lns = ProcessPoolExecutor(max_workers=args.procs)
    try:
        lns_jobs = []
        for idx_l, (_, n, K, D) in enumerate(my_targets[:8]):
            L0 = wins[n][0] if n in wins else cur[n][0]
            lns_jobs.append(
                (
                    n,
                    RE.normalize_to_origin(cur[n][1]),
                    L0,
                    95.0,
                    args.seed_off + args.shard * 1301 + idx_l * 79,
                )
            )
        futs_l = [ex_lns.submit(run_lns_screen_job, lj) for lj in lns_jobs]
        try:
            for fl in as_completed(futs_l, timeout=240.0):
                nl, l_list, n_tr = fl.result()
                L0 = wins[nl][0] if nl in wins else cur[nl][0]
                for gap_l, tg_l, Q_l in l_list:
                    Sl = RE.side_exact(Q_l)
                    if Sl < L0 - 1e-9:
                        Qp, Sp = w51d.deep_contact_polish(
                            nl, Q_l, deadline, slp_sec=36.0, fq_sec=22
                        )
                        ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(nl, Qp)
                        if ok0 and Lf0 < L0 - 1e-9:
                            ds = (Lf0 - L0) / math.sqrt(nl)
                            tot_ds += ds
                            wins[nl] = (float(Lf0), xf0, yf0, tf0)
                            L0 = float(Lf0)
                            print(
                                f"  [W60G DIRECT LNS WIN!!] n={nl:4d} [{tg_l}] {cur[nl][0]:.7f} -> {Lf0:.7f} (dS={ds:+.8f})!",
                                flush=True,
                            )
                            save_ckpt()
                    ad = w53h.topology_angle_dist(Q_l, cur[nl][1])
                    if Sl < L0 + 0.011 and ad > 0.035:
                        cands.append((Sl - L0, nl, Sl, ad, tg_l, Q_l))
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_lns)

    cands.sort(key=lambda z: z[0])
    seen_n = {}
    queued = []
    for item in cands:
        n = item[1]
        if seen_n.get(n, 0) >= 2:
            continue
        seen_n[n] = seen_n.get(n, 0) + 1
        queued.append(item)
        if len(queued) >= 6:
            break

    print(
        f"[W60G s={args.shard}] novel DP+LNS cands={len(cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, S_init, adist, dtag, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 65.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            best_Q, best_S = RE.normalize_to_origin(Q_init), float(S_init)

            per_slot = min(
                195.0, max(50.0, (rem - 35.0) / max(1, min(3, len(queued) - i)))
            )
            bseed = args.seed_off + 60000 + args.shard * 10009 + i * 733
            k_ceil = math.ceil(math.sqrt(n) - 1e-12)
            forbid_k = k_ceil if L0_cur < k_ceil + 1e-6 else 0
            ts_goal = min(best_S, L0_cur) - 4.5e-5

            rust_tasks = [
                ("hhop", best_Q, ts_goal, bseed + 1, forbid_k, False, per_slot),
                ("hhop", best_Q, ts_goal, bseed + 2, forbid_k, True, per_slot),
                ("hop", best_Q, ts_goal, bseed + 3, forbid_k, False, per_slot),
                ("hop", best_Q, ts_goal, bseed + 4, forbid_k, True, per_slot),
            ]
            futs_r = [ex_hop.submit(w59l.run_rust_worker_task, rt) for rt in rust_tasks]
            best_mode = "init"
            try:
                for fr in as_completed(futs_r, timeout=per_slot + 35.0):
                    rr = fr.result()
                    if rr is not None and rr[0] < best_S - 1e-10:
                        best_S, best_Q, best_mode = rr[0], rr[1], rr[2]
            except TimeoutError:
                w51d.terminate_pool(ex_hop)
                ex_hop = ProcessPoolExecutor(max_workers=args.procs)

            if best_S < L0_cur + 0.0018 or best_S < S_init - 2e-4:
                best_Q, best_S = w51d.deep_contact_polish(
                    n, best_Q, deadline, slp_sec=45.0, fq_sec=26
                )

            if best_S < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, best_Q)
                if ok and Lf < L0_cur - 1e-9:
                    ds = (Lf - L0_cur) / math.sqrt(n)
                    tot_ds += ds
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W60G RUST WIN!!] n={n:4d} [{dtag}->{best_mode}] dA={adist:.2f} {cur[n][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f})!",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W60G done] n={n:4d} [{dtag}->{best_mode}] dA={adist:.2f} gap0={gap0:+.5f} -> dEnd={best_S - L0_cur:+.6f}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W60G s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
