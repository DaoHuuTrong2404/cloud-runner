#!/usr/bin/env python3
"""Wave 72S (`w72s_seg5_dp_and_packer_hop_donor.py`):
64-Core Dual-Engine:
  1) Harvests the `7` brand-new Frontier-Gated `"seg"` gadgets forged in Wave 71S
     (`19x18+9 -> 0.757499`, `18x19+9 -> 0.759575`, `9x4+2 -> 0.720210`, `7x4+2 -> 0.720210`)
     via Sorted Seam-Melt `223`-Gadget 8-Sym DP + 5th-Gen Frontier-Gated `"seg"` Forge!
  2) Compiled Rust `packer hop` (descending-target region-move + `best_hole` + `jam_lines` row-break)
     + `packer hhop` (hard-square collision event-chains) on High-Tilt Donor-Basin Starts across
     Top-32 Gap Puzzles (`ranked[:32][shard :: 16]`)!
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

import bidir_transfer_w70 as btr
import dp8_cache_w68 as pdc
import fault_dp_8sym as fdp8
import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w67s_merged_seam_seg_forge_and_global_dp as w67s
import w69s_sorted_seam_melt_dp_and_3gen_forge as w69s
import w71s_frontier_gated_seg_dp_and_defect_unjam as w71s


def packer_hop_worker(job):
    """Run compiled Rust `packer hop` + `packer hhop` trajectory inside a donor basin!"""
    tag, n, L0, P_start, target_s, seed, hop_sec = job
    best_Q = P_start
    best_S = RE.side_exact(P_start)
    fz = RE.build_far_grid_freeze_mask(P_start, buffer_dist=3.8, min_free=110)

    r1 = RE.run_packer_hop(
        best_Q,
        target_s=min(L0 - 4e-5, best_S - 4e-5),
        seed=int(seed),
        iters=140000,
        temp=0.28,
        amp=1.05,
        maxit=2800,
        descend=4.5e-5,
        linepen=0.012,
        stall=1e-5,
        frozen_mask=fz if int(seed) % 2 == 1 else None,
        timeout_s=max(25.0, 0.65 * hop_sec),
    )
    if r1 is not None and r1[0] < best_S:
        best_S, best_Q = float(r1[0]), r1[1]

    r2 = RE.run_packer_hhop(
        best_Q,
        seed=int(seed) + 19,
        iters=110000,
        temp=1.3e-4,
        chains=8,
        chainlen=1.08,
        gmax=0.018,
        soft=0.18,
        frozen_mask=None,
        timeout_s=max(20.0, 0.35 * hop_sec),
    )
    if r2 is not None and r2[0] < best_S:
        best_S, best_Q = float(r2[0]), r2[1]

    if best_S < L0 + 0.00035:
        Sp, Qp = RE.kkt_sqp_polish_fq(best_Q, timeout_each=6.0)
        if Sp < best_S:
            best_S, best_Q = float(Sp), Qp
    return tag, n, best_S, best_Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--dp-cache-npz", required=True)
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=3450000)
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
    donors, donor_n, G, dp0, PG0, PM0, PQ0 = pdc.load_dp8_cache(args.dp_cache_npz)

    all_targets = []
    for n in range(20, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D <= 0 or cur[n][1] is None:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        all_targets.append((gl / math.sqrt(n), n, K, D))
    all_targets.sort(reverse=True)

    my_sweep = all_targets[args.shard :: args.num_shards]
    raw_chain_jobs = []
    for _, n, K, D in my_sweep:
        if K > fdp8.WMAX or D > fdp8.DMAX:
            continue
        L0 = cur[n][0]
        dp_ch = fdp8.trace_and_compose_v2(
            dp0, PG0, PM0, PQ0, G, donors, donor_n, K, D, top_m=5
        )
        for j_i, (L_est, dtag, P_comp) in enumerate(dp_ch):
            if L_est < L0 + 0.0045:
                raw_chain_jobs.append(
                    (
                        L_est - L0,
                        n,
                        L0,
                        dtag,
                        P_comp,
                        args.seed_off + args.shard * 6151 + n * 13 + j_i,
                    )
                )
    raw_chain_jobs.sort(key=lambda z: z[0])
    chain_jobs = raw_chain_jobs[:36]

    seg_all = w71s.collect_frontier_gated_seg_jobs(
        dp0, PG0, PM0, PQ0, G, donors, cur, all_targets[:560], max_sq=460
    )
    my_segs = seg_all[args.shard :: args.num_shards][:8]
    seg_jobs = [
        (k, f_r, q_s, args.seed_off + 300000 + args.shard * 4391 + si * 113)
        for si, (k, f_r, q_s) in enumerate(my_segs)
    ]

    print(
        f"[W72S-Seg5+PackerDonor s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"donors={len(donors)} G={len(G)} chain_jobs={len(chain_jobs)} gated_segs={len(seg_jobs)}/{len(seg_all)} ({time.time() - t0:.1f}s)",
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

    def record_win_n(n_w, Q_cand, S_cand, src_tag):
        nonlocal tot_ds
        L_ref = wins[n_w][0] if n_w in wins else cur[n_w][0]
        if S_cand >= L_ref - 1e-9:
            return
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n_w, Q_cand)
        if ok and Lf < L_ref - 1e-9:
            ds = (Lf - L_ref) / math.sqrt(n_w)
            tot_ds += ds
            wins[n_w] = (float(Lf), xf, yf, tf)
            print(
                f"  [W72S WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f})!",
                flush=True,
            )
            save_ckpt()

    upgraded_segs = []
    ex = ProcessPoolExecutor(max_workers=args.procs)
    try:
        # Phase 1A: 223-Gadget Seam-Melt DP
        if chain_jobs:
            futs_ch = [
                ex.submit(w69s.quench_seam_melt_chain_job, jb) for jb in chain_jobs
            ]
            try:
                for f in as_completed(futs_ch, timeout=360.0):
                    n_c, L0_c, dtag, Qc, Sc = f.result()
                    if Qc is None:
                        continue
                    L_now = wins[n_c][0] if n_c in wins else L0_c
                    if Sc < L_now + 0.00030:
                        Qp, Sp = w51d.deep_contact_polish(
                            n_c, Qc, deadline, slp_sec=40.0, fq_sec=25
                        )
                        record_win_n(n_c, Qp, Sp, dtag + "_seammelt")
            except TimeoutError:
                w51d.terminate_pool(ex)
                ex = ProcessPoolExecutor(max_workers=args.procs)

        # Phase 1B: 5th-Gen Frontier-Gated "seg" Forge (`max_sq <= 460`)
        if seg_jobs:
            futs_sg = [ex.submit(w67s.quench_merged_seg_job, jb) for jb in seg_jobs]
            try:
                for f in as_completed(futs_sg, timeout=300.0):
                    key, f_ref, best_f, best_Q = f.result()
                    if best_Q is not None and best_f < f_ref - 1e-7:
                        upgraded_segs.append((key, f_ref, best_f, best_Q))
                        print(
                            f"  [W72S GATED SEG UPGRADE!!] {key[0]}x{key[1]}+{key[2]}: {f_ref:.6f} -> {best_f:.6f} (df={best_f - f_ref:+.6f})!",
                            flush=True,
                        )
                        if key[0] == key[1]:
                            n_eq = key[0] * key[1] + key[2]
                            if 1 <= n_eq <= 1000:
                                S_eq = RE.side_exact(best_Q)
                                L_eq0 = wins[n_eq][0] if n_eq in wins else cur[n_eq][0]
                                if S_eq < L_eq0 + 0.00030:
                                    Qp_e, Sp_e = w51d.deep_contact_polish(
                                        n_eq, best_Q, deadline, slp_sec=36.0, fq_sec=22
                                    )
                                    record_win_n(n_eq, Qp_e, Sp_e, "seg5_sq_direct")
            except TimeoutError:
                w51d.terminate_pool(ex)
                ex = ProcessPoolExecutor(max_workers=args.procs)

        if upgraded_segs:
            gad_pl = {
                "keys": np.array([k for k, _, _, _ in upgraded_segs], dtype=np.int32)
            }
            for gi, (_, _, _, sq1) in enumerate(upgraded_segs):
                gad_pl[f"sq_{gi}"] = np.asarray(sq1, dtype=np.float64)
            np.savez_compressed(
                os.path.join(
                    os.path.dirname(args.out_npz), f"w72s_gadgets_s{args.shard}.npz"
                ),
                **gad_pl,
            )

        # Phase 2 (~1100s): Parallel 4-Core Rust `packer hop` + `packer hhop` on Top-32 Donor Basins!
        top32 = [
            all_targets[idx][1]
            for idx in (args.shard, 16 + args.shard)
            if idx < len(all_targets)
        ]
        for pi, n_t in enumerate(top32):
            rem = deadline - time.time()
            if rem < 140.0:
                break
            p_b = (rem - 15.0) / max(1, len(top32) - pi)
            bseed = args.seed_off + 600000 + args.shard * 4999 + pi * 613
            L0_t = wins[n_t][0] if n_t in wins else cur[n_t][0]
            P0_t = cur[n_t][1]

            d_jobs = btr.build_donor_invasion_jobs(n_t, cur, bseed)
            starts = []
            if d_jobs:
                futs_d = [ex.submit(btr.eval_deep_donor_job, jb) for jb in d_jobs]
                try:
                    for f in as_completed(futs_d, timeout=65.0):
                        _, tg_d, _, S_d, Q_d = f.result()
                        if Q_d is not None and S_d < L0_t + 0.022:
                            if S_d < L0_t + 0.00020:
                                Qp_d, Sp_d = w51d.deep_contact_polish(
                                    n_t, Q_d, deadline, slp_sec=36.0, fq_sec=20
                                )
                                record_win_n(n_t, Qp_d, Sp_d, f"p2_dir_{tg_d}")
                                Q_d, S_d = Qp_d, Sp_d
                            ad = w53h.topology_angle_dist(Q_d, P0_t)
                            if ad >= 0.30:
                                starts.append((S_d, tg_d, Q_d))
                except TimeoutError:
                    w51d.terminate_pool(ex)
                    ex = ProcessPoolExecutor(max_workers=args.procs)

            starts.sort(key=lambda z: z[0])
            chosen = [(tg, q) for _, tg, q in starts[:3]]
            chosen.append(("cur_own", P0_t))
            while len(chosen) < args.procs:
                chosen.append(chosen[0])
            chosen = chosen[: args.procs]

            hop_s = max(60.0, min(deadline - time.time() - 20.0, p_b - 65.0))
            p_jobs = [
                (
                    tg_k,
                    n_t,
                    L0_t,
                    Q_k,
                    L0_t - 2e-4,
                    bseed + 1000 + ki * 97,
                    hop_s,
                )
                for ki, (tg_k, Q_k) in enumerate(chosen)
            ]
            futs_p = [ex.submit(packer_hop_worker, pj) for pj in p_jobs]
            best_end = math.inf
            try:
                for f in as_completed(futs_p, timeout=hop_s + 45.0):
                    tg_o, _, S_o, Q_o = f.result()
                    best_end = min(best_end, S_o)
                    if S_o < (wins[n_t][0] if n_t in wins else L0_t) + 0.00025:
                        Qp_o, Sp_o = w51d.deep_contact_polish(
                            n_t, Q_o, deadline, slp_sec=40.0, fq_sec=24
                        )
                        record_win_n(n_t, Qp_o, Sp_o, f"pkhop_{tg_o}")
            except TimeoutError:
                w51d.terminate_pool(ex)
                ex = ProcessPoolExecutor(max_workers=args.procs)
            print(
                f"  [W72S pkhop] n={n_t:4d} starts={[c[0] for c in chosen]} best_dEnd={best_end - L0_t:+.6f}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex)

    save_ckpt()
    print(
        f"[W72S s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
