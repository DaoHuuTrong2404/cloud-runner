#!/usr/bin/env python3
"""Wave 69S (`w69s_sorted_seam_melt_dp_and_3gen_forge.py`):
1) Loads 211-Gadget (`W64G + W65F + W66G + W67S + W68S`!) Precomputed 8-Sym DP Cache (`pool/dp_cache.npz`) in 0.25s!
2) Sorts ALL candidate DP chains across `my_sweep` by `L_est - L0` ASCENDING so EVERY strictly-improving
   or near-tied (`< +0.002`) composition is quenched FIRST with zero timeout truncation!
3) Runs Rounded-Corner Nakajima Seam-Melt (`--round-r0 0.09 --round-press`) + `anneal gls` on tight
   multi-gadget DP compositions (`Sc < L0 + 0.0025`) so bottleneck gadgets spill across rigid fault seams!
4) 3rd-Generation Ultra-Compact `"seg"` Forge (`M_sq <= 200`, reusable small multi-gadget building blocks!)
   + 6-Round Continuous-Momentum WideHopper on top 2 novel basins!
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

import dp8_cache_w68 as pdc
import fault_dp_8sym as fdp8
import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w56g_unified_global_faultdp as w56g
import w63k_1puzzle_per_shard_killshot as w63k
import w67s_merged_seam_seg_forge_and_global_dp as w67s
import w68s_global_seg2_dp_and_small_seam_forge as w68s


def quench_seam_melt_chain_job(job):
    _, n, L0, dtag, P_comp, seed0 = job
    rh = w56g.quench_composed_chain(P_comp, L0)
    if rh is None:
        return n, L0, dtag, None, math.inf
    Qc, Sc = rh

    # If tight (`< L0 + 0.0025`), dissolve internal fault seams via Nakajima rounded-corner press + GLS!
    if Sc < L0 + 0.0025:
        rq_m = RE.run_fq_quench_args(
            Qc,
            [
                "--seed",
                str(int(seed0)),
                "--round-r0",
                "0.09",
                "--round-steps",
                "14",
                "--round-press",
                "--round-press-outer",
                "8",
                "--round-kick",
                "0.004",
                "--round-kick-peak",
                "0.055",
                "--loosen",
                "1.004",
                "--outer",
                "48",
                "--pit",
                "260",
                "--flip-top",
                "20",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "17.0",
            ],
            s_ref=max(Sc, L0),
            timeout_s=20.0,
        )
        if rq_m is not None and rq_m[0] < Sc:
            Sc, Qc = float(rq_m[0]), rq_m[1]

        rg = RE.run_anneal_gls(
            Qc,
            shrink=3.5e-4,
            ms=3200.0,
            seed=int(seed0) + 19,
            ng=36,
            nl=22,
            ring=1,
            relax=26,
            strike_iters=8,
            timeout_s=8.5,
        )
        if rg is not None and (rg[2] or rg[3] < 2.5e-3):
            rq_g = RE.run_fq_quench_args(
                rg[1],
                [
                    "--seed",
                    str(int(seed0) + 23),
                    "--loosen",
                    "1.0025",
                    "--outer",
                    "38",
                    "--pit",
                    "260",
                    "--flip-top",
                    "18",
                    "--finish",
                    "--stag-w",
                    "0",
                    "--deadline",
                    "11.0",
                ],
                s_ref=L0,
                timeout_s=13.5,
            )
            if rq_g is not None and rq_g[0] < Sc:
                Sc, Qc = float(rq_g[0]), rq_g[1]

    if Sc < L0 + 0.00045:
        Sp, Qp = RE.kkt_sqp_polish_fq(Qc, timeout_each=6.0)
        if Sp < Sc:
            Qc, Sc = Qp, Sp
    return n, L0, dtag, Qc, float(Sc)


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
    ap.add_argument("--seed-off", type=int, default=2850000)
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
        if D <= 0:
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
            if L_est < L0 + 0.014:
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
    # CRITICAL FIX: Sort by `L_est - L0` ascending so strictly-improving & tied chains run FIRST!
    raw_chain_jobs.sort(key=lambda z: z[0])
    chain_jobs = raw_chain_jobs[:64]

    seg_all = w68s.collect_compact_seg_jobs(
        dp0, PG0, PM0, PQ0, G, donors, all_targets[:360], max_sq=200
    )
    my_segs = seg_all[args.shard :: args.num_shards][:10]
    seg_jobs = [
        (k, f_r, q_s, args.seed_off + 300000 + args.shard * 4391 + si * 113)
        for si, (k, f_r, q_s) in enumerate(my_segs)
    ]

    min_est_gap = chain_jobs[0][0] if chain_jobs else 9.9
    print(
        f"[W69S-SortedSeamDP s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"donors={len(donors)} G={len(G)} chain_jobs={len(chain_jobs)}/{len(raw_chain_jobs)} "
        f"min_est_gap={min_est_gap:+.6f} ucompact_segs={len(seg_jobs)}/{len(seg_all)} ({time.time() - t0:.1f}s)",
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
            print(
                f"  [W69S SEAM-MELT DP WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    full_cands = []
    upgraded_segs = []
    ex_pool = ProcessPoolExecutor(max_workers=args.procs)
    try:
        # Submit in sorted order (`L_est - L0` ascending)!
        futs_ch = [ex_pool.submit(quench_seam_melt_chain_job, jb) for jb in chain_jobs]
        try:
            for f in as_completed(futs_ch, timeout=540.0):
                n, L0, dtag, Qc, Sc = f.result()
                if Qc is None:
                    continue
                L_now = wins[n][0] if n in wins else L0
                if Sc < L_now + 0.00030:
                    Qp, Sp = w51d.deep_contact_polish(
                        n, Qc, deadline, slp_sec=40.0, fq_sec=25
                    )
                    record_win_n(n, Qp, Sp, dtag + "_seammelt", 0)
                    Qc, Sc = Qp, Sp
                ad = w53h.topology_angle_dist(Qc, cur[n][1])
                if Sc < L_now + 0.0120 and ad > 0.035:
                    full_cands.append((Sc - L_now, n, dtag, Sc, ad, Qc))
        except TimeoutError:
            pass

        futs_sg = [ex_pool.submit(w67s.quench_merged_seg_job, jb) for jb in seg_jobs]
        try:
            for f in as_completed(futs_sg, timeout=340.0):
                key, f_ref, best_f, best_Q = f.result()
                if best_Q is not None and best_f < f_ref - 1e-7:
                    upgraded_segs.append((key, f_ref, best_f, best_Q))
                    print(
                        f"  [W69S UCOMPACT SEG UPGRADE!!] {key[0]}x{key[1]}+{key[2]}: {f_ref:.6f} -> {best_f:.6f} (df={best_f - f_ref:+.6f})!",
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
                                record_win_n(n_eq, Qp_e, Sp_e, "seg3_sq_direct", 0)
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_pool)

    if upgraded_segs:
        gad_pl = {"keys": np.array([k for k, _, _, _ in upgraded_segs], dtype=np.int32)}
        for gi, (_, _, _, sq1) in enumerate(upgraded_segs):
            gad_pl[f"sq_{gi}"] = np.asarray(sq1, dtype=np.float64)
        np.savez_compressed(
            os.path.join(
                os.path.dirname(args.out_npz), f"w69s_gadgets_s{args.shard}.npz"
            ),
            **gad_pl,
        )

    # Prefer genuinely novel topologies (`dA > 0.5`) when tied at `gap0 ~ 0`!
    full_cands.sort(key=lambda z: (round(z[0], 4), -min(z[4], 10.0)))
    queued = []
    seen_n = {}
    for item in full_cands:
        nn = item[1]
        if seen_n.get(nn, 0) >= 1:
            continue
        seen_n[nn] = 1
        queued.append(item)
        if len(queued) >= 2:
            break

    for _, nn, _, _ in my_sweep:
        if len(queued) >= 2:
            break
        if seen_n.get(nn, 0) >= 1:
            continue
        L0_fb = wins[nn][0] if nn in wins else cur[nn][0]
        P0_fb = cur[nn][1]
        queued.append((0.0, nn, "cur_direct", L0_fb, 0.0, P0_fb))
        seen_n[nn] = 1

    print(
        f"[W69S s={args.shard}] ucompact_segs={len(upgraded_segs)} screened={len(full_cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, tag, S_init, adist, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 80.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            sub_b = (rem - 20.0) / max(1, len(queued) - i)
            Qb, Sb, (_, _, wt, nr) = w63k.continuous_momentum_widehopper(
                ex_hop,
                n,
                Q_init,
                L0_cur,
                args.seed_off + 95000 + args.shard * 8009 + i * 401,
                sub_b,
                deadline,
                lambda Qc, Sc, stg, wc, _n=n, _tg=tag: record_win_n(
                    _n, Qc, Sc, f"{_tg}->{stg}", wc
                ),
            )
            record_win_n(n, Qb, Sb, f"{tag}->whFin", wt)
            print(
                f"  [W69S done] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W69S s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
