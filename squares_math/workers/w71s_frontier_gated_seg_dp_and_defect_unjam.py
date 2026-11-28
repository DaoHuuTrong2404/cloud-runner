#!/usr/bin/env python3
"""Wave 71S (`w71s_frontier_gated_seg_dp_and_defect_unjam.py`):
64-Core Triple-Engine:
  1) Sorted Seam-Melt 8-Sym DP over the updated `dp_cache.npz` (now incorporating `W70A`'s brand-new
     `n=333` `18.7632203` packing and all `216` forged gadgets!),
  2) FRONTIER-GATED Merged-Seam `"seg"` Forge (`K + f_dp <= cur[n][0] + 0.008`, fixing the `W67S-W69S`
     non-separable `+0.07` approximation leak so 100% of forged `"seg"` blocks lie on competitive chains!),
  3) Dedicated Low-Tilt Defect Unjammer + 6-Island `santa_ga` v5 on Ranks `#49..#80`!
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
import w63k_1puzzle_per_shard_killshot as w63k
import w67s_merged_seam_seg_forge_and_global_dp as w67s
import w69s_sorted_seam_melt_dp_and_3gen_forge as w69s


def collect_frontier_gated_seg_jobs(
    dp, PG, PM, PQ, G, donors, cur, targets, max_sq=380
):
    """Collect 2-gadget and 3-gadget sub-chains ONLY from DP chains within `+0.008` of `cur[n][0]`!"""
    seg_map = {}
    for _, n, K, D in targets:
        if K > fdp8.WMAX or D > fdp8.DMAX or cur[n][1] is None:
            continue
        L_cur = cur[n][0]
        sub = dp[: K + 1, : K + 1, D, :]
        if not np.isfinite(sub).any():
            continue
        flat_idx = np.argsort(sub.ravel())[:12]
        for fi in flat_idx:
            cw, ch, q = np.unravel_index(int(fi), sub.shape)
            f_chain = float(sub[cw, ch, q])
            if not math.isfinite(f_chain) or float(K) + f_chain > L_cur + 0.008:
                break
            t_cw, t_ch, t_d, t_q = int(cw), int(ch), int(D), int(q)
            links = []
            ok_tr = True
            while t_cw or t_ch or t_d:
                g = int(PG[t_cw, t_ch, t_d, t_q])
                if g < 0:
                    ok_tr = False
                    break
                mode = int(PM[t_cw, t_ch, t_d, t_q])
                qp = int(PQ[t_cw, t_ch, t_d, t_q])
                links.append((g, mode))
                w, h, dd = int(G[g, 0]), int(G[g, 1]), int(G[g, 2])
                t_cw -= w + (1 if mode == 1 else 0)
                t_ch -= h + (1 if mode == 2 else 0)
                t_d -= dd
                t_q = qp
            if not ok_tr or len(links) < 2:
                continue
            links = links[::-1]
            for span in (2, 3):
                for st in range(len(links) - span + 1):
                    res = w67s.compose_subchain_segment(
                        links[st : st + span], G, donors
                    )
                    if res is None:
                        continue
                    key, f_init, Q_seg = res
                    if len(Q_seg) > max_sq:
                        continue
                    rows = np.where(
                        (G[:, 0].astype(int) == key[0])
                        & (G[:, 1].astype(int) == key[1])
                        & (G[:, 2].astype(int) == key[2])
                    )[0]
                    f_lib = float(np.min(G[rows, 3])) if len(rows) else f_init
                    f_ref = min(f_lib, f_init)
                    if f_init <= f_ref + 0.009:
                        prev = seg_map.get(key)
                        if prev is None or f_init < prev[1]:
                            seg_map[key] = (
                                f_ref,
                                f_init,
                                Q_seg,
                                prev[3] + 1 if prev else 1,
                            )
                        else:
                            seg_map[key] = (prev[0], prev[1], prev[2], prev[3] + 1)
    ranked = sorted(seg_map.items(), key=lambda kv: (-kv[1][3], kv[1][1] - kv[1][0]))
    return [(k, v[0], v[2]) for k, v in ranked]


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
    ap.add_argument("--seed-off", type=int, default=3250000)
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
            if L_est < L0 + 0.0055:
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
    chain_jobs = raw_chain_jobs[:44]

    seg_all = collect_frontier_gated_seg_jobs(
        dp0, PG0, PM0, PQ0, G, donors, cur, all_targets[:520], max_sq=380
    )
    my_segs = seg_all[args.shard :: args.num_shards][:9]
    seg_jobs = [
        (k, f_r, q_s, args.seed_off + 300000 + args.shard * 4391 + si * 113)
        for si, (k, f_r, q_s) in enumerate(my_segs)
    ]

    min_est_gap = chain_jobs[0][0] if chain_jobs else 9.9
    print(
        f"[W71S-GatedSeg+DefectGAv5 s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"donors={len(donors)} G={len(G)} chain_jobs={len(chain_jobs)}/{len(raw_chain_jobs)} "
        f"min_est_gap={min_est_gap:+.6f} gated_segs={len(seg_jobs)}/{len(seg_all)} ({time.time() - t0:.1f}s)",
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
                f"  [W71S WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    upgraded_segs = []
    dp_nbrs = {}
    ex_pool = ProcessPoolExecutor(max_workers=args.procs)
    try:
        # Phase 1: Sorted Seam-Melt DP (with new n=333 18x18+9 gadget!)
        if chain_jobs:
            futs_ch = [
                ex_pool.submit(w69s.quench_seam_melt_chain_job, jb) for jb in chain_jobs
            ]
            try:
                for f in as_completed(futs_ch, timeout=420.0):
                    n_c, L0_c, dtag, Qc, Sc = f.result()
                    if Qc is None:
                        continue
                    L_now = wins[n_c][0] if n_c in wins else L0_c
                    if Sc < L_now + 0.00030:
                        Qp, Sp = w51d.deep_contact_polish(
                            n_c, Qc, deadline, slp_sec=40.0, fq_sec=25
                        )
                        record_win_n(n_c, Qp, Sp, dtag + "_seammelt", 0)
                        Qc, Sc = Qp, Sp
                    if Sc < L_now + 0.013:
                        dp_nbrs.setdefault(n_c, []).append(Qc)
            except TimeoutError:
                w51d.terminate_pool(ex_pool)
                ex_pool = ProcessPoolExecutor(max_workers=args.procs)

        # Phase 2: Frontier-Gated Merged-Seam "seg" Forge
        if seg_jobs:
            futs_sg = [
                ex_pool.submit(w67s.quench_merged_seg_job, jb) for jb in seg_jobs
            ]
            try:
                for f in as_completed(futs_sg, timeout=340.0):
                    key, f_ref, best_f, best_Q = f.result()
                    if best_Q is not None and best_f < f_ref - 1e-7:
                        upgraded_segs.append((key, f_ref, best_f, best_Q))
                        print(
                            f"  [W71S GATED SEG UPGRADE!!] {key[0]}x{key[1]}+{key[2]}: {f_ref:.6f} -> {best_f:.6f} (df={best_f - f_ref:+.6f})!",
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
                                    record_win_n(n_eq, Qp_e, Sp_e, "seg4_sq_direct", 0)
            except TimeoutError:
                w51d.terminate_pool(ex_pool)
                ex_pool = ProcessPoolExecutor(max_workers=args.procs)

        if upgraded_segs:
            gad_pl = {
                "keys": np.array([k for k, _, _, _ in upgraded_segs], dtype=np.int32)
            }
            for gi, (_, _, _, sq1) in enumerate(upgraded_segs):
                gad_pl[f"sq_{gi}"] = np.asarray(sq1, dtype=np.float64)
            np.savez_compressed(
                os.path.join(
                    os.path.dirname(args.out_npz), f"w71s_gadgets_s{args.shard}.npz"
                ),
                **gad_pl,
            )

        # Phase 3 (~900s): Untouched Ranks `#49..#80` (`ranked[48..80][shard :: 16]`) via Defect-Snap Santa-GA v5!
        ga_targets = [
            all_targets[idx][1]
            for idx in (48 + args.shard, 64 + args.shard)
            if idx < len(all_targets)
        ]
        print(
            f"[W71S s={args.shard}] gated_segs={len(upgraded_segs)} -> Phase-3 GA-v5 targets={ga_targets} ({time.time() - t0:.1f}s)",
            flush=True,
        )

        for pi, n_g in enumerate(ga_targets):
            rem = deadline - time.time()
            if rem < 140.0:
                break
            p_b = (rem - 15.0) / max(1, len(ga_targets) - pi)
            bseed = args.seed_off + 500000 + args.shard * 4099 + pi * 521
            L0_g = wins[n_g][0] if n_g in wins else cur[n_g][0]
            P0_g = cur[n_g][1]
            nbrs_g = list(dp_nbrs.get(n_g, [])[:3])

            ds_jobs = [
                ("defsnap", dtag, n_g, L0_g, P_ds, bseed + di)
                for di, (dtag, P_ds) in enumerate(btr.defect_snap_candidates(P0_g))
            ]
            if ds_jobs:
                futs_d = [ex_pool.submit(btr.eval_bidir_seed_job, jb) for jb in ds_jobs]
                try:
                    for f in as_completed(futs_d, timeout=28.0):
                        _, tg_d, _, S_d, Q_d = f.result()
                        if Q_d is not None and S_d < L0_g + 0.015:
                            if S_d < L0_g - 1e-9:
                                Qp_d, Sp_d = w51d.deep_contact_polish(
                                    n_g, Q_d, deadline, slp_sec=34.0, fq_sec=20
                                )
                                record_win_n(n_g, Qp_d, Sp_d, f"defsnap_{tg_d}", 0)
                                Q_d = Qp_d
                            nbrs_g.append(Q_d)
                except TimeoutError:
                    w51d.terminate_pool(ex_pool)
                    ex_pool = ProcessPoolExecutor(max_workers=args.procs)

            if n_g + 1 <= 1000 and cur[n_g + 1][1] is not None:
                nbrs_g.append(cur[n_g + 1][1])

            ga_sec = min(440.0, max(120.0, 0.65 * (p_b - 25.0)))
            Qw, Qc, js = RE.run_santa_ga(
                P0_g,
                nbr_packings=nbrs_g,
                seed=bseed + 77,
                sec=ga_sec,
                islands=6,
                pop=14,
                offspring=28,
                steps=125,
                threads=args.procs,
                timeout_s=ga_sec + 35.0,
            )
            print(
                f"  [W71S GA-v5] n={n_g:4d} gens={js.get('gens', 0)} swaps={js.get('swaps', 0)} "
                f"ga_wins={js.get('wins', 0)} s_best={js.get('s_best', L0_g):.7f} (L0={L0_g:.7f})",
                flush=True,
            )
            hop_P = P0_g
            hop_tg = "p3_cur"
            if Qw is not None:
                Qp_w, Sp_w = w51d.deep_contact_polish(
                    n_g, Qw, deadline, slp_sec=40.0, fq_sec=24
                )
                record_win_n(n_g, Qp_w, Sp_w, "w71s_ga_v5_direct", js.get("wins", 1))
                hop_P, hop_tg = Qp_w, "w71s_ga_v5_win"
            elif Qc is not None:
                rq_c = RE.run_fq_quench_args(
                    Qc,
                    [
                        "--seed",
                        str(bseed + 199),
                        "--loosen",
                        "1.0045",
                        "--outer",
                        "44",
                        "--pit",
                        "280",
                        "--flip-top",
                        "20",
                        "--finish",
                        "--stag-w",
                        "0",
                        "--deadline",
                        "13.5",
                    ],
                    s_ref=L0_g,
                    timeout_s=16.5,
                )
                if rq_c is not None:
                    Sc_p, Qc_p = RE.kkt_sqp_polish_fq(rq_c[1], timeout_each=6.0)
                    if Sc_p < L0_g - 1e-9:
                        Qp_c, Sp_c = w51d.deep_contact_polish(
                            n_g, Qc_p, deadline, slp_sec=38.0, fq_sec=22
                        )
                        record_win_n(n_g, Qp_c, Sp_c, "w71s_ga_v5_champ", 0)
                        Qc_p, Sc_p = Qp_c, Sp_c
                    if Sc_p < L0_g + 0.013:
                        hop_P, hop_tg = Qc_p, "w71s_ga_v5_champ"

            hop_b = max(70.0, min(deadline - time.time() - 12.0, p_b - ga_sec - 35.0))
            if hop_b >= 65.0:
                L0_now = wins[n_g][0] if n_g in wins else cur[n_g][0]
                Qb, Sb, (_, _, wt, _) = w63k.continuous_momentum_widehopper(
                    ex_pool,
                    n_g,
                    hop_P,
                    L0_now,
                    bseed + 90000,
                    hop_b,
                    deadline,
                    lambda Qx, Sx, stg, wc, _n=n_g, _tg=hop_tg: record_win_n(
                        _n, Qx, Sx, f"{_tg}->{stg}", wc
                    ),
                )
                record_win_n(n_g, Qb, Sb, f"{hop_tg}->whFin", wt)
    finally:
        w51d.terminate_pool(ex_pool)

    save_ckpt()
    print(
        f"[W71S s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
