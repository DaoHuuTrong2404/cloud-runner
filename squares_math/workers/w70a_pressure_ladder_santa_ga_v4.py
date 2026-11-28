#!/usr/bin/env python3
"""Wave 70A (`w70a_pressure_ladder_santa_ga_v4.py`):
6-Island Pressure-Ladder Replica-Exchange Rust Santa-GA v4 (`+0.0042` & `+0.0095` Breathing Arm-Growth Islands
with `arm_extend` 1..4 + `parallel_arm_spawn` + Warmup Novelty Selection + Odriozola Downward Replica Swaps!)
seeded with 5 distinct topology families on all Top-48 Gap Puzzles (`ranked[:48][shard :: 24]`):
  1) `cur[n]` incumbent,
  2) `n - 1 -> n` `insert_holes` + `target-hole insertion` (`malyshevdanil` `run_gpu.py` via Rust `fq --round-r0`),
  3) `(K + 1)^2 + D -> K^2 + D` 4-corner Soft L-Peel (`cur[n + 2K + 1]` healed via `harden reharden` + `fq`),
  4) `216`-Gadget (`W64..W69S`) 8-Sym DP compositions (`trace_and_compose_v2`),
  5) `cur[n + 1..n + 2]` neighbor drops -> `santa_ga` v4 -> KKT SQP -> Continuous WideHopper!
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
import w56g_unified_global_faultdp as w56g
import w63k_1puzzle_per_shard_killshot as w63k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--dp-cache-npz", default="")
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=24)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=2950000)
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

    dp_ctx = None
    if args.dp_cache_npz and os.path.exists(args.dp_cache_npz):
        try:
            dp_ctx = pdc.load_dp8_cache(args.dp_cache_npz)
        except Exception:
            dp_ctx = None

    ranked = []
    for n in range(95, 999):
        K = math.isqrt(n)
        if K * K == n or cur[n][1] is None:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gl > 2.5e-5:
            ranked.append((gl / math.sqrt(n), n))
    ranked.sort(reverse=True)
    my_two = [x[1] for x in ranked[: 2 * args.num_shards]][
        args.shard :: args.num_shards
    ]

    print(
        f"[W70A-SantaGAv4 s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"dp_cache={dp_ctx is not None} TARGETS={my_two} ({time.time() - t0:.1f}s)",
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
                f"  [W70A SANTA-GAv4 WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_two):
            rem_all = deadline - time.time()
            if rem_all < 120.0:
                break
            puz_b = (rem_all - 15.0) / max(1, len(my_two) - pi)
            bseed = args.seed_off + args.shard * 7919 + pi * 997

            L0 = wins[n][0] if n in wins else cur[n][0]
            P0 = cur[n][1]
            nbrs = []

            # Stage 1 (~35s): Parallel Bidirectional Seed Synthesis (`n-1` hole insertion + `(K+1)^2+D` 4-corner L-peel + `n+1` LP)
            seed_jobs = []
            if n > 4 and cur[n - 1][1] is not None:
                ins_cands = btr.insert_holes_candidates(
                    cur[n - 1][1], P_n_own=P0, seed=bseed + 3, n_holes=3, n_target=2
                )
                for ji, (itag, P_ins) in enumerate(ins_cands):
                    seed_jobs.append(("ins", itag, n, L0, P_ins, bseed + 50 + ji))

            K = math.isqrt(n)
            n_up = n + 2 * K + 1
            if (
                n_up <= 1000
                and cur[n_up][1] is not None
                and cur[n_up][0] - 1.0 < L0 + 0.008
            ):
                for ji, (ptag, P_peel) in enumerate(
                    btr.soft_corner_lpeel_candidates(cur[n_up][1], n)
                ):
                    seed_jobs.append(("lpeel", ptag, n, L0, P_peel, bseed + 90 + ji))

            if (
                n + 1 <= 1000
                and cur[n + 1][1] is not None
                and cur[n + 1][0] - L0 < 0.055
            ):
                del_cands = btr.delete_one_candidates(
                    cur[n + 1][1], max_boundary=8, max_sparse=5
                )
                seed_jobs.append(("del_lp", "del1", n, L0, del_cands, bseed + 130))

            if seed_jobs:
                futs = [ex_hop.submit(btr.eval_bidir_seed_job, jb) for jb in seed_jobs]
                cand_seeds = []
                try:
                    for f in as_completed(futs, timeout=44.0):
                        md, tg, _, S_s, Q_s = f.result()
                        if Q_s is None or not math.isfinite(S_s):
                            continue
                        if S_s < L0 - 1e-9:
                            Qp_s, Sp_s = w51d.deep_contact_polish(
                                n, Q_s, deadline, slp_sec=36.0, fq_sec=20
                            )
                            record_win_n(n, Qp_s, Sp_s, f"bidir_{tg}", 0)
                            Q_s, S_s = Qp_s, Sp_s
                        ad_s = w53h.topology_angle_dist(Q_s, P0)
                        if S_s < L0 + 0.015:
                            cand_seeds.append((S_s, ad_s, tg, Q_s))
                except TimeoutError:
                    w51d.terminate_pool(ex_hop)
                    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
                cand_seeds.sort(key=lambda z: z[0])
                for S_s, ad_s, tg, Q_s in cand_seeds[:3]:
                    nbrs.append(Q_s)

            # Stage 2: 216-Gadget (`W64..W69S`) 8-Sym DP compositions
            if dp_ctx is not None:
                donors_c, donor_n_c, G_c, dp_c, PG_c, PM_c, PQ_c = dp_ctx
                D = n - K * K
                if K <= fdp8.WMAX and 0 < D <= fdp8.DMAX:
                    dp_ch = fdp8.trace_and_compose_v2(
                        dp_c, PG_c, PM_c, PQ_c, G_c, donors_c, donor_n_c, K, D, top_m=3
                    )
                    for L_est, dtag, P_comp in dp_ch:
                        if L_est < L0 + 0.018:
                            rh = w56g.quench_composed_chain(P_comp, L0)
                            if rh is not None:
                                Qc_dp, Sc_dp = rh
                                if Sc_dp < L0 + 0.00030:
                                    Qp_dp, Sp_dp = w51d.deep_contact_polish(
                                        n, Qc_dp, deadline, slp_sec=36.0, fq_sec=22
                                    )
                                    record_win_n(n, Qp_dp, Sp_dp, dtag + "_prega", 0)
                                    Qc_dp, Sc_dp = Qp_dp, Sp_dp
                                if Sc_dp < L0 + 0.014:
                                    nbrs.append(Qc_dp)
                                if len(nbrs) >= 5:
                                    break

            for m in (n + 1, n + 2):
                if m <= 1000 and cur[m][1] is not None:
                    nbrs.append(cur[m][1])

            rem_puz = max(160.0, puz_b - 42.0)
            ga_sec = min(610.0 if pi == 0 else 470.0, max(130.0, 0.64 * rem_puz))
            L0_ga = wins[n][0] if n in wins else cur[n][0]
            P0_ga = (
                np.column_stack([wins[n][1], wins[n][2], wins[n][3]])
                if n in wins
                else P0
            )
            Qw, Qc, js = RE.run_santa_ga(
                P0_ga,
                nbr_packings=nbrs,
                seed=bseed,
                sec=ga_sec,
                islands=6,
                pop=14,
                offspring=28,
                steps=125,
                threads=args.procs,
                timeout_s=ga_sec + 35.0,
            )
            print(
                f"  [W70A GA-v4 done] n={n:4d} gens={js.get('gens', 0)} swaps={js.get('swaps', 0)} "
                f"ga_wins={js.get('wins', 0)} resets={js.get('resets', 0)} s_best={js.get('s_best', L0_ga):.7f} (L0={L0_ga:.7f})",
                flush=True,
            )

            hop_start_P = P0_ga
            hop_tag = "cur_after_ga"
            if Qw is not None:
                rq_w = RE.run_fq_quench_args(
                    Qw,
                    [
                        "--loosen",
                        "1.0003",
                        "--outer",
                        "25",
                        "--pit",
                        "450",
                        "--finish",
                        "--stag-w",
                        "0",
                        "--rmin",
                        "1e-12",
                        "--deadline",
                        "10.0",
                    ],
                    timeout_s=12.5,
                )
                if rq_w is not None:
                    Qp_w, Sp_w = w51d.deep_contact_polish(
                        n, rq_w[1], deadline, slp_sec=44.0, fq_sec=26
                    )
                    record_win_n(n, Qp_w, Sp_w, "santa_ga_v4_direct", js.get("wins", 1))
                    hop_start_P = Qp_w
                    hop_tag = "santa_ga_v4_win"

            if hop_tag == "cur_after_ga" and Qc is not None:
                rq_c = RE.run_fq_quench_args(
                    Qc,
                    [
                        "--seed",
                        str(bseed + 17),
                        "--loosen",
                        "1.005",
                        "--outer",
                        "48",
                        "--pit",
                        "300",
                        "--flip-top",
                        "22",
                        "--finish",
                        "--stag-w",
                        "0",
                        "--deadline",
                        "15.0",
                    ],
                    s_ref=L0_ga,
                    timeout_s=18.0,
                )
                if rq_c is not None:
                    Sc_pol, Qc_pol = RE.kkt_sqp_polish_fq(rq_c[1], timeout_each=6.5)
                    if Sc_pol < L0_ga - 1e-9:
                        Qp_c, Sp_c = w51d.deep_contact_polish(
                            n, Qc_pol, deadline, slp_sec=42.0, fq_sec=24
                        )
                        record_win_n(n, Qp_c, Sp_c, "santa_ga_v4_champ_dir", 0)
                        Qc_pol, Sc_pol = Qp_c, Sp_c
                    if Sc_pol < L0_ga + 0.0135:
                        hop_start_P = Qc_pol
                        hop_tag = "santa_ga_v4_champ"

            L0_cur = wins[n][0] if n in wins else cur[n][0]
            ad = w53h.topology_angle_dist(hop_start_P, P0)
            hop_b = max(85.0, min(deadline - time.time() - 15.0, puz_b - ga_sec - 55.0))
            Qb, Sb, (_, _, wt, nr) = w63k.continuous_momentum_widehopper(
                ex_hop,
                n,
                hop_start_P,
                L0_cur,
                bseed + 60000,
                hop_b,
                deadline,
                lambda Qx, Sx, stg, wc, _n=n, _tg=hop_tag: record_win_n(
                    _n, Qx, Sx, f"{_tg}->{stg}", wc
                ),
            )
            record_win_n(n, Qb, Sb, f"{hop_tag}->whFin", wt)
            print(
                f"  [W70A done] n={n:4d} [{hop_tag}] dA={ad:.2f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W70A s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
