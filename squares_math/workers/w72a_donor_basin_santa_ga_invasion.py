#!/usr/bin/env python3
"""Wave 72A (`w72a_donor_basin_santa_ga_invasion.py`):
PURE DONOR-BASIN INVASION VIA 6-ISLAND RUST SANTA-GA v5!

Root cause proven in `diagnose_top15_geometry.py`:
  - Top gap puzzles (`234 til=36`, `367 til=90`, `491 til=101`, `738 til=120`, `456 til=60`,
    `904 til=115`, `414 til=50`, `785/730`) are trapped in WRONG LOW-TILT BASINS!
  - Meanwhile their neighbours (`233/265 til=121`, `368 til=168`, `490/492/536 til=128..147`,
    `737/739/793 til=203..247`, `455/457 til=136..167`, `903/965 til=338..365`, `842 til=303`)
    already sit in the TRUE HIGH-TILT BASINS!
  - Previously, passing `cur[n]` as `--in` locked `min_l` at `cur[n][0]`, causing `far_behind`
    to execute every donor island before it could descend `+0.010` down below `cur[n][0]`!
  - FIX: Synthesize deep soft-healed high-tilt donor packings (`Q_donor_1`, `Q_donor_2`) and
    pass **`Q_donor` DIRECTLY AS `--in` (WITHOUT `cur[n]`!)** so all 6 `santa_ga` islands +
    subsequent `WideHopper` evolve 100% INSIDE the high-tilt donor basin down below `cur[n][0]`!
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
    ap.add_argument("--seed-off", type=int, default=3350000)
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
        f"[W72A-DonorInvasion s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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
                f"  [W72A DONOR-INVASION WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    ex = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_two):
            rem_all = deadline - time.time()
            if rem_all < 130.0:
                break
            puz_b = (rem_all - 15.0) / max(1, len(my_two) - pi)
            bseed = args.seed_off + args.shard * 8521 + pi * 997
            L0 = wins[n][0] if n in wins else cur[n][0]
            P0 = cur[n][1]

            # 1) Synthesize genuinely distinct donor topologies (`dA >= 0.40` vs `P0`)!
            d_jobs = btr.build_donor_invasion_jobs(n, cur, bseed)
            distinct_donors = []
            if d_jobs:
                futs = [ex.submit(btr.eval_deep_donor_job, jb) for jb in d_jobs]
                try:
                    for f in as_completed(futs, timeout=75.0):
                        fam, tg, _, S_d, Q_d = f.result()
                        if Q_d is None or not math.isfinite(S_d):
                            continue
                        if S_d < L0 + 0.00020:
                            Qp_d, Sp_d = w51d.deep_contact_polish(
                                n, Q_d, deadline, slp_sec=38.0, fq_sec=22
                            )
                            record_win_n(n, Qp_d, Sp_d, f"direct_{tg}", 0)
                            Q_d, S_d = Qp_d, Sp_d
                        ad_d = w53h.topology_angle_dist(Q_d, P0)
                        if ad_d >= 0.35 and S_d < L0 + 0.024:
                            distinct_donors.append((S_d - L0, ad_d, tg, Q_d))
                except TimeoutError:
                    w51d.terminate_pool(ex)
                    ex = ProcessPoolExecutor(max_workers=args.procs)

            # Also check 223-gadget DP distinct basins!
            if dp_ctx is not None:
                donors_c, donor_n_c, G_c, dp_c, PG_c, PM_c, PQ_c = dp_ctx
                K = math.isqrt(n)
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
                                ad_dp = w53h.topology_angle_dist(Qc_dp, P0)
                                if ad_dp >= 0.35 and Sc_dp < L0 + 0.020:
                                    distinct_donors.append(
                                        (Sc_dp - L0, ad_dp, dtag, Qc_dp)
                                    )

            distinct_donors.sort(key=lambda z: z[0])
            if distinct_donors:
                # PURE DONOR-BASIN INVASION: primary `--in` is the best distinct donor (NO `cur[n]` so `min_l` = `S_donor`!)
                gap_d0, ad_d0, main_tag, P_in_ga = distinct_donors[0]
                nbrs_ga = [q for _, _, _, q in distinct_donors[1:5]]
                if n + 1 <= 1000 and cur[n + 1][1] is not None:
                    nbrs_ga.append(cur[n + 1][1])
                print(
                    f"  [W72A invade] n={n:4d} donors={len(distinct_donors)} primary={main_tag} "
                    f"dStart={gap_d0:+.5f} dA={ad_d0:.2f}!",
                    flush=True,
                )
            else:
                main_tag = "cur_fallback"
                P_in_ga = P0
                nbrs_ga = [q for _, q in btr.defect_snap_candidates(P0)[:3]]
                if n + 1 <= 1000 and cur[n + 1][1] is not None:
                    nbrs_ga.append(cur[n + 1][1])

            rem_puz = max(150.0, min(deadline - time.time() - 15.0, puz_b - 65.0))
            ga_sec = min(590.0 if pi == 0 else 450.0, max(120.0, 0.65 * rem_puz))
            s_in_ga = RE.side_exact(P_in_ga)
            Qw, Qc, js = RE.run_santa_ga(
                P_in_ga,
                nbr_packings=nbrs_ga,
                seed=bseed + 777,
                sec=ga_sec,
                islands=6,
                pop=14,
                offspring=28,
                steps=125,
                threads=args.procs,
                timeout_s=ga_sec + 35.0,
            )
            print(
                f"  [W72A GA done] n={n:4d} [{main_tag}] gens={js.get('gens', 0)} ga_wins={js.get('wins', 0)} "
                f"s_in={s_in_ga:.7f} -> s_best={js.get('s_best', s_in_ga):.7f} (cur_L0={L0:.7f})",
                flush=True,
            )

            best_post_Q = Qw if Qw is not None else P_in_ga
            if Qw is not None:
                rq_w = RE.run_fq_quench_args(
                    Qw,
                    [
                        "--loosen",
                        "1.0004",
                        "--outer",
                        "28",
                        "--pit",
                        "450",
                        "--finish",
                        "--stag-w",
                        "0",
                        "--rmin",
                        "1e-12",
                        "--deadline",
                        "11.0",
                    ],
                    timeout_s=13.5,
                )
                if rq_w is not None:
                    best_post_Q = rq_w[1]
                    if rq_w[0] < (wins[n][0] if n in wins else L0) + 0.00045:
                        Qp_w, Sp_w = w51d.deep_contact_polish(
                            n, rq_w[1], deadline, slp_sec=44.0, fq_sec=26
                        )
                        record_win_n(
                            n, Qp_w, Sp_w, f"{main_tag}->ga_v5", js.get("wins", 1)
                        )
                        best_post_Q = Qp_w
            elif Qc is not None:
                rq_c = RE.run_fq_quench_args(
                    Qc,
                    [
                        "--seed",
                        str(bseed + 888),
                        "--loosen",
                        "1.0045",
                        "--outer",
                        "46",
                        "--pit",
                        "300",
                        "--flip-top",
                        "22",
                        "--finish",
                        "--stag-w",
                        "0",
                        "--deadline",
                        "14.0",
                    ],
                    s_ref=min(s_in_ga, L0 + 0.004),
                    timeout_s=17.0,
                )
                if rq_c is not None and rq_c[0] < RE.side_exact(best_post_Q):
                    best_post_Q = rq_c[1]
                    if rq_c[0] < (wins[n][0] if n in wins else L0) + 0.00045:
                        Qp_c, Sp_c = w51d.deep_contact_polish(
                            n, rq_c[1], deadline, slp_sec=42.0, fq_sec=24
                        )
                        record_win_n(n, Qp_c, Sp_c, f"{main_tag}->ga_champ", 0)
                        best_post_Q = Qp_c

            L0_cur = wins[n][0] if n in wins else cur[n][0]
            ad_fin = w53h.topology_angle_dist(best_post_Q, P0)
            hop_b = max(
                75.0, min(deadline - time.time() - 12.0, rem_puz - ga_sec - 20.0)
            )
            Qb, Sb, (_, _, wt, nr) = w63k.continuous_momentum_widehopper(
                ex,
                n,
                best_post_Q,
                L0_cur,
                bseed + 60000,
                hop_b,
                deadline,
                lambda Qx, Sx, stg, wc, _n=n, _tg=main_tag: record_win_n(
                    _n, Qx, Sx, f"{_tg}->{stg}", wc
                ),
            )
            record_win_n(n, Qb, Sb, f"{main_tag}->whFin", wt)
            print(
                f"  [W72A done] n={n:4d} [{main_tag}] dA={ad_fin:.2f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex)

    save_ckpt()
    print(
        f"[W72A s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
