#!/usr/bin/env python3
"""Wave 79B (`w79b_untouched_defect_twins_slice3_gav8.py`):
80-Core (`20` Shards x `2` Puzzles = `40` Slice-3 `[120:160]` UNTOUCHED `OLD_333`-Twin Defect Puzzles!)
6-Island Pressure-Ladder + Unscaled Hole-Squeeze + Defect-Snap Rust `santa_ga` v8!

Zero-overlap continuation of Wave 77 (`[0:80]`) & Wave 78B (`[80:120]`):
  - Targets the next 40 untouched `rank > 48` `OLD_333`-twin jammed low-tilt defect puzzles,
    strictly disjoint from W77, W78A, W78B, and W79A!
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
import kink_polish_and_strip_lifts_w76 as kps
import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w56g_unified_global_faultdp as w56g
import w63k_1puzzle_per_shard_killshot as w63k
import w77_untouched_defect_twins_santa_ga_v7 as w77
import w78a_unscaled_hole_squeeze_basin_invasion_gav8 as w78a
import w78b_untouched_defect_twins_slice2_gav8 as w78b
import w79a_large_trust_fixed_lp_dualwall_unlock_gav8 as w79a


def select_slice3_untouched_defect_twins(cur, front_L, extra_exclude=None):
    w77_80, _ = w77.select_80_untouched_defect_twins(cur, front_L)
    w78a_40 = w78a.select_40_upward_targets(cur, front_L, num_targets=40)
    w78b_40, _ = w78b.select_slice2_untouched_defect_twins(
        cur, front_L, exclude_set=set(w78a_40)
    )
    excl = set(w77_80) | set(w78a_40) | set(w78b_40) | set(extra_exclude or ())

    ranked_all = []
    for n in range(95, 999):
        K = math.isqrt(n)
        if K * K == n or cur[n][1] is None:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gl > 2.5e-5:
            ranked_all.append((gl / math.sqrt(n), n))
    ranked_all.sort(reverse=True)
    top48_set = {ni for _, ni in ranked_all[:48]}

    twins = []
    fallback = []
    for n in range(95, 999):
        K = math.isqrt(n)
        if K * K == n or cur[n][1] is None or n in top48_set or n in excl:
            continue
        L, P = cur[n]
        if L >= K + 1 - 1e-7:
            continue
        gl = max(0.0, L - front_L.get(n, L))
        if gl <= 2.5e-5:
            continue
        deg = np.degrees(P[:, 2]) % 90.0
        fd = np.minimum(deg, 90.0 - deg)
        low_def = int(((fd > 0.25) & (fd < 12.5)).sum())
        mid_def = int(((fd > 0.25) & (fd < 8.5)).sum())
        base_ds = gl / math.sqrt(n)
        if low_def >= 3:
            bonus = 1.38 if mid_def >= 8 else (1.18 if mid_def >= 4 else 1.0)
            twins.append((base_ds * bonus, n, low_def, mid_def))
        else:
            fallback.append((base_ds, n, low_def, mid_def))

    twins.sort(reverse=True)
    fallback.sort(reverse=True)
    combined = [x[1] for x in twins] + [x[1] for x in fallback]
    defect_info = {x[1]: (x[2], x[3]) for x in (twins + fallback)}
    return combined[:40], defect_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--dp-cache-npz", default="")
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=5150000)
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

    w79a_40 = w79a.select_40_dualwall_targets(cur, front_L, num_targets=40)
    slice3_twins, def_info = select_slice3_untouched_defect_twins(
        cur, front_L, extra_exclude=set(w79a_40)
    )
    my_two = slice3_twins[args.shard :: args.num_shards]

    print(
        f"[W79B-DefectTwinsSlice3-GAv8 s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"TARGETS={[(n, def_info.get(n, (0, 0))) for n in my_two]} ({time.time() - t0:.1f}s)",
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
                f"  [W79B DEFECT-TWIN-S3 WIN!!] n={n_w:4d} [{src_tag}] -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_two):
            rem_all = deadline - time.time()
            if rem_all < 130.0:
                break
            puz_b = (rem_all - 15.0) / max(1, len(my_two) - pi)
            puz_dl = time.time() + puz_b
            bseed = args.seed_off + args.shard * 8191 + pi * 997

            L0 = cur[n][0]
            P0 = cur[n][1]
            nbrs = []
            if (
                n > 4
                and math.isqrt(n - 1) ** 2 != (n - 1)
                and cur[n - 1][1] is not None
            ):
                nbrs.append(cur[n - 1][1])

            seed_jobs = []
            for di, (dtag, P_ds) in enumerate(btr.defect_snap_candidates(P0)):
                seed_jobs.append(("defsnap", dtag, n, L0, P_ds, bseed + 10 + di))

            for s_lf, tg_lf, q_lf in kps.sweep_strip_and_l_lifts_for_n(n, cur)[:2]:
                if s_lf < L0 + 0.015:
                    nbrs.append(q_lf)

            if seed_jobs:
                futs = [ex_hop.submit(btr.eval_bidir_seed_job, jb) for jb in seed_jobs]
                cand_seeds = []
                try:
                    for f in as_completed(futs, timeout=40.0):
                        md, tg, _, S_s, Q_s = f.result()
                        if Q_s is None or not math.isfinite(S_s):
                            continue
                        if S_s < cur[n][0] - 1e-9:
                            Qp_s, Sp_s = w51d.deep_contact_polish(
                                n, Q_s, puz_dl, slp_sec=36.0, fq_sec=20
                            )
                            record_win_n(n, Qp_s, Sp_s, f"seed_{tg}", 0)
                            Q_s, S_s = Qp_s, Sp_s
                        ad_s = w53h.topology_angle_dist(Q_s, P0)
                        if S_s < cur[n][0] + 0.016:
                            cand_seeds.append((S_s, -min(ad_s, 6.0), tg, Q_s))
                except TimeoutError:
                    w51d.terminate_pool(ex_hop)
                    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
                cand_seeds.sort(key=lambda z: (round(z[0], 4), z[1]))
                for _, _, _, Q_s in cand_seeds[:3]:
                    nbrs.append(Q_s)

            K = math.isqrt(n)
            D = n - K * K
            if dp_ctx is not None and K <= fdp8.WMAX and 0 < D <= fdp8.DMAX:
                donors_c, donor_n_c, G_c, dp_c, PG_c, PM_c, PQ_c = dp_ctx
                for L_est, dtag, P_comp in fdp8.trace_and_compose_v2(
                    dp_c, PG_c, PM_c, PQ_c, G_c, donors_c, donor_n_c, K, D, top_m=2
                ):
                    if L_est < cur[n][0] + 0.016:
                        rh = w56g.quench_composed_chain(P_comp, cur[n][0])
                        if rh is not None and rh[1] < cur[n][0] + 0.014:
                            nbrs.append(rh[0])

            if n + 1 <= 1000 and cur[n + 1][1] is not None:
                nbrs.append(cur[n + 1][1])

            rem_puz = max(160.0, puz_dl - time.time())
            ga_sec = min(580.0 if pi == 0 else 470.0, max(140.0, 0.65 * rem_puz))
            L0_ga = cur[n][0]
            P0_ga = cur[n][1]
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
                timeout_s=ga_sec + 65.0,
            )
            print(
                f"  [W79B GAv8 done] n={n:4d} gens={js.get('gens', 0)} swaps={js.get('swaps', 0)} "
                f"ga_wins={js.get('wins', 0)} resets={js.get('resets', 0)} s_best={js.get('s_best', L0_ga):.7f} (L0={L0_ga:.7f})",
                flush=True,
            )

            hop_P = cur[n][1]
            hop_tag = "cur_after_gav8"
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
                        "11.0",
                    ],
                    timeout_s=14.0,
                )
                if rq_w is not None:
                    Qp_w, Sp_w = w51d.deep_contact_polish(
                        n, rq_w[1], puz_dl, slp_sec=44.0, fq_sec=26
                    )
                    Sk_w, Qk_w = kps.kink_contact_lp_polish(
                        Qp_w, rounds=40, seed=bseed + 77, max_seconds=12.0
                    )
                    if Sk_w < Sp_w:
                        Qp_w, Sp_w = Qk_w, Sk_w
                    record_win_n(n, Qp_w, Sp_w, "santa_ga_v8_direct", js.get("wins", 1))
                    hop_P = Qp_w
                    hop_tag = "santa_ga_v8_win"

            if hop_tag == "cur_after_gav8" and Qc is not None:
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
                    if Sc_pol < cur[n][0] - 1e-9:
                        Qp_c, Sp_c = w51d.deep_contact_polish(
                            n, Qc_pol, puz_dl, slp_sec=42.0, fq_sec=24
                        )
                        record_win_n(n, Qp_c, Sp_c, "santa_ga_v8_champ_dir", 0)
                        Qc_pol, Sc_pol = Qp_c, Sp_c
                    if Sc_pol < cur[n][0] + 0.0135:
                        hop_P = Qc_pol
                        hop_tag = "santa_ga_v8_champ"

            L0_cur = cur[n][0]
            ad = w53h.topology_angle_dist(hop_P, P0)
            hop_b = max(85.0, min(deadline - time.time() - 15.0, puz_dl - time.time()))
            Qb, Sb, (_, _, wt, nr) = w63k.continuous_momentum_widehopper(
                ex_hop,
                n,
                hop_P,
                L0_cur,
                bseed + 60000,
                hop_b,
                deadline,
                lambda Qx, Sx, stg, wc, _n=n, _tg=hop_tag: record_win_n(
                    _n, Qx, Sx, f"{_tg}->{stg}", wc
                ),
            )
            S_fin, Q_fin = kps.kink_contact_lp_polish(
                Qb, rounds=38, seed=bseed + 131, max_seconds=12.0
            )
            if S_fin < Sb:
                Qb, Sb = Q_fin, S_fin
            record_win_n(n, Qb, Sb, f"{hop_tag}->wh->kink", wt)
            print(
                f"  [W79B done] n={n:4d} [{hop_tag}] dA={ad:.2f} -> dEnd={Sb - L0:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W79B s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
