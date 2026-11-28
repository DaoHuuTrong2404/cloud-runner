#!/usr/bin/env python3
"""Wave 78A (`w78a_unscaled_hole_squeeze_basin_invasion_gav8.py`):
80-Core (`20` Shards x `2` Puzzles = `40` Top-Gap Upward Basin-Invasion Puzzles!)
Unscaled 5-Corner + Seam-Split `16,000`-Iter `m=12` L-BFGS Hole-Insertion + Bisection Squeeze (`santa_ga` v8)!

Root-cause breakthroughs unlocked in Wave 78A:
  1. Fixed `rust_engines.py` line 789 (`len(Pn) >= max(1, len(P) - 4)`): previously `len(Pn) >= len(P)`
     silently dropped 100% of `m < n` (`cur[n - 1]`, `cur[n - 2]`) neighbors before `santa_ga` ran!
  2. Fixed primary `--in` anchoring: `P0 = cur[n][1]` (`L0 = cur[n][0]`) is ALWAYS passed as `--in` so
     `santa_ga` v8's `best_legal_s` starts at exact `L0` (never inflated by pre-healed donors).
  3. Upgraded `santa_ga.rs` -> v8 (`synthesize_upward_insert_seeds` + `bisection_squeeze_one`):
     places `cur[n - 1]` and `cur[n - 2]` UNSCALED at 5 corner/perimeter anchors + 2 interior seam-split
     trenches inside `[0, L_box]^2`, probes 640 Voronoi `free_spots` per square, runs `16,000`-step
     `m=12` L-BFGS + 7-round `Relaxer2W` (`mp < 0.08`), and executes 44-step multiplicative bisection
     `squeeze` down to `rel = 3e-10`!
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

PRIORITY_NM1_TARGETS = [
    234,
    491,
    738,
    456,
    261,
    904,
    414,
    295,
    367,
    332,
    804,
    914,
    748,
    371,
    273,
    377,
    976,
    500,
    209,
    303,
    632,
    240,
    499,
    540,
    911,
    638,
    747,
    926,
]


def center_preserving_frame_peels(P_big, n_target):
    """Peel outer upright border squares from `P_big` (`(K+1)^2 + D -> K^2 + D` or `n -> n - 1`)
    without chopping an interior centered diamond or tilted block.
    """
    P = RE.normalize_to_origin(P_big)
    if len(P) <= n_target:
        return []
    L = RE.side_exact(P)
    t_mod = np.mod(P[:, 2], 0.5 * math.pi)
    fold = np.minimum(t_mod, 0.5 * math.pi - t_mod)
    is_tilted = (fold > 0.015).astype(np.float64)

    cx, cy = 0.5 * L, 0.5 * L
    out = []
    score_sym = (
        np.maximum(np.abs(P[:, 0] - cx), np.abs(P[:, 1] - cy)) - 50.0 * is_tilted
    )
    keep_sym = np.argsort(score_sym)[:n_target]
    out.append(("cpeel_sym", RE.normalize_to_origin(P[keep_sym])))

    for tag, sx, sy in (
        ("cpeel_BL", P[:, 0], P[:, 1]),
        ("cpeel_BR", L - P[:, 0], P[:, 1]),
        ("cpeel_TL", P[:, 0], L - P[:, 1]),
        ("cpeel_TR", L - P[:, 0], L - P[:, 1]),
    ):
        sc = np.maximum(sx, sy) - 50.0 * is_tilted
        keep = np.argsort(sc)[:n_target]
        out.append((tag, RE.normalize_to_origin(P[keep])))
    return out


def select_40_upward_targets(cur, front_L, num_targets=40):
    ranked = []
    for n in range(95, 999):
        K = math.isqrt(n)
        if K * K == n or cur[n][1] is None or cur[n - 1][1] is None:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        d_m1 = cur[n][0] - cur[n - 1][0]
        d_m2 = (
            (cur[n][0] - cur[n - 2][0])
            if (n >= 6 and cur[n - 2][1] is not None)
            else 0.0
        )
        if gl > 2.5e-5 and (d_m1 > 0.010 or d_m2 > 0.022):
            bonus = 1.50 if d_m1 > 0.024 else (1.22 if d_m1 > 0.015 else 1.0)
            ranked.append(((gl / math.sqrt(n)) * bonus, n))
    ranked.sort(reverse=True)

    ordered = []
    seen = set()
    for n in PRIORITY_NM1_TARGETS:
        K = math.isqrt(n)
        if (
            K * K != n
            and cur[n][1] is not None
            and cur[n - 1][1] is not None
            and n not in seen
        ):
            ordered.append(n)
            seen.add(n)
    for _, n in ranked:
        if len(ordered) >= num_targets:
            break
        if n not in seen:
            ordered.append(n)
            seen.add(n)
    return ordered[:num_targets]


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
    ap.add_argument("--seed-off", type=int, default=4550000)
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

    ordered = select_40_upward_targets(cur, front_L, num_targets=2 * args.num_shards)
    my_two = ordered[args.shard :: args.num_shards]

    print(
        f"[W78A-UnscaledHoleSqueeze-GAv8 s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"TARGETS={my_two} ({time.time() - t0:.1f}s)",
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
                f"  [W78A HOLE-SQUEEZE WIN!!] n={n_w:4d} [{src_tag}] -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_two):
            rem_all = deadline - time.time()
            if rem_all < 140.0:
                break
            puz_b = (rem_all - 15.0) / max(1, len(my_two) - pi)
            puz_dl = time.time() + puz_b
            bseed = args.seed_off + args.shard * 8191 + pi * 997

            L0 = cur[n][0]
            P0 = cur[n][1]

            # Pass raw `cur[n - 1]` (`m = n - 1`) & `cur[n - 2]` (`m = n - 2`) FIRST into `nbrs`!
            # With `rust_engines.py` fixed (`len(Pn) >= n - 4`), `santa_ga` v8 executes 36-candidate
            # Unscaled 5-Corner + Seam-Split `16,000`-iter `m=12` L-BFGS + `Relaxer2W` + Bisection Squeeze!
            nbrs = []
            if (
                n > 4
                and math.isqrt(n - 1) ** 2 != (n - 1)
                and cur[n - 1][1] is not None
            ):
                nbrs.append(cur[n - 1][1])
            if (
                n >= 6
                and math.isqrt(n - 2) ** 2 != (n - 2)
                and cur[n - 2][1] is not None
            ):
                nbrs.append(cur[n - 2][1])

            K = math.isqrt(n)
            n_up = n + 2 * K + 1
            if n_up <= 1000 and cur[n_up][1] is not None:
                # Peel `cur[n + 2K + 1]` down to `n - 1` so `santa_ga` v8's unscaled hole-inserter + squeeze can place the final square cleanly!
                for _, P_cp1 in center_preserving_frame_peels(cur[n_up][1], n - 1)[:1]:
                    nbrs.append(P_cp1)

            # Stage 1 (~38s): Defect-Snap + Strip/L-Lift + Soft-Insert pre-seeds (`m == n`)
            seed_jobs = []
            for di, (dtag, P_ds) in enumerate(btr.defect_snap_candidates(P0)[:3]):
                seed_jobs.append(("defsnap", dtag, n, L0, P_ds, bseed + 10 + di))
            if n > 4 and cur[n - 1][1] is not None:
                for ji, (itag, P_ins) in enumerate(
                    btr.insert_holes_candidates(
                        cur[n - 1][1],
                        P_n_own=P0,
                        seed=bseed + 31,
                        n_holes=3,
                        n_target=2,
                    )
                ):
                    seed_jobs.append(("ins", itag, n, L0, P_ins, bseed + 50 + ji))

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
                        if S_s < cur[n][0] + 0.018:
                            cand_seeds.append((S_s, -min(ad_s, 6.0), tg, Q_s))
                except TimeoutError:
                    w51d.terminate_pool(ex_hop)
                    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
                cand_seeds.sort(key=lambda z: (round(z[0], 4), z[1]))
                for _, _, _, Q_s in cand_seeds[:2]:
                    nbrs.append(Q_s)

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

            # Stage 2 (~560s): Always pass `cur[n][1]` as `--in` so `best_legal_s == cur[n][0]`!
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
                timeout_s=ga_sec + 95.0,
            )
            print(
                f"  [W78A GAv8 done] n={n:4d} d_m1=+{L0_ga - cur[n - 1][0]:.5f} nbrs={len(nbrs)} "
                f"gens={js.get('gens', 0)} swaps={js.get('swaps', 0)} ga_wins={js.get('wins', 0)} "
                f"s_best={js.get('s_best', L0_ga):.7f} (L0={L0_ga:.7f})",
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
                f"  [W78A done] n={n:4d} [{hop_tag}] dA={ad:.2f} -> dEnd={Sb - L0:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W78A s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
