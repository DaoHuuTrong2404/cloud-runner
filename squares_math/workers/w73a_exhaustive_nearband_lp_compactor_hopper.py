#!/usr/bin/env python3
"""Wave 73A (`w73a_exhaustive_nearband_lp_compactor_hopper.py`):
Exhaustive Near-Band `sq_engine` Contact-LP Compactor (`n + 1 -> n` & `n + 2 -> n`)
-> Angle-Pinned Rust `santa_ga` v6 -> Multi-Round Continuous Momentum WideHopper!

Implements `goodpjw2008`'s `NOTES.md` line 35 (`n=654` `-0.00224` method) + fixes Wave 72's
`n=234 (+0.000161)` / `n=295 (+0.000072)` / `n=632 (w=29 cut off at r=2)` bottlenecks:
  1. Isolated `<22s` Two-Stage Near-Band Contact-LP Screening (`ex_screen` -> `terminate_pool`):
     - Stage 1 (`max_iter=15, patience=2`): screens up to 148 near-band squares of `cur[n + 1]`
       PLUS top 16 `gen_del2_pairs` of `cur[n + 2]` via rotating+sliding `w51d.compact_deleted`!
     - Stage 2 (`max_iter=60, patience=2`): refines the Top-16 Stage-1 cavities!
  2. Combines top compacted cavities + `225`-Gadget (`W64..W72`) 8-Sym DP donor seeds,
     runs Angle-Pinned `santa_ga` v6 (`~290s`), and then unleashes `~520s` (up to 6 full rounds)
     of `w63k.continuous_momentum_widehopper` directly inside the tightest cavity basin!
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
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w56g_unified_global_faultdp as w56g
import w63k_1puzzle_per_shard_killshot as w63k


def select_nearband_indices(P_src, max_cands=148):
    """Select all tilted squares + upright squares within 2.55 units of any tilted square,
    prioritized by proximity to tilted bands and interior void freedom.
    """
    P = E.to_origin(P_src)
    n_m = len(P)
    tilts = np.abs(E.fold(P[:, 2]))
    tilted_idx = np.where(tilts > 0.012)[0]
    if len(tilted_idx) == 0:
        # Pure grid or near-grid: sample boundary + interior checkerboard
        cx, cy = float(np.mean(P[:, 0])), float(np.mean(P[:, 1]))
        d2 = (P[:, 0] - cx) ** 2 + (P[:, 1] - cy) ** 2
        order = np.argsort(d2)
        return [int(i) for i in order[:max_cands]]

    tx = P[tilted_idx, 0][:, None]
    ty = P[tilted_idx, 1][:, None]
    d2_min = np.min((P[:, 0][None, :] - tx) ** 2 + (P[:, 1][None, :] - ty) ** 2, axis=0)
    near_mask = d2_min <= 2.55**2
    cand = np.where(near_mask)[0]
    if len(cand) > max_cands:
        # Prioritize squares right on or immediately bordering the tilted band!
        score = d2_min[cand] - 0.35 * (tilts[cand] > 0.012).astype(np.float64)
        cand = cand[np.argsort(score)[:max_cands]]
    return [int(i) for i in cand]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--dp-cache-npz", default="")
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=3550000)
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

    # Rank puzzles where cur[n+1] is close enough for n+1 -> n cavity collapse OR top gap!
    ranked = []
    for n in range(95, 999):
        K = math.isqrt(n)
        if K * K == n or cur[n][1] is None or cur[n + 1][1] is None:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        step_up = cur[n + 1][0] - cur[n][0]
        if gl > 2.5e-5 and step_up < 0.029:
            # Bonus weight when cur[n+1] is extra tight relative to cur[n] (wrong-basin signature!)
            eff = (gl / math.sqrt(n)) * (1.35 if step_up < 0.015 else 1.0)
            ranked.append((eff, n))
    ranked.sort(reverse=True)
    my_two = [x[1] for x in ranked[: 2 * args.num_shards]][
        args.shard :: args.num_shards
    ]

    print(
        f"[W73A-NearBandLP+GAv6 s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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
                f"  [W73A CAVITY WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    for pi, n in enumerate(my_two):
        rem_all = deadline - time.time()
        if rem_all < 140.0:
            break
        puz_b = (rem_all - 15.0) / max(1, len(my_two) - pi)
        puz_dl = time.time() + puz_b
        bseed = args.seed_off + args.shard * 8191 + pi * 997

        L0 = wins[n][0] if n in wins else cur[n][0]
        P0 = cur[n][1]
        P1 = cur[n + 1][1]

        # Phase 1: Isolated Two-Stage Exhaustive Near-Band `compact_deleted` Screening!
        t_scr = time.time()
        nb_idx = select_nearband_indices(P1, max_cands=148)
        s1_jobs = [(P1, (idx,), f"nb1_{idx}") for idx in nb_idx]
        if n + 2 <= 1000 and cur[n + 2][1] is not None and cur[n + 2][0] - L0 < 0.026:
            for u, v in w51d.gen_del2_pairs(cur[n + 2][1], max_pairs=16):
                s1_jobs.append((cur[n + 2][1], (u, v), f"nb2_{u}_{v}"))

        s1_res = []
        s2_res = []
        ex_screen = ProcessPoolExecutor(max_workers=args.procs)
        try:
            futs1 = [ex_screen.submit(w51d.eval_stage1, jb) for jb in s1_jobs]
            try:
                for f in as_completed(futs1, timeout=32.0):
                    tg, S1, Q1 = f.result()
                    if Q1 is not None and math.isfinite(S1):
                        s1_res.append((S1, tg, Q1))
            except TimeoutError:
                pass
            s1_res.sort(key=lambda z: z[0])
            if s1_res:
                s2_jobs = [(tg, Q1) for _, tg, Q1 in s1_res[:16]]
                futs2 = [ex_screen.submit(w51d.eval_stage2, jb) for jb in s2_jobs]
                try:
                    for f in as_completed(futs2, timeout=22.0):
                        tg, S2, Q2 = f.result()
                        if Q2 is not None and math.isfinite(S2):
                            s2_res.append((S2, tg, Q2))
                except TimeoutError:
                    pass
        finally:
            w51d.terminate_pool(ex_screen)

        cand_cavities = s2_res if s2_res else s1_res
        cand_cavities.sort(key=lambda z: z[0])

        # Also evaluate top 225-Gadget 8-Sym DP donor seeds!
        K = math.isqrt(n)
        D = n - K * K
        if dp_ctx is not None and K <= fdp8.WMAX and 0 < D <= fdp8.DMAX:
            donors_c, donor_n_c, G_c, dp_c, PG_c, PM_c, PQ_c = dp_ctx
            for L_est, dtag, P_comp in fdp8.trace_and_compose_v2(
                dp_c, PG_c, PM_c, PQ_c, G_c, donors_c, donor_n_c, K, D, top_m=3
            ):
                if L_est < L0 + 0.018:
                    rh = w56g.quench_composed_chain(P_comp, L0)
                    if rh is not None:
                        Qc_dp, Sc_dp = rh
                        if Sc_dp < L0 + 0.014:
                            cand_cavities.append((Sc_dp, dtag, Qc_dp))

        cand_cavities.sort(key=lambda z: z[0])

        # Polish top-2 cavities immediately if ultra-close!
        polished_donors = []
        for S_c, tg_c, Q_c in cand_cavities[:6]:
            if S_c < L0 + 0.0012:
                Qp_c, Sp_c = w51d.deep_contact_polish(
                    n, Q_c, puz_dl, slp_sec=34.0, fq_sec=18
                )
                record_win_n(n, Qp_c, Sp_c, f"cav_{tg_c}", 0)
                Q_c, S_c = Qp_c, Sp_c
            ad_c = w53h.topology_angle_dist(Q_c, P0)
            polished_donors.append((S_c, ad_c, tg_c, Q_c))

        polished_donors.sort(key=lambda z: z[0])

        # Choose primary starting basin: best distinct-topology cavity if within +0.012, else best overall!
        distinct = [d for d in polished_donors if d[1] >= 0.08 and d[0] < L0 + 0.012]
        if distinct:
            S_prim, ad_prim, tg_prim, P_prim = distinct[0]
        elif polished_donors and polished_donors[0][0] < L0 + 0.008:
            S_prim, ad_prim, tg_prim, P_prim = polished_donors[0]
        else:
            P_prim = (
                np.column_stack([wins[n][1], wins[n][2], wins[n][3]])
                if n in wins
                else P0
            )
            S_prim = wins[n][0] if n in wins else L0
            ad_prim = 0.0
            tg_prim = "cur_own"

        nbrs = [d[3] for d in polished_donors[:5]]
        nbrs.append(P0)
        if n + 1 <= 1000 and cur[n + 1][1] is not None:
            nbrs.append(cur[n + 1][1])

        print(
            f"  [W73A screen] n={n:4d} s1={len(s1_res)}/{len(s1_jobs)} s2={len(s2_res)} "
            f"prim={tg_prim} dStart={S_prim - L0:+.5f} dA={ad_prim:.2f} ({time.time() - t_scr:.1f}s)",
            flush=True,
        )

        # Phase 2A (~285s): Angle-Pinned Rust `santa_ga` v6 inside the compacted cavity basin!
        rem_puz = max(140.0, puz_dl - time.time())
        ga_sec = min(295.0, max(110.0, 0.35 * rem_puz))
        Qw, Qc, js = RE.run_santa_ga(
            P_prim,
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
            f"  [W73A GAv6 done] n={n:4d} [{tg_prim}] gens={js.get('gens', 0)} "
            f"ga_wins={js.get('wins', 0)} s_in={S_prim:.7f} -> s_best={js.get('s_best', S_prim):.7f} (L0={L0:.7f})",
            flush=True,
        )

        hop_P = P_prim
        hop_tag = tg_prim
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
                    n, rq_w[1], puz_dl, slp_sec=42.0, fq_sec=24
                )
                record_win_n(n, Qp_w, Sp_w, f"{tg_prim}->gav6", js.get("wins", 1))
                hop_P = Qp_w
                hop_tag = f"{tg_prim}->gav6"

        # Phase 2B (~520s = 4..6 full rounds!): Continuous Momentum Contact-LP WideHopper!
        ex_hop = ProcessPoolExecutor(max_workers=args.procs)
        try:
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            hop_b = max(95.0, min(deadline - time.time() - 12.0, puz_dl - time.time()))
            Qb, Sb, (_, _, wt, nr) = w63k.continuous_momentum_widehopper(
                ex_hop,
                n,
                hop_P,
                L0_cur,
                bseed + 70000,
                hop_b,
                deadline,
                lambda Qx, Sx, stg, wc, _n=n, _tg=hop_tag: record_win_n(
                    _n, Qx, Sx, f"{_tg}->{stg}", wc
                ),
            )
            record_win_n(n, Qb, Sb, f"{hop_tag}->whFin", wt)
            ad_end = w53h.topology_angle_dist(Qb, P0)
            print(
                f"  [W73A done] n={n:4d} [{hop_tag}] dA={ad_end:.2f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
        finally:
            w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W73A s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
