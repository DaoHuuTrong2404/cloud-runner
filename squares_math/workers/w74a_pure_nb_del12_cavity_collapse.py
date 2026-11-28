#!/usr/bin/env python3
"""Wave 74A (`w74a_pure_nb_del12_cavity_collapse.py`):
80-Core (`20` Shards) PURE Near-Band `cur[n + 1]` / `cur[n + 2]` Contact-LP Cavity Collapse
(`0%` `fdp8` Hijacking!) -> Multi-Round WideHopper -> Angle-Pinned `santa_ga` v6!

Fixes the exact root cause discovered in Wave 73A empirical logs:
  - In W73A, mixing `fdp8v2_m1` (`dp_cache`, `0` void left to collapse, `dA = 0.20`) into
    `cand_cavities` beat every fresh `nb1_*` (`cur[n+1]` delete-1) cavity in raw pre-hop `S`,
    so `nb1_*` NEVER ran `WideHopper` or `santa_ga` v6 on `367<-368`, `491<-492`, `738<-739`,
    `456<-457`, `234<-235`!
  - Here, ONLY genuine `cur[n+1]` (`nb1_*`) and `cur[n+2]` (`nb2_*`) Contact-LP compacted cavities
    are selected as primary starters (`cav_0` and spatially/topologically distinct `cav_1`),
    with adaptive `max_cands = min(132, max(44, int(42000 / n)))` (`< 28s` screen even at `N = 990`!),
    running 4-5 full rounds of `continuous_momentum_widehopper` FIRST to collapse the deleted
    square's 1.0-unit void, followed by Angle-Pinned `santa_ga` v6 + `deep_contact_polish`!
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

import rust_engines as RE
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w63k_1puzzle_per_shard_killshot as w63k
import w73a_exhaustive_nearband_lp_compactor_hopper as w73a

# Proven high-tilt neighbor mismatches & freshly jumped n+1 -> n cascade targets!
PRIORITY_CAVITY_TARGETS = [
    367,  # cur[368] has tilted=168 vs cur[367] tilted=90!
    738,  # cur[739] has tilted=222 vs cur[738] tilted=120!
    491,  # cur[492] has tilted=147 vs cur[491] tilted=101!
    456,  # cur[457] has tilted=167 vs cur[456] tilted=60!
    234,  # cur[235] has tilted=121 vs cur[234] tilted=36 (+0.0037)!
    414,  # cur[415] has tilted=136 vs cur[414] tilted=50!
    904,  # cur[905] high-tilt donor!
    747,  # cur[748] JUST jumped in W71 & W73A (27.8219927)!
    926,  # cur[927] JUST jumped -0.000902 in pub_v18!
    697,  # cur[698] jumped in pub_v18!
    753,  # cur[754] jumped in pub_v18!
    990,  # cur[991] jumped in pub_v18!
    539,  # cur[540] jumped in pub_v18!
    332,  # cur[333] jumped -0.007765 in W70A!
    804,
    914,
    295,
    632,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=3750000)
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

    ranked = []
    for n in range(95, 999):
        K = math.isqrt(n)
        if K * K == n or cur[n][1] is None or cur[n + 1][1] is None:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        step_up = cur[n + 1][0] - cur[n][0]
        if gl > 2.5e-5 and step_up < 0.030:
            eff = (gl / math.sqrt(n)) * (1.40 if step_up < 0.015 else 1.0)
            ranked.append((eff, n))
    ranked.sort(reverse=True)

    ordered = []
    seen = set()
    for n in PRIORITY_CAVITY_TARGETS:
        if (
            math.isqrt(n) ** 2 != n
            and cur[n][1] is not None
            and cur[n + 1][1] is not None
            and n not in seen
        ):
            ordered.append(n)
            seen.add(n)
    for _, n in ranked:
        if len(ordered) >= 2 * args.num_shards:
            break
        if n not in seen:
            ordered.append(n)
            seen.add(n)

    my_two = ordered[: 2 * args.num_shards][args.shard :: args.num_shards]

    print(
        f"[W74A-PureNBCavity s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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
            print(
                f"  [W74A PURE-CAVITY WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
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

        # Adaptive near-band count so Stage 1 + Stage 2 finishes in < 32s for ALL N <= 999!
        max_nb = min(132, max(44, int(42000 // max(100, n))))
        t_scr = time.time()
        nb_idx = w73a.select_nearband_indices(P1, max_cands=max_nb)
        s1_jobs = [(P1, (idx,), f"nb1_{idx}") for idx in nb_idx]
        if n + 2 <= 1000 and cur[n + 2][1] is not None and cur[n + 2][0] - L0 < 0.028:
            for u, v in w51d.gen_del2_pairs(cur[n + 2][1], max_pairs=14):
                s1_jobs.append((cur[n + 2][1], (u, v), f"nb2_{u}_{v}"))

        s1_res = []
        s2_res = []
        ex_screen = ProcessPoolExecutor(max_workers=args.procs)
        try:
            futs1 = [ex_screen.submit(w51d.eval_stage1, jb) for jb in s1_jobs]
            try:
                for f in as_completed(futs1, timeout=34.0):
                    tg, S1, Q1 = f.result()
                    if Q1 is not None and math.isfinite(S1):
                        s1_res.append((S1, tg, Q1))
            except TimeoutError:
                pass
            s1_res.sort(key=lambda z: z[0])
            if s1_res:
                n_s2 = min(14, max(8, int(6000 // max(100, n))))
                s2_jobs = [(tg, Q1) for _, tg, Q1 in s1_res[:n_s2]]
                futs2 = [ex_screen.submit(w51d.eval_stage2, jb) for jb in s2_jobs]
                try:
                    for f in as_completed(futs2, timeout=24.0):
                        tg, S2, Q2 = f.result()
                        if Q2 is not None and math.isfinite(S2):
                            s2_res.append((S2, tg, Q2))
                except TimeoutError:
                    pass
        finally:
            w51d.terminate_pool(ex_screen)

        pure_cavs = s2_res if s2_res else s1_res
        pure_cavs.sort(key=lambda z: z[0])

        if not pure_cavs:
            print(f"  [W74A skip] n={n:4d} 0 cavities returned", flush=True)
            continue

        # Select Cav #0 (tightest pure cavity!) and Cav #1 (distinct deletion index & angle distance >= 0.08 if available!)
        S_c0, tg_c0, Q_c0 = pure_cavs[0]
        ad_c0 = w53h.topology_angle_dist(Q_c0, P0)
        cav_pair = [(S_c0, ad_c0, tg_c0, Q_c0)]
        for S_ck, tg_ck, Q_ck in pure_cavs[1:]:
            if w53h.topology_angle_dist(Q_ck, Q_c0) >= 0.06 or len(cav_pair) == 0:
                cav_pair.append((S_ck, w53h.topology_angle_dist(Q_ck, P0), tg_ck, Q_ck))
                break
        if len(cav_pair) < 2 and len(pure_cavs) >= 2:
            S_ck, tg_ck, Q_ck = pure_cavs[1]
            cav_pair.append((S_ck, w53h.topology_angle_dist(Q_ck, P0), tg_ck, Q_ck))

        print(
            f"  [W74A pure-screen] n={n:4d} s1={len(s1_res)}/{len(s1_jobs)} s2={len(s2_res)} "
            f"cav0={tg_c0}({S_c0 - L0:+.5f},dA={ad_c0:.2f}) ({time.time() - t_scr:.1f}s)",
            flush=True,
        )

        # Phase 2A: Collapse Cav #0 (~440s) and Cav #1 (~200s) directly with Contact-LP WideHopper!
        best_after_hop = []
        ex_hop = ProcessPoolExecutor(max_workers=args.procs)
        try:
            for ci, (S_ci, ad_ci, tg_ci, Q_ci) in enumerate(cav_pair):
                rem_puz = puz_dl - time.time()
                if rem_puz < 110.0:
                    break
                frac = (
                    0.52
                    if (ci == 0 and len(cav_pair) > 1)
                    else (0.72 if ci == 0 else 0.55)
                )
                hop_b = max(95.0, min(460.0 if ci == 0 else 230.0, frac * rem_puz))
                L0_cur = wins[n][0] if n in wins else cur[n][0]
                Qb_c, Sb_c, (_, _, wt_c, nr_c) = w63k.continuous_momentum_widehopper(
                    ex_hop,
                    n,
                    Q_ci,
                    L0_cur,
                    bseed + 40000 + ci * 13007,
                    hop_b,
                    deadline,
                    lambda Qx, Sx, stg, wc, _n=n, _tg=tg_ci: record_win_n(
                        _n, Qx, Sx, f"{_tg}->{stg}", wc
                    ),
                )
                record_win_n(n, Qb_c, Sb_c, f"{tg_ci}->whFin", wt_c)
                best_after_hop.append((Sb_c, tg_ci, Qb_c))
                print(
                    f"  [W74A cav-hop] n={n:4d} [{tg_ci}] dIn={S_ci - L0:+.5f} -> dOut={Sb_c - L0_cur:+.6f} r={nr_c} w={wt_c}",
                    flush=True,
                )
        finally:
            w51d.terminate_pool(ex_hop)

        # Phase 2B (~190s): Angle-Pinned `santa_ga` v6 + KKT Polish on the collapsed cavity champion!
        rem_ga = min(deadline - time.time() - 15.0, puz_dl - time.time())
        if best_after_hop and rem_ga >= 95.0:
            best_after_hop.sort(key=lambda z: z[0])
            S_ch, tg_ch, Q_ch = best_after_hop[0]
            nbrs = (
                [z[2] for z in best_after_hop[1:]]
                + [z[2] for z in pure_cavs[:3]]
                + [P0, P1]
            )
            ga_sec = max(75.0, min(220.0, rem_ga - 28.0))
            Qw, Qc, js = RE.run_santa_ga(
                Q_ch,
                nbr_packings=nbrs,
                seed=bseed + 901,
                sec=ga_sec,
                islands=6,
                pop=14,
                offspring=28,
                steps=125,
                threads=args.procs,
                timeout_s=ga_sec + 30.0,
            )
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
                        n, rq_w[1], deadline, slp_sec=36.0, fq_sec=20
                    )
                    record_win_n(n, Qp_w, Sp_w, f"{tg_ch}->wh->gav6", js.get("wins", 1))
            print(
                f"  [W74A GAv6-fin] n={n:4d} [{tg_ch}] gens={js.get('gens', 0)} ga_wins={js.get('wins', 0)} s_best={js.get('s_best', S_ch):.7f}",
                flush=True,
            )

    save_ckpt()
    print(
        f"[W74A s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
