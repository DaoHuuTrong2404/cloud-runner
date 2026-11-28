#!/usr/bin/env python3
"""Wave 66V (`w66v_dedicated_vise_sched_8round_killshot.py`): Dedicated 2-Puzzles-Per-Shard
Hydraulic-Vise `--round-press` + `anneal sched` Regional NPT -> 8-Round (`~750s`!) Continuous WideHopper!

Fixes the exact time-budget truncation observed in Wave 65F:
  - In Wave 65F, Hydraulic-Vise (`vise_r0.13..0.27`) produced genuine non-clone (`dA = 0.26..1.09`) active cascades
    that were still plummeting when cut off at Round 2 (`~280s`):
      * `n = 738`: `dStart = +0.00543 -> dEnd = +0.000001` (`r = 2, w = 14, dA = 0.88`) — 1e-6 from beating R35!
      * `n = 499`: `dStart = +0.00760 -> dEnd = +0.000332` (`r = 2, w = 49, dA = 1.09`) — 49 active jumps!
      * `n = 977`: `dStart = +0.00072 -> dEnd = +0.000070` (`r = 2, w = 6,  dA = 0.73`)
      * `n = 333`: `dStart = +0.00002 -> dEnd = -0.000000` (`r = 3, w = 4,  dA = 0.27`)
      * `n = 376`: `dStart = +0.00035 -> dEnd = -0.000000` (`r = 2, w = 20, dA = 0.26`)
    while `anneal sched` (`sched_r3.4_p0.15`) scored a direct strict R35 win on `n = 380`!
  - Each `W66V` shard owns ONLY `2` priority puzzles (`900s` = `3,600` core-seconds per puzzle!):
      1) `~125s` parallel 8-way screen (`4` Hydraulic-Vise `--round-press` + `4` `anneal sched` Regional NPT).
      2) `~750s` **8-Round Continuous-Momentum `eta <= 0.28` LP WideHopper** (`155s`/round + KKT SQP polish)!
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
import w50d_exhaustive_private_del1_widehopper as w50d
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w56f_frozen_grid_channel_remelt as w56f
import w65f_hydraulic_vise_channel_and_large_strip as w65f
import w65s_sparrow_gls_and_sched_npt as w65s


HOT_VISE_36 = [
    738,
    499,
    977,
    333,
    376,
    234,
    367,
    491,
    456,
    730,
    261,
    904,
    962,
    414,
    332,
    804,
    914,
    785,
    371,
    530,
    273,
    377,
    976,
    500,
    401,
    209,
    303,
    380,
    685,
    240,
    338,
    452,
    937,
    418,
    652,
    465,
]


def eight_round_momentum_widehopper(
    ex, n, P_start, L0, base_seed, max_budget_sec, global_deadline, on_win_cb
):
    """8-Round Continuous-Momentum `eta <= 0.28` LP WideHopper (`~750s` budget per puzzle!)."""
    t_start = time.time()
    sub_deadline = min(global_deadline, t_start + max_budget_sec)
    best_P = w32a.safe_open(P_start)
    best_S = float(E.side(best_P))
    tot_tr, tot_fit, tot_w, rounds_run = 0, 0, 0, 0

    for rnd in range(8):
        rem = sub_deadline - time.time()
        if rem < 52.0:
            break
        rounds_run += 1
        s_before = best_S
        sec_r = min(145, max(38, int(rem - 34.0)))
        if sec_r < 25:
            break

        seed0 = base_seed + rnd * 50021
        futs = [
            ex.submit(
                w50d.hop_seed_worker,
                (best_P, seed0 + k * 1009, sec_r, sub_deadline),
            )
            for k in range(4)
        ]
        rnd_w = 0
        try:
            for f in as_completed(futs, timeout=sec_r + 45.0):
                _, Q, S, (tr, ft, wn) = f.result()
                tot_tr += tr
                tot_fit += ft
                tot_w += wn
                rnd_w += wn
                if S < best_S - 1e-10 and E.min_gap(Q) >= -1e-12:
                    best_S, best_P = S, Q
        except TimeoutError:
            for f in futs:
                f.cancel()

        if best_S < L0 + 0.0035 or best_S < s_before - 1e-8:
            use_fq = 30 if best_S < L0 + 0.0012 else 15
            best_P, best_S = w51d.deep_contact_polish(
                n, best_P, sub_deadline, slp_sec=48.0, fq_sec=use_fq
            )

        print(
            f"    [W66V R{rnd + 1}] n={n:4d} dCur={best_S - L0:+.6f} (step={best_S - s_before:+.6f}, rw={rnd_w}, tot_w={tot_w}, {time.time() - t_start:.0f}s)",
            flush=True,
        )

        if best_S < L0 - 1e-9:
            on_win_cb(best_P, best_S, f"whR{rnd + 1}", tot_w)

        if (
            rnd >= 2
            and best_S >= L0 + 0.0022
            and (s_before - best_S) < 5e-7
            and rnd_w == 0
        ):
            break

    return best_P, best_S, (tot_tr, tot_fit, tot_w, rounds_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=18)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1800)
    ap.add_argument("--seed-off", type=int, default=2180000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )
    my_two = HOT_VISE_36[args.shard :: args.num_shards][:2]

    print(
        f"[W66V-8RoundViseSched s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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
                f"  [W66V KILLSHOT WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    rng = np.random.default_rng(args.seed_off + args.shard * 977)
    ex = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_two):
            rem_all = deadline - time.time()
            if rem_all < 120.0:
                break
            puz_budget = (rem_all - 15.0) / max(1, len(my_two) - pi)
            L0 = wins[n][0] if n in wins else cur[n][0]
            P0 = cur[n][1]
            bseed = args.seed_off + args.shard * 6007 + pi * 911

            anc_t = w56f.compute_crystal_anchors(P0, buffer_dist=1.65)
            anc_w = w56f.compute_crystal_anchors(P0, buffer_dist=2.55)
            anc = anc_t or anc_w

            futs_v = []
            if anc is not None:
                for ci, (ac, r0, rk) in enumerate(
                    [
                        (anc, 0.12, 0.0050),
                        (anc, 0.17, 0.0068),
                        (anc_w or anc, 0.22, 0.0085),
                        (anc_w or anc, 0.28, 0.0105),
                        (anc, 0.14, 0.0058),
                        (anc_w or anc, 0.19, 0.0075),
                    ]
                ):
                    Pn, Sn, fs, axs, ays = ac
                    futs_v.append(
                        ex.submit(
                            w65f.eval_hydraulic_vise_job,
                            (
                                n,
                                f"vise_r{r0:.2f}_{ci}",
                                Pn,
                                Sn,
                                fs,
                                axs,
                                ays,
                                bseed + ci * 19,
                                r0,
                                rk,
                                0.0,
                                False,
                            ),
                        )
                    )

            centers = w65s.pick_channel_centers(P0, rng)
            futs_s = [
                ex.submit(
                    w65s.eval_sched_regional_job,
                    (n, P0, L0, cx, cy, rad, rp, bseed + 400 + si * 31),
                )
                for si, ((cx, cy), (rad, rp)) in enumerate(
                    zip(
                        centers[:4],
                        [(2.9, 0.13), (3.6, 0.15), (4.4, 0.18), (3.3, 0.14)],
                    )
                )
            ]

            cands = []
            try:
                for f in as_completed(futs_v, timeout=95.0):
                    _, tg, Sq, Qq = f.result()
                    if Qq is None:
                        continue
                    if Sq < L0 - 1e-9:
                        Qp0, Sp0 = w51d.deep_contact_polish(
                            n, Qq, deadline, slp_sec=40.0, fq_sec=24
                        )
                        record_win_n(n, Qp0, Sp0, tg + "_dir", 0)
                        Qq, Sq = Qp0, Sp0
                    ad = w53h.topology_angle_dist(Qq, P0)
                    if Sq < L0 + 0.0125 and ad > 0.08:
                        cands.append((Sq - L0, tg, Sq, ad, Qq))
            except TimeoutError:
                pass

            try:
                for f in as_completed(futs_s, timeout=55.0):
                    for _, Sq, tg, Qq in f.result():
                        if Sq < L0 - 1e-9:
                            Qp0, Sp0 = w51d.deep_contact_polish(
                                n, Qq, deadline, slp_sec=40.0, fq_sec=24
                            )
                            record_win_n(n, Qp0, Sp0, tg + "_dir", 0)
                            Qq, Sq = Qp0, Sp0
                        ad = w53h.topology_angle_dist(Qq, P0)
                        if Sq < L0 + 0.0125 and ad > 0.04:
                            cands.append((Sq - L0, tg, Sq, ad, Qq))
            except TimeoutError:
                pass

            cands.sort(key=lambda z: z[0])
            if cands:
                gap0, tag, S_init, adist, Q_init = cands[0]
            else:
                gap0, tag, S_init, adist, Q_init = 0.0, "cur_direct", L0, 0.0, P0

            L0_cur = wins[n][0] if n in wins else cur[n][0]
            hop_b = max(120.0, min(deadline - time.time() - 15.0, puz_budget - 110.0))
            print(
                f"  [W66V start] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} hop_budget={hop_b:.0f}s (cands={len(cands)})",
                flush=True,
            )
            Qb, Sb, (_, _, wt, nr) = eight_round_momentum_widehopper(
                ex,
                n,
                Q_init,
                L0_cur,
                bseed + 50000,
                hop_b,
                deadline,
                lambda Qc, Sc, stg, wc, _n=n, _tg=tag: record_win_n(
                    _n, Qc, Sc, f"{_tg}->{stg}", wc
                ),
            )
            record_win_n(n, Qb, Sb, f"{tag}->wh8Fin", wt)
            print(
                f"  [W66V done] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex)

    save_ckpt()
    print(
        f"[W66V s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
