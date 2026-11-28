#!/usr/bin/env python3
"""Wave 76A (`w76a_twoband_breadth_hopper_kink_polish.py`):
112-Core (`28` Shards x `5` Puzzles = `140` Untouched Two-Band Puzzles!) Breadth-First
Contact-LP WideHopper + Kink-Aware Free-Angle Sequential LP + Strip/L-Band Structural Lifts!

Directly implements `goodpjw2008`'s #1 highest-yield empirical CPU allocation (`NOTES.md` line 96:
Breadth-First Two-Band Hopper across `n >= 300, f in [0.58, 0.997]` gave `0.9e-6` score/proc-hr
vs `0` on `f < 0.55` brick walls) + `malyshev_atlas` / `prvsiyan` Kink-Aware Sequential LP!
  - Skips the 12 over-mined brick walls (`234, 367, 491, 738, 730, 962, ...`) and fans 112 cores
    out across Ranks `#11 .. #150` of the Two-Band Family (`977, 981, 913, 698, 849, 378, 862,
    506, 759, 585, 495, 792, 758, 602, 449, 591, 685, 853, 457, 338, 859, 805, 749, 460, ...`
    PLUS direct strip/L-lift neighbors of our private `333`, `748`, `927` wins!).
  - On each of its 5 puzzles (`~355s` x 4 cores = `1,420` core-seconds per puzzle!):
    1) Evaluates all `E.transplant` + `E.change_length` forward/reverse strip & L-band lifts,
    2) Applies `hull_rotate_exact` + `kink_contact_lp_polish` (`0/90 deg` cusp subgradients!),
    3) Runs `continuous_momentum_widehopper` (`4` parallel seeds, escalating budget automatically
       whenever `w > 0` jumps a basin!) + final `kink_contact_lp_polish` & `deep_contact_polish`!
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

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

import kink_polish_and_strip_lifts_w76 as kps
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w63k_1puzzle_per_shard_killshot as w63k

# Direct strip/L-lift & +/-1 neighbors of our freshest breakthroughs (333, 748, 927, 698, 754, 991, 540)
FRESH_CASCADE_TWOBAND = [
    314,
    315,
    332,
    334,
    351,
    352,
    370,
    720,
    721,
    747,
    748,
    749,
    775,
    776,
    896,
    897,
    926,
    927,
    928,
    957,
    958,
    697,
    698,
    699,
    753,
    754,
    755,
    990,
    991,
    992,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=28)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=4150000)
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

    # Build Two-Band Breadth-First Target List (`n >= 300, f in [0.58, 0.997]`), skipping over-mined top-10 brick walls!
    OVERMINED = {234, 367, 491, 738, 456, 730, 261, 904, 962, 414}
    twoband = []
    for n in range(300, 1000):
        K = math.isqrt(n)
        if K * K == n or cur[n][1] is None or n in OVERMINED:
            continue
        f = cur[n][0] - K
        if 0.58 <= f <= 0.997:
            gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
            # Include both frontier-deficit two-band puzzles AND zero-deficit two-band puzzles (`+2e-4` floor)!
            twoband.append(((gl + 2.0e-4) / math.sqrt(n), n))
    twoband.sort(reverse=True)

    ordered = []
    seen = set()
    for n in FRESH_CASCADE_TWOBAND:
        K = math.isqrt(n)
        if (
            300 <= n <= 999
            and K * K != n
            and cur[n][1] is not None
            and n not in OVERMINED
            and 0.56 <= cur[n][0] - K <= 0.998
            and n not in seen
        ):
            ordered.append(n)
            seen.add(n)
    for _, n in twoband:
        if len(ordered) >= 5 * args.num_shards:
            break
        if n not in seen:
            ordered.append(n)
            seen.add(n)

    my_five = ordered[: 5 * args.num_shards][args.shard :: args.num_shards]

    print(
        f"[W76A-TwoBandBreadth s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"TARGETS={my_five} ({time.time() - t0:.1f}s)",
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
                f"  [W76A TWOBAND WIN!!] n={n_w:4d} [{src_tag}] -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_five):
            rem_all = deadline - time.time()
            if rem_all < 85.0:
                break
            puz_b = (rem_all - 12.0) / max(1, len(my_five) - pi)
            bseed = args.seed_off + args.shard * 8191 + pi * 997

            L0 = cur[n][0]
            P0 = cur[n][1]

            # Step 1 (~15s): Check Strip & L-Band Lifts + Kink-Aware Contact-LP Polish on `cur[n]`!
            S_kp, Q_kp = kps.kink_contact_lp_polish(
                P0, rounds=42, seed=bseed, max_seconds=15.0
            )
            if S_kp < L0 - 1e-9:
                Qp_k, Sp_k = w51d.deep_contact_polish(
                    n, Q_kp, deadline, slp_sec=28.0, fq_sec=16
                )
                record_win_n(n, Qp_k, Sp_k, "kink_lp_own", 0)
                P_start = Qp_k
                start_tag = "kink_own"
            else:
                P_start = Q_kp if S_kp <= L0 + 1e-11 else P0
                start_tag = "own"

            lifts = kps.sweep_strip_and_l_lifts_for_n(n, cur)
            for s_lf, tg_lf, q_lf in lifts[:2]:
                s_lfk, q_lfk = kps.kink_contact_lp_polish(
                    q_lf, rounds=32, seed=bseed + 7, max_seconds=10.0
                )
                if s_lfk < cur[n][0] - 1e-9:
                    Qp_l, Sp_l = w51d.deep_contact_polish(
                        n, q_lfk, deadline, slp_sec=32.0, fq_sec=18
                    )
                    record_win_n(n, Qp_l, Sp_l, tg_lf, 0)
                    P_start = Qp_l
                    start_tag = tg_lf

            # Step 2 (~315s): Multi-Seed Force-Guided Contact-LP WideHopper + Final Kink Polish!
            L0_cur = cur[n][0]
            hop_b = max(75.0, min(deadline - time.time() - 14.0, puz_b - 22.0))
            Qb, Sb, (_, _, wt, nr) = w63k.continuous_momentum_widehopper(
                ex_hop,
                n,
                P_start,
                L0_cur,
                bseed + 50000,
                hop_b,
                deadline,
                lambda Qx, Sx, stg, wc, _n=n, _tg=start_tag: record_win_n(
                    _n, Qx, Sx, f"{_tg}->{stg}", wc
                ),
            )
            # Final kink-aware LP + hull-rotation polish on WideHopper output!
            S_fin, Q_fin = kps.kink_contact_lp_polish(
                Qb, rounds=38, seed=bseed + 99, max_seconds=12.0
            )
            if S_fin < Sb:
                Qb, Sb = Q_fin, S_fin
            record_win_n(n, Qb, Sb, f"{start_tag}->wh->kink", wt)
            ad_end = w53h.topology_angle_dist(Qb, P0)
            print(
                f"  [W76A done] n={n:4d} [{start_tag}] dA={ad_end:.2f} -> dEnd={Sb - L0:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W76A s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
