#!/usr/bin/env python3
"""Wave 67A (`w67a_rust_santa_island_ga_and_hopper.py`): Compiled Rust 1st-Place Santa 2025
Island Genetic Algorithm (`santa_ga`: `arm_extend` + `mirror_seed` + `released_mirror` + `crossover`)
+ KKT SQP Finish + Continuous-Momentum WideHopper!

Deploys Jeroen Cottaar's 1st-Place Santa 2025 Island Genetic Algorithm adapted to Squares in Squares
(`grennwolf` empirical RTX-5090 ruleset, compiled to native Rust LTO across `4` `std::thread::scope` threads):
  - `arm_extend` (grew tilted arms on **17 of 22** overnight RTX-5090 wins!)
  - `mirror_seed` + `mirror_symmetrize` + `released_mirror` (`22` improvements/100 mirror islands + `3` of top-9 wins!)
  - Exact empirical move weights: `row_slide (4.56)`, `band_reangle (3.98)`, `jiggle_small (3.07)`,
    `arm_extend (2.80)`, `twist (2.33)`, `jiggle_big (2.08)`, `block_rotate (1.10)`, `crossover (6%)`,
    and `OVERLAP_TARGETING = 0.80`!
  - Seeded with `cur[n]` + `cur[n+1]` + `cur[n+2]` (`n+1/n+2` hole redistribution seeds)!
  - Harvests both direct `santa_ga` strict wins (`--out`) AND evolved near-feasible island champions (`--out.champ`),
    quenches via `fq` + two-pass KKT SQP, and escalates via Continuous-Momentum `eta <= 0.28` LP WideHopper!
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

import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w63k_1puzzle_per_shard_killshot as w63k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=24)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=2350000)
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
    for n in range(150, 999):
        K = math.isqrt(n)
        if K * K == n or cur[n][1] is None:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gl > 5.0e-5:
            ranked.append((gl / math.sqrt(n), n))
    ranked.sort(reverse=True)
    my_two = [x[1] for x in ranked[: args.num_shards * 2]][
        args.shard :: args.num_shards
    ][:2]

    print(
        f"[W67A-RustSantaGA s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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
                f"  [W67A SANTA-GA WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
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
            ga_sec = min(540.0, max(120.0, 0.60 * puz_b))

            L0 = wins[n][0] if n in wins else cur[n][0]
            P0 = cur[n][1]
            nbrs = []
            for m in (n + 1, n + 2):
                if m <= 1000 and cur[m][1] is not None:
                    nbrs.append(cur[m][1])

            bseed = args.seed_off + args.shard * 7919 + pi * 997
            Qw, Qc, js = RE.run_santa_ga(
                P0,
                nbr_packings=nbrs,
                seed=bseed,
                sec=ga_sec,
                islands=6,
                pop=28,
                offspring=112,
                steps=260,
                threads=args.procs,
                timeout_s=ga_sec + 35.0,
            )
            print(
                f"  [W67A GA done] n={n:4d} gens={js.get('gens', 0)} ga_wins={js.get('wins', 0)} "
                f"resets={js.get('resets', 0)} s_best={js.get('s_best', L0):.7f} (L0={L0:.7f})",
                flush=True,
            )

            hop_start_P = P0
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
                    record_win_n(n, Qp_w, Sp_w, "santa_ga_direct", js.get("wins", 1))
                    hop_start_P = Qp_w
                    hop_tag = "santa_ga_win"

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
                    s_ref=L0,
                    timeout_s=18.0,
                )
                if rq_c is not None:
                    Sc_pol, Qc_pol = RE.kkt_sqp_polish_fq(rq_c[1], timeout_each=6.5)
                    if Sc_pol < L0 - 1e-9:
                        Qp_c, Sp_c = w51d.deep_contact_polish(
                            n, Qc_pol, deadline, slp_sec=42.0, fq_sec=24
                        )
                        record_win_n(n, Qp_c, Sp_c, "santa_ga_champ_dir", 0)
                        Qc_pol, Sc_pol = Qp_c, Sp_c
                    if Sc_pol < L0 + 0.0135:
                        hop_start_P = Qc_pol
                        hop_tag = "santa_ga_champ"

            L0_cur = wins[n][0] if n in wins else cur[n][0]
            ad = w53h.topology_angle_dist(hop_start_P, P0)
            hop_b = max(90.0, min(deadline - time.time() - 15.0, puz_b - ga_sec - 25.0))
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
                f"  [W67A done] n={n:4d} [{hop_tag}] dA={ad:.2f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W67A s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
