#!/usr/bin/env python3
"""Wave 75A (`w75a_upward_nm1_basin_invasion_gav7.py`):
80-Core (`20` Shards) Upward `cur[n - 1] -> n` (and `cur[n - 2] -> n` + Center-Preserving Frame Peel)
Two-Arm / Diamond Basin Invasion powered by Rust `santa_ga` v7 (`m < n` Native `Relaxer2W` GLS Legalizer)!

Directly attacks the #1 empirical discovery from `inspect_neighbor_wall_locks.py`:
  - Almost every remaining big gap puzzle is trapped in a low-tilt 1-arm basin while `cur[n - 1]`
    sits `0.024 .. 0.045` BELOW `cur[n]` in the true two-arm / 45-deg diamond basin:
      * `n=234` (`til=36, 15.8229`) <- `cur[233]` (`til=121, 15.7782, d_m1=+0.0447`!) & `cur[265]` (`til=121`)
      * `n=491` (`til=101, 22.6768`) <- `cur[490]` (`til=128, 22.6370, d_m1=+0.0398`!)
      * `n=738` (`til=120, 27.6730`) <- `cur[737]` (`til=203, 27.6388, d_m1=+0.0342`!)
      * `n=456` (`til=60, 21.8467`)  <- `cur[455]` (`til=136, 21.8182, d_m1=+0.0285`!)
      * `n=261` (`til=105, 16.6788`) <- `cur[260]` (`til=122, 16.6335, d_m1=+0.0453`!)
      * `n=904` (`til=115, 30.5629`) <- `cur[903]` (`til=338, 30.5332, d_m1=+0.0297`!) & `cur[965]` (`til=365`)
      * `n=414` (`til=50, 20.8407`)  <- `cur[413]` (`til=114, 20.8170, d_m1=+0.0237`!)
      * `n=295` (`til=72, 17.6980`)  <- `cur[294]` (`til=120, 17.6569, d_m1=+0.0412`!)
  - Passes pre-healed high-tilt `cur[n-1]->n` donor as `--in` AND passes raw `cur[n-1]` (`m = n - 1`) +
    `cur[n-2]` (`m = n - 2`) via `--nbr` into `santa_ga` v7 (`synthesize_upward_insert_seeds` 20-item
    `Relaxer2W` GLS legalization batch!), keeping all 6 islands inside the high-tilt basin!
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
import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
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
    """Peel outer upright border squares from `P_big` (`(K+1)^2 + D -> K^2 + D`) WITHOUT chopping
    an interior centered diamond/tilted block (fixes `lpeel` chopping `cur[265]`'s `11x11` diamond!).
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
    # Mode 1: protect tilted squares + squares closest to box center (peels all 4 outer upright borders evenly!)
    score_sym = (
        np.maximum(np.abs(P[:, 0] - cx), np.abs(P[:, 1] - cy)) - 50.0 * is_tilted
    )
    keep_sym = np.argsort(score_sym)[:n_target]
    out.append(("cpeel_sym", RE.normalize_to_origin(P[keep_sym])))

    # Modes 2..5: protect tilted squares + peel predominantly from (R+T, L+T, R+B, L+B) upright borders!
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


def eval_soft_nm1_heal_job(job):
    tag, n, L0, P_cand, seed = job
    try:
        rh = RE.run_harden_reharden(
            P_cand,
            s_ref=L0 + 0.004,
            seed=int(seed),
            r0=0.22,
            stiff0=0.28,
            pre=2600,
            hold=9000,
            release=4200,
            settle=3200,
            hold_f=1.0025,
            noise=0.0055,
            timeout_s=14.0,
        )
        P_mid = rh[1] if rh is not None else P_cand
        rq = RE.run_fq_quench_args(
            P_mid,
            [
                "--seed",
                str(int(seed) + 19),
                "--round-r0",
                "0.09",
                "--round-steps",
                "14",
                "--round-press",
                "--round-press-outer",
                "6",
                "--loosen",
                "1.008",
                "--outer",
                "48",
                "--pit",
                "300",
                "--flip-top",
                "20",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "11.0",
            ],
            s_ref=L0,
            timeout_s=14.0,
        )
        if rq is None:
            return tag, math.inf, None
        Sp, Qp = RE.kkt_sqp_polish_fq(rq[1], timeout_each=5.0)
        return tag, float(Sp), Qp
    except Exception:
        return tag, math.inf, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=3950000)
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
        if K * K == n or K * K + 1 == n or cur[n][1] is None or cur[n - 1][1] is None:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        d_m1 = cur[n][0] - cur[n - 1][0]
        if gl > 2.5e-5 and d_m1 > 0.014:
            ranked.append(((gl / math.sqrt(n)) * (1.45 if d_m1 > 0.024 else 1.0), n))
    ranked.sort(reverse=True)

    ordered = []
    seen = set()
    for n in PRIORITY_NM1_TARGETS:
        K = math.isqrt(n)
        if (
            K * K != n
            and K * K + 1 != n
            and cur[n][1] is not None
            and cur[n - 1][1] is not None
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
        f"[W75A-UpwardNM1-GAv7 s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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
                f"  [W75A UPWARD-NM1 WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
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

            L0 = wins[n][0] if n in wins else cur[n][0]
            P0 = cur[n][1]
            Pm1 = cur[n - 1][1]

            # Stage 1 (~42s): Pre-heal `cur[n-1] -> n` holes + Center-Preserving Frame Peels of `cur[n + 2K + 1]`!
            t_s1 = time.time()
            heal_jobs = []
            for ji, (itag, P_ins) in enumerate(
                btr.insert_holes_candidates(
                    Pm1, P_n_own=P0, seed=bseed + 11, n_holes=5, n_target=3
                )
            ):
                heal_jobs.append((f"nm1_{itag}", n, L0, P_ins, bseed + 100 + ji))

            K = math.isqrt(n)
            n_up = n + 2 * K + 1
            if n_up <= 1000 and cur[n_up][1] is not None:
                for ji, (ptag, P_cp) in enumerate(
                    center_preserving_frame_peels(cur[n_up][1], n)[:4]
                ):
                    heal_jobs.append(
                        (f"{ptag}_from{n_up}", n, L0, P_cp, bseed + 200 + ji)
                    )

            healed = []
            futs = [ex_hop.submit(eval_soft_nm1_heal_job, jb) for jb in heal_jobs]
            try:
                for f in as_completed(futs, timeout=46.0):
                    tg, Sh, Qh = f.result()
                    if Qh is not None and math.isfinite(Sh):
                        if Sh < L0 + 0.0006:
                            Qp_h, Sp_h = w51d.deep_contact_polish(
                                n, Qh, puz_dl, slp_sec=36.0, fq_sec=20
                            )
                            record_win_n(n, Qp_h, Sp_h, tg, 0)
                            Qh, Sh = Qp_h, Sp_h
                        ad_h = w53h.topology_angle_dist(Qh, P0)
                        healed.append((Sh, ad_h, tg, Qh))
            except TimeoutError:
                w51d.terminate_pool(ex_hop)
                ex_hop = ProcessPoolExecutor(max_workers=args.procs)

            healed.sort(key=lambda z: z[0])

            # Pick primary `--in` for `santa_ga` v7: best distinct high-tilt donor if within L0 + 0.022!
            distinct = [h for h in healed if h[1] >= 0.25 and h[0] < L0 + 0.022]
            if distinct:
                S_prim, ad_prim, tg_prim, P_prim = distinct[0]
            elif healed and healed[0][0] < L0 + 0.015:
                S_prim, ad_prim, tg_prim, P_prim = healed[0]
            else:
                P_prim = (
                    np.column_stack([wins[n][1], wins[n][2], wins[n][3]])
                    if n in wins
                    else P0
                )
                S_prim = wins[n][0] if n in wins else L0
                ad_prim = 0.0
                tg_prim = "cur_own_with_nm1"

            # Pass raw `cur[n - 1]` and `cur[n - 2]` FIRST in `--nbr` so `santa_ga` v7 runs its native
            # 20-item `Relaxer2W` `synthesize_upward_insert_seeds` directly inside Rust!
            nbrs = [Pm1]
            if (
                n >= 6
                and math.isqrt(n - 2) ** 2 != (n - 2)
                and cur[n - 2][1] is not None
            ):
                nbrs.append(cur[n - 2][1])
            for h in healed[:3]:
                nbrs.append(h[3])
            if tg_prim != "cur_own_with_nm1":
                nbrs.append(P0)

            print(
                f"  [W75A nm1-stage1] n={n:4d} d_m1=+{L0 - cur[n - 1][0]:.5f} healed={len(healed)}/{len(heal_jobs)} "
                f"prim={tg_prim}({S_prim - L0:+.5f},dA={ad_prim:.2f}) ({time.time() - t_s1:.1f}s)",
                flush=True,
            )

            # Stage 2 (~490s): `santa_ga` v7 with native `m < n` `Relaxer2W` upward insertion!
            rem_puz = max(140.0, puz_dl - time.time())
            ga_sec = min(500.0 if pi == 0 else 410.0, max(125.0, 0.58 * rem_puz))
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
                timeout_s=ga_sec + 38.0,
            )
            print(
                f"  [W75A GAv7 done] n={n:4d} [{tg_prim}] gens={js.get('gens', 0)} swaps={js.get('swaps', 0)} "
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
                        n, rq_w[1], puz_dl, slp_sec=44.0, fq_sec=26
                    )
                    record_win_n(n, Qp_w, Sp_w, f"{tg_prim}->gav7", js.get("wins", 1))
                    hop_P = Qp_w
                    hop_tag = f"{tg_prim}->gav7"

            L0_cur = wins[n][0] if n in wins else cur[n][0]
            hop_b = max(90.0, min(deadline - time.time() - 14.0, puz_dl - time.time()))
            Qb, Sb, (_, _, wt, nr) = w63k.continuous_momentum_widehopper(
                ex_hop,
                n,
                hop_P,
                L0_cur,
                bseed + 90000,
                hop_b,
                deadline,
                lambda Qx, Sx, stg, wc, _n=n, _tg=hop_tag: record_win_n(
                    _n, Qx, Sx, f"{_tg}->{stg}", wc
                ),
            )
            record_win_n(n, Qb, Sb, f"{hop_tag}->whFin", wt)
            ad_end = w53h.topology_angle_dist(Qb, P0)
            print(
                f"  [W75A done] n={n:4d} [{hop_tag}] dA={ad_end:.2f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W75A s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
