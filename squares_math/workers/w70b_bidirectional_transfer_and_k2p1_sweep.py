#!/usr/bin/env python3
"""Wave 70B (`w70b_bidirectional_transfer_and_k2p1_sweep.py`):
64-Core All-1000 Bidirectional Neighbour Transfer (`n+1 -> n` HiGHS LP + `fq`, `n-1 -> n` `insert_holes`,
4-Corner Soft L-Peel `(K+1)^2+D -> K^2+D`) + Dedicated `K^2 + 1` & `K^2 + 2` Sub-Half Cascade Engine!

Why this is orthogonal to `W70A` (`Top-48` Santa-GA v4) and `W67S-W69S` (Fault-DP):
  - Sweeps ALL `~960` non-square puzzles (`shard :: 16`, `~60` puzzles/shard) with:
    1) `n + 1 -> n` boundary + sparse-interior deletions (`fixed_transfer.py`) + sparse HiGHS LP + Rust `fq`,
    2) `n - 1 -> n` `insert_holes` + `target-hole insertion` (`run_gpu.py` ported to Rust `fq --round-r0`),
    3) `(K + 1)^2 + D -> K^2 + D` 4-corner Soft L-Peel (`BL, BR, TL, TR` via `harden reharden` + `fq`)
       whenever `L_{n+2K+1} - 1.0 < L_n + 0.004` (hits `n=785 <- 842`, `n=730 <- 785/842`, `n=362 <- 401`, etc.!).
  - Phase 2 spends the remaining `~900s` per shard running 6-Island Pressure-Ladder `santa_ga` v4 +
    WideHopper on the `K^2 + 1` / `K^2 + 2` / untouched Rank `#49..#96` targets assigned to each shard,
    automatically cascading any `K^2 + D` win upward via exact upright L-extension (`n -> n + 2K + 1`)!
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


def exact_add_upright_L(P_in):
    """Exact `n -> n + 2*floor(S) + 1` upright L-border extension (`S -> S + 1.0`)."""
    P = RE.normalize_to_origin(P_in)
    s0 = RE.side_exact(P)
    s1 = s0 + 1.0
    m = int(math.floor(s1 + 1e-11))
    rows = []
    for j in range(m):
        rows.append((s1 - 0.5, j + 0.5, 0.0))
    for i in range(m - 1):
        rows.append((i + 0.5, s1 - 0.5, 0.0))
    if not rows:
        return None, math.inf
    Q = np.vstack([P, np.asarray(rows, dtype=np.float64)])
    return Q, RE.side_exact(Q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=3050000)
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

    all_non_grid = [
        n for n in range(5, 1000) if math.isqrt(n) ** 2 != n and cur[n][1] is not None
    ]
    # Order by frontier gap descending so highest-leverage transfers run first!
    all_non_grid.sort(
        key=lambda n: max(0.0, cur[n][0] - front_L.get(n, cur[n][0])) / math.sqrt(n),
        reverse=True,
    )
    my_sweep = all_non_grid[args.shard :: args.num_shards]

    print(
        f"[W70B-BidirTransfer s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"sweep_puzzles={len(my_sweep)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    deadline = t0 + args.budget_sec
    wins = {}
    tot_ds = 0.0

    def get_L(n_q):
        return wins[n_q][0] if n_q in wins else cur[n_q][0]

    def get_P(n_q):
        if n_q in wins:
            return np.column_stack([wins[n_q][1], wins[n_q][2], wins[n_q][3]])
        return cur[n_q][1]

    def save_ckpt():
        pl = {}
        for kn, (Lf, xf, yf, tf) in sorted(wins.items()):
            pl[f"L_{kn}"] = np.array([Lf], dtype=np.float64)
            pl[f"x_{kn}"] = np.asarray(xf, dtype=np.float64)
            pl[f"y_{kn}"] = np.asarray(yf, dtype=np.float64)
            pl[f"t_{kn}"] = np.asarray(tf, dtype=np.float64)
        np.savez_compressed(args.out_npz, **pl)

    def record_win_and_cascade(n_w, Q_cand, S_cand, src_tag):
        nonlocal tot_ds
        L_ref = get_L(n_w)
        if S_cand >= L_ref - 1e-9:
            return False
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n_w, Q_cand)
        if not (ok and Lf < L_ref - 1e-9):
            return False
        ds = (Lf - L_ref) / math.sqrt(n_w)
        tot_ds += ds
        wins[n_w] = (float(Lf), xf, yf, tf)
        print(
            f"  [W70B WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f})!",
            flush=True,
        )
        save_ckpt()

        # Immediate upward upright-L cascade (`n -> n + 2*floor(S) + 1 -> ...`)!
        Q_cur = np.column_stack([xf, yf, tf])
        m_cur = n_w
        while True:
            Q_ext, S_ext = exact_add_upright_L(Q_cur)
            if Q_ext is None or len(Q_ext) > 1000:
                break
            m_next = len(Q_ext)
            if S_ext < get_L(m_next) - 1e-9:
                ok_e, L_e, xe, ye, te = f2d.stabilize_decimal(m_next, Q_ext)
                if ok_e and L_e < get_L(m_next) - 1e-9:
                    ds_e = (L_e - get_L(m_next)) / math.sqrt(m_next)
                    tot_ds += ds_e
                    wins[m_next] = (float(L_e), xe, ye, te)
                    print(
                        f"    [W70B L-CASCADE WIN!!] n={m_next:4d} [Lext<-{m_cur}] {cur[m_next][0]:.7f} -> {L_e:.7f} (dS={ds_e:+.8f})!",
                        flush=True,
                    )
                    save_ckpt()
            Q_cur = Q_ext
            m_cur = m_next
        return True

    # =========================================================================
    # PHASE 1 (~850s): All-1000 Bidirectional Transfer (`del_lp` + `ins` + `lpeel`)
    # =========================================================================
    transfer_jobs = []
    bseed0 = args.seed_off + args.shard * 9973
    for idx_s, n in enumerate(my_sweep):
        L0 = get_L(n)
        P0 = get_P(n)
        s_base = bseed0 + idx_s * 37

        # 1) `n + 1 -> n` boundary + sparse-interior deletions + HiGHS LP + `fq`
        if n + 1 <= 1000 and get_P(n + 1) is not None and get_L(n + 1) - L0 < 0.055:
            d_cands = btr.delete_one_candidates(
                get_P(n + 1), max_boundary=10, max_sparse=6
            )
            transfer_jobs.append((0, "del_lp", "del1", n, L0, d_cands, s_base + 1))

        # 2) `(K + 1)^2 + D -> K^2 + D` (and `(K + 2)^2 + D`) 4-corner Soft L-Peel!
        K = math.isqrt(n)
        for step_k in (1, 2):
            K_up = K + step_k
            n_up = K_up * K_up + (n - K * K)
            if n_up <= 1000 and get_P(n_up) is not None:
                if get_L(n_up) - float(step_k) < L0 + 0.0035:
                    for pi_c, (ptag, P_peel) in enumerate(
                        btr.soft_corner_lpeel_candidates(get_P(n_up), n)
                    ):
                        transfer_jobs.append(
                            (
                                1,
                                "lpeel",
                                f"{ptag}_from{n_up}",
                                n,
                                L0,
                                P_peel,
                                s_base + 9 + 4 * step_k + pi_c,
                            )
                        )

        # 3) `n - 1 -> n` `insert_holes` + `target-hole insertion` when step > 0.0025
        if n > 5 and get_P(n - 1) is not None and L0 - get_L(n - 1) > 0.0025:
            ins_cands = btr.insert_holes_candidates(
                get_P(n - 1), P_n_own=P0, seed=s_base + 21, n_holes=2, n_target=1
            )
            for ii_c, (itag, P_ins) in enumerate(ins_cands[:2]):
                transfer_jobs.append((2, "ins", itag, n, L0, P_ins, s_base + 25 + ii_c))

    # Execute Phase 1 priority-ordered (`del_lp` & `lpeel` first, then `ins`)
    transfer_jobs.sort(key=lambda z: z[0])
    novel_pool = []
    p1_budget = min(900.0, 0.50 * args.budget_sec)
    ex = ProcessPoolExecutor(max_workers=args.procs)
    done_jobs = 0
    try:
        for b0 in range(0, len(transfer_jobs), 24):
            if time.time() - t0 > p1_budget or time.time() > deadline - 180.0:
                break
            batch = transfer_jobs[b0 : b0 + 24]
            futs = [ex.submit(btr.eval_bidir_seed_job, jb[1:]) for jb in batch]
            try:
                for f in as_completed(futs, timeout=135.0):
                    done_jobs += 1
                    md, tg, n_j, S_j, Q_j = f.result()
                    if Q_j is None or not math.isfinite(S_j):
                        continue
                    L_cur = get_L(n_j)
                    if S_j < L_cur + 0.00025:
                        Qp_j, Sp_j = w51d.deep_contact_polish(
                            n_j, Q_j, deadline, slp_sec=34.0, fq_sec=20
                        )
                        record_win_and_cascade(n_j, Qp_j, Sp_j, f"p1_{tg}")
                        Q_j, S_j = Qp_j, Sp_j
                    ad_j = w53h.topology_angle_dist(Q_j, get_P(n_j))
                    if S_j < get_L(n_j) + 0.011 and ad_j > 0.12:
                        novel_pool.append((S_j - get_L(n_j), n_j, tg, Q_j))
            except TimeoutError:
                w51d.terminate_pool(ex)
                ex = ProcessPoolExecutor(max_workers=args.procs)

        print(
            f"[W70B s={args.shard}] Phase-1 transfer done={done_jobs}/{len(transfer_jobs)} "
            f"wins={len(wins)} novel={len(novel_pool)} ({time.time() - t0:.1f}s)",
            flush=True,
        )

        # =========================================================================
        # PHASE 2 (~900s): Dedicated `K^2 + 1` / `K^2 + 2` Sub-Half Ladder + Fresh Ranks `#49..#64`
        # =========================================================================
        k2p1_targets = [
            730,
            962,
            785,
            901,
            530,
            401,
            842,
            577,
            362,
            677,
            485,
            442,
            731,
            786,
            902,
            963,
        ]
        my_special = [k2p1_targets[args.shard % len(k2p1_targets)]]
        # Add best novel basin from Phase 1 (if distinct puzzle) or fresh Rank #49..#64 target
        novel_pool.sort(key=lambda z: z[0])
        nbr_extra = {}
        for _, n_nv, _, Q_nv in novel_pool:
            nbr_extra.setdefault(n_nv, []).append(Q_nv)
        if novel_pool and novel_pool[0][1] not in my_special:
            my_special.append(novel_pool[0][1])
        else:
            ranked_gap = [
                n
                for n in all_non_grid
                if max(0.0, get_L(n) - front_L.get(n, get_L(n))) > 5e-5
            ]
            idx_alt = 48 + args.shard
            if idx_alt < len(ranked_gap) and ranked_gap[idx_alt] not in my_special:
                my_special.append(ranked_gap[idx_alt])

        for pi, n_sp in enumerate(my_special):
            rem = deadline - time.time()
            if rem < 130.0:
                break
            sp_b = (rem - 15.0) / max(1, len(my_special) - pi)
            ga_s = min(440.0, max(120.0, 0.62 * sp_b))
            L0_sp = get_L(n_sp)
            P0_sp = get_P(n_sp)
            nbrs_sp = list(nbr_extra.get(n_sp, [])[:3])

            # Cross-K L-peel donors from `K+1` and `K+2` (critical for `730 <- 785, 842`!)
            K_sp = math.isqrt(n_sp)
            for sk in (1, 2):
                m_up = (K_sp + sk) ** 2 + (n_sp - K_sp * K_sp)
                if m_up <= 1000 and get_P(m_up) is not None:
                    for _, P_pl in btr.soft_corner_lpeel_candidates(get_P(m_up), n_sp)[
                        :2
                    ]:
                        rh_p = RE.run_harden_reharden(
                            P_pl,
                            s_ref=L0_sp,
                            seed=bseed0 + 700 + sk,
                            r0=0.14,
                            timeout_s=9.5,
                        )
                        if rh_p is not None and rh_p[0] < L0_sp + 0.020:
                            nbrs_sp.append(rh_p[1])
            if n_sp + 1 <= 1000 and get_P(n_sp + 1) is not None:
                nbrs_sp.append(get_P(n_sp + 1))

            Qw, Qc, js = RE.run_santa_ga(
                P0_sp,
                nbr_packings=nbrs_sp,
                seed=bseed0 + 5000 + pi * 313,
                sec=ga_s,
                islands=6,
                pop=14,
                offspring=28,
                steps=125,
                threads=args.procs,
                timeout_s=ga_s + 35.0,
            )
            print(
                f"  [W70B GA-v4] n={n_sp:4d} gens={js.get('gens', 0)} swaps={js.get('swaps', 0)} "
                f"ga_wins={js.get('wins', 0)} s_best={js.get('s_best', L0_sp):.7f} (L0={L0_sp:.7f})",
                flush=True,
            )
            hop_P = P0_sp
            hop_tg = "p2_cur"
            if Qw is not None:
                Qp_w, Sp_w = w51d.deep_contact_polish(
                    n_sp, Qw, deadline, slp_sec=40.0, fq_sec=24
                )
                record_win_and_cascade(n_sp, Qp_w, Sp_w, "p2_ga_v4_win")
                hop_P, hop_tg = Qp_w, "p2_ga_v4_win"
            elif Qc is not None:
                rq_c = RE.run_fq_quench_args(
                    Qc,
                    [
                        "--seed",
                        str(bseed0 + 5111 + pi),
                        "--loosen",
                        "1.0045",
                        "--outer",
                        "44",
                        "--pit",
                        "280",
                        "--flip-top",
                        "20",
                        "--finish",
                        "--stag-w",
                        "0",
                        "--deadline",
                        "14.0",
                    ],
                    s_ref=L0_sp,
                    timeout_s=17.0,
                )
                if rq_c is not None:
                    Sc_p, Qc_p = RE.kkt_sqp_polish_fq(rq_c[1], timeout_each=6.0)
                    if Sc_p < get_L(n_sp) - 1e-9:
                        Qp_c, Sp_c = w51d.deep_contact_polish(
                            n_sp, Qc_p, deadline, slp_sec=38.0, fq_sec=22
                        )
                        record_win_and_cascade(n_sp, Qp_c, Sp_c, "p2_ga_v4_champ")
                        Qc_p, Sc_p = Qp_c, Sp_c
                    if Sc_p < get_L(n_sp) + 0.013:
                        hop_P, hop_tg = Qc_p, "p2_ga_v4_champ"

            hop_sec = max(75.0, min(deadline - time.time() - 12.0, sp_b - ga_s - 30.0))
            if hop_sec >= 65.0:
                Qb, Sb, _ = w63k.continuous_momentum_widehopper(
                    ex,
                    n_sp,
                    hop_P,
                    get_L(n_sp),
                    bseed0 + 80000 + pi * 97,
                    hop_sec,
                    deadline,
                    lambda Qx, Sx, stg, _wc, _n=n_sp, _tg=hop_tg: (
                        record_win_and_cascade(_n, Qx, Sx, f"{_tg}->{stg}")
                    ),
                )
                record_win_and_cascade(n_sp, Qb, Sb, f"{hop_tg}->whFin")
    finally:
        w51d.terminate_pool(ex)

    save_ckpt()
    print(
        f"[W70B s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
