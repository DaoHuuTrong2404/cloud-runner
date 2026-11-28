#!/usr/bin/env python3
"""Wave 63K (`w63k_1puzzle_per_shard_killshot.py`): 1-Puzzle-Per-Shard 160-Core Laser Killshot
(`7,200` Core-Seconds / Puzzle) on the Top 40 Proven Active `d >= 2K + 1` Inexact L-Cut Cascades!

Directly harvests the 18 active `w > 0` cascades + 22 sub-`0.010` line-free `jam = 0` basins discovered in Wave 61L:
  - In `W61L`, 4 puzzles shared one shard (`~400s` per basin), cutting off active descents right before record
    breakthrough (`n=904 +0.000285 / w=183`, `n=377 +0.000294 / w=126`, `n=748 +0.000000 / w=75`,
    `n=240 +0.000361 / w=31`, `n=937 +0.000249`, `n=685 +0.000348`, `n=376 +0.000320`, `n=415 +0.000469`,
    `n=234 +0.000161`, `n=418 w=274`, `n=380 w=40`, `n=401 +0.000436`, ...).
  - Here, Shard `s` (`0..39`) owns **EXACTLY ONE PUZZLE** `n = HOT_40[s]` for the entire `1,800s` (`4` cores):
    1. Phase A (`~135s`): Evaluates `42` ultra-fine `step=0.04, sep=0.10` `d >= 2K + 1` cuts across
       `loosen in {1.018, 1.032, 1.050}` (`jam == 0` prioritized!).
    2. Phase B (`~1,645s` = `27.4` minutes!): Runs Native Rust `packer hhop`/`hop` -> **6-Round Continuous
       Momentum `eta <= 0.28` LP WideHopper** (`24+` seeds + interleaved `SLP` + `Residual Contact Continuation`
       + two-pass `fq --sqp --finish --stag-w 0 --rmin 1e-12`) that NEVER stops early while `w > 0` is descending!
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

import lcut as LC
import rust_engines as RE
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w50d_exhaustive_private_del1_widehopper as w50d
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w59l_faithful_lcut_and_rust_hop as w59l

# Ranked Top-40 proven active `d >= 2K + 1` lcut targets from Wave 61L empirical logs (`task-7434` & `task-7449`)
HOT_40 = [
    904,
    377,
    748,
    240,
    937,
    685,
    376,
    415,
    234,
    418,
    380,
    401,
    338,
    449,
    544,
    789,
    333,
    632,
    308,
    273,
    367,
    640,
    456,
    303,
    746,
    806,
    591,
    577,
    506,
    653,
    759,
    852,
    859,
    371,
    414,
    805,
    295,
    498,
    530,
    644,
]


def build_fine_single_puzzle_lcut_jobs(n, cur, bseed):
    K = math.isqrt(n)
    if K >= 31:
        return [], []
    D = n - K * K
    Kp = K + 1
    L0 = cur[n][0]

    jobs = []
    don_ids = []
    j_idx = 0
    # Dense quotas focused on extra=1 (`2K+2`) and extra=2 (`2K+3`), which produced 90% of W61L's active cascades!
    quotas = {0: 4, 1: 7, 2: 6, 3: 3}
    for extra in (1, 2, 0, 3):
        n0 = Kp * Kp + D + extra
        if n0 > 1000 or n0 >= (Kp + 1) * (Kp + 1) or cur[n0][1] is None:
            continue
        if math.isqrt(n0) ** 2 == n0:
            continue
        P_don = RE.normalize_to_origin(cur[n0][1])
        S_don = RE.side_exact(P_don)
        if S_don - 1.0 > L0 + 0.038 + 0.025 * extra:
            continue
        P_deg = P_don.copy()
        P_deg[:, 2] = np.degrees(P_deg[:, 2]) % 90.0
        m_want = quotas[extra]
        cuts = LC.best_cuts(P_deg, S_don, n0 - n, step=0.04, m=m_want, sep=0.10)
        if not cuts:
            continue
        don_ids.append(n0)
        for ci, (sc, cx, cy, mx_f) in enumerate(cuts):
            for lo, kk in ((1.02, 0.0), (1.05, 0.0015)):
                jobs.append(
                    (
                        f"lcut{n0}_d{n0 - n}_c{ci}_l{int(round(lo * 100))}",
                        P_deg,
                        S_don,
                        float(cx),
                        float(cy),
                        n0 - n,
                        n,
                        lo,
                        kk,
                        bseed + j_idx,
                    )
                )
                j_idx += 1
            if ci < 2:
                jobs.append(
                    (
                        f"lcut{n0}_d{n0 - n}_c{ci}_l103",
                        P_deg,
                        S_don,
                        float(cx),
                        float(cy),
                        n0 - n,
                        n,
                        1.032,
                        0.0008,
                        bseed + j_idx,
                    )
                )
                j_idx += 1
    return jobs[:42], don_ids


def continuous_momentum_widehopper(
    ex, n, P_start, L0, base_seed, max_budget_sec, global_deadline, on_win_cb
):
    """6-Round Continuous Momentum `eta <= 0.28` LP WideHopper!
    Keeps launching fresh 4-seed rounds (`~150s` each) + full KKT `deep_contact_polish` as long as
    either `best_S < L0 + 0.0018` OR the previous round made > `5e-7` progress!
    """
    t_start = time.time()
    sub_deadline = min(global_deadline, t_start + max_budget_sec)
    best_P = w32a.safe_open(P_start)
    best_S = float(E.side(best_P))
    tot_tr, tot_fit, tot_w, rounds_run = 0, 0, 0, 0

    for rnd in range(6):
        rem = sub_deadline - time.time()
        if rem < 55.0:
            break
        rounds_run += 1
        s_before = best_S
        sec_r = min(155, max(40, int(0.65 * (rem - 35.0)) if rnd == 5 else 150))
        sec_r = min(sec_r, int(rem - 35.0))
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

        if best_S < L0 + 0.0032 or best_S < s_before - 1e-8:
            use_fq = 28 if best_S < L0 + 0.0012 else 14
            best_P, best_S = w51d.deep_contact_polish(
                n, best_P, sub_deadline, slp_sec=46.0, fq_sec=use_fq
            )

        print(
            f"    [W63K R{rnd + 1}] n={n:4d} dCur={best_S - L0:+.6f} (step={best_S - s_before:+.6f}, rw={rnd_w}, tot_w={tot_w}, {time.time() - t_start:.0f}s)",
            flush=True,
        )

        if best_S < L0 - 1e-9:
            on_win_cb(best_P, best_S, f"whR{rnd + 1}", tot_w)

        # Stop early ONLY after >= 2 rounds if far from L0 AND zero step progress!
        if (
            rnd >= 1
            and best_S >= L0 + 0.0018
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
    ap.add_argument("--num-shards", type=int, default=40)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=1450000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )

    n = HOT_40[args.shard % len(HOT_40)]
    L0 = cur[n][0]
    P0 = cur[n][1]
    deadline = t0 + args.budget_sec

    print(
        f"[W63K-1PuzzleKillshot s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"TARGET n={n} L0={L0:.8f} ({time.time() - t0:.1f}s)",
        flush=True,
    )

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

    def record_win_if_valid(Q_cand, S_cand, src_tag, w_cnt):
        nonlocal tot_ds
        L_ref = wins[n][0] if n in wins else cur[n][0]
        if S_cand >= L_ref - 1e-9:
            return
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, Q_cand)
        if ok and Lf < L_ref - 1e-9:
            ds = (Lf - L_ref) / math.sqrt(n)
            tot_ds += ds
            wins[n] = (float(Lf), xf, yf, tf)
            print(
                f"  [W63K KILLSHOT WIN!!] n={n:4d} [{src_tag}] {cur[n][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    bseed = args.seed_off + args.shard * 13007
    jobs, don_ids = build_fine_single_puzzle_lcut_jobs(n, cur, bseed)

    basins = []
    ex_sc = ProcessPoolExecutor(max_workers=args.procs)
    try:
        ts0 = time.time()
        futs = [ex_sc.submit(w59l.eval_faithful_lcut_job, jb) for jb in jobs]
        hits = []
        try:
            for f in as_completed(futs, timeout=270.0):
                tg, Sq, Qq, njam = f.result()
                if Qq is None:
                    continue
                gap0 = Sq - L0
                ad = w53h.topology_angle_dist(Qq, P0) if P0 is not None else 1.0
                if gap0 < -1e-9:
                    Qp0, Sp0 = w51d.deep_contact_polish(
                        n, Qq, deadline, slp_sec=42.0, fq_sec=25
                    )
                    record_win_if_valid(Qp0, Sp0, tg + "_direct", 0)
                if gap0 < -1e-9 or (gap0 < 0.0175 and ad > 0.025):
                    hits.append(
                        (gap0 + 0.0020 * min(njam, 2), gap0, tg, Sq, ad, njam, Qq)
                    )
        except TimeoutError:
            pass
        hits.sort(key=lambda z: z[0])
        for h in hits:
            if all(w53h.topology_angle_dist(h[6], u[6]) > 0.020 for u in basins):
                basins.append(h)
            if len(basins) >= 3:
                break
        b_str = (
            f"{basins[0][1]:+.6f}(dA={basins[0][4]:.2f},jam={basins[0][5]})@{basins[0][2]}"
            if basins
            else "none"
        )
        print(
            f"  [W63K screen] n={n:4d}<-{don_ids} jobs={len(jobs)} hits={len(hits)} uniq_basins={len(basins)} best={b_str} ({time.time() - ts0:.1f}s)",
            flush=True,
        )
    finally:
        w51d.terminate_pool(ex_sc)

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for bi, (_, gap0, tag, S_init, adist, njam, Q_init) in enumerate(basins):
            rem = deadline - time.time()
            if rem < 90.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            best_Q, best_S = RE.normalize_to_origin(Q_init), float(S_init)

            # Give Basin #0 65% of remaining time (~1,050s!), Basin #1 65% of what's left (~550s!)
            basin_budget = rem - 20.0 if bi == len(basins) - 1 else rem * 0.65
            rust_slot = min(135.0, max(45.0, basin_budget * 0.14))
            k_ceil = math.ceil(math.sqrt(n) - 1e-12)
            forbid_k = k_ceil if (L0_cur < k_ceil + 1e-6 and njam == 0) else 0
            ts_goal = min(best_S, L0_cur) - 4.5e-5
            rseed = bseed + 40000 + bi * 7919

            rust_tasks = [
                ("hhop", best_Q, ts_goal, rseed + 1, forbid_k, False, rust_slot),
                ("hhop", best_Q, ts_goal, rseed + 2, forbid_k, True, rust_slot),
                ("hop", best_Q, ts_goal, rseed + 3, forbid_k, False, rust_slot),
                ("hop", best_Q, ts_goal, rseed + 4, forbid_k, True, rust_slot),
            ]
            futs_r = [ex_hop.submit(w59l.run_rust_worker_task, rt) for rt in rust_tasks]
            best_mode = tag
            try:
                for fr in as_completed(futs_r, timeout=rust_slot + 35.0):
                    rr = fr.result()
                    if rr is not None and rr[0] < best_S - 1e-10:
                        best_S, best_Q, best_mode = rr[0], rr[1], f"{tag}->{rr[2]}"
            except TimeoutError:
                w51d.terminate_pool(ex_hop)
                ex_hop = ProcessPoolExecutor(max_workers=args.procs)

            if best_S < L0_cur - 1e-9:
                record_win_if_valid(best_Q, best_S, best_mode, 0)
                L0_cur = wins[n][0] if n in wins else cur[n][0]

            wh_budget = max(80.0, basin_budget - rust_slot - 15.0)
            best_Q, best_S, (_, _, w_tot, n_rnd) = continuous_momentum_widehopper(
                ex_hop,
                n,
                best_Q,
                L0_cur,
                rseed + 1000,
                wh_budget,
                deadline,
                lambda Qc, Sc, stg, wc: record_win_if_valid(
                    Qc, Sc, f"{best_mode}->{stg}", wc
                ),
            )
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            record_win_if_valid(best_Q, best_S, f"{best_mode}->whFinal", w_tot)
            print(
                f"  [W63K basin#{bi} done] n={n:4d} [{best_mode}] dStart={gap0:+.5f} -> dEnd={best_S - L0_cur:+.6f} rounds={n_rnd} w={w_tot}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W63K s={args.shard}] DONE in {time.time() - t0:.1f}s | n={n} | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
