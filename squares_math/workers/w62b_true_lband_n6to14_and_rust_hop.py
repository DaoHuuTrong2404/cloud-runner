#!/usr/bin/env python3
"""Wave 62B (`w62b_true_lband_n6to14_and_rust_hop.py`): Faithful Exact-Angle Two-Arm L-Band MILP
(`N_arm = 5..14`, `mode in {'right_rot', 'right_ref'}`) + Hybrid Rust `packer` -> LP WideHopper!

Fixes ALL FIVE bugs discovered by comparing `w57t` line-by-line against reference `lband.py` & `lband2.py`:
  1. Exact mode names `('right_rot', 'right_ref')` (`w57t` passed `'right'`/`'right_opp'`, so same-tilt `'right_rot'`
     was NEVER tested once!).
  2. Full deep-arm range `N_arm = 5..14` (`w57t` capped at `3..6`, which cannot pack `250..404` tilted squares
     required on `n = 730, 785, 901, 962, 914, 976, 738, 804, 491, 367` at small overhang `f`!).
  3. Reference excess-square trimming (`np.argsort(-np.abs(sol[:, 0] - sol[:, 1]))[:n]` from `lband.py` line 152,
     instead of `pick_near_band` which punched holes in the middle of the tilted arm!).
  4. NO `relocate_crushed_fast` on `tol = 0.010` near-feasible MILP solutions!
  5. Plain `fq quench --loosen 1.015` / `1.035` (`s_ref = K + f_try`, `--deadline 25.0`, NO `--round-r0`),
     plus `lband.assemble_L(..., extra=(-3, 3, 6))` steeper-tilt sliding end columns, escalated through
     Hybrid Rust `packer` -> `w51d.deep_3round_widehopper` + `Decimal64`!
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
import scipy.optimize._milp as _milp

if hasattr(_milp, "_highs_wrapper"):
    _hw = _milp._highs_wrapper
    _milp._highs_wrapper = lambda *a: _hw(*a[:-1], dict(a[-1], threads=1))

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
TOOLS_DIR = os.path.abspath(os.path.join(HERE, "..", "tools"))
for _d in (DS_DIR, TOOLS_DIR, HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import lband as LB
import lband2 as LB2
import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w59l_faithful_lcut_and_rust_hop as w59l


def trim_to_n_reference(sol, n):
    """Exact `lband.py` line 152 trimming: keep squares closest to the top-left / bottom-right band ends or core."""
    sol = np.asarray(sol, dtype=np.float64)
    if len(sol) <= n:
        return sol
    keep = np.sort(np.argsort(-np.abs(sol[:, 0] - sol[:, 1]))[:n])
    return sol[keep].copy()


def eval_puzzle_lband_sweep(job_tup):
    n, K, f_front, f_cur, seed0 = job_tup
    L0 = K + f_cur
    # Target f values strictly below f_cur (from f_front - 0.002 up to f_cur - 0.001)
    f_lo = max(0.32, min(f_front - 0.001, f_cur - 0.012))
    f_hi = max(f_lo + 0.002, f_cur - 0.0005)
    f_grid = list(np.linspace(f_lo, f_hi, 3))

    max_N = min(14, max(6, K - 3))
    min_N = 4 if K <= 17 else 5
    Ns = list(range(min_N, max_N + 1))

    hits_raw = []
    # 1) Fast Split-MILP (`lband2.assemble_split` with `rich=0.25`, `tol=0.010`)
    for f_try in f_grid:
        for N_arm in Ns:
            if len(hits_raw) >= 6:
                break
            for mode in ("right_rot", "right_ref"):
                for pt, pr in ((0.0, 0.0), (0.0, 0.5), (0.5, 0.5)):
                    try:
                        cnt, sol = LB2.assemble_split(
                            K,
                            float(f_try),
                            int(N_arm),
                            pt,
                            pr,
                            mode,
                            ec=3,
                            phases=5,
                            jm=2.6,
                            rich=0.25,
                            time_limit=2.5,
                            tol=0.010,
                        )
                        if sol is not None and cnt >= n:
                            hits_raw.append(
                                (
                                    float(f_try),
                                    f"lb2_N{N_arm}_{mode[-3:]}_{pt:.1f}_{pr:.1f}",
                                    trim_to_n_reference(sol, n),
                                )
                            )
                            break
                    except Exception:
                        continue

    # 2) Monolithic `lband.assemble_L` with steeper-tilt sliding end columns `extra=(-3, 3, 6)`!
    if len(hits_raw) < 4:
        f_mid = 0.5 * (f_lo + f_hi)
        for N_arm in Ns[::2]:
            for mode in ("right_rot", "right_ref"):
                try:
                    cnt, sol = LB.assemble_L(
                        K,
                        float(f_mid),
                        int(N_arm),
                        0.0,
                        0.0,
                        mode,
                        ec=3,
                        phases=5,
                        time_limit=3.5,
                        jm=2.0,
                        extra=(-3, 3, 6),
                    )
                    if sol is not None and cnt >= n:
                        hits_raw.append(
                            (
                                float(f_mid),
                                f"lb1x_N{N_arm}_{mode[-3:]}",
                                trim_to_n_reference(sol, n),
                            )
                        )
                        if len(hits_raw) >= 6:
                            break
                except Exception:
                    continue

    quenched = []
    for qi, (f_try, tag, Q_sol) in enumerate(hits_raw[:6]):
        s_box = float(K) + float(f_try)
        Q_in = RE.normalize_to_origin(Q_sol)
        Q_in[:, 0] = np.clip(Q_in[:, 0], 0.5, max(s_box, RE.side_exact(Q_in)) - 0.5)
        Q_in[:, 1] = np.clip(Q_in[:, 1], 0.5, max(s_box, RE.side_exact(Q_in)) - 0.5)
        for rep_i, lo in enumerate((1.015, 1.035)):
            rq = RE.run_fq_quench_args(
                Q_in,
                [
                    "--seed",
                    str(int(seed0) + qi * 7 + rep_i),
                    "--loosen",
                    repr(float(lo)),
                    "--outer",
                    "60",
                    "--pit",
                    "320",
                    "--flip-top",
                    "24",
                    "--finish",
                    "--stag-w",
                    "0",
                    "--deadline",
                    "22.0",
                ],
                s_ref=s_box,
                timeout_s=25.0,
            )
            if rq is not None:
                Sq, Qq = float(rq[0]), rq[1]
                if Sq < L0 + 0.0175:
                    quenched.append((Sq, f"{tag}_l{int(round(lo * 1000))}", Qq))
    quenched.sort(key=lambda z: z[0])
    return n, len(hits_raw), quenched[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=22)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=1250000)
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

    targets = []
    for n in range(120, 1001):
        K = math.isqrt(n)
        if K * K == n:
            continue
        gap_l = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gap_l > 8.0e-5:
            targets.append((gap_l / math.sqrt(n), n))
    targets.sort(reverse=True)
    my_slice = [x[1] for x in targets[: args.num_shards * 4]][
        args.shard :: args.num_shards
    ]

    print(
        f"[W62B-TrueLBandN5to14 s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_slice={my_slice} ({time.time() - t0:.1f}s)",
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

    screened = []
    ex_sc = ProcessPoolExecutor(max_workers=args.procs)
    try:
        sc_tasks = []
        for pi, n in enumerate(my_slice):
            K = math.isqrt(n)
            f_cur = cur[n][0] - K
            f_front = front_L.get(n, cur[n][0]) - K
            sc_tasks.append(
                (n, K, f_front, f_cur, args.seed_off + args.shard * 3301 + pi * 173)
            )
        futs = [ex_sc.submit(eval_puzzle_lband_sweep, tk) for tk in sc_tasks]
        try:
            for f in as_completed(futs, timeout=540.0):
                n, n_raw, q_list = f.result()
                L0 = wins[n][0] if n in wins else cur[n][0]
                P0 = cur[n][1]
                for Sq, tg, Qq in q_list:
                    if Sq < L0 - 1e-9:
                        Qp0, Sp0 = w51d.deep_contact_polish(
                            n, Qq, deadline, slp_sec=40.0, fq_sec=24
                        )
                        ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                        if ok0 and Lf0 < L0 - 1e-9:
                            ds0 = (Lf0 - L0) / math.sqrt(n)
                            tot_ds += ds0
                            wins[n] = (float(Lf0), xf0, yf0, tf0)
                            L0 = float(Lf0)
                            print(
                                f"  [W62B DIRECT LBAND WIN!!] n={n:4d} [{tg}] {cur[n][0]:.7f} -> {Lf0:.7f} (dS={ds0:+.8f})!",
                                flush=True,
                            )
                            save_ckpt()
                    ad = w53h.topology_angle_dist(Qq, P0) if P0 is not None else 1.0
                    if Sq < L0 - 1e-9 or (Sq < L0 + 0.0165 and ad > 0.030):
                        screened.append((Sq - L0, n, tg, Sq, ad, Qq))
                b_str = (
                    f"{q_list[0][0] - cur[n][0]:+.6f}@{q_list[0][1]}"
                    if q_list
                    else "none"
                )
                print(
                    f"  [W62B screen] n={n:4d} milp_hits={n_raw} quenched={len(q_list)} best={b_str}",
                    flush=True,
                )
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_sc)

    screened.sort(key=lambda z: z[0])
    queued = screened[:5]
    print(
        f"[W62B s={args.shard}] Phase B: hunting {len(queued)}/{len(screened)} L-band basins ({time.time() - t0:.1f}s)!",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, tag, S_init, adist, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 75.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            best_Q, best_S = RE.normalize_to_origin(Q_init), float(S_init)

            per_slot = min(125.0, max(40.0, (rem - 60.0) * 0.25))
            bseed = args.seed_off + 60000 + args.shard * 9001 + i * 313
            k_ceil = math.ceil(math.sqrt(n) - 1e-12)
            forbid_k = k_ceil if L0_cur < k_ceil + 1e-6 else 0
            ts_goal = min(best_S, L0_cur) - 4.5e-5

            rust_tasks = [
                ("hhop", best_Q, ts_goal, bseed + 1, forbid_k, False, per_slot),
                ("hhop", best_Q, ts_goal, bseed + 2, forbid_k, True, per_slot),
                ("hop", best_Q, ts_goal, bseed + 3, forbid_k, False, per_slot),
                ("hop", best_Q, ts_goal, bseed + 4, forbid_k, True, per_slot),
            ]
            futs_r = [ex_hop.submit(w59l.run_rust_worker_task, rt) for rt in rust_tasks]
            best_mode = "init"
            try:
                for fr in as_completed(futs_r, timeout=per_slot + 35.0):
                    rr = fr.result()
                    if rr is not None and rr[0] < best_S - 1e-10:
                        best_S, best_Q, best_mode = rr[0], rr[1], rr[2]
            except TimeoutError:
                w51d.terminate_pool(ex_hop)
                ex_hop = ProcessPoolExecutor(max_workers=args.procs)

            rem2 = deadline - time.time()
            wn_cnt = 0
            if best_S < L0_cur + 0.0048 and rem2 > 85.0:
                r1 = min(155, max(35, int(0.35 * (rem2 - 25.0))))
                r2 = min(140, max(30, int(0.30 * (rem2 - 25.0))))
                r3 = min(140, max(30, int(0.30 * (rem2 - 25.0))))
                Qw, Sw, (_, _, wn_cnt) = w51d.deep_3round_widehopper(
                    ex_hop, n, best_Q, L0_cur, bseed + 700, r1, r2, r3, deadline
                )
                if Sw < best_S - 1e-10:
                    best_S, best_Q, best_mode = Sw, Qw, best_mode + "->wh3r"
            elif best_S < L0_cur + 0.0020 or best_S < S_init - 2e-4:
                best_Q, best_S = w51d.deep_contact_polish(
                    n, best_Q, deadline, slp_sec=44.0, fq_sec=26
                )

            if best_S < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, best_Q)
                if ok and Lf < L0_cur - 1e-9:
                    ds = (Lf - L0_cur) / math.sqrt(n)
                    tot_ds += ds
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W62B LBAND WIN!!] n={n:4d} [{tag}->{best_mode}] dA={adist:.2f} {cur[n][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={wn_cnt})!",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W62B done] n={n:4d} [{tag}->{best_mode}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={best_S - L0_cur:+.6f} w={wn_cnt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W62B s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
