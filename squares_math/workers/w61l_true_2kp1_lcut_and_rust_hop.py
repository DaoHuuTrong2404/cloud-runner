#!/usr/bin/env python3
"""Wave 61L (`w61l_true_2kp1_lcut_and_rust_hop.py`): Exact Cross-Occupancy `d >= 2K + 1` Inexact L-Cut
(`n0 in [Kp^2 + D .. Kp^2 + D + 4]`) + Hybrid Rust `packer` -> `w51d.deep_3round_widehopper`!

Fixes the exact donor-crowding bug discovered in Wave 59L logs:
  - In `W59L`, `lo = Kp^2 + D - 4` allowed `d = 2K - 3 .. 2K - 1` (`261..263 -> 234`, `402 -> 367`,
    `532..534 -> 491`, `366 -> 333`). Because those smaller-`n0` packings had smaller `S_don`, their 24 cuts
    crowded out 100% of `d >= 2K + 1`, while leaving 2..4 unremoved squares jammed inside the box (`0` hits)!
  - Conversely, `n = 401` (`D = 1`) clamped `lo = Kp^2 + 1 = Kp^2 + D` (`d = 41 = 2K + 1`, `42 = 2K + 2`),
    immediately scoring `18 / 26` hits and reaching `dEnd = +0.000436` via `hop_inf_fq_fz_r2`!
  - By First-Principles Cross Area `Area([cx, cx+1) u [cy, cy+1)) = 2*S_don - 1 = 2K + 1 + 2*f_don`,
    valid inexact L-cuts ALWAYS satisfy `d in {2K + 1, 2K + 2, 2K + 3, 2K + 4}` (`n0 = Kp^2 + D + {0, 1, 2, 3}`)!
  - Enforces strict per-donor cut quotas across `d = 2K + 1 .. 2K + 4` (`loosen = 1.02` & `1.05`), then chains
    compiled Rust `packer hhop`/`hop` (macro barrier crossing) directly into `w51d.deep_3round_widehopper`
    (exact LP contact-hopping + Residual Contact Continuation + `fq --sqp --finish`)!
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
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w59l_faithful_lcut_and_rust_hop as w59l


def build_2kp1_donor_jobs(n, cur, bseed):
    """Strictly enforce `d = n0 - n >= 2*K + 1` (`d in {2K+1, 2K+2, 2K+3, 2K+4}`) with per-donor quotas!"""
    K = math.isqrt(n)
    if K >= 31:
        return [], []
    D = n - K * K
    Kp = K + 1
    L0 = cur[n][0]

    # Exact cross-occupancy law: d = 2K + 1 + extra, extra in {0, 1, 2, 3} -> n0 = Kp^2 + D + extra!
    jobs = []
    don_ids = []
    j_idx = 0
    quotas = {0: 7, 1: 5, 2: 4, 3: 3}
    for extra in (0, 1, 2, 3):
        n0 = Kp * Kp + D + extra
        if n0 > 1000 or n0 >= (Kp + 1) * (Kp + 1) or cur[n0][1] is None:
            continue
        if math.isqrt(n0) ** 2 == n0:
            continue
        P_don = RE.normalize_to_origin(cur[n0][1])
        S_don = RE.side_exact(P_don)
        # Allow wider S_don when removing extra > 0 squares since excess area compresses during fq quench!
        if S_don - 1.0 > L0 + 0.032 + 0.025 * extra:
            continue
        P_deg = P_don.copy()
        P_deg[:, 2] = np.degrees(P_deg[:, 2]) % 90.0
        m_want = quotas[extra]
        cuts = LC.best_cuts(P_deg, S_don, n0 - n, step=0.05, m=m_want, sep=0.14)
        if not cuts:
            continue
        don_ids.append(n0)
        for ci, (sc, cx, cy, mx_f) in enumerate(cuts):
            jobs.append(
                (
                    f"lcut{n0}_d{n0 - n}_c{ci}_l102",
                    P_deg,
                    S_don,
                    float(cx),
                    float(cy),
                    n0 - n,
                    n,
                    1.02,
                    0.0,
                    bseed + j_idx,
                )
            )
            j_idx += 1
            if ci < max(2, m_want - 2):
                jobs.append(
                    (
                        f"lcut{n0}_d{n0 - n}_c{ci}_l105",
                        P_deg,
                        S_don,
                        float(cx),
                        float(cy),
                        n0 - n,
                        n,
                        1.05,
                        0.0015,
                        bseed + j_idx,
                    )
                )
                j_idx += 1
    return jobs, don_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=24)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=1100000)
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
    for n in range(100, 961):
        K = math.isqrt(n)
        if K * K == n or K >= 31:
            continue
        gap_l = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gap_l > 8.0e-5:
            targets.append((gap_l / math.sqrt(n), n))
    targets.sort(reverse=True)
    my_slice = [x[1] for x in targets[: args.num_shards * 4]][
        args.shard :: args.num_shards
    ]

    print(
        f"[W61L-True2Kp1LCut s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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
        for pi, n in enumerate(my_slice):
            if time.time() - t0 > 580.0:
                break
            L0 = cur[n][0]
            P0 = cur[n][1]
            bseed = args.seed_off + args.shard * 4801 + pi * 233
            jobs, don_ids = build_2kp1_donor_jobs(n, cur, bseed)
            if not jobs:
                continue

            ts0 = time.time()
            futs = [ex_sc.submit(w59l.eval_faithful_lcut_job, jb) for jb in jobs]
            hits = []
            direct_best = None
            try:
                for f in as_completed(futs, timeout=240.0):
                    tg, Sq, Qq, njam = f.result()
                    if Qq is None:
                        continue
                    gap0 = Sq - L0
                    ad = w53h.topology_angle_dist(Qq, P0) if P0 is not None else 1.0
                    if gap0 < -1e-9:
                        if direct_best is None or Sq < direct_best[0]:
                            direct_best = (Sq, tg, Qq, ad, njam)
                    if gap0 < -1e-9 or (gap0 < 0.0175 and ad > 0.030):
                        hits.append(
                            (
                                gap0 + 0.0015 * min(njam, 2),
                                gap0,
                                n,
                                tg,
                                Sq,
                                ad,
                                njam,
                                Qq,
                            )
                        )
            except TimeoutError:
                w51d.terminate_pool(ex_sc)
                ex_sc = ProcessPoolExecutor(max_workers=args.procs)

            if direct_best is not None:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    n, direct_best[2], deadline, slp_sec=42.0, fq_sec=25
                )
                L0_cur = wins[n][0] if n in wins else cur[n][0]
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        ds0 = (Lf0 - L0_cur) / math.sqrt(n)
                        tot_ds += ds0
                        wins[n] = (float(Lf0), xf0, yf0, tf0)
                        print(
                            f"  [W61L DIRECT LCUT WIN!!] n={n:4d} [{direct_best[1]}] {cur[n][0]:.7f} -> {Lf0:.7f} (dS={ds0:+.8f})!",
                            flush=True,
                        )
                        save_ckpt()

            hits.sort(key=lambda z: z[0])
            uniq_n = []
            for h in hits:
                if all(abs(h[4] - u[4]) > 4e-6 for u in uniq_n):
                    uniq_n.append(h)
                if len(uniq_n) >= 3:
                    break
            b_str = (
                f"{uniq_n[0][1]:+.6f}(dA={uniq_n[0][5]:.2f},jam={uniq_n[0][6]})@{uniq_n[0][3]}"
                if uniq_n
                else "none"
            )
            print(
                f"  [W61L screen] n={n:4d}<-{don_ids} jobs={len(jobs)} hits={len(hits)} best={b_str} ({time.time() - ts0:.1f}s)",
                flush=True,
            )
            for u in uniq_n[:2]:
                screened.append(u)
    finally:
        w51d.terminate_pool(ex_sc)

    screened.sort(key=lambda z: z[0])
    queued = screened[:5]
    print(
        f"[W61L s={args.shard}] Phase B (Hybrid Rust packer -> LP WideHopper): hunting {len(queued)}/{len(screened)} basins ({time.time() - t0:.1f}s)!",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (_, gap0, n, tag, S_init, adist, njam, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 75.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            best_Q, best_S = RE.normalize_to_origin(Q_init), float(S_init)

            # Stage 1: Native Rust `packer hhop` + `packer hop` (`125s` across 4 cores)
            per_slot = min(125.0, max(40.0, (rem - 60.0) * 0.25))
            bseed = args.seed_off + 50000 + args.shard * 9001 + i * 313
            k_ceil = math.ceil(math.sqrt(n) - 1e-12)
            forbid_k = k_ceil if (L0_cur < k_ceil + 1e-6 and njam == 0) else 0
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

            # Stage 2: Exact LP Contact WideHopper (`w51d.deep_3round_widehopper`) when within +0.0048!
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
                        f"  [W61L HYBRID WIN!!] n={n:4d} [{tag}->{best_mode}] dA={adist:.2f} {cur[n][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={wn_cnt})!",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W61L done] n={n:4d} [{tag}->{best_mode}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={best_S - L0_cur:+.6f} w={wn_cnt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W61L s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
