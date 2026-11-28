#!/usr/bin/env python3
"""Wave 59L (`w59l_faithful_lcut_and_rust_hop.py`): Faithful Reference Inexact L-Cut (`lcut.py`)
+ Compiled Rust `packer hhop` / `packer hop` + KKT `deep_contact_polish`!

Fixes ALL THREE root causes diagnosed in `w57l_inexact_lcut_exact_n.py`:
  1. Exact donor target side `s1 = S_don - 1.0` (matching `lcut.py` line 98 — NEVER inflating to `L0_want`!).
  2. NO `relocate_crushed_fast` (which destroyed the grid by yanking gentle seam squares out of place!).
  3. Plain `fq quench --loosen 1.02` and `--loosen 1.05` (`--outer 60 --pit 320 --finish --stag-w 0 --deadline 25.0`),
     NO `--round-r0` time-starvation, evaluating up to `24` top-scored `(donor, cx, cy)` cuts per puzzle across
     ALL candidate donors in `[Kp^2 + max(1, D - 4), min(1000, Kp^2 + D + 9, (Kp+1)^2 - 1)]`!
  4. Escalates the sharpest new `lcut` basins through compiled Rust `packer hhop` (event-chain + `--soft 0.20`)
     AND `packer hop` (descending-target region-move + `best_hole` + `jam_lines` row-break), both unfrozen and
     far-grid-frozen (`build_far_grid_freeze_mask`), followed by KKT `deep_contact_polish` + `Decimal64`!
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
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h


def count_jam_lines_k(P_rad, need_k):
    """Count axis-aligned lines crossed by >= `need_k` near-axis squares (port of `packer::jam_lines_k`)."""
    half_pi = 0.5 * math.pi
    a_mod = np.mod(P_rad[:, 2], half_pi)
    tilts = np.minimum(a_mod, half_pi - a_mod)
    ax = np.where(tilts < 0.02)[0]
    if len(ax) < need_k:
        return 0
    found = []
    for dim in (1, 0):
        coords = P_rad[ax, dim]
        for v0 in coords:
            for off in (-0.45, 0.0, 0.45):
                y0 = float(v0) + off
                mem = tuple(ax[np.abs(coords - y0) < 0.5].tolist())
                if len(mem) >= need_k and (dim, mem) not in found:
                    found.append((dim, mem))
    return len(found)


def eval_faithful_lcut_job(job_tup):
    tag, P_don_deg, S_don, cx, cy, d_rem, n_want, loosen, kick, seed = job_tup
    try:
        C_deg = LC.apply_cut(P_don_deg, float(cx), float(cy), int(d_rem))
        if len(C_deg) != n_want:
            return tag, 1e9, None, 99
        # Faithful `lcut.py` line 98-99: exact `s1 = S_don - 1.0`, clip to `[0.5, s1 - 0.5]`, NO relocate_crushed!
        s1 = float(S_don) - 1.0
        Q_raw = np.asarray(C_deg, dtype=np.float64)
        Q_raw[:, 0] = np.clip(Q_raw[:, 0], 0.5, s1 - 0.5)
        Q_raw[:, 1] = np.clip(Q_raw[:, 1], 0.5, s1 - 0.5)
        Q_raw[:, 2] = np.radians(Q_raw[:, 2])

        extra = [
            "--seed",
            str(int(seed)),
            "--loosen",
            repr(float(loosen)),
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
            "24.5",
        ]
        if kick > 0.0:
            extra += ["--kick", repr(float(kick))]

        rq = RE.run_fq_quench_args(Q_raw, extra, s_ref=s1, timeout_s=27.5)
        if rq is None:
            return tag, 1e9, None, 99
        Sq, Qq = float(rq[0]), rq[1]
        k_ceil = math.ceil(math.sqrt(n_want) - 1e-12)
        n_jam = count_jam_lines_k(Qq, k_ceil)
        return tag, Sq, Qq, n_jam
    except Exception:
        return tag, 1e9, None, 99


def run_rust_worker_task(task_tup):
    mode, Q_in, target_s, seed, forbid_k, use_freeze, timeout_s = task_tup
    try:
        fz = (
            RE.build_far_grid_freeze_mask(Q_in, buffer_dist=3.6) if use_freeze else None
        )
        if mode == "hhop":
            rh = RE.run_packer_hhop(
                Q_in,
                seed=seed,
                iters=95000,
                temp=1.3e-4 if not use_freeze else 1.8e-4,
                chains=8 if not use_freeze else 10,
                chainlen=1.05,
                gmax=0.018 if not use_freeze else 0.022,
                soft=0.20,
                forbid_lines=forbid_k,
                frozen_mask=fz,
                timeout_s=timeout_s,
            )
            if rh is None:
                return None
            return (rh[0], rh[1], f"hhop_{'fz' if use_freeze else 'all'}")
        else:
            rp = RE.run_packer_hop(
                Q_in,
                target_s=target_s,
                seed=seed,
                iters=95000,
                temp=0.25 if not use_freeze else 0.30,
                amp=0.95 if not use_freeze else 1.10,
                maxit=2600,
                descend=4.5e-5,
                linepen=0.01,
                frozen_mask=fz,
                timeout_s=timeout_s,
            )
            if rp is None:
                return None
            return (rp[0], rp[1], f"{rp[2]}_{'fz' if use_freeze else 'all'}")
    except Exception:
        return None


def enumerate_scored_donor_cuts(n, cur, max_cuts=20):
    K = math.isqrt(n)
    if K >= 31:
        return []
    D = n - K * K
    Kp = K + 1
    L0 = cur[n][0]
    lo = max(Kp * Kp + 1, Kp * Kp + D - 4)
    hi = min(1000, Kp * Kp + D + 9, (Kp + 1) * (Kp + 1) - 1)

    scored_cuts = []
    for n0 in range(lo, hi + 1):
        if n0 <= n or cur[n0][1] is None or math.isqrt(n0) ** 2 == n0:
            continue
        P_don = RE.normalize_to_origin(cur[n0][1])
        S_don = RE.side_exact(P_don)
        if S_don - 1.0 > L0 + 0.008:
            continue
        P_deg = P_don.copy()
        P_deg[:, 2] = np.degrees(P_deg[:, 2]) % 90.0
        cuts = LC.best_cuts(P_deg, S_don, n0 - n, step=0.05, m=8, sep=0.14)
        for ci, (sc, cx, cy, mx_f) in enumerate(cuts):
            # Rank jointly by donor target `S_don - 1` plus seam-overlap penalty!
            rank_val = (S_don - 1.0 - L0) + 0.022 * float(sc)
            scored_cuts.append(
                (rank_val, float(sc), n0, ci, P_deg, S_don, float(cx), float(cy))
            )
    scored_cuts.sort(key=lambda x: x[0])
    return scored_cuts[:max_cuts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1820)
    ap.add_argument("--seed-off", type=int, default=850000)
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
        f"[W59L-FaithfulLCut s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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

    # PHASE A: Faithful reference `lcut` screening (`s1 = S_don - 1`, `loosen = 1.02 / 1.05`, NO relocate_crushed!)
    screened = []
    ex_sc = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_slice):
            if time.time() - t0 > 560.0:
                break
            L0 = cur[n][0]
            P0 = cur[n][1]
            top_cuts = enumerate_scored_donor_cuts(n, cur, max_cuts=18)
            if not top_cuts:
                continue
            bseed = args.seed_off + args.shard * 4001 + pi * 197
            jobs = []
            for idx_c, (_, sc, n0, ci, P_deg, S_don, cx, cy) in enumerate(top_cuts):
                jobs.append(
                    (
                        f"lcut{n0}_c{ci}_l102",
                        P_deg,
                        S_don,
                        cx,
                        cy,
                        n0 - n,
                        n,
                        1.02,
                        0.0,
                        bseed + idx_c * 2,
                    )
                )
                if idx_c < 10:
                    jobs.append(
                        (
                            f"lcut{n0}_c{ci}_l105",
                            P_deg,
                            S_don,
                            cx,
                            cy,
                            n0 - n,
                            n,
                            1.05,
                            0.0015,
                            bseed + idx_c * 2 + 1,
                        )
                    )

            ts0 = time.time()
            futs = [ex_sc.submit(eval_faithful_lcut_job, jb) for jb in jobs]
            hits = []
            direct_best = None
            try:
                for f in as_completed(futs, timeout=225.0):
                    tg, Sq, Qq, njam = f.result()
                    if Qq is None:
                        continue
                    gap0 = Sq - L0
                    ad = w53h.topology_angle_dist(Qq, P0) if P0 is not None else 1.0
                    if gap0 < -1e-9:
                        if direct_best is None or Sq < direct_best[0]:
                            direct_best = (Sq, tg, Qq, ad, njam)
                    if gap0 < -1e-9 or (gap0 < 0.0165 and ad > 0.035):
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
                    n, direct_best[2], deadline, slp_sec=40.0, fq_sec=24
                )
                L0_cur = wins[n][0] if n in wins else cur[n][0]
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        ds0 = (Lf0 - L0_cur) / math.sqrt(n)
                        tot_ds += ds0
                        wins[n] = (float(Lf0), xf0, yf0, tf0)
                        print(
                            f"  [W59L DIRECT LCUT WIN!!] n={n:4d} [{direct_best[1]}] {cur[n][0]:.7f} -> {Lf0:.7f} (dS={ds0:+.8f})!",
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
            don_set = sorted({tc[2] for tc in top_cuts[:8]})
            print(
                f"  [W59L screen] n={n:4d}<-{don_set} jobs={len(jobs)} hits={len(hits)} best={b_str} ({time.time() - ts0:.1f}s)",
                flush=True,
            )
            for u in uniq_n[:2]:
                screened.append(u)
    finally:
        w51d.terminate_pool(ex_sc)

    screened.sort(key=lambda z: z[0])
    queued = screened[:6]
    print(
        f"[W59L s={args.shard}] Phase B (Rust packer hhop + hop): hunting {len(queued)}/{len(screened)} basins ({time.time() - t0:.1f}s)!",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (_, gap0, n, tag, S_init, adist, njam, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 65.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            best_Q, best_S = Q_init.copy(), float(S_init)

            if best_S < L0_cur + 0.0005:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    n, best_Q, deadline, slp_sec=36.0, fq_sec=22
                )
                if Sp0 < best_S:
                    best_Q, best_S = Qp0, Sp0

            rem = deadline - time.time()
            per_slot = min(
                175.0, max(45.0, (rem - 35.0) / max(1, min(3, len(queued) - i)))
            )
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
            futs_r = [ex_hop.submit(run_rust_worker_task, rt) for rt in rust_tasks]
            best_mode = "init"
            try:
                for fr in as_completed(futs_r, timeout=per_slot + 35.0):
                    rr = fr.result()
                    if rr is not None and rr[0] < best_S - 1e-10:
                        best_S, best_Q, best_mode = rr[0], rr[1], rr[2]
            except TimeoutError:
                w51d.terminate_pool(ex_hop)
                ex_hop = ProcessPoolExecutor(max_workers=args.procs)

            # Deep KKT polish if competitive or improved
            if best_S < L0_cur + 0.0018 or best_S < S_init - 2e-4:
                best_Q, best_S = w51d.deep_contact_polish(
                    n, best_Q, deadline, slp_sec=45.0, fq_sec=26
                )

            # Sniper second pass if within +0.0006 of record!
            if best_S < L0_cur + 0.0006 and (deadline - time.time()) > 130.0:
                s2_slot = min(130.0, (deadline - time.time()) * 0.55)
                ts2 = min(best_S, L0_cur) - 3.0e-5
                rt2 = [
                    ("hhop", best_Q, ts2, bseed + 101, forbid_k, False, s2_slot),
                    ("hhop", best_Q, ts2, bseed + 102, forbid_k, True, s2_slot),
                    ("hop", best_Q, ts2, bseed + 103, forbid_k, False, s2_slot),
                    ("hop", best_Q, ts2, bseed + 104, forbid_k, True, s2_slot),
                ]
                futs_r2 = [ex_hop.submit(run_rust_worker_task, rt) for rt in rt2]
                try:
                    for fr2 in as_completed(futs_r2, timeout=s2_slot + 35.0):
                        rr2 = fr2.result()
                        if rr2 is not None and rr2[0] < best_S - 1e-10:
                            best_S, best_Q, best_mode = rr2[0], rr2[1], rr2[2] + "_r2"
                except TimeoutError:
                    w51d.terminate_pool(ex_hop)
                    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
                best_Q, best_S = w51d.deep_contact_polish(
                    n, best_Q, deadline, slp_sec=48.0, fq_sec=28
                )

            if best_S < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, best_Q)
                if ok and Lf < L0_cur - 1e-9:
                    ds = (Lf - L0_cur) / math.sqrt(n)
                    tot_ds += ds
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W59L RUST WIN!!] n={n:4d} [{tag}->{best_mode}] dA={adist:.2f} {cur[n][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f})!",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W59L done] n={n:4d} [{tag}->{best_mode}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={best_S - L0_cur:+.6f}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W59L s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
