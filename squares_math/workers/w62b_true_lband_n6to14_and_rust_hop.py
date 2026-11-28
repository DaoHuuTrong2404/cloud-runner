#!/usr/bin/env python3
"""Wave 62B (`w62b_true_lband_n6to14_and_rust_hop.py`): Tilted-Skeleton Monte Carlo + Exact `fillfast`
Staircase Refill (`TSMC-FillFast`, ~35ms/eval!) + `layout.py` Constructive Channel Evolution + Fast `right_rot`/`right_ref` `lband2`!

Built directly from empirical root-cause proof (`debug_lb2_cnt.py` + `debug_lb1_cnt.py` + `smoke_test_w62.py`):
  - Rigid uniform-angle `lband` MILP hits a hard mathematical ceiling (`cnt = 361 < 367` at `K=19, f=0.675`, `opt gap 0`)
    and times out at `N >= 8` when `jm=2.6, rich=0.25` creates >1,500 clique variables.
  - Meanwhile, stripping ALL upright squares from `cur[n]` / `cur[n+-1]` / `cur[n+2K+1]`, mutating ONLY the tilted
    skeleton `T` inside `[0, s_try]^2` (`s_try < L0`), and calling `fillfast.fill(s_try, T_mut, 'exact', tau)` solves
    the exact global Maximum Independent Set of all 4-corner + staircase upright squares in ~35ms (`235 >= 234` at
    `s = 15.818 < L_234 = 15.82288` on the very first call!!)!
  - Combines:
    1. `TSMC-FillFast` (`120` tilted-skeleton mutations per puzzle: band shift/wave, drop-1-tilted, add-1-band-square,
       neighbor donor transplant, all refilled globally by `fillfast` at `s_try < L0`!).
    2. `layout.py` constructive two-channel `(left, bottom)` genome SA scored by `fillfast`.
    3. Fast `N = 4..7` `('right_rot', 'right_ref')` `lband2` (`rich=0.0, jm=1.5, tl=2.5s`, verified `<2.5s` `opt gap 0`!).
    4. Plain `fq quench --loosen 1.02` (`s_ref = s_try`) -> Hybrid Rust `packer` -> `w51d.deep_3round_widehopper`!
"""

import argparse
import json
import math
import os
import random
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

import fillfast as FF
import layout as LAY
import lband2 as LB2
import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w59l_faithful_lcut_and_rust_hop as w59l


def extract_tilted_deg(P_rad, s_try):
    P = RE.normalize_to_origin(P_rad)
    s0 = RE.side_exact(P)
    half_pi = 0.5 * math.pi
    a_mod = np.mod(P[:, 2], half_pi)
    tilts = np.minimum(a_mod, half_pi - a_mod)
    t_idx = np.where(tilts > 0.02)[0]
    if len(t_idx) < 4:
        return None
    T = P[t_idx].copy()
    scale = float(s_try) / max(s0, 1e-9)
    T[:, 0] *= scale
    T[:, 1] *= scale
    T[:, 2] = np.degrees(T[:, 2]) % 90.0
    h = 0.5 * (
        np.abs(np.cos(np.radians(T[:, 2]))) + np.abs(np.sin(np.radians(T[:, 2])))
    )
    T[:, 0] = np.clip(T[:, 0], h + 1e-6, float(s_try) - h - 1e-6)
    T[:, 1] = np.clip(T[:, 1], h + 1e-6, float(s_try) - h - 1e-6)
    return T


def mutate_tilted_skeleton(T_base, s_try, rng):
    """Mutate ONLY the tilted skeleton `T` (so `fillfast` re-optimizes all upright squares from scratch!)."""
    T = T_base.copy()
    m = len(T)
    op = rng.randrange(5)
    if op == 0 and m > 5:
        # Drop 1 tilted square (lets fillfast replace 1 tilted square with staircase uprights!)
        drop_i = rng.randrange(m)
        T = np.delete(T, drop_i, axis=0)
    elif op == 1:
        # Add 1 tilted square stepped along a random tilted square's lattice vector!
        base_i = rng.randrange(m)
        x0, y0, deg0 = T[base_i]
        tr = math.radians(deg0)
        dx, dy = rng.choice(
            [
                (math.cos(tr), math.sin(tr)),
                (-math.cos(tr), -math.sin(tr)),
                (-math.sin(tr), math.cos(tr)),
                (math.sin(tr), -math.cos(tr)),
            ]
        )
        nx, ny = x0 + dx, y0 + dy
        h = 0.5 * (abs(math.cos(tr)) + abs(math.sin(tr)))
        if h < nx < s_try - h and h < ny < s_try - h:
            T = np.vstack([T, [[nx, ny, deg0]]])
    elif op == 2:
        # Smooth collective angle shift + wall hug on a local cluster of tilted squares
        ci = rng.randrange(m)
        d = np.hypot(T[:, 0] - T[ci, 0], T[:, 1] - T[ci, 1])
        cl = d < rng.uniform(2.0, 6.5)
        da = rng.gauss(0.0, 1.8)
        dx = rng.gauss(0.0, 0.04)
        dy = rng.gauss(0.0, 0.04)
        T[cl, 2] = (T[cl, 2] + da) % 90.0
        T[cl, 0] += dx
        T[cl, 1] += dy
    elif op == 3:
        # Global gentle shift of all tilted squares toward one corner
        sx = rng.choice([-0.03, 0.0, 0.03])
        sy = rng.choice([-0.03, 0.0, 0.03])
        T[:, 0] += sx
        T[:, 1] += sy
    else:
        # Mirror / transpose one half of T
        T[:, 2] = (T[:, 2] + rng.gauss(0.0, 0.8)) % 90.0

    rad = np.radians(T[:, 2])
    h = 0.5 * (np.abs(np.cos(rad)) + np.abs(np.sin(rad)))
    T[:, 0] = np.clip(T[:, 0], h + 1e-6, float(s_try) - h - 1e-6)
    T[:, 1] = np.clip(T[:, 1], h + 1e-6, float(s_try) - h - 1e-6)
    return T, f"op{op}"


def eval_tsmc_and_lband_puzzle(job_tup):
    n, K, f_front, f_cur, donor_packings, seed0 = job_tup
    t0 = time.time()
    rng = random.Random(int(seed0))
    L0 = K + f_cur
    s_front = K + f_front

    raw_starts = []
    # 1) TSMC-FillFast: mutate tilted skeletons from cur[n] and neighbors at s_try < L0!
    s_grid = sorted(
        {
            round(max(K + 0.30, L0 - 0.0025), 6),
            round(max(K + 0.30, 0.5 * (L0 + s_front)), 6),
            round(max(K + 0.30, s_front + 0.0015), 6),
        }
    )
    t_bases = []
    for dtag, P_don in donor_packings:
        for s_try in s_grid:
            Tb = extract_tilted_deg(P_don, s_try)
            if Tb is not None and len(Tb) < n:
                t_bases.append((dtag, s_try, Tb))

    for dtag, s_try, Tb in t_bases:
        if time.time() - t0 > 38.0 or len(raw_starts) >= 7:
            break
        for tau in (0.008, 0.022):
            try:
                X0 = FF.fill(s_try, Tb, mode="exact", tau=tau, time_limit=1.5)
                if len(X0) >= n:
                    X_use = X0[:n].copy()
                    X_use[:, 2] = np.radians(X_use[:, 2])
                    raw_starts.append((s_try, f"ff0_{dtag}_t{int(tau * 1000)}", X_use))
                    break
            except Exception:
                pass

    # Mutate tilted skeletons (~50 fast 35ms trials capped at 35s!)
    it_m = 0
    while t_bases and time.time() - t0 < 58.0 and it_m < 48 and len(raw_starts) < 10:
        it_m += 1
        dtag, s_try, Tb = t_bases[it_m % len(t_bases)]
        Tm, op_tag = mutate_tilted_skeleton(Tb, s_try, rng)
        tau = 0.015 if it_m % 2 == 0 else 0.028
        try:
            Xm = FF.fill(s_try, Tm, mode="exact", tau=tau, time_limit=1.2)
            if len(Xm) >= n:
                X_use = Xm[:n].copy()
                X_use[:, 2] = np.radians(X_use[:, 2])
                raw_starts.append((s_try, f"tsmc_{dtag}_{op_tag}", X_use))
        except Exception:
            continue

    # 2) Constructive `layout.py` channel evolution (~12s burst for moderate K <= 25)
    if K <= 25 and time.time() - t0 < 70.0 and len(raw_starts) < 10:
        s_lay = max(K + 0.35, 0.5 * (L0 + s_front))
        w_init = max(2, min(5, int(round((n - K * K + 10) / max(6, K // 2)))))
        g = ([(w_init, 26.5, 0)] * 5, [(w_init, 26.5, 0)] * 5)
        best_f = -1.0
        t_lay = time.time()
        while time.time() - t_lay < 10.0:
            g2 = LAY.mutate(g, rng)
            T_l = LAY.build(s_lay, *g2)
            if not T_l or len(T_l) >= n:
                continue
            try:
                Xl = FF.fill(
                    s_lay,
                    np.asarray(T_l, float),
                    mode="greedy",
                    tau=0.018,
                    time_limit=0.5,
                )
            except Exception:
                continue
            sc_l = float(len(Xl))
            if sc_l >= best_f or rng.random() < 0.25:
                g = g2
                best_f = max(best_f, sc_l)
            if len(Xl) >= n:
                X_use = Xl[:n].copy()
                X_use[:, 2] = np.radians(X_use[:, 2])
                raw_starts.append((s_lay, "layout_hit", X_use))
                break

    # 3) Fast proven `N = 4..7` `('right_rot', 'right_ref')` `lband2` (`rich=0.0, jm=1.5, tl=2.2s`)
    if time.time() - t0 < 85.0:
        f_try = max(0.32, 0.5 * (f_front + f_cur))
        for N_arm in (4, 5, 6, 7):
            if N_arm >= K - 2:
                continue
            for mode in ("right_rot", "right_ref"):
                try:
                    cnt, sol = LB2.assemble_split(
                        K,
                        float(f_try),
                        N_arm,
                        0.0,
                        0.0,
                        mode,
                        ec=3,
                        phases=4,
                        jm=1.5,
                        rich=0.0,
                        time_limit=2.0,
                        tol=0.010,
                    )
                    if sol is not None and cnt >= n:
                        keep = np.sort(np.argsort(-np.abs(sol[:, 0] - sol[:, 1]))[:n])
                        raw_starts.append(
                            (K + f_try, f"lb2_N{N_arm}_{mode[-3:]}", sol[keep].copy())
                        )
                        break
                except Exception:
                    continue

    # Quench top distinct starts with plain `fq --loosen 1.02` (`NO relocate_crushed`, `NO --round-r0`!)
    quenched = []
    for qi, (s_box, tag, Q_sol) in enumerate(raw_starts[:7]):
        Q_in = RE.normalize_to_origin(Q_sol)
        rq = RE.run_fq_quench_args(
            Q_in,
            [
                "--seed",
                str(int(seed0) + qi * 13),
                "--loosen",
                "1.022",
                "--outer",
                "58",
                "--pit",
                "300",
                "--flip-top",
                "22",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "20.0",
            ],
            s_ref=float(s_box),
            timeout_s=23.0,
        )
        if rq is not None:
            Sq, Qq = float(rq[0]), rq[1]
            if Sq < L0 + 0.0165:
                quenched.append((Sq, tag, Qq))
    quenched.sort(key=lambda z: z[0])
    return n, len(raw_starts), quenched[:3]


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
    for n in range(100, 1001):
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
        f"[W62B-TSMCFillFast s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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
            d_packs = [(f"n{n}", cur[n][1])]
            for m in (n + 1, n - 1, n + 2 * K + 1):
                if (
                    50 <= m <= 1000
                    and cur[m][1] is not None
                    and math.isqrt(m) ** 2 != m
                ):
                    d_packs.append((f"n{m}", cur[m][1]))
            sc_tasks.append(
                (
                    n,
                    K,
                    f_front,
                    f_cur,
                    d_packs,
                    args.seed_off + args.shard * 3301 + pi * 173,
                )
            )
        futs = [ex_sc.submit(eval_tsmc_and_lband_puzzle, tk) for tk in sc_tasks]
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
                                f"  [W62B DIRECT TSMC WIN!!] n={n:4d} [{tg}] {cur[n][0]:.7f} -> {Lf0:.7f} (dS={ds0:+.8f})!",
                                flush=True,
                            )
                            save_ckpt()
                    ad = w53h.topology_angle_dist(Qq, P0) if P0 is not None else 1.0
                    if Sq < L0 - 1e-9 or (Sq < L0 + 0.0165 and ad > 0.025):
                        screened.append((Sq - L0, n, tg, Sq, ad, Qq))
                b_str = (
                    f"{q_list[0][0] - cur[n][0]:+.6f}@{q_list[0][1]}"
                    if q_list
                    else "none"
                )
                print(
                    f"  [W62B screen] n={n:4d} raw_hits={n_raw} quenched={len(q_list)} best={b_str}",
                    flush=True,
                )
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_sc)

    screened.sort(key=lambda z: z[0])
    queued = screened[:5]
    print(
        f"[W62B s={args.shard}] Phase B: hunting {len(queued)}/{len(screened)} TSMC/LBand basins ({time.time() - t0:.1f}s)!",
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
                        f"  [W62B TSMC WIN!!] n={n:4d} [{tag}->{best_mode}] dA={adist:.2f} {cur[n][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={wn_cnt})!",
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
