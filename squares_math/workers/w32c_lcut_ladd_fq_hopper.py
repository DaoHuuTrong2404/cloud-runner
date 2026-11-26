#!/usr/bin/env python3
"""Wave 32C: Evan Daniel's Vectorized 7x7 Inexact L-Cut (lcut) & Approximate L-Add (ladd) -> Rust fq -> Fast Stock Hopper!

Replaces slow Python L-BFGS with vectorized 49-point overlap scoring + 100x faster Rust fq quench:
  - lcut (K+1 -> K): For donor m ~ n + 2K + 1 (e.g. 455->414, 499->456, 265->234, 405->367, 818->762),
    scores every (cx, cy) cross cut on a 0.25 grid in ~2s, keeps gentle poking boundary squares,
    resolves tiny seam overlaps via Rust fq (--loosen 1.03 -> 1.00), then unleashes FAST Stock Hopper!
  - ladd (K-1 -> K): Opens cleanest torn cross (cx, cy) of width w in [1.02, 1.16] in donor m ~ n - (2K-1)
    (e.g. 842->901, 901->962, 730->785, 938->1000), fills tilted/staggered squares along the cross,
    quenches with Rust fq (--loosen 1.04 -> 1.00), then unleashes FAST Stock Hopper!
"""

import argparse
import json
import math
import os
import random
import sys
import tempfile
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

import sq_engine as E
import lcut as LC
import ladd as LA
import slp as SLP
import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12
import w32a_fresh_basin_fast_hopper as w32a


def arr_rad_to_deg_tuples(arr):
    P = E.to_origin(np.asarray(arr, dtype=np.float64))
    return [
        (float(x), float(y), float(math.degrees(t) % 90.0)) for x, y, t in P
    ], float(E.side(P))


def deg_tuples_to_rad_arr(C):
    a = np.array(C, dtype=np.float64)
    a[:, 2] = np.radians(a[:, 2])
    return E.to_origin(a)


def fq_quench_seed(n, seed_arr, S_hint, seed, tmp, loosen="1.035"):
    """Fast 2-stage Rust fq quench for an inexact lcut/ladd seed."""
    if not os.path.exists(f2d.FQ_BIN):
        return None, 1e9
    r1 = f2d.run_fq(
        S_hint,
        seed_arr,
        tmp,
        [
            "--loosen",
            loosen,
            "--round-r0",
            "0.06",
            "--round-compress",
            "--round-press",
            "--round-steps",
            "8",
            "--round-press-outer",
            "12",
            "--seed",
            str(seed),
            "--flip-top",
            "44",
            "--pit",
            "340",
        ],
        timeout=45,
    )
    if r1 is None:
        return None, 1e9
    cur_s, cur_a = r1[0], r1[1]
    r2 = f2d.run_fq(
        cur_s,
        cur_a,
        tmp,
        ["--loosen", "1.0", "--no-alm", "--flip-top", "52", "--pit", "480"],
        timeout=45,
    )
    if r2 is not None and r2[0] < cur_s:
        cur_s, cur_a = r2[0], r2[1]
    if E.min_gap(cur_a) < -1e-6:
        return None, 1e9
    cur_a = w32a.safe_open(cur_a)
    return cur_a, float(E.side(cur_a))


def worker_lcut_ladd(job):
    n, L0, arr0, cut_donors, add_donors, seed, hop_sec = job
    rng = random.Random(seed)
    best_S = L0
    best_P = None
    best_tag = ""
    quenched = []  # (S_fq, tag, arr)

    with tempfile.TemporaryDirectory(prefix=f"w32c_{n}_") as tmp:
        t_start = time.time()
        # 1) Vectorized 7x7 Inexact L-Cut (m > n -> n)
        for m, m_arr in cut_donors:
            if time.time() - t_start > 280.0:
                break
            d = m - n
            if d <= 0:
                continue
            sq_deg, s_m = arr_rad_to_deg_tuples(m_arr)
            try:
                cuts = LC.best_cuts(sq_deg, s_m, d, step=0.25, m=8, sep=1.5)
            except Exception:
                continue
            for ci, (score, cx, cy, mxf) in enumerate(cuts[:6]):
                if time.time() - t_start > 280.0:
                    break
                C = LC.apply_cut(sq_deg, cx, cy, d)
                if len(C) != n:
                    continue
                seed_arr = deg_tuples_to_rad_arr(C)
                Q, Sq = fq_quench_seed(
                    n,
                    seed_arr,
                    max(s_m - 1.0, L0) * 1.005,
                    seed + m * 17 + ci * 131,
                    tmp,
                    loosen="1.03",
                )
                if Q is not None:
                    tag = f"lcut_{m}_c{ci}"
                    if Sq < best_S - 1e-8:
                        best_S, best_P, best_tag = Sq, Q, tag
                    if Sq < best_S + 0.009:
                        quenched.append((Sq, tag, Q))

        # 2) Approximate L-Add (m < n -> n)
        for m, m_arr in add_donors:
            if time.time() - t_start > 420.0:
                break
            d = n - m
            if d <= 0:
                continue
            sq_deg, s_m = arr_rad_to_deg_tuples(m_arr)
            try:
                cuts = LA.best_cuts(sq_deg, s_m, step=0.5, m=6, sep=1.5)
            except Exception:
                continue
            for ci, (tear_s, cx, cy) in enumerate(cuts[:4]):
                if time.time() - t_start > 420.0:
                    break
                w_use = rng.uniform(1.01, 1.12)
                S_est, C = LA.add_l(sq_deg, s_m, d, cx, cy, w_use, rng, tmax=24.0)
                if len(C) != n:
                    continue
                seed_arr = deg_tuples_to_rad_arr(C)
                Q, Sq = fq_quench_seed(
                    n,
                    seed_arr,
                    max(S_est, L0) * 1.006,
                    seed + 50000 + m * 19 + ci * 137,
                    tmp,
                    loosen="1.045",
                )
                if Q is not None:
                    tag = f"ladd_{m}_c{ci}"
                    if Sq < best_S - 1e-8:
                        best_S, best_P, best_tag = Sq, Q, tag
                    if Sq < best_S + 0.009:
                        quenched.append((Sq, tag, Q))

        # 3) Feed top-2 distinct competitive quenched basins into FAST Stock Hopper!
        quenched.sort(key=lambda z: z[0])
        picked = []
        for Sq, tag, Q in quenched:
            if all(abs(Sq - ps) > 5e-6 for ps, _, _ in picked):
                picked.append((Sq, tag, Q))
            if len(picked) >= 2:
                break
        for pi, (Sq, tag, Q) in enumerate(picked):
            Qh, Sh, _ = w32a.run_stock_hopper_chain(
                Q,
                seed + 88000 + pi * 3571,
                first_sec=hop_sec,
                cascade_sec=140,
                max_cascades=3,
            )
            if Sh < best_S - 1e-8:
                best_S, best_P, best_tag = Sh, Qh, f"{tag}_sh"

        # 4) Final SLP + fq polish + Decimal64 >= 1.05e-9 cert
        if best_P is not None and best_S < L0 - 1e-9:
            cur_a = w32a.safe_open(best_P)
            cur_s = float(E.side(cur_a))
            try:
                xs, ys, ts, Ls = SLP.polish(
                    cur_a[:, 0],
                    cur_a[:, 1],
                    cur_a[:, 2],
                    iters=100,
                    r0=2e-3,
                    rmin=1e-10,
                    tol_sep=-1e-13,
                    time_limit=22.0,
                )
                Qs = w32a.safe_open(np.column_stack([xs, ys, ts]))
                Ss = float(E.side(Qs))
                if Ss < cur_s - 1e-10 and E.min_gap(Qs) >= -1e-12:
                    cur_a, cur_s = Qs, Ss
            except Exception:
                pass
            r_fin = f2d.run_fq(
                cur_s,
                cur_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "56", "--pit", "550"],
                timeout=55,
            )
            if r_fin is not None and r_fin[0] < cur_s:
                cur_s, cur_a = r_fin[0], r_fin[1]
            try:
                cur_a = w12.lp_rot_polish(cur_a)
            except Exception:
                pass
            ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_a)
            if ok and Lf < L0 - 1e-9:
                return n, best_tag, (Lf, xf, yf, tf)
    return n, best_tag, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2700)
    ap.add_argument("--hop-sec", type=int, default=150)
    ap.add_argument("--max-targets", type=int, default=64)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    front_rows = json.load(open(args.frontier_json))
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    grid_breaks = {605, 762, 819, 877, 1000}
    scored = []
    for n in range(150, 1001):
        K = math.isqrt(n)
        if K * K == n:
            continue
        is_gb = n in grid_breaks and cur[n][0] >= K + 1 - 1e-6
        if not is_gb and cur[n][0] >= K + 1 - 1e-6:
            continue
        gap_ds = max(0.0, (cur[n][0] - front_L.get(n, cur[n][0])) / math.sqrt(n))
        if gap_ds > 1e-6 or is_gb:
            scored.append((gap_ds + (0.01 if is_gb else 0.0), n))
    scored.sort(reverse=True)
    picked = [n for _, n in scored[: args.max_targets]]
    my_slice = picked[args.shard :: args.num_shards]

    jobs = []
    for i, n in enumerate(my_slice):
        k_cur = int(math.floor(cur[n][0]))
        cut_donors = []
        for dm in (0, -1, 1, -2):
            m = n + 2 * k_cur + 1 + dm
            if m <= 1000:
                km = math.isqrt(m)
                if km * km != m and cur[m][0] < km + 1 - 1e-6:
                    if cur[m][0] - 1.0 < cur[n][0] + 0.008:
                        cut_donors.append((m, cur[m][1]))
        cut_donors = cut_donors[:2]
        add_donors = []
        for dm in (0, -1, 1):
            m = n - (2 * k_cur - 1) + dm
            if m >= 150:
                km = math.isqrt(m)
                if km * km != m and cur[m][0] < km + 1 - 1e-6:
                    if cur[m][0] + 1.0 < cur[n][0] + 0.012:
                        add_donors.append((m, cur[m][1]))
        add_donors = add_donors[:2]
        jobs.append(
            (
                n,
                cur[n][0],
                cur[n][1],
                cut_donors,
                add_donors,
                3230001 + args.shard * 10169 + i * 331,
                args.hop_sec,
            )
        )

    print(
        f"[W32C-LCutLAdd s={args.shard}/{args.num_shards}] Base={tot_score:.10f} "
        f"(loaded {n_files} win files), my_targets({len(my_slice)})={my_slice} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0
    done = 0

    def save_ckpt():
        payload = {}
        for k_n, (Lf, xf, yf, tf) in sorted(wins.items()):
            payload[f"L_{k_n}"] = np.array([Lf], dtype=np.float64)
            payload[f"x_{k_n}"] = np.asarray(xf, dtype=np.float64)
            payload[f"y_{k_n}"] = np.asarray(yf, dtype=np.float64)
            payload[f"t_{k_n}"] = np.asarray(tf, dtype=np.float64)
        np.savez_compressed(args.out_npz, **payload)

    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_lcut_ladd, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W32C ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                dL = Lf - cur[n][0]
                if dL < -1e-9:
                    dS = dL / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W32C WIN!!] n={n:4d} {tag:18s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W32C s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
