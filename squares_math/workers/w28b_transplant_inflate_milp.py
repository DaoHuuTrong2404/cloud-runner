#!/usr/bin/env python3
"""Wave 28B: Cross-K Band Transplant + Diagonal/Grid Inflation + HiGHS 0-1 MILP Refill & GRID_BREAK Assault.

Combines three orthogonal structural operators from Ross Wade's sq_engine.py + sq_research.py (CC BY 4.0)
never previously executed on our 22.086 R20 packings:
  Part 1 (Cross-K Structural Growth/Shrink across ALL 560+ non-grid donors):
    - sq_engine.transplant(P[m], up=True/False) (exact bar+row cut & insert/remove across tilted bands)
    - sq_research.best_inflation2(P[m], k=1,2) + Compactor
    - sq_research.best_diag(P[m], k=1,2) + Compactor
    Any resulting packing n_new <= 1000 beating or within 0.006 of cur[n_new] gets Hopper(110s) + fq polish!
  Part 2 (HiGHS 0-1 Exact Independent-Set MILP Multi-Lattice Window Refill):
    - Attacks ALL 5 GRID_BREAK targets (605, 762, 819, 877, 1000) + Top-80 tight n-1 -> n insertions!
    - Extracts empirical multi-tilt lattice vectors in windows around high-empty-area/band-end pockets,
      packs +1 extra square via HiGHS 0-1 MILP, then compacts & hops into a new lower basin!
"""

import argparse
import json
import math
import os
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
import sq_research as R
import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12


def warm_numba():
    P = E.to_origin(
        np.array(
            [
                [0.5, 0.5, 0.0],
                [1.5, 0.5, 0.0],
                [0.5, 1.5, 0.0],
                [1.5, 1.5, 0.0],
                [2.5, 0.5, 0.15],
            ],
            dtype=np.float64,
        )
    )
    E.min_gap(P)
    E.side(P)
    E.open_contacts(P)
    E.empty_area(P, 3.0, 4)
    R.clash_pairs(P, E.neighbour_pairs(P, 1.45), np.full(5, -1, np.int64), 0.0)
    H = E.Hopper(P, seed=0)
    if H.applicable():
        H.run(1)


def polish_and_certify(n, arr, L_ref, hopper_sec=100, seed=0):
    cur_P = E.to_origin(arr)
    try:
        Q, S, ov = E.Compactor(cur_P).compact(cur_P, max_iter=120, patience=3)
        if ov < 2e-9 and S < E.side(cur_P) + 1e-6:
            cur_P = E.to_origin(Q)
    except Exception:
        pass

    cur_S = float(E.side(cur_P))
    if cur_S > L_ref + 0.008:
        return None

    # Run Force-Guided Hopper on promising new topology!
    if hopper_sec > 0:
        try:
            H = E.Hopper(cur_P, seed=seed)
            if H.applicable():
                Q2, S2 = H.run(hopper_sec)
                Q2 = E.to_origin(E.open_contacts(Q2))
                if E.side(Q2) < cur_S and E.min_gap(Q2) >= -1e-12:
                    cur_P = Q2
                    cur_S = float(E.side(Q2))
        except Exception:
            pass

    if cur_S >= L_ref - 1e-9:
        return None

    with tempfile.TemporaryDirectory(prefix=f"w28b_{n}_") as tmp:
        cur_a = E.to_origin(E.open_contacts(cur_P))
        cur_s = float(E.side(cur_a))
        if os.path.exists(f2d.FQ_BIN):
            r_fin = f2d.run_fq(
                cur_s,
                cur_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "48", "--pit", "450"],
                timeout=55,
            )
            if r_fin is not None and r_fin[0] < cur_s:
                cur_s, cur_a = r_fin[0], r_fin[1]
        try:
            cur_a = w12.lp_rot_polish(cur_a)
        except Exception:
            pass
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_a)
        if ok and Lf < L_ref - 1e-9:
            return (Lf, xf, yf, tf)
    return None


def donor_worker(job):
    m, arr_m, cur_L_map, seed = job
    rng = np.random.default_rng(seed)
    cands = []  # (tag, Q)

    # 1. Exact Band Transplant (up & down)
    for up in (True, False):
        try:
            out = E.transplant(arr_m, up)
            if out is not None:
                Q, S, _, _ = out
                cands.append((f"tr{'U' if up else 'D'}_{m}", Q))
        except Exception:
            pass

    # 2. Grid-Cut Inflation (k=1, 2)
    for k in (1, 2):
        try:
            Q, S2, _ = R.best_inflation2(arr_m, k, tol=-1e-12, rng=rng, n_random=90)
            if Q is not None and len(Q) <= 1003 and E.min_gap(Q) >= -1e-9:
                cands.append((f"inf{k}_{m}", Q))
        except Exception:
            pass

    # 3. Diagonal Band-Aware Inflation (k=1, 2)
    for k in (1, 2):
        try:
            Q, S2, _ = R.best_diag(arr_m, k, tol=-1e-12, steps=36)
            if Q is not None and len(Q) <= 1003 and E.min_gap(Q) >= -1e-9:
                cands.append((f"diag{k}_{m}", Q))
        except Exception:
            pass

    found = []
    for tag, Q in cands:
        Q = E.to_origin(Q)
        # Try exact size and trim by 1..2 if needed
        for drop in (0, 1, 2):
            n_new = len(Q) - drop
            if not (150 <= n_new <= 1000):
                continue
            K_new = math.isqrt(n_new)
            if K_new * K_new == n_new:
                continue
            L_ref = cur_L_map.get(n_new, float(K_new + 1))
            Qsub = Q if drop == 0 else Q[:n_new].copy()
            S_raw = float(E.side(Qsub))
            if S_raw > L_ref + 0.015:
                continue
            # Quick compact first
            try:
                Qc, Sc, ov = E.Compactor(Qsub).compact(Qsub, max_iter=80, patience=2)
                if ov < 2e-9 and Sc < S_raw:
                    Qsub, S_raw = E.to_origin(Qc), float(Sc)
            except Exception:
                pass
            if S_raw < L_ref + 0.005:
                res = polish_and_certify(
                    n_new,
                    Qsub,
                    L_ref,
                    hopper_sec=110 if S_raw < L_ref + 0.003 else 60,
                    seed=seed + n_new * 17 + drop,
                )
                if res is not None:
                    found.append((n_new, f"{tag}_d{drop}", res))
    return m, found


def milp_refill_worker(job):
    n, L_ref, arr_prev, L_prev, is_grid, seed = job
    K = math.isqrt(n)
    rng = np.random.default_rng(seed)
    P0 = E.to_origin(arr_prev)

    # Target box side S_try strictly below L_ref (for GRID_BREAK: strictly below K+1!)
    if is_grid:
        S_list = [min(K + 0.996, L_prev + 0.018), min(K + 0.9985, L_prev + 0.030)]
    else:
        if L_prev >= L_ref - 1e-6:
            return n, None, ""
        S_list = [min(L_ref - 0.0005, L_prev + 0.015)]

    # Locate top empty-area pockets and tilted band endpoints
    try:
        V = E.empty_area(P0, L_prev, 8)
        flat = np.argsort(-V.ravel())[:12]
        ny = V.shape[1]
        pockets = [(int(idx // ny) + 0.5, int(idx % ny) + 0.5) for idx in flat]
    except Exception:
        pockets = [(L_prev * 0.5, L_prev * 0.5)]

    tilts = np.abs(E.fold(P0[:, 2]))
    til = np.where(tilts > 0.02)[0]
    if len(til) > 0:
        for q in rng.choice(til, size=min(8, len(til)), replace=False):
            pockets.append((float(P0[q, 0]), float(P0[q, 1])))

    best_res = None
    best_tag = ""
    best_L = L_ref

    for S_try in S_list:
        for pi, (cx, cy) in enumerate(pockets[:16]):
            w = float(rng.choice([5.5, 7.0, 8.5]))
            h = float(rng.choice([5.5, 7.0, 8.5]))
            x0 = max(0.0, min(S_try - w, cx - 0.5 * w))
            y0 = max(0.0, min(S_try - h, cy - 0.5 * h))
            try:
                C, inwin = R.candidates(P0, S_try, x0, y0, w, h, gens=5, margin=3.0)
                n_in = int(inwin.sum())
                if len(C) <= n_in or len(C) > 360:
                    continue
                sel, val, st = R.max_set(C, time_limit=8.0)
                if val >= n_in + 1:
                    Q = np.vstack([P0[~inwin], C[sel]])
                    if len(Q) > n:
                        Q = Q[:n]
                    if len(Q) == n and E.min_gap(Q) >= -1e-9:
                        res = polish_and_certify(
                            n, Q, best_L, hopper_sec=130, seed=seed + pi * 97
                        )
                        if res is not None and res[0] < best_L - 1e-9:
                            best_res = res
                            best_L = res[0]
                            best_tag = f"milp_{int(w)}x{int(h)}_p{pi}"
            except Exception:
                continue
    return n, best_res, best_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2280)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    warm_numba()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    front_rows = json.load(open(args.frontier_json))
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}
    cur_L_map = {n: cur[n][0] for n in range(1, 1001)}

    # Part 1 donors: ALL non-grid packings n >= 150
    donors = [
        n
        for n in range(150, 999)
        if math.isqrt(n) ** 2 != n and cur[n][0] < math.isqrt(n) + 1 - 1e-6
    ]
    my_donors = donors[args.shard :: args.num_shards]

    # Part 2 MILP targets: ALL 5 GRID_BREAK + Top-80 tight n-1 -> n targets
    grid_breaks = []
    tight_ins = []
    for n in range(151, 1001):
        K = math.isqrt(n)
        if K * K == n:
            continue
        Ln = cur[n][0]
        lf = front_L.get(n, Ln)
        if Ln >= K + 1 - 1e-5 and lf < K + 1 - 1e-5:
            grid_breaks.append((n, True))
        elif Ln < K + 1 - 1e-6 and (Ln - lf) > 1e-5:
            Lp = cur[n - 1][0]
            if Lp < Ln - 1e-4:
                tight_ins.append(((Ln - lf) / math.sqrt(n), n))
    tight_ins.sort(reverse=True)
    milp_all = grid_breaks + [(n, False) for _, n in tight_ins[:80]]
    # Ensure every GRID_BREAK target gets tried on at least 1 shard (distribute cleanly)
    my_milp = milp_all[args.shard :: args.num_shards]

    print(
        f"[W28B-TransplantInflateMILP s={args.shard}/{args.num_shards}] Base={tot_score:.10f}, "
        f"my_donors={len(my_donors)}, my_milp={len(my_milp)} (grid_breaks={[n for n, g in my_milp if g]}) ({time.time() - t0:.1f}s)",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0

    def save_ckpt():
        payload = {}
        for k_n, (Lf, xf, yf, tf) in sorted(wins.items()):
            payload[f"L_{k_n}"] = np.array([Lf], dtype=np.float64)
            payload[f"x_{k_n}"] = np.asarray(xf, dtype=np.float64)
            payload[f"y_{k_n}"] = np.asarray(yf, dtype=np.float64)
            payload[f"t_{k_n}"] = np.asarray(tf, dtype=np.float64)
        np.savez_compressed(args.out_npz, **payload)

    def record_win(n, tag, res):
        nonlocal tot_ds
        Lf, xf, yf, tf = res
        ref = wins[n][0] if n in wins else cur[n][0]
        if Lf < ref - 1e-9:
            dS = (Lf - ref) / math.sqrt(n)
            tot_ds += dS
            wins[n] = res
            print(
                f"  [W28B WIN!!] n={n:4d} {tag:18s} {ref:.7f} -> {Lf:.7f} "
                f"dL={Lf - ref:.7f} dS={dS:.8f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
            save_ckpt()

    # Run Part 2 FIRST (high-ROI GRID_BREAK & MILP refill, ~10m)
    milp_jobs = [
        (
            n,
            cur[n][0],
            cur[n - 1][1],
            cur[n - 1][0],
            is_g,
            2820001 + args.shard * 10009 + i * 313,
        )
        for i, (n, is_g) in enumerate(my_milp)
    ]
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(milp_refill_worker, j) for j in milp_jobs]
        for fut in as_completed(futs):
            try:
                n, res, tag = fut.result()
                if res is not None:
                    record_win(n, tag, res)
            except Exception as exc:
                print(f"  [W28B MILP ERR] {exc}", flush=True)
            if time.time() - t0 > args.budget_sec * 0.40:
                for f_rem in futs:
                    f_rem.cancel()
                break

    # Run Part 1 (Cross-K Transplant + Grid/Diag Inflation across all donors!)
    if time.time() - t0 < args.budget_sec - 60:
        d_jobs = [
            (m, cur[m][1], cur_L_map, 2825001 + args.shard * 10037 + i * 199)
            for i, m in enumerate(my_donors)
        ]
        with ProcessPoolExecutor(max_workers=args.procs) as ex:
            futs = [ex.submit(donor_worker, j) for j in d_jobs]
            for fut in as_completed(futs):
                try:
                    m, found = fut.result()
                    for n_new, tag, res in found:
                        record_win(n_new, tag, res)
                except Exception as exc:
                    print(f"  [W28B DONOR ERR] {exc}", flush=True)
                if time.time() - t0 > args.budget_sec:
                    print(
                        f"[W28B s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                        flush=True,
                    )
                    for f_rem in futs:
                        f_rem.cancel()
                    break

    save_ckpt()
    print(
        f"[W28B s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
