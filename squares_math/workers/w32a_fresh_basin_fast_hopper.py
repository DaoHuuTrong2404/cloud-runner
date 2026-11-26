#!/usr/bin/env python3
"""Wave 32A: FAST Stock sq_engine.Hopper + Fast n+1 Removal Compaction on ALL Fresh R23 Basins & Neighbors!

Empirical lessons from Waves 28-31:
  - Stock sq_engine.Hopper (fast small-LP core) beat 320-single HopperV2 by >10x eval throughput!
  - Freshly discovered/published basins (jazivxt Nov26 n=809/603/927, pub_v6 n=816/995/932/864,
    W28-W30 wins n=692/694/794/793/683/545/499/502/...) AND their n-1/n+1 neighbors are the
    #1 highest-yield targets for deep multi-seed Stock Hopper + n+1 removal screening!
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
import slp as SLP
import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12


def safe_open(P):
    try:
        return E.to_origin(E.open_contacts(P))
    except Exception:
        return E.to_origin(P)


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
    safe_open(P)
    H = E.Hopper(P, seed=0)
    if H.applicable():
        H.run(1)
    SLP.polish(P[:, 0], P[:, 1], P[:, 2], iters=2, r0=1e-3, time_limit=3.0)


def run_stock_hopper_chain(
    P_start, seed_base, first_sec=220, cascade_sec=160, max_cascades=3
):
    best_P = E.to_origin(P_start)
    best_S = float(E.side(best_P))
    start_S = best_S
    try:
        H = E.Hopper(best_P, seed=seed_base)
        if not H.applicable():
            return best_P, best_S, 0
        Q, S = H.run(first_sec)
        Q = safe_open(Q)
        S = float(E.side(Q))
        if S < best_S - 1e-10 and E.min_gap(Q) >= -1e-12:
            best_P, best_S = Q, S
    except Exception:
        return best_P, best_S, 0

    wins = 1 if best_S < start_S - 1e-9 else 0
    if wins:
        for c in range(max_cascades):
            prev_S = best_S
            # Quick SLP polish between cascades to unstick contact locks!
            try:
                xs, ys, ts, Ls = SLP.polish(
                    best_P[:, 0],
                    best_P[:, 1],
                    best_P[:, 2],
                    iters=60,
                    r0=3e-3,
                    rmin=1e-10,
                    tol_sep=-1e-13,
                    time_limit=15.0,
                )
                Qs = safe_open(np.column_stack([xs, ys, ts]))
                Ss = float(E.side(Qs))
                if Ss < best_S - 1e-10 and E.min_gap(Qs) >= -1e-12:
                    best_P, best_S = Qs, Ss
            except Exception:
                pass
            try:
                H2 = E.Hopper(best_P, seed=seed_base + (c + 1) * 1009)
                if not H2.applicable():
                    break
                Q2, S2 = H2.run(cascade_sec)
                Q2 = safe_open(Q2)
                S2 = float(E.side(Q2))
                if S2 < best_S - 1e-10 and E.min_gap(Q2) >= -1e-12:
                    best_P, best_S = Q2, S2
                    wins += 1
                if prev_S - best_S < 5e-7:
                    break
            except Exception:
                break
    return best_P, best_S, wins


def fast_delete_and_compact(P1, idx):
    """Fast stock Compactor removal screening (10x faster than 320-single HopperV2!)."""
    P = E.to_origin(np.delete(P1, idx, axis=0))
    body, _, _, size = E.find_bodies(P, 1e-3)
    core = size[body] <= 2
    free_ = core.copy()
    for u, v in E.neighbour_pairs(np.ascontiguousarray(P), 1.45):
        if core[u] and size[body[v]] <= 12:
            free_[v] = True
        if core[v] and size[body[u]] <= 12:
            free_[u] = True
    loose = np.where(free_)[0]
    Q, S, ov = E.Compactor(P, 1e-3, singles=loose).compact(P, max_iter=45, patience=2)
    if ov > 1e-7:
        return None, 1e9
    return Q, float(S)


def pick_near_band(P1, max_cand=180):
    P1 = E.to_origin(P1)
    n1 = len(P1)
    tilts = np.abs(E.fold(P1[:, 2]))
    tilted = np.where(tilts > 0.015)[0]
    near = set(int(i) for i in tilted)
    pairs = E.neighbour_pairs(np.ascontiguousarray(P1), 1.70)
    if len(tilted) > 0:
        for u, v in pairs:
            if u in near:
                near.add(int(v))
            if v in near:
                near.add(int(u))
    scores = np.zeros(n1, dtype=np.float64)
    try:
        H = E.Hopper(P1, seed=0)
        if H.L.ok:
            loads = np.maximum(
                np.nan_to_num(H.L.loads(), nan=0.0, posinf=0.0, neginf=0.0), 0.0
            )
            scores += loads * 12.0
    except Exception:
        pass
    for i in near:
        scores[i] += 0.30
    for i in tilted:
        scores[i] += 0.45
    return [int(i) for i in np.argsort(-scores)[: min(n1, max_cand)]]


def worker_fresh_hopper(job):
    n, L0, arr0, arr_p1, L_p1, seed, hop_sec, del_sec, n_seeds = job
    best_S = L0
    best_P = None
    best_tag = ""

    # Phase 1: n_seeds diverse seeds of FAST Stock sq_engine.Hopper + SLP cascade
    for si in range(n_seeds):
        P_src = best_P if best_P is not None else arr0
        Q, S, _ = run_stock_hopper_chain(
            P_src,
            seed + si * 7919,
            first_sec=hop_sec,
            cascade_sec=min(140, hop_sec),
            max_cascades=3,
        )
        if S < best_S - 1e-8:
            best_S, best_P, best_tag = S, Q, f"sh_s{si}"

    # Phase 2: Fast ALL-near-band n+1 removal screening (up to 180 squares!) -> Stock Hopper!
    if arr_p1 is not None and (L_p1 - best_S) < 0.035:
        cands = pick_near_band(arr_p1, max_cand=180)
        holes = []
        t_scr = time.time()
        for idx in cands:
            if time.time() - t_scr > 75.0:
                break
            try:
                Qh, Sh = fast_delete_and_compact(arr_p1, idx)
                if Qh is not None and Sh < best_S + 0.012:
                    holes.append((Sh, idx, Qh))
            except Exception:
                continue
        holes.sort(key=lambda z: z[0])
        picked = []
        for Sh, idx, Qh in holes:
            if all(abs(Sh - ps) > 2e-6 for ps, _, _ in picked):
                picked.append((Sh, idx, Qh))
            if len(picked) >= 3:
                break
        for hi, (Sh, idx, Qh) in enumerate(picked):
            if Sh < best_S - 1e-8:
                best_S, best_P, best_tag = Sh, Qh, f"fdel_{idx}"
            Q2, S2, _ = run_stock_hopper_chain(
                Qh,
                seed + 61000 + hi * 3571,
                first_sec=del_sec,
                cascade_sec=150,
                max_cascades=3,
            )
            if S2 < best_S - 1e-8:
                best_S, best_P, best_tag = S2, Q2, f"fdel_sh_{idx}"
                # Extra 2 seeds on a brand-new removal jump (goodpjw n=654 magic!)
                for r2 in range(2):
                    Q3, S3, _ = run_stock_hopper_chain(
                        best_P,
                        seed + 91000 + hi * 997 + r2 * 2111,
                        first_sec=180,
                        cascade_sec=140,
                        max_cascades=2,
                    )
                    if S3 < best_S - 1e-8:
                        best_S, best_P, best_tag = S3, Q3, f"fdel_sh2_{idx}"

    # Phase 3: Final SLP + fq polish + exact Decimal64 >= 1.05e-9 certification
    if best_P is not None and best_S < L0 - 1e-9:
        cur_a = safe_open(best_P)
        cur_s = float(E.side(cur_a))
        try:
            xs, ys, ts, Ls = SLP.polish(
                cur_a[:, 0],
                cur_a[:, 1],
                cur_a[:, 2],
                iters=120,
                r0=2e-3,
                rmin=1e-10,
                tol_sep=-1e-13,
                time_limit=25.0,
            )
            Qs = safe_open(np.column_stack([xs, ys, ts]))
            Ss = float(E.side(Qs))
            if Ss < cur_s - 1e-10 and E.min_gap(Qs) >= -1e-12:
                cur_a, cur_s = Qs, Ss
        except Exception:
            pass
        if os.path.exists(f2d.FQ_BIN):
            with tempfile.TemporaryDirectory(prefix=f"w32a_{n}_") as tmp:
                r_fin = f2d.run_fq(
                    cur_s,
                    cur_a,
                    tmp,
                    ["--loosen", "1.0", "--no-alm", "--flip-top", "56", "--pit", "550"],
                    timeout=60,
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
    ap.add_argument("--hop-sec", type=int, default=200)
    ap.add_argument("--del-sec", type=int, default=180)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-off", type=int, default=0)
    ap.add_argument("--max-targets", type=int, default=96)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    warm_numba()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    front_rows = json.load(open(args.frontier_json))
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    # VIP fresh basins (W35 jumps 694/809/807/452/785 + W34 496/988 + W33 502/817/815/451 + neighbors!)
    vip_seeds = [
        694,
        693,
        809,
        808,
        807,
        806,
        496,
        495,
        452,
        451,
        450,
        502,
        501,
        785,
        784,
        988,
        987,
        695,
        503,
        817,
        816,
        815,
        814,
        654,
        653,
        655,
        603,
        602,
        808,
        603,
        602,
        927,
        926,
        692,
        691,
        694,
        693,
        794,
        793,
        792,
        683,
        682,
        816,
        815,
        995,
        994,
        545,
        544,
        499,
        498,
        502,
        501,
        864,
        863,
        932,
        931,
        654,
        653,
        992,
        991,
        451,
        450,
        904,
        903,
        818,
        817,
        915,
        914,
        965,
        964,
        862,
        861,
        460,
        459,
        370,
        369,
        456,
        455,
        414,
        413,
        804,
        803,
        748,
        747,
        976,
        975,
        500,
        923,
        922,
        638,
        637,
        911,
        910,
        540,
        539,
        981,
        980,
        698,
        697,
        859,
        858,
    ]
    targets = []
    seen = set()
    for n in vip_seeds:
        if n in seen or n < 150 or n >= 1000:
            continue
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        seen.add(n)
        targets.append(n)
    # Pad up to 64 targets with top remaining two-band (f in 0.58..0.97) gaps
    extra = []
    for n in range(400, 1000):
        if n in seen:
            continue
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        f = cur[n][0] - math.floor(cur[n][0])
        if 0.58 <= f <= 0.97:
            gap = max(0.0, (cur[n][0] - front_L.get(n, cur[n][0])) / math.sqrt(n))
            if gap > 1e-6:
                extra.append((gap, n))
    extra.sort(reverse=True)
    for _, n in extra:
        if len(targets) >= args.max_targets:
            break
        targets.append(n)

    my_slice = targets[args.shard :: args.num_shards]
    jobs = []
    for i, n in enumerate(my_slice):
        arr_p1, L_p1 = None, 1e9
        if n + 1 <= 1000:
            k1 = math.isqrt(n + 1)
            if k1 * k1 != (n + 1) and cur[n + 1][0] < k1 + 1 - 1e-6:
                arr_p1, L_p1 = cur[n + 1][1], cur[n + 1][0]
        jobs.append(
            (
                n,
                cur[n][0],
                cur[n][1],
                arr_p1,
                L_p1,
                3210001 + args.seed_off + args.shard * 10169 + i * 331,
                args.hop_sec,
                args.del_sec,
                args.n_seeds,
            )
        )

    print(
        f"[W32A-FreshHop s={args.shard}/{args.num_shards}] Base={tot_score:.10f} "
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
        futs = [ex.submit(worker_fresh_hopper, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W32A ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                dL = Lf - cur[n][0]
                if dL < -1e-9:
                    dS = dL / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W32A WIN!!] n={n:4d} {tag:16s} {cur[n][0]:.7f} -> {Lf:.7f} "
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
        f"[W32A s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
