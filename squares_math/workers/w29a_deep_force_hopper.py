#!/usr/bin/env python3
"""Wave 29A: Ultra-Deep Multi-Seed Force-Guided LP Hopper + Exhaustive n+1 Removal Screening + Micro-Kick Escape.

Wave 28A proved Force-Guided LP Hopper (sq_engine.py, CC BY 4.0) is our #1 breakout engine on 22.086
(n=545 -0.00359, n=794 -0.00196, 12 wins with only 110s x 2 seeds!).
Wave 29A scales 4x deeper across 20 shards (80 free cores, only 8 puzzles/shard -> ~18 core-min/puzzle!):
  1. 4 diverse direct Force-Guided LP Hopper seeds x 210s each (cascading +2x180s on any jump!)
  2. Exhaustive n+1 removal-compaction screening (up to 64 highest-load/near-band removals)
     -> Top-3 deepest compacted holes -> 180s Hopper each (cascading +2x150s on any jump!)
  3. Micro-loosen kick (fq --loosen 1.012 --round-r0 0.04) -> Hopper(150s) to jump out of jammed basins!
  4. Strict fq polish + Contact LP rotation polish + Decimal64 >= 1.05e-9 certification.
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


def delete_and_compact(P1, idx):
    P = E.to_origin(np.delete(P1, idx, axis=0))
    body, kind, start, size = E.find_bodies(P, 1e-3)
    core = size[body] <= 2
    free_ = core.copy()
    for u, v in E.neighbour_pairs(np.ascontiguousarray(P), 1.45):
        if core[u] and size[body[v]] <= 12:
            free_[v] = True
        if core[v] and size[body[u]] <= 12:
            free_[u] = True
    Q, S, ov = E.Compactor(P, 1e-3, singles=np.where(free_)[0]).compact(
        P, max_iter=65, patience=2
    )
    if ov > 1e-7:
        return None, 1e9
    return Q, float(S)


def pick_removal_candidates(P1, max_cand=64):
    P1 = E.to_origin(P1)
    n1 = len(P1)
    tilts = np.abs(E.fold(P1[:, 2]))
    tilted = np.where(tilts > 0.02)[0]
    near_band = set(int(i) for i in tilted)
    if len(tilted) > 0:
        for u, v in E.neighbour_pairs(np.ascontiguousarray(P1), 1.75):
            if u in near_band:
                near_band.add(int(v))
            if v in near_band:
                near_band.add(int(u))
    scores = np.zeros(n1, dtype=np.float64)
    try:
        body, kind, start, size = E.find_bodies(P1, 1e-3)
        core = size[body] <= 2
        free_ = core.copy()
        for u, v in E.neighbour_pairs(np.ascontiguousarray(P1), 1.45):
            if core[u] and size[body[v]] <= 12:
                free_[v] = True
            if core[v] and size[body[u]] <= 12:
                free_[u] = True
        C = E.Compactor(P1, 1e-3, singles=np.where(free_)[0])
        loc = E.LocalLP(C)
        if loc.ok:
            loads = np.maximum(
                np.nan_to_num(loc.loads(), nan=0.0, posinf=0.0, neginf=0.0), 0.0
            )
            scores += loads * 10.0
    except Exception:
        pass
    for i in near_band:
        scores[i] += 0.25
    for i in tilted:
        scores[i] += 0.35
    order = np.argsort(-scores)
    return [int(i) for i in order[:max_cand]]


def run_hopper_chain(P_start, seed_base, first_sec, cascade_sec, max_cascades=3):
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


def worker_deep_hopper(job):
    n, L0, arr0, arr_p1, L_p1, seed, direct_sec, del_sec = job
    best_S = L0
    best_P = None
    best_tag = ""

    # Phase 1: 4 diverse direct Force-Guided LP Hopper seeds (always restart from best_P!)
    for si in range(4):
        P_start = best_P if best_P is not None else arr0
        Q, S, w = run_hopper_chain(
            P_start, seed + si * 7919, direct_sec, min(160, direct_sec), max_cascades=3
        )
        if S < best_S - 1e-8:
            best_S, best_P, best_tag = S, Q, f"dhop_s{si}"

    # Phase 2: Exhaustive n+1 removal-compaction screening (up to 64 removals -> top-3 holes -> Hopper!)
    if arr_p1 is not None and (L_p1 - best_S) < 0.036:
        cands = pick_removal_candidates(arr_p1, max_cand=64)
        holes = []
        t_scr = time.time()
        for idx in cands:
            if time.time() - t_scr > 55.0:
                break
            try:
                Qh, Sh = delete_and_compact(arr_p1, idx)
                if Qh is not None and Sh < best_S + 0.014:
                    holes.append((Sh, idx, Qh))
            except Exception:
                continue
        holes.sort(key=lambda x: x[0])
        for hi, (Sh, idx, Qh) in enumerate(holes[:3]):
            if Sh < best_S - 1e-8:
                best_S, best_P, best_tag = Sh, Qh, f"delc29_{idx}"
            if Sh < best_S + 0.009:
                Q2, S2, w2 = run_hopper_chain(
                    Qh,
                    seed + 60000 + hi * 3571,
                    del_sec,
                    min(140, del_sec),
                    max_cascades=2,
                )
                if S2 < best_S - 1e-8:
                    best_S, best_P, best_tag = S2, Q2, f"delhop29_{idx}"

    # Phase 3: Micro-loosen fq kick -> Hopper escape (breaks jammed force-chains!)
    if os.path.exists(f2d.FQ_BIN):
        with tempfile.TemporaryDirectory(prefix=f"w29a_k_{n}_") as tmp:
            P_src = best_P if best_P is not None else arr0
            S_src = best_S
            r_k = f2d.run_fq(
                S_src + 0.008,
                P_src,
                tmp,
                [
                    "--loosen",
                    "1.0",
                    "--round-r0",
                    "0.045",
                    "--round-compress",
                    "--round-press",
                    "--round-steps",
                    "6",
                    "--round-press-outer",
                    "10",
                    "--seed",
                    str(seed + 8888),
                    "--flip-top",
                    "32",
                    "--pit",
                    "180",
                ],
                timeout=45,
            )
            if r_k is not None and r_k[0] < best_S + 0.005:
                if r_k[0] < best_S - 1e-8:
                    best_S, best_P, best_tag = r_k[0], r_k[1], "fqkick29"
                Q3, S3, _ = run_hopper_chain(
                    r_k[1], seed + 99991, 150, 120, max_cascades=2
                )
                if S3 < best_S - 1e-8:
                    best_S, best_P, best_tag = S3, Q3, "kickhop29"

    # Phase 4: Final Polish + Strict Decimal64 >= 1.05e-9 Certification
    if best_P is not None and best_S < L0 - 1e-9:
        with tempfile.TemporaryDirectory(prefix=f"w29a_fin_{n}_") as tmp:
            cur_a = safe_open(best_P)
            cur_s = float(E.side(cur_a))
            if os.path.exists(f2d.FQ_BIN):
                r_fin = f2d.run_fq(
                    cur_s,
                    cur_a,
                    tmp,
                    ["--loosen", "1.0", "--no-alm", "--flip-top", "56", "--pit", "550"],
                    timeout=65,
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
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2800)
    ap.add_argument("--direct-sec", type=int, default=200)
    ap.add_argument("--del-sec", type=int, default=170)
    ap.add_argument("--min-rank", type=int, default=0)
    ap.add_argument("--max-targets", type=int, default=160)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    warm_numba()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    front_rows = json.load(open(args.frontier_json))
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    fresh_r21 = {
        545,
        794,
        692,
        816,
        995,
        370,
        460,
        818,
        371,
        915,
        785,
        793,
        683,
        923,
        405,
        642,
        593,
    }
    scored = []
    for n in range(150, 1000):
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        gap_ds = max(0.0, (cur[n][0] - front_L.get(n, cur[n][0])) / math.sqrt(n))
        if gap_ds > 1e-7 or n in fresh_r21:
            prio = gap_ds + (0.0004 if (args.min_rank == 0 and n in fresh_r21) else 0.0)
            scored.append((prio, n))
    scored.sort(reverse=True)
    targets = [n for _, n in scored[args.min_rank : args.min_rank + args.max_targets]]
    my_slice = targets[args.shard :: args.num_shards]
    print(
        f"[W29A-DeepForceHopper s={args.shard}/{args.num_shards}] Base={tot_score:.10f} "
        f"(loaded {n_files} win files), my_targets({len(my_slice)})={my_slice} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    jobs = []
    for i, n in enumerate(my_slice):
        arr_p1 = None
        L_p1 = 1e9
        if n + 1 <= 1000:
            kp1 = math.isqrt(n + 1)
            if kp1 * kp1 != (n + 1) and cur[n + 1][0] < kp1 + 1 - 1e-6:
                arr_p1 = cur[n + 1][1]
                L_p1 = cur[n + 1][0]
        jobs.append(
            (
                n,
                cur[n][0],
                cur[n][1],
                arr_p1,
                L_p1,
                2910001 + args.shard * 10099 + i * 317,
                args.direct_sec,
                args.del_sec,
            )
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
        futs = [ex.submit(worker_deep_hopper, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W29A ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                dL = Lf - cur[n][0]
                if dL < -1e-9:
                    dS = dL / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W29A DEEP-HOP WIN!!] n={n:4d} {tag:14s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W29A s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W29A s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
