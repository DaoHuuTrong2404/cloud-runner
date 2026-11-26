#!/usr/bin/env python3
"""Wave 31A: Trust-Region Contact SLP Polish + Exhaustive ALL-Square n+1 Removal + HopperV2 Ping-Pong.

Two high-yield orthogonal modes:
  1. --mode polish_all: Sweeps ALL non-grid packings (n=15..1000) with Numba+HiGHS Trust-Region
     Contact SLP (goodpjw_ds/tools/slp.py with kink-branch splitting at theta=0 and exact
     quadratic safety margin 2*rt^2 + r*rt). Whenever SLP moves L by > 1e-7, chains
     HopperV2 -> 2nd SLP polish to trigger secondary basin cascades!
  2. --mode exh_remove: For Top-160 frontier gaps, screens ALL near-band squares (up to 260!)
     of packing n+1 via fast LP compaction, feeds top-3 tightest holes (even if L_comp <= L_cur+0.014)
     into SLP -> HopperV2 -> SLP -> 2nd HopperV2 -> final Decimal64 >= 1.05e-9 certification!
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
import w30a_hopper_v2_unlocked as w30


def warm_all():
    w30.warm_numba()
    x = np.array([0.5, 1.5, 0.5, 1.5], dtype=np.float64)
    y = np.array([0.5, 0.5, 1.5, 1.5], dtype=np.float64)
    t = np.array([0.0, 0.0, 0.0, 0.1], dtype=np.float64)
    SLP.polish(x, y, t, iters=3, r0=1e-3, time_limit=5.0)


def run_slp_polish(P, iters=120, r0=5e-3, time_limit=35.0):
    """Run Trust-Region SLP contact polish and return (Q_arr, L_val) or (P, side(P))."""
    P0 = E.to_origin(np.asarray(P, dtype=np.float64))
    L0 = float(E.side(P0))
    try:
        x, y, t, L1 = SLP.polish(
            P0[:, 0],
            P0[:, 1],
            P0[:, 2],
            iters=iters,
            r0=r0,
            rmin=1e-10,
            tol_sep=-1e-13,
            time_limit=time_limit,
            kappa=0.35,
        )
        Q = E.to_origin(np.column_stack([x, y, t]))
        Q = w30.safe_open(Q)
        Lq = float(E.side(Q))
        if Lq < L0 - 1e-11 and E.min_gap(Q) >= -1e-12:
            return Q, Lq
    except Exception:
        pass
    return P0, L0


def finalize_candidate(n, L0, best_P, tag):
    """Run final SLP + fq micro-polish + lp_rot_polish + exact Decimal64 >= 1.05e-9 cert."""
    cur_a = w30.safe_open(best_P)
    cur_s = float(E.side(cur_a))
    # Final SLP polish
    q_slp, s_slp = run_slp_polish(cur_a, iters=120, r0=2e-3, time_limit=30.0)
    if s_slp < cur_s - 1e-10:
        cur_a, cur_s = q_slp, s_slp
    # Optional fq micro-polish
    if os.path.exists(f2d.FQ_BIN):
        with tempfile.TemporaryDirectory(prefix=f"w31a_fin_{n}_") as tmp:
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
        return n, tag, (Lf, xf, yf, tf)
    return n, tag, None


def worker_polish_all(job):
    n, L0, arr0, seed, slp_sec, hop_sec = job
    best_P = E.to_origin(arr0)
    best_S = L0
    best_tag = ""

    # Pass 1: Trust-Region Contact SLP with two step scales (moderate + fine)
    for r0, iters, tl, ttag in [
        (8e-3, 140, slp_sec, "slp8e3"),
        (1.5e-3, 120, slp_sec * 0.7, "slp15e4"),
    ]:
        Q, S = run_slp_polish(best_P, iters=iters, r0=r0, time_limit=tl)
        if S < best_S - 1e-10:
            best_P, best_S, best_tag = Q, S, ttag

    # If SLP opened a meaningful move (> 1e-7), unleash HopperV2 -> 2nd SLP cascade!
    if best_S < L0 - 1e-7:
        for rnd in range(2):
            prev_S = best_S
            Qh, Sh, _ = w30.run_v2_chain(
                best_P,
                seed + rnd * 7919,
                hop_sec,
                min(120, hop_sec),
                unlock_bands=(rnd == 0),
                max_cascades=3,
            )
            if Sh < best_S - 1e-9:
                best_P, best_S, best_tag = Qh, Sh, f"slp_hv2_r{rnd}"
            Qs, Ss = run_slp_polish(
                best_P, iters=120, r0=3e-3, time_limit=slp_sec * 0.7
            )
            if Ss < best_S - 1e-10:
                best_P, best_S, best_tag = Qs, Ss, f"slp_hv2_slp_r{rnd}"
            if prev_S - best_S < 5e-7:
                break

    if best_S < L0 - 1e-9:
        return finalize_candidate(n, L0, best_P, best_tag)
    return n, best_tag, None


def pick_all_near_band_squares(P1, max_cand=260):
    """Return ALL tilted + 2-hop near-band squares sorted by LP dual load + band proximity."""
    P1 = E.to_origin(P1)
    n1 = len(P1)
    tilts = np.abs(E.fold(P1[:, 2]))
    tilted = np.where(tilts > 0.015)[0]
    hop1 = set(int(i) for i in tilted)
    pairs = E.neighbour_pairs(np.ascontiguousarray(P1), 1.65)
    if len(tilted) > 0:
        for u, v in pairs:
            if u in hop1:
                hop1.add(int(v))
            if v in hop1:
                hop1.add(int(u))
    hop2 = set(hop1)
    for u, v in pairs:
        if u in hop1:
            hop2.add(int(v))
        if v in hop1:
            hop2.add(int(u))

    scores = np.zeros(n1, dtype=np.float64)
    try:
        H = w30.HopperV2(P1, seed=0, unlock_bands=True)
        if H.L.ok:
            loads = np.maximum(
                np.nan_to_num(H.L.loads(), nan=0.0, posinf=0.0, neginf=0.0), 0.0
            )
            scores += loads * 12.0
    except Exception:
        pass
    for i in hop2:
        scores[i] += 0.15
    for i in hop1:
        scores[i] += 0.30
    for i in tilted:
        scores[i] += 0.45
    # Only keep squares in hop2 (or top max_cand overall if hop2 small)
    order = np.argsort(-scores)
    return [int(i) for i in order[: min(n1, max_cand)]]


def worker_exh_remove(job):
    n, L0, arr0, arr_p1, L_p1, arr_p2, L_p2, seed, hop_sec, slp_sec = job
    best_S = L0
    best_P = None
    best_tag = ""

    # Phase 1: SLP pre-polish of current n (in case a fresh basin from W30 just landed!)
    Q0, S0 = run_slp_polish(arr0, iters=100, r0=5e-3, time_limit=slp_sec)
    if S0 < best_S - 1e-9:
        best_S, best_P, best_tag = S0, Q0, "pre_slp"

    # Phase 2: Exhaustive ALL-near-band n+1 removal compaction screening (up to 260 squares!)
    holes = []
    if arr_p1 is not None and (L_p1 - best_S) < 0.065:
        cands = pick_all_near_band_squares(arr_p1, max_cand=260)
        t_scr = time.time()
        for idx in cands:
            if time.time() - t_scr > 140.0:
                break
            try:
                Qh, Sh = w30.delete_and_compact_v2(arr_p1, idx)
                if Qh is not None and Sh < best_S + 0.020:
                    holes.append((Sh, f"del1_{idx}", Qh))
            except Exception:
                continue

    # Also test top-20 double-removals from n+2 when L_p2 is close!
    if arr_p2 is not None and (L_p2 - best_S) < 0.045:
        cands2 = pick_all_near_band_squares(arr_p2, max_cand=28)
        t_scr2 = time.time()
        for a_i in cands2[:12]:
            if time.time() - t_scr2 > 45.0:
                break
            try:
                Qa, Sa = w30.delete_and_compact_v2(arr_p2, a_i)
                if Qa is None or Sa > best_S + 0.035:
                    continue
                sub_c = pick_all_near_band_squares(Qa, max_cand=6)
                for b_i in sub_c[:4]:
                    Qb, Sb = w30.delete_and_compact_v2(Qa, b_i)
                    if Qb is not None and Sb < best_S + 0.015:
                        holes.append((Sb, f"del2_{a_i}_{b_i}", Qb))
            except Exception:
                continue

    holes.sort(key=lambda z: z[0])
    # Deduplicate holes by compacted side (within 1e-6) so we explore distinct basins!
    picked = []
    for Sh, htag, Qh in holes:
        if all(abs(Sh - ps) > 2e-6 for ps, _, _ in picked):
            picked.append((Sh, htag, Qh))
        if len(picked) >= 4:
            break

    for hi, (Sh, htag, Qh) in enumerate(picked):
        # First apply fast SLP to pull the hole tight before HopperV2!
        Qs, Ss = run_slp_polish(Qh, iters=60, r0=5e-3, time_limit=min(20.0, slp_sec))
        if Ss < Sh:
            Qh, Sh = Qs, Ss
        if Sh < best_S - 1e-8:
            best_S, best_P, best_tag = Sh, Qh, f"{htag}_slp"
        if Sh < best_S + 0.014:
            Q2, S2, w = w30.run_v2_chain(
                Qh,
                seed + 31000 + hi * 4999,
                hop_sec,
                min(150, hop_sec),
                unlock_bands=True,
                max_cascades=3,
            )
            if S2 < best_S - 1e-8:
                best_S, best_P, best_tag = S2, Q2, f"{htag}_hv2"
                # Re-polish with SLP + 2nd multi-seed HopperV2 cascade (goodpjw n=654 recipe!)
                Qs2, Ss2 = run_slp_polish(
                    best_P, iters=100, r0=3e-3, time_limit=slp_sec
                )
                if Ss2 < best_S - 1e-9:
                    best_S, best_P, best_tag = Ss2, Qs2, f"{htag}_hv2_slp"
                for s2_idx in range(2):
                    Q3, S3, _ = w30.run_v2_chain(
                        best_P,
                        seed + 77000 + hi * 997 + s2_idx * 3571,
                        int(hop_sec * 0.9),
                        120,
                        unlock_bands=(s2_idx == 0),
                        max_cascades=2,
                    )
                    if S3 < best_S - 1e-8:
                        best_S, best_P, best_tag = S3, Q3, f"{htag}_hv2x2"

    if best_P is not None and best_S < L0 - 1e-9:
        return finalize_candidate(n, L0, best_P, best_tag)
    return n, best_tag, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["polish_all", "exh_remove"], required=True)
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2800)
    ap.add_argument("--slp-sec", type=float, default=35.0)
    ap.add_argument("--hop-sec", type=int, default=150)
    ap.add_argument("--min-rank", type=int, default=0)
    ap.add_argument("--max-targets", type=int, default=160)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    warm_all()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    front_rows = json.load(open(args.frontier_json))
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    if args.mode == "polish_all":
        # ALL non-grid packings from n=15..1000, prioritized by gap to frontier + size
        scored = []
        for n in range(15, 1001):
            K = math.isqrt(n)
            if K * K == n or cur[n][0] >= K + 1 - 1e-6:
                continue
            gap_ds = max(0.0, (cur[n][0] - front_L.get(n, cur[n][0])) / math.sqrt(n))
            # Put recently improved / gapped puzzles first, still sweep all 560+!
            prio = gap_ds + (1e-4 if n in fresh else 0.0) + 1e-9 * n
            scored.append((prio, n))
        scored.sort(reverse=True)
        targets = [n for _, n in scored]
        my_slice = targets[args.shard :: args.num_shards]
        jobs = [
            (
                n,
                cur[n][0],
                cur[n][1],
                3110001 + args.shard * 10169 + i * 331,
                args.slp_sec,
                args.hop_sec,
            )
            for i, n in enumerate(my_slice)
        ]
        worker_fn = worker_polish_all
    else:
        scored = []
        for n in range(150, 1000):
            K = math.isqrt(n)
            if K * K == n or cur[n][0] >= K + 1 - 1e-6:
                continue
            gap_ds = max(0.0, (cur[n][0] - front_L.get(n, cur[n][0])) / math.sqrt(n))
            if gap_ds > 1e-7:
                scored.append((gap_ds, n))
        scored.sort(reverse=True)
        targets = [
            n for _, n in scored[args.min_rank : args.min_rank + args.max_targets]
        ]
        my_slice = targets[args.shard :: args.num_shards]
        jobs = []
        for i, n in enumerate(my_slice):
            arr_p1, L_p1 = None, 1e9
            if n + 1 <= 1000:
                k1 = math.isqrt(n + 1)
                if k1 * k1 != (n + 1) and cur[n + 1][0] < k1 + 1 - 1e-6:
                    arr_p1, L_p1 = cur[n + 1][1], cur[n + 1][0]
            arr_p2, L_p2 = None, 1e9
            if n + 2 <= 1000:
                k2 = math.isqrt(n + 2)
                if k2 * k2 != (n + 2) and cur[n + 2][0] < k2 + 1 - 1e-6:
                    arr_p2, L_p2 = cur[n + 2][1], cur[n + 2][0]
            jobs.append(
                (
                    n,
                    cur[n][0],
                    cur[n][1],
                    arr_p1,
                    L_p1,
                    arr_p2,
                    L_p2,
                    3120001 + args.shard * 10169 + i * 331,
                    args.hop_sec,
                    args.slp_sec,
                )
            )
        worker_fn = worker_exh_remove

    print(
        f"[W31A-{args.mode} s={args.shard}/{args.num_shards}] Base={tot_score:.10f} "
        f"(loaded {n_files} win files), my_targets({len(my_slice)})={my_slice[:15]}... ({time.time() - t0:.1f}s)",
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
        futs = [ex.submit(worker_fn, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W31A ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                dL = Lf - cur[n][0]
                if dL < -1e-9:
                    dS = dL / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W31A-{args.mode} WIN!!] n={n:4d} {tag:18s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W31A-{args.mode} s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W31A-{args.mode} s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
