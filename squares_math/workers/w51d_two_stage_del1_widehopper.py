#!/usr/bin/env python3
"""Wave 51D: Isolated Two-Stage Exhaustive Near-Band Delete-1/Delete-2 Map
-> 4x2-Seed True `eta<=0.28` Wider-Hopper + Residual Contact Continuation!

Fixes the exact root cause discovered in Wave 50D (where a 150s Phase-A timeout on n=965 left
unstarted `eval_single_delete` tasks in a shared `ProcessPoolExecutor` and starved Phase B):
  1. Strict Pool Isolation (`terminate_pool`):
     Phase A runs in its own `ex_screen` pool, which is explicitly terminated BEFORE Phase B
     creates a brand-new `ex_hop` pool!
  2. Two-Stage `<12s` Exhaustive Near-Band + `n+2->n` Screening:
     - Stage 1 (`max_iter=15, patience=2`): screens ALL ~280 near-band squares of `P[n+1]` PLUS
       top 18 `E.skeleton` 2-square pairs of `P[n+2]` (when `L[n+2] - L[n] < 0.024`) in 4-8s!
     - Stage 2 (`max_iter=65, patience=2`): refines the Top-24 Stage-1 holes in 2-4s!
  3. Interleaved Residual Contact Continuation (`w51r.residual_continuation_polish`):
     Between Round 1 (`4 seeds x 210s`) and Round 2 (`4 seeds x 175s`), applies BOTH `SLP.polish`
     AND exact trigonometric Residual Contact Continuation (`biased_contact_lp` + octagon + wall_slack)!
"""

import argparse
import csv
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
import w32a_fresh_basin_fast_hopper as w32a
import w50d_exhaustive_private_del1_widehopper as w50d
import w51r_residual_contact_continuation as w51r


def terminate_pool(ex):
    if ex is None:
        return
    procs = list(getattr(ex, "_processes", {}).values())
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.join(timeout=1.0)
        except Exception:
            pass


def compact_deleted(P_del, max_iter=60, patience=2):
    P = E.to_origin(P_del)
    body, _, _, size = E.find_bodies(P, 1e-3)
    core = size[body] <= 2
    free_ = core.copy()
    for u, v in E.neighbour_pairs(np.ascontiguousarray(P), 1.45):
        if core[u] and size[body[v]] <= 12:
            free_[v] = True
        if core[v] and size[body[u]] <= 12:
            free_[u] = True
    Q, S, ov = E.Compactor(P, 1e-3, singles=np.where(free_)[0]).compact(
        P, max_iter=max_iter, patience=patience
    )
    if ov > 5e-8:
        return 1e9, None
    return float(S), Q.astype(np.float64)


def eval_stage1(args_tuple):
    P_src, rem_tup, tag = args_tuple
    try:
        P_del = np.delete(P_src, list(rem_tup), axis=0)
        S1, Q1 = compact_deleted(P_del, max_iter=15, patience=2)
        return tag, S1, Q1
    except Exception:
        return tag, 1e9, None


def eval_stage2(args_tuple):
    tag, Q1 = args_tuple
    try:
        S2, Q2 = compact_deleted(Q1, max_iter=60, patience=2)
        return tag, S2, Q2
    except Exception:
        return tag, 1e9, None


def gen_del2_pairs(P2, max_pairs=18):
    """Select top 2-square deletions from P[n+2] using force skeleton + near-band neighbors."""
    P2 = E.to_origin(P2)
    try:
        _, _, _, _, load = E.skeleton(P2)
    except Exception:
        load = np.zeros(len(P2))
    tilts = np.abs(E.fold(P2[:, 2]))
    near = np.where((tilts > 0.015) | (load > np.percentile(load, 80)))[0]
    if len(near) < 2:
        return []
    # Sort by lowest load near tilted bands (easiest voids to collapse!)
    order = near[np.argsort(load[near])]
    pairs = []
    # Pair 1: spatially adjacent low-load near-band squares
    nb_pairs = E.neighbour_pairs(np.ascontiguousarray(P2), 1.65)
    near_set = set(int(i) for i in order[:60])
    adj = []
    for u, v in nb_pairs:
        if int(u) in near_set and int(v) in near_set:
            adj.append((float(load[u] + load[v]), int(min(u, v)), int(max(u, v))))
    adj.sort()
    for _, u, v in adj[: max_pairs // 2]:
        pairs.append((u, v))
    # Pair 2: two independent lowest-load near-band squares (e.g. one from each band!)
    top_k = [int(i) for i in order[:10]]
    for a_i in range(len(top_k)):
        for b_i in range(a_i + 1, len(top_k)):
            pr = (min(top_k[a_i], top_k[b_i]), max(top_k[a_i], top_k[b_i]))
            if pr not in pairs:
                pairs.append(pr)
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def deep_contact_polish(n, P_in, deadline, slp_sec=48.0, fq_sec=28):
    """Full convergence SLP (iters=240, up to 48s) + Residual Contact Continuation + Rust fq + lp_rot!
    Closes the exact 1e-5 .. 2.5e-4 side gap left by W50's 22s SLP cap!
    """
    cur_a = w32a.safe_open(P_in)
    cur_s = float(E.side(cur_a))
    rem = deadline - time.time() - 5.0
    if rem < 4.0:
        return cur_a, cur_s

    # 1) Deep HiGHS SLP down to r = 1e-11
    try:
        import slp as SLP

        xs, ys, ts, _ = SLP.polish(
            cur_a[:, 0],
            cur_a[:, 1],
            cur_a[:, 2],
            iters=240,
            r0=3e-3,
            rmin=1e-11,
            tol_sep=-1e-13,
            time_limit=min(float(slp_sec), rem * 0.45),
            kappa=0.32,
        )
        qs = w32a.safe_open(np.column_stack([xs, ys, ts]))
        ss = float(E.side(qs))
        if ss < cur_s - 1e-11 and E.min_gap(qs) >= -1e-12:
            cur_a, cur_s = qs, ss
    except Exception:
        pass

    # 2) Exact Trigonometric Residual Contact Continuation (box + octagon + wall_slack)
    rem = deadline - time.time() - 5.0
    if rem > 6.0:
        try:
            qr, sr = w51r.residual_continuation_polish(
                cur_a, budget_s=min(18.0, rem * 0.35)
            )
            if sr < cur_s - 1e-11 and E.min_gap(qr) >= -1e-12:
                cur_a, cur_s = qr, sr
        except Exception:
            pass

    # 3) Rust fq nonlinear L-BFGS + flip-top polish
    rem = deadline - time.time() - 5.0
    if fq_sec > 0 and os.path.exists(f2d.FQ_BIN) and rem > 10.0:
        with tempfile.TemporaryDirectory(prefix=f"w51dp_{n}_") as tmp:
            r_fin = f2d.run_fq(
                cur_s,
                cur_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "750"],
                timeout=min(int(fq_sec), max(8, int(rem * 0.45))),
            )
            if r_fin is not None and r_fin[0] < cur_s - 1e-11:
                cur_s, cur_a = float(r_fin[0]), w32a.safe_open(r_fin[1])

    # 4) Final rotation LP polish
    try:
        ql = w32a.safe_open(w12.lp_rot_polish(cur_a))
        sl = float(E.side(ql))
        if sl < cur_s - 1e-11 and E.min_gap(ql) >= -1e-12:
            cur_a, cur_s = ql, sl
    except Exception:
        pass

    return cur_a, cur_s


def certify_with_residual(n, P_best, L0, deadline):
    cur_a, cur_s = deep_contact_polish(n, P_best, deadline, slp_sec=48.0, fq_sec=32)
    ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_a)
    if ok and Lf < L0 - 1e-9:
        return (float(Lf), xf, yf, tf)
    return None


def deep_3round_widehopper(
    ex, n, P_start, L0, base_seed, r1_sec, r2_sec, r3_sec, deadline
):
    """3-Round Adaptive Escalation `eta<=0.28` Wider-Hopper WITH Interleaved `deep_contact_polish`!
    - Round 1: 4 seeds x r1_sec -> if < L0 + 0.0035: FULL `deep_contact_polish` (SLP 45s + Residual + fq + lp_rot)!
    - Round 2: 4 seeds x r2_sec from tightened contacts -> if < L0 + 0.0012: FULL `deep_contact_polish`!
    - Round 3: IF < L0 + 0.00045 (all near-misses like n=652 +1.0e-5, n=553 +1.1e-5, n=756 +4.3e-5!):
               4 NEW seeds x r3_sec -> FULL `deep_contact_polish`!
    """
    P_cur = w32a.safe_open(P_start)
    S_start = float(E.side(P_cur))
    best_P, best_S = P_cur, S_start
    tot_tr, tot_fit, tot_w = 0, 0, 0

    def _run_4seeds(P_in, seed0, sec_budget):
        nonlocal best_P, best_S, tot_tr, tot_fit, tot_w
        futs = [
            ex.submit(
                w50d.hop_seed_worker, (P_in, seed0 + s * 1009, sec_budget, deadline)
            )
            for s in range(4)
        ]
        try:
            for f in as_completed(futs, timeout=max(25.0, sec_budget + 45.0)):
                _, Q, S, (tr, ft, wn) = f.result()
                tot_tr += tr
                tot_fit += ft
                tot_w += wn
                if S < best_S - 1e-10 and E.min_gap(Q) >= -1e-12:
                    best_S, best_P = S, Q
        except TimeoutError:
            for f in futs:
                f.cancel()

    # Round 1
    _run_4seeds(P_cur, base_seed, r1_sec)
    if best_S < L0 + 0.0035 or (S_start - best_S) > 2e-4:
        use_fq1 = 22 if best_S < L0 + 0.0008 else 0
        best_P, best_S = deep_contact_polish(
            n, best_P, deadline, slp_sec=42.0, fq_sec=use_fq1
        )

    # Round 2
    if (
        r2_sec >= 20
        and (deadline - time.time()) > 45.0
        and (best_S < L0 + 0.0030 or (S_start - best_S) > 2.5e-4)
    ):
        s_pre2 = best_S
        _run_4seeds(best_P, base_seed + 50000, r2_sec)
        if best_S < L0 + 0.0012 or best_S < s_pre2 - 1e-8:
            use_fq2 = 28 if best_S < L0 + 0.0010 else 0
            best_P, best_S = deep_contact_polish(
                n, best_P, deadline, slp_sec=48.0, fq_sec=use_fq2
            )

        # Round 3 Sniper Escalation on ALL `< +0.00045` ultra-near-misses (or active jumpers)!
        if (
            r3_sec >= 20
            and (deadline - time.time()) > 50.0
            and (best_S < L0 + 0.00045 or (s_pre2 - best_S) > 1.5e-4)
        ):
            s_pre3 = best_S
            _run_4seeds(best_P, base_seed + 100000, r3_sec)
            if best_S < L0 + 0.00050 or best_S < s_pre3 - 1e-9:
                best_P, best_S = deep_contact_polish(
                    n, best_P, deadline, slp_sec=52.0, fq_sec=32
                )

    return best_P, best_S, (tot_tr, tot_fit, tot_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--targets-csv", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--r1-sec", type=int, default=210)
    ap.add_argument("--r2-sec", type=int, default=175)
    ap.add_argument("--max-puzzles", type=int, default=160)
    ap.add_argument("--seed-off", type=int, default=510000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    base_only, _, _, _ = w12.load_frontier_with_wins(args.union_npz, [])
    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    tg_hints = {}
    if os.path.exists(args.targets_csv):
        with open(args.targets_csv) as f:
            for row in csv.DictReader(f):
                tg_hints.setdefault(int(row["n"]), []).append(int(row["removed"]))

    cand_ns = []
    for n in range(180, 1000):
        K = math.isqrt(n)
        K1 = math.isqrt(n + 1)
        if K * K == n or K1 * K1 == (n + 1):
            continue
        if cur[n + 1][0] >= K1 + 1 - 1e-6:
            continue
        step_up = cur[n + 1][0] - cur[n][0]
        step_up2 = (cur[n + 2][0] - cur[n][0]) if (n + 2 <= 1000) else 1.0
        if step_up > 0.036 and step_up2 > 0.024:
            continue
        fl = front_L.get(n, cur[n][0])
        gap_wr = cur[n][0] - fl
        if gap_wr < 5e-6 and min(step_up, step_up2) > 0.010:
            continue
        priv_adv = max(0.0, base_only[n + 1][0] - cur[n + 1][0])
        prio = (
            (gap_wr / math.sqrt(n))
            + (0.00065 if priv_adv > 1e-4 else 0.0)
            + max(0.0, 0.026 - min(step_up, step_up2)) * 0.025
            + (0.00015 if n in tg_hints else 0.0)
        )
        cand_ns.append((prio, n))

    cand_ns.sort(reverse=True)
    selected = [x[1] for x in cand_ns[: args.max_puzzles]]
    my_slice = selected[args.shard :: args.num_shards]
    deadline = t0 + args.budget_sec

    print(
        f"[W51D-TwoStageDel12 s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_slice({len(my_slice)})={my_slice} ({time.time() - t0:.1f}s)",
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

    # PHASE A: Strictly isolated screening pool (terminated cleanly before Phase B!)
    screened_starts = []
    ex_screen = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for n in my_slice:
            if time.time() > t0 + min(420.0, args.budget_sec * 0.25):
                break
            P1 = cur[n + 1][1]
            L0 = cur[n][0]
            idxs = list(
                dict.fromkeys(
                    list(tg_hints.get(n, []))
                    + w50d.all_near_band_indices(P1, max_cand=280)
                )
            )
            idxs = [i for i in idxs if 0 <= i < len(P1)]
            s1_tasks = [(P1, (idx,), f"del1_{idx}") for idx in idxs]
            if (
                n + 2 <= 1000
                and cur[n + 2][1] is not None
                and (cur[n + 2][0] - L0) < 0.024
            ):
                P2 = cur[n + 2][1]
                for u, v in gen_del2_pairs(P2, max_pairs=18):
                    s1_tasks.append((P2, (u, v), f"del2_{u}_{v}"))

            ts0 = time.time()
            futs1 = [ex_screen.submit(eval_stage1, tk) for tk in s1_tasks]
            stage1_hits = []
            try:
                for f in as_completed(futs1, timeout=45.0):
                    tag, S1, Q1 = f.result()
                    if Q1 is not None and S1 < L0 + 0.025:
                        stage1_hits.append((S1, tag, Q1))
            except TimeoutError:
                # Recreate clean pool so next puzzle has 0 leftover tasks!
                terminate_pool(ex_screen)
                ex_screen = ProcessPoolExecutor(max_workers=args.procs)

            stage1_hits.sort(key=lambda z: z[0])
            top_s1 = stage1_hits[:24]
            holes = []
            if top_s1:
                futs2 = [
                    ex_screen.submit(eval_stage2, (tag, Q1)) for _, tag, Q1 in top_s1
                ]
                try:
                    for f in as_completed(futs2, timeout=35.0):
                        tag, S2, Q2 = f.result()
                        if Q2 is not None and S2 < L0 + 0.015:
                            holes.append((S2 - L0, S2, tag, Q2))
                except TimeoutError:
                    terminate_pool(ex_screen)
                    ex_screen = ProcessPoolExecutor(max_workers=args.procs)

            holes.sort(key=lambda z: z[0])
            best_str = f"{holes[0][0]:+.6f}@{holes[0][2]}" if holes else "none(<+0.015)"
            print(
                f"  [W51D screen] n={n:4d} s1={len(s1_tasks)}->top{len(top_s1)}->holes={len(holes)} best={best_str} ({time.time() - ts0:.1f}s)",
                flush=True,
            )
            picked_n = []
            for gap0, S_del, tag, Q_del in holes:
                if all(abs(S_del - ps) > 3e-6 for _, ps, _, _ in picked_n):
                    picked_n.append((gap0, S_del, tag, Q_del))
                max_keep = 3 if gap0 < 0.009 else 2
                if len(picked_n) >= max_keep:
                    break
            for gap0, S_del, tag, Q_del in picked_n:
                screened_starts.append((gap0, n, tag, S_del, Q_del))
    finally:
        terminate_pool(ex_screen)

    screened_starts.sort(key=lambda z: z[0])
    print(
        f"[W51D s={args.shard}] Phase B (CLEAN POOL): hunting {len(screened_starts)} best starts ({time.time() - t0:.1f}s)!",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for rank_i, (gap0, n, tag, S_del, Q_del) in enumerate(screened_starts):
            rem = deadline - time.time()
            if rem < 45.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            if S_del < L0_cur + 0.00025:
                res0 = certify_with_residual(n, Q_del, L0_cur, deadline)
                if res0 is not None:
                    Lf = res0[0]
                    tot_ds += (Lf - L0_cur) / math.sqrt(n)
                    wins[n] = res0
                    L0_cur = Lf
                    print(
                        f"  [W51D DIRECT COMPACT WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf:.7f}!",
                        flush=True,
                    )
                    save_ckpt()

            r1 = min(args.r1_sec, max(30, int(0.35 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(25, int(0.30 * (rem - 20.0))))
            r3 = min(args.r2_sec, max(25, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + args.shard * 10007 + rank_i * 601
            Qb, Sb, (tr, ft, wn) = deep_3round_widehopper(
                ex_hop, n, Q_del, L0_cur, bseed, r1, r2, r3, deadline
            )
            if Sb < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, Qb)
                if ok and Lf < L0_cur - 1e-9:
                    dL = Lf - cur[n][0]
                    tot_ds += (Lf - L0_cur) / math.sqrt(n)
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W51D WIN!!] n={n:4d} [{tag}_wh28_3r] gap0={gap0:+.5f} tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{rank_i + 1}/{len(screened_starts)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W51D done] n={n:4d} [{tag}] gap0={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{rank_i + 1}/{len(screened_starts)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W51D s={args.shard} EXC] {exc}", flush=True)
    finally:
        terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W51D s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
