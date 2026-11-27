#!/usr/bin/env python3
"""Wave 51L: Crystalline Diagonal-Staircase Splices + Band-Wave Macro-LNS (`lns_band`)
-> 4x2-Seed True `eta<=0.28` Wider-Hopper + Residual Contact Continuation!

Combines TWO non-local macro-topology generators with our fast `eta<=0.28` Wider-Hopper:
  1. Sub-track L1 (`w49k` Healed Crystalline Staircase Starts):
     Generates `(V_4, V_5, V_1)` diagonal staircase splices across `K=18..31` (now starting from
     `load_full_closed_frontier` where `442/485/530` are already Window-Peeled to `.52461175`!),
     and feeds every healed candidate (`gap < +0.022`) into fast `eta<=0.28` Wider-Hopper (`40,000+`
     trials instead of W49K's 30-80 slow FaultOpenSA trials!).
  2. Sub-track L2 (`lns_band` Band-Wave / Reseed / Pinned-Angle Numba L-BFGS `squeeze2`):
     Runs 4 parallel seeds x 65s of `lns_band.lns_burst` (`band_angles` sinusoidal/ramp/alternating
     wave along tilted bands, `reseed` tilted strip lattice replacement, `PIN` 8..32 deg pinned-angle
     relaxation) on Top-64 high-gap puzzles (`K^2+d` & stalled two-band packings), harvesting both
     direct LNS improvements AND fresh distinct-angle basins (`< +0.0045`) for `eta<=0.28` Wider-Hopper!
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

import lns_band
import w32a_fresh_basin_fast_hopper as w32a
import w49k_diagonal_staircase_splicer as w49k
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r


def _lns_worker(args_tuple):
    n, P_in, budget_s, seed = args_tuple
    try:
        best_P, best_S, near_cands, tr = lns_band.lns_burst(
            P_in, budget_s=budget_s, seed=seed, near_tol=0.0045
        )
        return n, best_P, best_S, near_cands, tr
    except Exception:
        return n, P_in, float("inf"), [], 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--r1-sec", type=int, default=185)
    ap.add_argument("--r2-sec", type=int, default=155)
    ap.add_argument("--seed-off", type=int, default=515000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )
    cur_L_map = {n: cur[n][0] for n in range(1, 1001)}
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    # --- PART 1: Isolated Staircase Splice Generation (<= 260s) ---
    priority_donors, other_donors = [], []
    for n in range(324, 999):
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        if (n - K * K) in (1, 2, 3, 4):
            priority_donors.append(n)
        else:
            other_donors.append(n)
    all_donors = priority_donors + other_donors
    my_donors = all_donors[args.shard :: args.num_shards]
    st_tasks = [(n, cur[n][1], cur_L_map) for n in my_donors]

    cands = []
    ex_st = ProcessPoolExecutor(max_workers=args.procs)
    try:
        t_st = time.time()
        futs_st = [ex_st.submit(w49k.generate_splices_for_donor, tk) for tk in st_tasks]
        try:
            for fi in as_completed(futs_st, timeout=260.0):
                try:
                    for nt, St, tag, Qt in fi.result():
                        gap = St - cur[nt][0]
                        if gap < 0.022:
                            cands.append((gap, nt, St, tag, Qt))
                except Exception:
                    continue
                if time.time() - t_st > 250.0:
                    break
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_st)

    # --- PART 2: Isolated Macro Band-LNS (`band_angles` / `reseed` / `PIN`) Burst (<= 280s) ---
    gap_ranked = []
    for n in range(200, 1001):
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        fl = front_L.get(n, cur[n][0])
        gds = (cur[n][0] - fl) / math.sqrt(n)
        if gds > 5e-6:
            gap_ranked.append((gds, n))
    gap_ranked.sort(reverse=True)
    lns_targets = [n for _, n in gap_ranked[:96]][args.shard :: args.num_shards][:4]

    ex_lns = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for li, n_lns in enumerate(lns_targets):
            if time.time() - t0 > 520.0:
                break
            b_seed = args.seed_off + args.shard * 7919 + li * 101
            futs_l = [
                ex_lns.submit(
                    _lns_worker, (n_lns, cur[n_lns][1], 55.0, b_seed + s * 37)
                )
                for s in range(4)
            ]
            tot_ltr = 0
            try:
                for fl in as_completed(futs_l, timeout=75.0):
                    _, bP, bS, near_list, ltr = fl.result()
                    tot_ltr += ltr
                    if bS < cur[n_lns][0] - 1e-9:
                        cands.append((bS - cur[n_lns][0], n_lns, bS, "lns_direct", bP))
                    for g_near, ktag, q_near in near_list[:2]:
                        cands.append(
                            (g_near, n_lns, cur[n_lns][0] + g_near, ktag, q_near)
                        )
            except TimeoutError:
                w51d.terminate_pool(ex_lns)
                ex_lns = ProcessPoolExecutor(max_workers=args.procs)
            print(
                f"  [W51L LNS s={args.shard}] n={n_lns:4d} lns_trials={tot_ltr} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_lns)

    def sort_key(item):
        gap, nt, _, _, _ = item
        K = math.isqrt(nt)
        bonus = -0.006 if (nt == K * K + 1) else (-0.0025 if (nt == K * K + 2) else 0.0)
        return gap + bonus

    cands.sort(key=sort_key)
    seen = set()
    queued = []
    for gap, nt, St, tag, Qt in cands:
        t_fam = tag.split("_")[0] + "_" + tag.split("_")[-1][:3]
        if (nt, t_fam) in seen:
            continue
        seen.add((nt, t_fam))
        queued.append((gap, nt, St, tag, Qt))
        if len(queued) >= 12:
            break

    print(
        f"[W51L-StairLNS s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"raw_cands={len(cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
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

    # --- PART 3: Clean 4-Core `eta<=0.28` Wider-Hopper + Residual Continuation! ---
    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, S_init, tag, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 40.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            if S_init < L0_cur - 1e-8:
                res0 = w51d.certify_with_residual(n, Q_init, L0_cur, deadline)
                if res0 is not None:
                    Lf = res0[0]
                    tot_ds += (Lf - L0_cur) / math.sqrt(n)
                    wins[n] = res0
                    L0_cur = Lf
                    print(
                        f"  [W51L DIRECT WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf:.7f}!",
                        flush=True,
                    )
                    save_ckpt()

            r1 = min(args.r1_sec, max(30, int(0.35 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(25, int(0.30 * (rem - 20.0))))
            r3 = min(args.r2_sec, max(25, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + args.shard * 10007 + i * 601
            Qb, Sb, (tr, ft, wn) = w51d.deep_3round_widehopper(
                ex_hop, n, Q_init, L0_cur, bseed, r1, r2, r3, deadline
            )
            if Sb < L0_cur - 1e-9:
                import w5b_2d_faultdp_gadgetq as f2d

                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, Qb)
                if ok and Lf < L0_cur - 1e-9:
                    dL = Lf - cur[n][0]
                    tot_ds += (Lf - L0_cur) / math.sqrt(n)
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W51L WIN!!] n={n:4d} [{tag}_wh28_3r] gap0={gap0:+.5f} tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W51L done] n={n:4d} [{tag}] gap0={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W51L s={args.shard} EXC] {exc}", flush=True)
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W51L s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
