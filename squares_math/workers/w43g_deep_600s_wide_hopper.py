#!/usr/bin/env python3
"""Wave 43G: Deep 450s-600s Multi-Round Wider-Move Hopper (max_cl=18!) + Contact-Polish Protocol!

Directly implements goodpjw2008's v50 world-record protocol (which jumped n=545 by -0.0073 to
equal the #1 per-puzzle leaderboard record to 9 digits):
  - Targets our EXACT top-64 freshest soft designs (W42 Bar-Transplant hits 378/758/875,
    goodpjw 545/556/698, W40-41 816/995/851/694/785/678/731/786/548/910/862/603/298/264)
    AND our Top-40 biggest live-gap puzzles (234, 367, 491, 333, 738, 456, 804, 730, 261, 904, ...).
  - Runs 4 rounds of DEEP Wider-Move Hopper (300s-450s per round, max_cluster=18 on 50% of trials)
    interleaved with SLP contact polish to break out of deep local basins!
  - Also screens up to 240 near-band n+1 -> n removals and runs deep Wider-Move Hopper on the
    best near-miss hole!
  - Strictly enforces internal deadline clamping so 100% of workers return & checkpoint before 60m!
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
import w32a_fresh_basin_fast_hopper as w32a


def _slp_polish(P):
    cur_a = w32a.safe_open(P)
    cur_s = float(E.side(cur_a))
    try:
        xs, ys, ts, Ls = SLP.polish(
            cur_a[:, 0],
            cur_a[:, 1],
            cur_a[:, 2],
            iters=100,
            r0=2.5e-3,
            rmin=1e-10,
            tol_sep=-1e-13,
            time_limit=18.0,
        )
        Qs = w32a.safe_open(np.column_stack([xs, ys, ts]))
        Ss = float(E.side(Qs))
        if Ss < cur_s - 1e-10 and E.min_gap(Qs) >= -1e-12:
            return Qs, Ss
    except Exception:
        pass
    return cur_a, cur_s


def worker_deep_wide(job):
    n, L0, arr0, arr_p1, L_p1, seed, round_sec, n_rounds, deadline = job
    best_S = L0
    best_P = None
    best_tag = ""

    # Phase 1: Deep multi-round Wider-Move Hopper (max_cl=18) interleaved with SLP contact polish!
    cur_P = w32a.safe_open(arr0)
    for r in range(n_rounds):
        rem = deadline - time.time()
        if rem < 15.0:
            break
        sec = min(round_sec, rem - 8.0)
        try:
            H = E.Hopper(cur_P, seed=seed + r * 9973, wide=True)
            if not H.applicable():
                break
            Q, S = H.run(sec)
            Q, S = _slp_polish(Q)
            if S < best_S - 1e-9 and E.min_gap(Q) >= -1e-12:
                best_S, best_P, best_tag = S, Q, f"dwh_r{r}"
                cur_P = Q
            elif S < float(E.side(cur_P)) - 1e-9:
                cur_P = Q
        except Exception:
            break

    # Phase 2: Exhaustive n+1 -> n removal screening (up to 240 near-band squares!) -> Deep Wide-Hopper!
    if (
        arr_p1 is not None
        and (L_p1 - best_S) < 0.040
        and (deadline - time.time()) > 60.0
    ):
        cands = w32a.pick_near_band(arr_p1, max_cand=240)
        holes = []
        t_scr = time.time()
        for idx in cands:
            if time.time() - t_scr > 90.0 or (deadline - time.time()) < 40.0:
                break
            try:
                Qh, Sh = w32a.fast_delete_and_compact(arr_p1, idx)
                if Qh is not None and Sh < best_S + 0.014:
                    holes.append((Sh, idx, Qh))
            except Exception:
                continue
        holes.sort(key=lambda z: z[0])
        picked = []
        for Sh, idx, Qh in holes:
            if all(abs(Sh - ps) > 2e-6 for ps, _, _ in picked):
                picked.append((Sh, idx, Qh))
            if len(picked) >= 2:
                break
        for hi, (Sh, idx, Qh) in enumerate(picked):
            rem = deadline - time.time()
            if rem < 20.0:
                break
            if Sh < best_S - 1e-9:
                best_S, best_P, best_tag = Sh, Qh, f"ddel_{idx}"
            cur_h = w32a.safe_open(Qh)
            for rr in range(2):
                rem = deadline - time.time()
                if rem < 15.0:
                    break
                sec = min(round_sec, rem - 8.0)
                try:
                    H = E.Hopper(
                        cur_h, seed=seed + 70000 + hi * 5003 + rr * 1997, wide=True
                    )
                    if not H.applicable():
                        break
                    Q2, S2 = H.run(sec)
                    Q2, S2 = _slp_polish(Q2)
                    if S2 < best_S - 1e-9 and E.min_gap(Q2) >= -1e-12:
                        best_S, best_P, best_tag = S2, Q2, f"ddel_wh_{idx}_r{rr}"
                        cur_h = Q2
                    elif S2 < float(E.side(cur_h)) - 1e-9:
                        cur_h = Q2
                except Exception:
                    break

    # Phase 3: Final SLP + Rust fq + lp_rot + exact Decimal64 >= 1.05e-9 certification!
    if best_P is not None and best_S < L0 - 1e-9:
        cur_a, cur_s = _slp_polish(best_P)
        if os.path.exists(f2d.FQ_BIN):
            with tempfile.TemporaryDirectory(prefix=f"w43g_{n}_") as tmp:
                r_fin = f2d.run_fq(
                    cur_s,
                    cur_a,
                    tmp,
                    ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
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
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2650)
    ap.add_argument("--round-sec", type=int, default=320)
    ap.add_argument("--n-rounds", type=int, default=3)
    ap.add_argument("--seed-off", type=int, default=430000)
    ap.add_argument("--max-targets", type=int, default=64)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    # Priority 1: Freshest soft designs (W42 Bar-Transplant 378/758/875 + goodpjw 545/556/698 + W40-41 816/995/851/694/785...)
    fresh_vip = [
        378,
        377,
        379,
        758,
        757,
        759,
        875,
        874,
        876,
        545,
        544,
        546,
        816,
        815,
        817,
        995,
        994,
        851,
        850,
        556,
        555,
        698,
        697,
        694,
        693,
        785,
        786,
        678,
        731,
        548,
        910,
        862,
        603,
        298,
        264,
    ]
    targets = []
    seen = set()
    for n in fresh_vip:
        if 80 <= n <= 1000 and n not in seen:
            K = math.isqrt(n)
            if K * K != n and cur[n][0] < K + 1 - 1e-6:
                targets.append(n)
                seen.add(n)

    # Priority 2: Top live-gap puzzles (234, 367, 491, 333, 738, 456, 804, 730, 261, 904, 748, 962, ...)
    gap_ranked = []
    for n in range(120, 1001):
        if n in seen:
            continue
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        fl = front_L.get(n, cur[n][0])
        gds = (cur[n][0] - fl) / math.sqrt(n)
        if gds > 1e-6:
            gap_ranked.append((gds, n))
    gap_ranked.sort(reverse=True)
    for _, n in gap_ranked:
        if len(targets) >= args.max_targets:
            break
        targets.append(n)
        seen.add(n)

    my_slice = targets[args.shard :: args.num_shards]
    deadline = t0 + args.budget_sec
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
                args.seed_off + args.shard * 10169 + i * 521,
                args.round_sec,
                args.n_rounds,
                deadline,
            )
        )

    print(
        f"[W43G-DeepWideHop s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_targets({len(my_slice)})={my_slice} round_sec={args.round_sec}x{args.n_rounds} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0
    done = 0

    def save_ckpt():
        pl = {}
        for kn, (Lf, xf, yf, tf) in sorted(wins.items()):
            pl[f"L_{kn}"] = np.array([Lf], dtype=np.float64)
            pl[f"x_{kn}"] = np.asarray(xf, dtype=np.float64)
            pl[f"y_{kn}"] = np.asarray(yf, dtype=np.float64)
            pl[f"t_{kn}"] = np.asarray(tf, dtype=np.float64)
        np.savez_compressed(args.out_npz, **pl)

    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_deep_wide, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W43G ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                dL = Lf - cur[n][0]
                if dL < -1e-9:
                    dS = dL / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W43G WIN!!] n={n:4d} [{tag:16s}] {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() > deadline:
                print(
                    f"[W43G s={args.shard}] Reached graceful deadline {args.budget_sec}s.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W43G-DeepWideHop s={args.shard} DONE] {len(wins)} certified wins, "
        f"total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
