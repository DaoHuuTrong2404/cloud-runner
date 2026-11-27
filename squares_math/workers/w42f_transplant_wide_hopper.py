#!/usr/bin/env python3
"""Wave 42F: Ross Wade Bar-Transplant + goodpjw2008 Wider-Move Hopper (max_cluster=18) Engine!

Combines BOTH freshly validated mathematical breakthroughs into a dedicated 160c wave pillar:
  1. Runs E.transplant(P, up=True) and E.transplant(P, up=False) across ALL non-grid packings
     in our frontier (inserting/removing a tilted bar + upright row at density ~0.991 instead of
     L-border's 0.970!), generating BRAND-NEW SOFT BAND DESIGNS at target sizes n' in [100..1000]!
  2. Immediately attacks EVERY winning OR near-miss fresh transplant (side < cur[n'] + 0.008)
     with goodpjw2008's Wider-Move Hopper (50% trials move up to 18 squares!) + SLP + Rust fq
     + exact Decimal64 >= 1.05e-9 certification!
  3. Also runs deep Wider-Move Hopper directly on the highest-gap & freshest basins (n=545, 816,
     694, 785, 963, 902, 548, 678, 731, 786, 910, 862, 234, 367, 491, 333, 738, ...)!
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


def _try_transplant_one(args_tuple):
    src_n, P, up = args_tuple
    out_list = []
    try:
        out = E.transplant(P, up)
        if out is None:
            return out_list
        Q, S, _, _ = out
        n_new = int(len(Q))
        if (80 <= n_new <= 1000) and E.min_gap(Q) >= -1e-8:
            out_list.append(
                (
                    n_new,
                    float(S),
                    f"tp1_{src_n}_{'u' if up else 'd'}",
                    Q.astype(np.float64),
                )
            )
            # Bonus 1: Transplant + Delete-1 (instant soft design for n_new - 1!)
            if n_new - 1 >= 80:
                try:
                    cands = w32a.pick_near_band(Q, max_cand=12)
                    best_d = None
                    for idx in cands[:6]:
                        Qd, Sd = w32a.fast_delete_and_compact(Q, idx)
                        if Qd is not None and (best_d is None or Sd < best_d[0]):
                            best_d = (Sd, idx, Qd)
                    if best_d is not None:
                        out_list.append(
                            (
                                n_new - 1,
                                float(best_d[0]),
                                f"tpd1_{src_n}_{'u' if up else 'd'}",
                                best_d[2].astype(np.float64),
                            )
                        )
                except Exception:
                    pass
            # Bonus 2: 2-step chained Bar-Transplant ((m, j) -> (m+-2, j+-4)!)
            try:
                out2 = E.transplant(Q, up)
                if out2 is not None:
                    Q2, S2, _, _ = out2
                    n2 = int(len(Q2))
                    if (80 <= n2 <= 1000) and E.min_gap(Q2) >= -1e-8:
                        out_list.append(
                            (
                                n2,
                                float(S2),
                                f"tp2_{src_n}_{'u' if up else 'd'}",
                                Q2.astype(np.float64),
                            )
                        )
            except Exception:
                pass
    except Exception:
        pass
    return out_list


def worker_wide_hopper_polish(job):
    n, L0, P_init, tag_in, seed, hop_sec, n_seeds, deadline = job
    best_S = L0
    best_P = None
    best_tag = tag_in

    # Check if P_init itself already beats L0!
    P_start = w32a.safe_open(P_init)
    S_start = float(E.side(P_start))
    if S_start < best_S - 1e-9 and E.min_gap(P_start) >= -1e-12:
        best_S, best_P, best_tag = S_start, P_start, f"{tag_in}_raw"

    # Run goodpjw2008 Wider-Move Hopper (wide=True: 50% stock <=9, 50% wide <=18!)
    for si in range(n_seeds):
        if time.time() > deadline - 10.0:
            break
        P_src = best_P if best_P is not None else P_start
        Q, S, _ = w32a.run_stock_hopper_chain(
            P_src,
            seed + si * 7919,
            first_sec=hop_sec,
            cascade_sec=min(120, hop_sec),
            max_cascades=3,
            wide=True,
            deadline=deadline,
        )
        if S < best_S - 1e-9:
            best_S, best_P, best_tag = S, Q, f"{tag_in}_wh_s{si}"

    # Final SLP + Rust fq + lp_rot + exact Decimal64 >= 1.05e-9 certification!
    if best_P is not None and best_S < L0 - 1e-9:
        cur_a = w32a.safe_open(best_P)
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
            Qs = w32a.safe_open(np.column_stack([xs, ys, ts]))
            Ss = float(E.side(Qs))
            if Ss < cur_s - 1e-10 and E.min_gap(Qs) >= -1e-12:
                cur_a, cur_s = Qs, Ss
        except Exception:
            pass
        if os.path.exists(f2d.FQ_BIN):
            with tempfile.TemporaryDirectory(prefix=f"w42f_{n}_") as tmp:
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
    ap.add_argument("--budget-sec", type=int, default=2200)
    ap.add_argument("--hop-sec", type=int, default=110)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-off", type=int, default=420000)
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

    # Phase 1: Parallel Bar-Transplant sweep over this shard's slice of ALL non-grid packings [100..1000]!
    srcs = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        if cur[n][0] < K + 1 - 1e-6 and n != K * K:
            srcs.append(n)
    my_srcs = srcs[args.shard :: args.num_shards]
    tp_tasks = [(n, cur[n][1], up) for n in my_srcs for up in (True, False)]

    tp_cands = []  # (gap, n_new, S, tag, Q)
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        for res_list in ex.map(_try_transplant_one, tp_tasks):
            for n_new, S, tag, Q in res_list:
                gap = S - cur[n_new][0]
                if gap < 0.018:
                    tp_cands.append((gap, n_new, S, tag, Q))
    tp_cands.sort(key=lambda z: z[0])
    print(
        f"[W42F-SuperTransplant+WideHop s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"swept={len(tp_tasks)} 1+2-step+del1 transplants -> promising={len(tp_cands)} "
        f"(direct_wins={sum(1 for z in tp_cands if z[0] < -1e-7)}) ({time.time() - t0:.1f}s)",
        flush=True,
    )

    # Deduplicate best transplant per target n_new, then fill remaining queue with top live-gap Wide-Hopper targets!
    deadline = t0 + args.budget_sec
    jobs = []
    seen_tp_n = set()
    for i, (gap, n_new, S, tag, Q) in enumerate(tp_cands):
        if n_new in seen_tp_n:
            continue
        seen_tp_n.add(n_new)
        jobs.append(
            (
                n_new,
                cur[n_new][0],
                Q,
                tag,
                args.seed_off + args.shard * 10007 + i * 97,
                args.hop_sec,
                args.n_seeds,
                deadline,
            )
        )

    # Also add top live-gap puzzles (ranked by (cur_L - front_L)/sqrt(n)) for pure Wider-Move Hopper (max_cluster=18)!
    gap_ranked = []
    for n in range(120, 1001):
        K = math.isqrt(n)
        if n == K * K or cur[n][0] >= K + 1 - 1e-6:
            continue
        fl = front_L.get(n, cur[n][0])
        gds = (cur[n][0] - fl) / math.sqrt(n)
        if gds > 1e-6:
            gap_ranked.append((gds, n))
    gap_ranked.sort(reverse=True)
    top_gap_ns = [n for _, n in gap_ranked[:160]][args.shard :: args.num_shards]
    for j_i, n in enumerate(top_gap_ns):
        jobs.append(
            (
                n,
                cur[n][0],
                cur[n][1],
                "wide_gap",
                args.seed_off + 500000 + args.shard * 10007 + j_i * 131,
                args.hop_sec,
                args.n_seeds,
                deadline,
            )
        )

    print(
        f"[W42F s={args.shard}] Queued {len(jobs)} jobs ({len(seen_tp_n)} fresh transplants + {len(top_gap_ns)} wide-move gap targets)",
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

    done = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_wide_hopper_polish, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W42F ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                prev_L = wins[n][0] if n in wins else cur[n][0]
                if Lf < prev_L - 1e-10:
                    dL = Lf - cur[n][0]
                    dS = (Lf - prev_L) / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W42F WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"(dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W42F s={args.shard}] Graceful budget {args.budget_sec}s reached.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W42F-Transplant+WideHop s={args.shard} DONE] {len(wins)} certified wins, "
        f"total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
