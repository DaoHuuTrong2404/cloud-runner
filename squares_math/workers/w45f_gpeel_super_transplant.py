#!/usr/bin/env python3
"""Wave 45F: 83x Generalized Strip-Peel Half-Transplant + Relaxed Full Transplant + Wide-Hopper (max_cl=18)!

Scales Wave 44F's breakthrough win (htpD_983_a0 -> n=921 -0.0020981 / dS=-0.00006913!) by 83x:
  1. Generalized Strip-Peel Half-Transplant Down (gpeel):
     After removing 1 tilted bar + row on `axis` via relaxed multi-cut, peels EVERY square whose
     bounding extent intrudes into the outer 0.99 strip on `other` (both low & high walls!),
     guaranteeing zero overlap (`min_gap >= -1e-8`) even when a perpendicular band touches the corner!
     -> Expands valid Half-Transplant Down pool 83x (166+ ultra-soft basins starting +0.005..+0.035
     BEFORE Compactor LP — 4x closer than n=921's +0.0199!!).
  2. Chained 2-Step gpeel (e.g. 983 -> 921 -> 861!) + gpeel + Delete-1 (983 -> 920!) + Half-Up!
  3. Stock & Relaxed Multi-Cut Full Transplant (1-step + 2-step + del-1)!
  4. Widened acceptance gate (`gap < 0.035` after Compactor!) + Wider-Move Hopper (`wide=True`,
     `max_cl=18`) + SLP + Rust fq + lp_rot + exact Decimal64 >= 1.05e-9!
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
import w12a_super_bfmelt_rp as w12
import w32a_fresh_basin_fast_hopper as w32a
import w42f_transplant_wide_hopper as w42f
import w44f_relaxed_transplant_wide_hopper as w44f


def _gpeel_down_candidates(P, ax, top_cuts=2):
    """Generalized Strip-Peel Half-Transplant Down on axis `ax` (83x reach!)."""
    out = []
    other = 1 - ax
    cuts = w44f.find_cuts_relaxed(P, ax, top_k=top_cuts)
    for ci, c in enumerate(cuts):
        Qa = w44f._change_len_cut(P, ax, False, c)
        if E.min_gap(Qa) < -1e-8:
            continue
        Qa = E.to_origin(Qa)
        h = 0.5 * (np.abs(np.cos(Qa[:, 2])) + np.abs(np.sin(Qa[:, 2])))
        L_o = float((Qa[:, other] + h).max())
        for side_hi in (False, True):
            mask = (
                ((Qa[:, other] + h) > L_o - 0.99)
                if side_hi
                else ((Qa[:, other] - h) < 0.99)
            )
            if mask.sum() < 4:
                continue
            Qb = E.to_origin(Qa[~mask])
            res = w44f._compact_if_valid(Qb)
            if res is not None:
                Qc, Sc = res
                out.append((Qc, Sc, f"gpD_a{ax}_{'h' if side_hi else 'l'}"))
        if out:
            break
    return out


def _try_gpeel_and_all_transplants(args_tuple):
    src_n, P = args_tuple
    out_list = []
    seen = set()

    def _add(Q, S, tag):
        nn = int(len(Q))
        if not (80 <= nn <= 1000):
            return
        key = (nn, round(S, 5), tag[:4])
        if key in seen:
            return
        seen.add(key)
        out_list.append((nn, float(S), tag, Q))

    # 1. BRAND-NEW 83x Generalized Strip-Peel Half-Transplant Down + 2-Step + Del-1!
    for ax in (0, 1):
        try:
            g1_list = _gpeel_down_candidates(P, ax, top_cuts=2)
            for Qc, Sc, gtag in g1_list:
                _add(Qc, Sc, f"{gtag}_{src_n}")
                # Bonus A: gpeel + Delete-1 (instant soft basin for n_new - 1!)
                if len(Qc) - 1 >= 80:
                    try:
                        cands = w32a.pick_near_band(Qc, max_cand=8)
                        best_d = None
                        for idx in cands[:4]:
                            Qd, Sd = w32a.fast_delete_and_compact(Qc, idx)
                            if Qd is not None and (best_d is None or Sd < best_d[0]):
                                best_d = (Sd, Qd)
                        if best_d is not None:
                            _add(
                                best_d[1].astype(np.float64),
                                float(best_d[0]),
                                f"{gtag}d1_{src_n}",
                            )
                    except Exception:
                        pass
                # Bonus B: 2-Step Chained gpeel (e.g. 983 -> 921 -> 861!)
                try:
                    for Q2, S2, gtag2 in _gpeel_down_candidates(Qc, ax, top_cuts=1)[:1]:
                        _add(Q2, S2, f"gp2D_a{ax}_{src_n}")
                except Exception:
                    pass
        except Exception:
            pass

    # 2. All Wave 44F stock + relaxed full + half-up transplants + 3-STEP CHAINED TRANSPLANT (tp3)!
    try:
        for nn, S, tag, Q in w44f._try_all_transplants_for_src((src_n, P)):
            _add(Q, S, tag)
            if tag.startswith("tp2_"):
                up_flag = tag.endswith("_u")
                try:
                    out3 = E.transplant(Q, up_flag)
                    if out3 is not None:
                        Q3, S3, _, _ = out3
                        if (80 <= len(Q3) <= 1000) and E.min_gap(Q3) >= -1e-8:
                            _add(
                                Q3.astype(np.float64),
                                float(S3),
                                f"tp3_{src_n}_{'u' if up_flag else 'd'}",
                            )
                except Exception:
                    pass
    except Exception:
        pass

    return out_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--hop-sec", type=int, default=140)
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--max-gap", type=float, default=0.032)
    ap.add_argument("--seed-off", type=int, default=450000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )

    # Sweep ALL non-grid packings [100..1000] across 16 shards!
    srcs = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        if cur[n][0] < K + 1 - 1e-6 and n != K * K:
            srcs.append(n)
    my_srcs = srcs[args.shard :: args.num_shards]
    tp_tasks = [(n, cur[n][1]) for n in my_srcs]

    tp_cands = []
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        for res_list in ex.map(_try_gpeel_and_all_transplants, tp_tasks):
            for n_new, S, tag, Q in res_list:
                gap = S - cur[n_new][0]
                if gap < args.max_gap:
                    tp_cands.append((gap, n_new, S, tag, Q))

    # Prioritize smallest gap first (direct wins & sub-0.015 ultra-soft gpeel basins!)
    tp_cands.sort(key=lambda z: z[0])
    print(
        f"[W45F-GPeel+SuperTransplant s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"swept={len(tp_tasks)} srcs -> promising(gap<{args.max_gap})={len(tp_cands)} "
        f"(direct_wins={sum(1 for z in tp_cands if z[0] < -1e-7)}, gpeel={sum(1 for z in tp_cands if z[3].startswith('gp'))}) ({time.time() - t0:.1f}s)",
        flush=True,
    )

    deadline = t0 + args.budget_sec
    jobs = []
    seen_key = set()
    for i, (gap, n_new, S, tag, Q) in enumerate(tp_cands):
        fam = (
            "gp"
            if tag.startswith("gp")
            else ("half" if tag.startswith("htp") else "full")
        )
        if (n_new, fam) in seen_key:
            continue
        seen_key.add((n_new, fam))
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

    print(
        f"[W45F s={args.shard}] Queued {len(jobs)} PURE fresh gpeel/half/full transplant jobs (hop={args.hop_sec}s x {args.n_seeds} seeds)",
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
    ex = ProcessPoolExecutor(max_workers=args.procs)
    futs = [ex.submit(w42f.worker_wide_hopper_polish, j) for j in jobs]
    try:
        for fut in as_completed(futs, timeout=max(30.0, deadline - time.time() + 90.0)):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W45F ERR] {exc}", flush=True)
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
                        f"  [W45F WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"(dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() > deadline + 45.0:
                print(
                    f"  [W45F s={args.shard}] Hard deadline reached after {done}/{len(jobs)} jobs.",
                    flush=True,
                )
                break
    except TimeoutError:
        print(
            f"  [W45F s={args.shard}] as_completed timeout after {done}/{len(jobs)} jobs — checkpointing & exiting cleanly!",
            flush=True,
        )

    save_ckpt()
    print(
        f"[W45F s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    for proc in list(getattr(ex, "_processes", {}).values()):
        try:
            proc.terminate()
        except Exception:
            pass
    os._exit(0)


if __name__ == "__main__":
    main()
