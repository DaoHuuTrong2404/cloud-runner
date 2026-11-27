#!/usr/bin/env python3
"""Wave 49F: 83x Generalized Strip-Peel & Chained Transplants -> Fault-Open SA-Unwelded!

Upgrades Wave 48F with:
  1. `w49a.load_closed_frontier` (full L-border + best-boundary-square delete-1 closure!)
  2. `w49a.run_fault_open_pipeline` (Dual-Path Fault-Open Global Compaction + Full-Unwelding
     for `n <= 460` + 2-Cycle Reheating SA + mid-cycle `SLP.polish`!)
  3. Complete per-job near-miss telemetry (`dBest`, `tr`, `fit`, `fop`, `acc`) so no run is ever opaque!
"""

import argparse
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

import w32a_fresh_basin_fast_hopper as w32a
import w45f_gpeel_super_transplant as w45f
import w49_fault_open_sa_engine as w49a


def worker_tp_fop(job):
    n, L0, Q_init, tag_in, seed, sa_sec, quench_sec, wide_sec, deadline = job
    out_tag, st, cS, res = w49a.run_fault_open_pipeline(
        Q_init, L0, seed, sa_sec, quench_sec, wide_sec, deadline, tag_prefix=tag_in
    )
    return n, out_tag if res is not None else tag_in, st, cS - L0, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--sa-sec", type=int, default=100)
    ap.add_argument("--quench-sec", type=int, default=70)
    ap.add_argument("--wide-sec", type=int, default=45)
    ap.add_argument("--max-gap", type=float, default=0.024)
    ap.add_argument("--seed-off", type=int, default=498000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, n_files = w49a.load_closed_frontier(
        args.union_npz, args.win_glob
    )

    srcs = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        if cur[n][0] < K + 1 - 1e-6 and n != K * K:
            srcs.append(n)
    my_srcs = srcs[args.shard :: args.num_shards]
    tp_tasks = [(n, cur[n][1]) for n in my_srcs]

    tp_cands = []
    with ProcessPoolExecutor(max_workers=args.procs) as ex_init:
        for res_list in ex_init.map(w45f._try_gpeel_and_all_transplants, tp_tasks):
            for n_new, S, tag, Q in res_list:
                gap = S - cur[n_new][0]
                if gap < args.max_gap:
                    tp_cands.append((gap, n_new, S, tag, Q))

    tp_cands.sort(key=lambda z: z[0])
    print(
        f"[W49F-GPeel+FaultOpenSA s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"swept={len(tp_tasks)} srcs -> promising(gap<{args.max_gap})={len(tp_cands)} "
        f"(gpeel={sum(1 for z in tp_cands if z[3].startswith('gp'))}) ({time.time() - t0:.1f}s)",
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
                args.sa_sec,
                args.quench_sec,
                args.wide_sec,
                deadline,
            )
        )
        if len(jobs) >= 28:
            break

    print(
        f"[W49F s={args.shard}] Queued {len(jobs)} top fresh gpeel/half/full transplants into FaultOpenSA pipeline!",
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
    futs = [ex.submit(worker_tp_fop, j) for j in jobs]
    try:
        for fut in as_completed(futs, timeout=max(30.0, deadline - time.time() + 90.0)):
            done += 1
            try:
                n, tag, st, d_best, res = fut.result()
            except Exception as exc:
                print(f"  [W49F ERR] {exc}", flush=True)
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
                        f"  [W49F WIN!!] n={n:4d} [{tag}] tr={st[0]} fit={st[1]} fop={st[2]} acc={st[3]} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
            else:
                print(
                    f"  [W49F done] n={n:4d} [{tag}] tr={st[0]} fit={st[1]} fop={st[2]} acc={st[3]} dBest={d_best:+.6f} [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                    flush=True,
                )
            if time.time() > deadline + 45.0:
                break
    except TimeoutError:
        print(
            f"  [W49F s={args.shard}] Timeout after {done}/{len(jobs)} — exiting cleanly!",
            flush=True,
        )

    save_ckpt()
    print(
        f"[W49F s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
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
