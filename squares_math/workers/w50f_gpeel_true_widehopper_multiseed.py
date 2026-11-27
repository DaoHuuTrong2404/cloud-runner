#!/usr/bin/env python3
"""Wave 50F: 4-Seed True `eta<=0.28` Fast Wider-Move Hopper + Interleaved SLP on Sub-0.016 GPeel & Transplants!

Directly exploits Wave 49F's near-miss discovery (`n=701 +0.000118`, `n=754 +0.000275`,
`n=549 +0.000341`, `n=870 +0.000668`, `n=598 +0.000745`, `n=931 +0.000750`, `n=811 +0.000949`),
which only got 30-62 trials in W49F due to 460-single LP rebuilds:
  - Now every ultra-close `gpeel`, `htp`, and `tp1-3` fresh transplant gets 4 parallel seeds x 190s
    of FAST Stock Wider-Move Hopper WITH `eta <= 0.28` (~50,000+ trials across 4 seeds!) ->
    interleaved `SLP.polish(iters=115)` -> 4 new parallel seeds x 160s -> `fq` + `lp_rot` + Decimal64!
"""

import argparse
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

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
import w50d_exhaustive_private_del1_widehopper as w50d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--r1-sec", type=int, default=190)
    ap.add_argument("--r2-sec", type=int, default=160)
    ap.add_argument("--max-gap", type=float, default=0.016)
    ap.add_argument("--seed-off", type=int, default=504000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, _ = w49a.load_closed_frontier(args.union_npz, args.win_glob)

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
    seen_key = set()
    queued = []
    for gap, n_new, S, tag, Q in tp_cands:
        fam = (
            "gp"
            if tag.startswith("gp")
            else ("half" if tag.startswith("htp") else "full")
        )
        if (n_new, fam) in seen_key:
            continue
        seen_key.add((n_new, fam))
        queued.append((gap, n_new, S, tag, Q))
        if len(queued) >= 12:
            break

    print(
        f"[W50F-GPeelWide28 s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"srcs={len(tp_tasks)} -> cands(gap<{args.max_gap})={len(tp_cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
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

    ex = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, S_init, tag, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 40.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            if S_init < L0_cur - 1e-8:
                res0 = w50d.certify_winner(n, Q_init, L0_cur, deadline)
                if res0 is not None:
                    Lf = res0[0]
                    tot_ds += (Lf - L0_cur) / math.sqrt(n)
                    wins[n] = res0
                    L0_cur = Lf
                    print(
                        f"  [W50F DIRECT WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf:.7f}!",
                        flush=True,
                    )
                    save_ckpt()

            r1 = min(args.r1_sec, max(30, int(0.42 * (rem - 15.0))))
            r2 = min(args.r2_sec, max(20, int(0.35 * (rem - 15.0))))
            bseed = args.seed_off + args.shard * 10007 + i * 401
            Qb, Sb, (tr, ft, wn) = w50d.run_4seed_2round_widehopper(
                ex, Q_init, L0_cur, bseed, r1, r2, deadline
            )
            if Sb < L0_cur - 1e-9:
                res = w50d.certify_winner(n, Qb, L0_cur, deadline)
                if res is not None:
                    Lf = res[0]
                    dL = Lf - cur[n][0]
                    tot_ds += (Lf - L0_cur) / math.sqrt(n)
                    wins[n] = res
                    print(
                        f"  [W50F WIN!!] n={n:4d} [{tag}_wh28] gap0={gap0:+.5f} tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W50F done] n={n:4d} [{tag}] gap0={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W50F s={args.shard} EXC] {exc}", flush=True)

    save_ckpt()
    print(
        f"[W50F s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
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
