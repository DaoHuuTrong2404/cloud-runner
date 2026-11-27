#!/usr/bin/env python3
"""Wave 51F: Sub-0.008 GPeel & Transplants -> 3-Round `eta<=0.28` Wider-Hopper
WITH Interleaved Deep Contact Polish (`55s SLP` + `Residual Continuation` + `fq` + `lp_rot`)!

Directly cashes in on Wave 50F's huge telemetry breakthrough:
  - In Wave 50F, `eta<=0.28` Wider-Hopper made 15-180 jumps (`w=15..180`) on `n=652 (+0.000010)`,
    `n=378 (+0.000026)`, `n=418 (+0.000029)`, `n=756 (+0.000043, w=101!)`, `n=870 (+0.000077, w=48!)`,
    `n=701 (+0.000128, w=88!)`, `n=875 (+0.000134)`, `n=705 (+0.000136)`, `n=377 (+0.000142)`,
    `n=812 (+0.000148, w=90!)`, `n=651 (+0.000171, w=108!)` — stopping just `1e-5 .. 1.9e-4` above
    our fully-`fq`-polished R32 records ONLY because:
      (1) `SLP.polish` had a 22s cap (~65 iters, stopping at `r~1e-5` instead of `r~1e-10`),
      (2) `fq` + `lp_rot` + `Residual Continuation` was NEVER run when `Sb >= L0 - 1e-9`!
      (3) Round 3 was never run from the fully-polished state!
  - Now `w51d.deep_3round_widehopper` runs FULL `deep_contact_polish` (`55s SLP` + `Residual` +
    `fq --flip-top 64 --pit 750` + `lp_rot`) BETWEEN rounds AND on every `< +0.0010` near-miss,
    plus unleashes Round 3 (4 new seeds) whenever `best_S < L0 + 0.00040`!
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

import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w45f_gpeel_super_transplant as w45f
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--r1-sec", type=int, default=175)
    ap.add_argument("--r2-sec", type=int, default=160)
    ap.add_argument("--r3-sec", type=int, default=175)
    ap.add_argument("--max-gap", type=float, default=0.0085)
    ap.add_argument("--seed-off", type=int, default=520000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )

    srcs = []
    for n in range(150, 1001):
        K = math.isqrt(n)
        if cur[n][0] < K + 1 - 1e-6 and n != K * K:
            srcs.append(n)
    my_srcs = srcs[args.shard :: args.num_shards]
    tp_tasks = [(n, cur[n][1]) for n in my_srcs]

    # Strictly isolated & time-bounded transplant generator (`<= 210s`, terminated cleanly!)
    tp_cands = []
    ex_init = ProcessPoolExecutor(max_workers=args.procs)
    try:
        t_init = time.time()
        futs_i = [
            ex_init.submit(w45f._try_gpeel_and_all_transplants, tk) for tk in tp_tasks
        ]
        try:
            for fi in as_completed(futs_i, timeout=210.0):
                try:
                    for n_new, S, tag, Q in fi.result():
                        gap = S - cur[n_new][0]
                        if gap < args.max_gap:
                            tp_cands.append((gap, n_new, S, tag, Q))
                except Exception:
                    continue
                if time.time() - t_init > 200.0:
                    break
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_init)

    tp_cands.sort(key=lambda z: z[0])
    seen_key = set()
    queued = []
    for gap, n_new, S, tag, Q in tp_cands:
        fam = (
            "gp"
            if tag.startswith("gp")
            else (
                "half"
                if tag.startswith("htp")
                else ("rtp" if tag.startswith("rtp") else "tp")
            )
        )
        if (n_new, fam) in seen_key:
            continue
        seen_key.add((n_new, fam))
        queued.append((gap, n_new, S, tag, Q))
        if len(queued) >= 10:
            break

    print(
        f"[W51F-GPeelDeep3R s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, S_init, tag, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 45.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            if S_init < L0_cur + 0.00025:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    n, Q_init, deadline, slp_sec=35.0, fq_sec=25
                )
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        tot_ds += (Lf0 - L0_cur) / math.sqrt(n)
                        wins[n] = (float(Lf0), xf0, yf0, tf0)
                        L0_cur = float(Lf0)
                        print(
                            f"  [W51F DIRECT POLISH WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf0:.7f}!",
                            flush=True,
                        )
                        save_ckpt()

            r1 = min(args.r1_sec, max(30, int(0.35 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(25, int(0.30 * (rem - 20.0))))
            r3 = min(args.r3_sec, max(25, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + args.shard * 10007 + i * 601
            Qb, Sb, (tr, ft, wn) = w51d.deep_3round_widehopper(
                ex_hop, n, Q_init, L0_cur, bseed, r1, r2, r3, deadline
            )
            if Sb < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, Qb)
                if ok and Lf < L0_cur - 1e-9:
                    dL = Lf - cur[n][0]
                    tot_ds += (Lf - L0_cur) / math.sqrt(n)
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W51F WIN!!] n={n:4d} [{tag}_wh28_3r] gap0={gap0:+.5f} tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W51F done] n={n:4d} [{tag}] gap0={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W51F s={args.shard} EXC] {exc}", flush=True)
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W51F s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
