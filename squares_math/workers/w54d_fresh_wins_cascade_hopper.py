#!/usr/bin/env python3
"""Wave 54D: Secondary Basin-Jump Wider-Hopper + Exhaustive `delete-1` Cascade on Newly-Improved R33/W53 Packings!

Implements `goodpjw2008` v50's highest-yield empirical rule (`NOTES.md` line 228: `n=545` jumped `-3.0e-4`
on a second round of wider-move hopper from a newly discovered basin + `delete-1` compaction from fresh winners!):
  - Targets ALL newly improved packings (`n=380`, `n=702`, `n=758`, plus any fresh Wave 53 winners where
    `cur[n][0] < base_only[n][0] - 1e-6`), running:
    1. Multi-seed secondary `eta<=0.28` Wider-Hopper (`r1=280s, r2=240s, r3=240s`) directly on `P[n]`
       (8 independent seeds sharded across 8 shards for `380, 702, 758`!)
    2. Exhaustive all-near-band `delete-1` compaction (`380 -> 379`, `702 -> 701`, `758 -> 757`, plus
       our top private donors `817->816, 915->914, 794->793, 633->632, 500->499, 456->455, 299->298`)
       followed by 3-Round Deep-Polish `eta<=0.28` Wider-Hopper!
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
import w50d_exhaustive_private_del1_widehopper as w50d
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r

FRESH_RECORD_NS = [
    380,
    702,
    758,
    703,
    921,
    556,
    451,
    378,
    851,
    978,
    499,
    548,
    804,
    699,
    748,
    376,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2220)
    ap.add_argument("--r1-sec", type=int, default=280)
    ap.add_argument("--r2-sec", type=int, default=240)
    ap.add_argument("--r3-sec", type=int, default=240)
    ap.add_argument("--seed-off", type=int, default=590000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )

    # Combine explicit recent winners + any fresh wins loaded in megapool
    pool_ns = []
    for n in FRESH_RECORD_NS + sorted(fresh, reverse=True):
        if n not in pool_ns and 100 <= n <= 1000:
            K = math.isqrt(n)
            if K * K != n and cur[n][0] < K + 1 - 1e-6:
                pool_ns.append(n)

    # Every shard gets `n=380` (with shard-unique seed!) PLUS its own slice of `pool_ns` for direct + delete-1!
    my_direct = [380] if args.shard < 4 else [702 if args.shard < 6 else 758]
    for n in pool_ns[args.shard :: args.num_shards]:
        if n not in my_direct:
            my_direct.append(n)

    # Phase A: Screen exhaustive `delete-1` (`n_donor -> n_donor - 1`) from `my_direct[:4]`!
    del1_cands = []
    ex_del = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for dn in my_direct[:4]:
            nt = dn - 1
            if nt < 100 or math.isqrt(nt) ** 2 == nt:
                continue
            P_dn = cur[dn][1]
            L0_nt = cur[nt][0]
            idxs = w50d.all_near_band_indices(P_dn, max_cand=240)
            futs = [
                ex_del.submit(w51d.eval_stage1, (P_dn, (idx,), f"del1_{dn}_i{idx}"))
                for idx in idxs
            ]
            st1 = []
            try:
                for f in as_completed(futs, timeout=45.0):
                    tg, s1, q1 = f.result()
                    if q1 is not None and s1 < L0_nt + 0.025:
                        st1.append((s1, tg, q1))
            except TimeoutError:
                w51d.terminate_pool(ex_del)
                ex_del = ProcessPoolExecutor(max_workers=args.procs)
            st1.sort(key=lambda z: z[0])
            futs2 = [
                ex_del.submit(w51d.eval_stage2, (tg, q1)) for _, tg, q1 in st1[:12]
            ]
            st2 = []
            try:
                for f in as_completed(futs2, timeout=35.0):
                    tg, s2, q2 = f.result()
                    if q2 is not None and s2 < L0_nt + 0.012:
                        st2.append((s2 - L0_nt, nt, s2, tg, q2))
            except TimeoutError:
                w51d.terminate_pool(ex_del)
                ex_del = ProcessPoolExecutor(max_workers=args.procs)
            st2.sort(key=lambda z: z[0])
            for item in st2[:2]:
                del1_cands.append(item)
    finally:
        w51d.terminate_pool(ex_del)

    # Build Phase B queue: first top `delete-1` cascades (`380->379`, `702->701`, `758->757`), then fresh-seed secondary jumps on `380, 702, 758, ...`!
    del1_cands.sort(key=lambda z: z[0])
    queued = list(del1_cands[:4])
    for dn in my_direct[:5]:
        queued.append((0.0, dn, cur[dn][0], f"secjump_{dn}_s{args.shard}", cur[dn][1]))

    print(
        f"[W54D-FreshCascade s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"del1_hits={len(del1_cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
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
            if rem < 50.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            r1 = min(args.r1_sec, max(35, int(0.36 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(30, int(0.32 * (rem - 20.0))))
            r3 = min(args.r3_sec, max(30, int(0.32 * (rem - 20.0))))
            bseed = args.seed_off + args.shard * 15013 + i * 709
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
                        f"  [W54D WIN!!] n={n:4d} [{tag}_wh28] gap0={gap0:+.5f} tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W54D done] n={n:4d} [{tag}] gap0={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W54D s={args.shard} EXC] {exc}", flush=True)
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W54D s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
