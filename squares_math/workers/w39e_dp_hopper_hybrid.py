#!/usr/bin/env python3
"""Wave 39E: 18-Bucket 4D-DP -> Stock sq_engine.Hopper HYBRID ENGINE!

Fuses our TWO strongest proven global + mesoscale engines in a single zero-startup pass:
  1. Loads the precomputed 18-bucket Numba 4D-DP cache in <1 second (0s startup!).
  2. Extracts diverse 4D-DP staircase topologies across all lagging puzzles (both exact
     wins AND near-misses within gap < +0.0075 that pure Rust `fq` alone threw away!).
  3. Seam-welds each fresh 4D-DP topology with fast Rust `fq` (closing the E-epsilon DP
     seam gap), then IMMEDIATELY unleashes FAST Stock sq_engine.Hopper (rigid-body contact LP
     band tilts + loose seam hops!) + SLP cascades + exact Decimal64 >= 1.05e-9 certification!
"""

import argparse
import json
import math
import os
import random
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
import w8b_megapool_dp_seamfq as w8b
import w32a_fresh_basin_fast_hopper as w32a
import precompute_dp_cache as pc


def worker_dp_hopper(job):
    n, s0, sq_comp, f_dp, seed, hop_sec = job
    rng = random.Random(seed)
    K = math.isqrt(n)
    best_s = s0
    best_a = None
    best_tag = ""

    with tempfile.TemporaryDirectory(prefix=f"w39e_{n}_") as tmp:
        # Step 1: Fast 2-schedule fq seam-weld on the freshly composed 4D-DP packing
        s_in = K + f_dp + 1e-5
        welded = None
        for sched in (
            ["--loosen", "1.0", "--flip-top", "32", "--pit", "180"],
            [
                "--loosen",
                "1.015",
                "--kick",
                "0.005",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "32",
                "--pit",
                "200",
            ],
        ):
            r = f2d.run_fq(s_in, sq_comp, tmp, sched, timeout=40)
            if r is not None:
                if welded is None or r[0] < welded[0]:
                    welded = r
                if r[0] < best_s - 1e-10:
                    best_s, best_a, best_tag = r[0], r[1], "dp_fq"

        # Step 2: Feed seam-welded DP topology (or raw sq_comp if close) into FAST Stock Hopper!
        if welded is not None and welded[0] < s0 + 0.0085:
            P_hop_in = w32a.safe_open(welded[1])
        else:
            P_hop_in = w32a.safe_open(sq_comp)

        if float(E.side(P_hop_in)) < s0 + 0.010:
            for si in range(2):
                P_src = w32a.safe_open(best_a) if best_a is not None else P_hop_in
                Q, S, _ = w32a.run_stock_hopper_chain(
                    P_src,
                    seed + si * 7919,
                    first_sec=hop_sec,
                    cascade_sec=min(120, hop_sec),
                    max_cascades=3,
                )
                if S < best_s - 1e-9:
                    best_s, best_a, best_tag = S, Q, f"dp_hop_s{si}"

        # Step 3: Final SLP + fq + lp_rot + Decimal64 >= 1.05e-9 Certification!
        if best_a is not None and best_s < s0 - 1e-10:
            cur_a = w32a.safe_open(best_a)
            cur_s = float(E.side(cur_a))
            try:
                xs, ys, ts, Ls = SLP.polish(
                    cur_a[:, 0],
                    cur_a[:, 1],
                    cur_a[:, 2],
                    iters=100,
                    r0=2e-3,
                    rmin=1e-10,
                    tol_sep=-1e-13,
                    time_limit=20.0,
                )
                Qs = w32a.safe_open(np.column_stack([xs, ys, ts]))
                Ss = float(E.side(Qs))
                if Ss < cur_s - 1e-10 and E.min_gap(Qs) >= -1e-12:
                    cur_a, cur_s = Qs, Ss
            except Exception:
                pass
            if os.path.exists(f2d.FQ_BIN):
                r_fin = f2d.run_fq(
                    cur_s,
                    cur_a,
                    tmp,
                    ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "550"],
                    timeout=65,
                )
                if r_fin is not None and r_fin[0] < cur_s:
                    cur_s, cur_a = r_fin[0], r_fin[1]
            try:
                cur_a = w8b.lp_rot_polish(cur_a)
            except Exception:
                pass
            ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_a)
            if ok and Lf < s0 - 1e-10:
                return n, best_tag, (Lf, xf, yf, tf)
    return n, best_tag, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp-cache", required=True)
    ap.add_argument("--lags-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2650)
    ap.add_argument("--hop-sec", type=int, default=85)
    ap.add_argument("--seed-off", type=int, default=390000)
    ap.add_argument("--max-cands", type=int, default=240)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cache = pc.load_dp_cache(args.dp_cache)
    cur, donors = cache["cur"], cache["donors"]
    G, dp, PG, PM, PQ = cache["G"], cache["dp"], cache["PG"], cache["PM"], cache["PQ"]
    score = sum(cur[n][0] / math.sqrt(n) - 1.0 for n in range(1, 1001))

    lags = json.load(open(args.lags_json)) if os.path.exists(args.lags_json) else []
    live_lag = {}
    for it in lags:
        nn = int(it["n"])
        lb = float(it.get("Lb", cur[nn][0]))
        ds = max(0.0, (cur[nn][0] - lb) / math.sqrt(nn))
        if ds > 1e-7:
            live_lag[nn] = ds
    grid_breaks = {605, 762, 819, 877, 1000}

    cands = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D == 0 or D > f2d.DMAX:
            continue
        s0 = cur[n][0]
        if s0 >= K + 1 - 1e-7 and n not in grid_breaks:
            continue
        is_vip = (n in live_lag) or (n in grid_breaks)
        chains = w8b.extract_diverse_chains(
            dp, PG, PM, PQ, G, K, D, max_chains=2 if is_vip else 1
        )
        for ci, (f_dp, links, fp) in enumerate(chains):
            gap = (K + f_dp) - s0
            lim = 0.0075 if is_vip else 0.0030
            if gap < lim:
                sq_comp = f2d.compose_packing(K, D, f_dp, links, G, donors)
                if sq_comp is not None and len(sq_comp) == n:
                    # Prioritize exact wins & smallest gap weighted by live lag!
                    prio = gap - 15.0 * live_lag.get(n, 0.0)
                    cands.append((prio, gap, n, s0, sq_comp, f_dp, ci))

    cands.sort(key=lambda z: (z[0], z[1], -z[2]))
    cands = cands[: args.max_cands]
    my_cands = cands[args.shard :: args.num_shards]
    print(
        f"[W39E-DPHopper s={args.shard}/{args.num_shards}] INSTANT dp_cache loaded "
        f"(G={len(G)}, score={score:.10f}, total_cands={len(cands)}, my_slice={len(my_cands)}, {time.time() - t0:.1f}s)",
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

    job_args = [
        (
            n,
            s0,
            sq_comp,
            f_dp,
            args.seed_off + args.shard * 10007 + i * 73,
            args.hop_sec,
        )
        for i, (_, _, n, s0, sq_comp, f_dp, _) in enumerate(my_cands)
    ]

    done = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_dp_hopper, j) for j in job_args]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W39E ERR] {exc}", flush=True)
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
                        f"  [W39E WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"(dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{done}/{len(job_args)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W39E s={args.shard}] Graceful budget {args.budget_sec}s reached.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W39E-DPHopper s={args.shard} DONE] {len(wins)} certified wins, "
        f"total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
