#!/usr/bin/env python3
"""Wave 6B Engine (48 Cores / 12 Shards): Unified Pooled 4D Staircase Multi-Gadget DP + Deep Seam-fq Quench!
Loads ALL 1000 R5 packings PLUS ALL ~5,000 pooled tightened gadgets from Wave 6A into ONE SINGLE global
Pareto gadget table, runs ONE global 4D Staircase DP, and fans out deep seam-fq + kick-chain quenching
across all lagging puzzles n=100..1000!
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
sys.path.insert(0, DS_DIR)
sys.path.insert(0, HERE)

import w5b_2d_faultdp_gadgetq as f2d


def quench_composed_deep(job):
    n, s0, sq_comp, f_dp, seed = job
    rng = random.Random(seed)
    K = math.isqrt(n)
    best_s = s0
    best_a = None
    with tempfile.TemporaryDirectory() as tmp:
        # 1) Exact composed packing directly
        ok0, L0, x0, y0, t0 = f2d.stabilize_decimal(n, sq_comp)
        if ok0 and L0 < best_s - 1e-10:
            best_s = L0
            best_a = np.column_stack([x0, y0, t0])
        # 2) Multi-schedule seam-fq quench
        s_in = K + f_dp + 1e-5
        local_best = None
        for sched in (
            ["--loosen", "1.0", "--flip-top", "32", "--pit", "200"],
            [
                "--loosen",
                "1.012",
                "--kick",
                "0.004",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "32",
                "--pit",
                "220",
            ],
            [
                "--loosen",
                "1.025",
                "--kick",
                "0.01",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "32",
                "--pit",
                "220",
            ],
            [
                "--loosen",
                "1.0",
                "--round-r0",
                "0.08",
                "--round-compress",
                "--round-press",
                "--round-steps",
                "5",
                "--round-press-outer",
                "8",
                "--flip-top",
                "24",
                "--pit",
                "120",
            ],
        ):
            r = f2d.run_fq(s_in, sq_comp, tmp, sched, timeout=55)
            if r is not None:
                if local_best is None or r[0] < local_best[0]:
                    local_best = r
                if r[0] < best_s - 1e-10:
                    best_s, best_a = r[0], r[1]
        # 3) Follow-up kick/loosen chain if won or within +0.006 near-miss!
        if local_best is not None and local_best[0] < best_s + 0.006:
            ls, la = local_best
            for _ in range(3):
                r2 = f2d.run_fq(
                    ls,
                    la,
                    tmp,
                    [
                        "--loosen",
                        rng.choice(("1.005", "1.012", "1.022")),
                        "--kick",
                        rng.choice(("0.002", "0.006")),
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "48",
                        "--pit",
                        "300",
                    ],
                    timeout=55,
                )
                if r2 and r2[0] < ls:
                    ls, la = r2
                if r2 and r2[0] < best_s - 1e-10:
                    best_s, best_a = r2[0], r2[1]
        if best_a is not None and best_s < s0 - 1e-10:
            r_fin = f2d.run_fq(
                best_s,
                best_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
                timeout=80,
            )
            if r_fin and r_fin[0] < best_s:
                best_s, best_a = r_fin[0], r_fin[1]
            ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, best_a)
            if ok and Lf < s0 - 1e-10:
                return n, (Lf, xf, yf, tf)
    return n, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--gadget-glob", required=True)
    ap.add_argument("--lags-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    d = np.load(args.union_npz)
    x, y, t, L = d["x"], d["y"], d["t"], d["L"]
    offs = [0]
    for n in range(1, 1001):
        offs.append(offs[-1] + n)

    donors = []
    cur = {}
    for n in range(1, 1001):
        s0, e0 = offs[n - 1], offs[n]
        sq = np.column_stack([x[s0:e0], y[s0:e0], t[s0:e0]])
        cur[n] = (float(L[n - 1]), sq)
        if float(L[n - 1]) < math.ceil(math.sqrt(n)) - 1e-7:
            donors.append(sq)

    # Load ALL pooled tightened gadgets from Wave 6A shards!
    g_files = sorted(glob.glob(args.gadget_glob, recursive=True))
    n_pooled = 0
    for gf in g_files:
        gd = np.load(gf)
        for k in gd.files:
            donors.append(gd[k])
            n_pooled += 1
    print(
        f"[W6B Shard {args.shard}/{args.num_shards}] Loaded {len(donors)} total donors ({n_pooled} pooled shrunk gadgets from {len(g_files)} files, {time.time() - t0:.1f}s)",
        flush=True,
    )

    G = f2d.extract_2d_gadgets(donors)
    dp, PG, PM, PQ = f2d.run_2d_dp(G)
    print(
        f"[W6B Shard {args.shard}] Global Pareto 2D gadgets: {len(G)}, 4D DP solved ({time.time() - t0:.1f}s)",
        flush=True,
    )

    # Check ALL n=100..1000 (not just lags!) where 4D DP lands within +0.024 of current L!
    lags = {r["n"]: r["dS"] for r in json.load(open(args.lags_json))}
    all_comp = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D == 0 or K > f2d.WMAX or D > f2d.DMAX:
            continue
        s0 = cur[n][0]
        f_dp, links = f2d.extract_chain(dp, PG, PM, PQ, G, K, D)
        if f_dp is not None and K + f_dp < s0 + 0.024:
            # Sort by (K+f_dp - s0) ascending so exact DP wins & closest near-misses run first!
            all_comp.append((K + f_dp - s0, -lags.get(n, 0.0), n, s0, f_dp, links))
    all_comp.sort()
    my_comp = [
        c for idx, c in enumerate(all_comp) if idx % args.num_shards == args.shard
    ]
    jobs = []
    for gap_est, _, n, s0, f_dp, links in my_comp:
        sq_c = f2d.compose_packing(
            math.isqrt(n), n - math.isqrt(n) ** 2, f_dp, links, G, donors
        )
        jobs.append((n, s0, sq_c, f_dp, 20261129 + 919 * n + args.shard))

    print(
        f"[W6B Shard {args.shard}] Deep seam-fq quenching {len(jobs)}/{len(all_comp)} global 4D-DP compositions (best raw gap={my_comp[0][0] if my_comp else 0:.6f})...",
        flush=True,
    )
    improved = {}
    tot_gain = 0.0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(quench_composed_deep, j) for j in jobs]
        for fut in as_completed(futs):
            n, res = fut.result()
            if res is not None:
                Lf, xf, yf, tf = res
                s0 = cur[n][0]
                gain = (s0 - Lf) / math.sqrt(n)
                tot_gain += gain
                improved[n] = (Lf, xf, yf, tf)
                print(
                    f"W6B_POOLED_WIN n={n:4d} {s0:.9f} -> {Lf:.9f} dL={s0 - Lf:.6f} dS={gain:.6f} tot={tot_gain:.6f}",
                    flush=True,
                )

    payload = {}
    for n, (Lf, xf, yf, tf) in improved.items():
        payload[f"L_{n}"] = np.array([Lf], dtype=np.float64)
        payload[f"x_{n}"] = xf
        payload[f"y_{n}"] = yf
        payload[f"t_{n}"] = tf
    np.savez_compressed(args.out_npz, **payload)
    print(
        f"DONE W6B Pooled 4D-DP Shard {args.shard}: {len(improved)} wins, total_gain={tot_gain:.6f} -> {args.out_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
