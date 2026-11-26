#!/usr/bin/env python3
"""Wave 36B: Hopper-Powered 2D Bottleneck Gadget Surgery (BRAND-NEW ORTHOGONAL ENGINE!).

Combines our TWO strongest proven mechanisms into a single compounding weapon:
  1. Solves the live-weighted 13-bucket 4D Staircase DP over the full megapool to identify the
     EXACT top bottleneck gadgets (w, h, dm) blocking our current R26 frontier.
  2. Pads each bottleneck gadget (M = max(w, h) in [12..30]) to a valid (M + f0) square packing
     and attacks it directly with FAST Stock sq_engine.Hopper (2 seeds x 65s + cascade) +
     Trust-Region SLP + Rust fq micro-polish!
  3. Breaks bottleneck gadgets that fq/anneal alone could never move — and saves every improved
     padded gadget natively so Wave 37's Grand 18-Wave 4D-DP cascades it across 10-40 puzzles!
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
import w9a_bottleneck_surgery as w9a
import w32a_fresh_basin_fast_hopper as w32a

# Match Wave 36 13-bucket THR
f2d.THR = np.array(
    [
        0.0,
        0.30,
        0.45,
        0.52,
        0.58,
        0.64,
        0.70,
        0.75,
        0.80,
        0.85,
        0.89,
        0.93,
        0.96,
    ]
)
f2d.NB = len(f2d.THR)


def hopper_shrink_gadget(job):
    w, h, dm, f0, sq_crop, seed, hop_sec = job
    full, M = w9a.pad_to_square(sq_crop, w, h, f0)
    full = w32a.safe_open(full)
    L0 = float(E.side(full))
    best_S = L0
    best_P = None

    # Quick initial fq seam-weld (especially helpful for fused pairs!) before Stock Hopper
    if os.path.exists(f2d.FQ_BIN):
        with tempfile.TemporaryDirectory(prefix=f"w36b_pre_{w}_{h}_") as tmp_pre:
            r_pre = f2d.run_fq(
                L0 + 1e-4,
                full,
                tmp_pre,
                [
                    "--loosen",
                    "1.010",
                    "--kick",
                    "0.003",
                    "--seed",
                    str(seed & 0x3FFFFFFF),
                    "--flip-top",
                    "32",
                    "--pit",
                    "180",
                ],
                timeout=22,
            )
            if r_pre is not None and r_pre[0] < best_S - 1e-6:
                best_S, best_P = r_pre[0], w32a.safe_open(r_pre[1])

    # 2 diverse seeds of FAST Stock sq_engine.Hopper + SLP cascade directly on the padded gadget!
    for si in range(2):
        P_src = best_P if best_P is not None else full
        Q, S, _ = w32a.run_stock_hopper_chain(
            P_src,
            seed + si * 7919,
            first_sec=hop_sec,
            cascade_sec=min(90, hop_sec),
            max_cascades=2,
        )
        if S < best_S - 1e-6:
            best_S, best_P = S, Q

    if best_P is None or (L0 - best_S) < 5e-5:
        return None

    # Final SLP + fq polish
    cur_a = w32a.safe_open(best_P)
    cur_s = float(E.side(cur_a))
    try:
        xs, ys, ts, Ls = SLP.polish(
            cur_a[:, 0],
            cur_a[:, 1],
            cur_a[:, 2],
            iters=80,
            r0=2e-3,
            rmin=1e-10,
            tol_sep=-1e-13,
            time_limit=15.0,
        )
        Qs = w32a.safe_open(np.column_stack([xs, ys, ts]))
        Ss = float(E.side(Qs))
        if Ss < cur_s - 1e-9 and E.min_gap(Qs) >= -1e-12:
            cur_a, cur_s = Qs, Ss
    except Exception:
        pass
    if os.path.exists(f2d.FQ_BIN):
        with tempfile.TemporaryDirectory(prefix=f"w36b_{w}_{h}_") as tmp:
            r_fin = f2d.run_fq(
                cur_s,
                cur_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "48", "--pit", "400"],
                timeout=40,
            )
            if r_fin is not None and r_fin[0] < cur_s:
                cur_s, cur_a = r_fin[0], r_fin[1]

    df = L0 - float(E.side(cur_a))
    if df > 5e-5 and E.min_gap(cur_a) >= -1e-10:
        return (w, h, dm, df, cur_a.astype(np.float64))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--prev-glob", action="append", default=[])
    ap.add_argument("--dp-cache", default="")
    ap.add_argument("--lags-json", required=True)
    ap.add_argument("--mode", choices=("single", "pairs"), default="single")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2600)
    ap.add_argument("--hop-sec", type=int, default=65)
    ap.add_argument("--seed-off", type=int, default=360000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    if args.dp_cache and os.path.exists(args.dp_cache):
        import precompute_dp_cache as pc

        cache = pc.load_dp_cache(args.dp_cache)
        donors, cur_L = cache["donors"], cache["cur_L"]
        G, dp, PG, PM, PQ = (
            cache["G"],
            cache["dp"],
            cache["PG"],
            cache["PM"],
            cache["PQ"],
        )
    else:
        donors, cur_L, n_base, n_prev = w9a.load_pooled_donors(
            args.union_npz, args.prev_glob, args.win_glob
        )
        G = f2d.extract_2d_gadgets(donors)
        dp, PG, PM, PQ = f2d.run_2d_dp(G)

    lags = json.load(open(args.lags_json)) if os.path.exists(args.lags_json) else []
    lag_wt = {}
    for it in lags:
        nn = int(it["n"])
        lb = float(it.get("Lb", cur_L.get(nn, 0.0)))
        live_ds = max(0.0, (cur_L.get(nn, lb) - lb) / math.sqrt(nn))
        if live_ds > 1e-7:
            lag_wt[nn] = live_ds
    grid_breaks = {605, 762, 819, 877, 1000}

    bn_weight = {}
    pair_weight = {}
    for n in range(120, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D == 0 or D > f2d.DMAX:
            continue
        if cur_L[n] >= K + 1 - 1e-7 and n not in grid_breaks:
            continue
        wt = lag_wt.get(n, 2e-6)
        chains = w9a.extract_top_chains(
            dp,
            PG,
            PM,
            PQ,
            G,
            K,
            D,
            max_chains=3 if (n in lag_wt or n in grid_breaks) else 1,
        )
        for f_ch, links in chains:
            if (K + f_ch) > cur_L[n] + 0.015:
                continue
            for idx_l, (g, mode) in enumerate(links):
                fg = float(G[g, 3])
                if fg >= f_ch - 0.009:
                    bn_weight[g] = bn_weight.get(g, 0.0) + wt
                if idx_l + 1 < len(links):
                    g2, mode2 = links[idx_l + 1]
                    if max(fg, float(G[g2, 3])) >= f_ch - 0.009:
                        pk = (g, g2, mode2)
                        pair_weight[pk] = pair_weight.get(pk, 0.0) + wt

    jobs = []
    if args.mode == "pairs":
        ranked = sorted(pair_weight.items(), key=lambda x: -x[1])
        my_slice = ranked[args.shard :: args.num_shards][:120]
        for i, ((g1, g2, mode2), wt) in enumerate(my_slice):
            comp = w9a.compose_pair_rect(donors, G, g1, g2, mode2)
            if comp is not None:
                W, H, dm_sum, f0, sq_pair = comp
                jobs.append(
                    (
                        W,
                        H,
                        dm_sum,
                        float(f0),
                        sq_pair,
                        args.seed_off + args.shard * 10007 + i * 211,
                        args.hop_sec,
                    )
                )
    else:
        # Focus on M in [7..30] gadgets (unlocking K^2+1 sub-0.50 & mid-size high-leverage bottlenecks!)
        ranked = []
        for gi, wt in sorted(bn_weight.items(), key=lambda x: -x[1]):
            w, h, dm = int(G[gi, 0]), int(G[gi, 1]), int(G[gi, 2])
            if 7 <= max(w, h) <= 30 and float(G[gi, 3]) >= 0.45:
                ranked.append((gi, wt))
        my_slice = ranked[args.shard :: args.num_shards][:140]
        for i, (gi, wt) in enumerate(my_slice):
            w, h, dm, f, _, _, di, si = G[gi]
            w, h, dm, di, si = int(w), int(h), int(dm), int(di), int(si)
            sq_crop, _ = f2d.crop_gadget(donors[di], si, w, h, dm)
            jobs.append(
                (
                    w,
                    h,
                    dm,
                    float(f),
                    sq_crop,
                    args.seed_off + args.shard * 10007 + i * 197,
                    args.hop_sec,
                )
            )

    print(
        f"[W36B-HopGadget-{args.mode} s={args.shard}/{args.num_shards}] donors={len(donors)} G={len(G)} "
        f"eligible={len(ranked)} queued={len(jobs)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    out_dict = {}
    dfs = []
    done = 0

    def save_ckpt():
        np.savez_compressed(args.out_npz, **out_dict)

    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(hopper_shrink_gadget, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                res = fut.result()
            except Exception as exc:
                print(f"  [W36B ERR] {exc}", flush=True)
                continue
            if res is not None:
                w, h, dm, df, arr = res
                key = f"ghop_{w}x{h}_d{dm}_{len(out_dict)}"
                out_dict[key] = arr
                dfs.append(df)
                print(
                    f"  [W36B HIT!!] {w:2d}x{h:2d} dm={dm:2d} df=-{df:.5f} (hits={len(out_dict)}, {done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                    flush=True,
                )
                save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W36B s={args.shard}] Graceful budget {args.budget_sec}s reached.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    max_df = max(dfs) if dfs else 0.0
    avg_df = (sum(dfs) / len(dfs)) if dfs else 0.0
    print(
        f"[W36B-HopGadget s={args.shard} DONE] Saved {len(out_dict)}/{done} Hopper gadget breakers -> {args.out_npz} "
        f"({time.time() - t0:.1f}s), max_df={max_df:.5f}, avg_df={avg_df:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
