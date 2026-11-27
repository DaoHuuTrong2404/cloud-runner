#!/usr/bin/env python3
"""Wave 56 Stage 1 (`w56_gadget_forge.py`): Global Distributed Rectangle-Gadget Forge!

16 shards x 4 cores = 64 cores x 6 minutes:
  - Extracts all ~16,515 8-sym `(w, h, dm)` gadgets from `pub_base_v5.npz` + `megapool_w56.tar`.
  - Ranks the Top 640 active & near-active Pareto rectangle gadgets across all multi-gadget DP chains.
  - Each shard quenches 40 gadgets across 6 schedules in compiled Rust `fq quench --rect-dx (w-m) --rect-dy (h-m)`
    INCLUDING rounded-square soft-prestage `--round-r0 0.08 --round-steps 8 --finish --stag-w 0` so jammed
    sub-rectangles can rearrange internal squares and project onto their exact KKT minimum!
  - Writes strictly improved `(w, h, dm)` square arrays into `--out-npz` for Stage 2 global unification!
"""

import argparse
import json
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

import fault_dp_8sym as fdp8
import rust_engines as RE
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53g_rect_quench_8sym_faultdp as w53g


def quench_rect_gadget_deep(args_tuple):
    key, sq_init, seed0 = args_tuple
    w, h, dm = key
    m = min(int(w), int(h))
    dx, dy = float(w - m), float(h - m)
    f0 = RE.gadget_overhang(sq_init, w, h)
    best_sq, best_f = RE.normalize_to_origin(sq_init), f0

    # 4 rigid kick/loosen passes + 2 rounded-square soft-prestage passes, all with `--finish --stag-w 0`!
    trials = [
        [
            "--rect-dx",
            repr(dx),
            "--rect-dy",
            repr(dy),
            "--loosen",
            "1.000",
            "--no-alm",
            "--finish",
            "--stag-w",
            "0",
            "--rmin",
            "1e-12",
            "--flip-top",
            "32",
            "--pit",
            "450",
            "--deadline",
            "10.0",
        ],
        [
            "--rect-dx",
            repr(dx),
            "--rect-dy",
            repr(dy),
            "--seed",
            str(seed0 + 11),
            "--loosen",
            "1.010",
            "--kick",
            "0.0045",
            "--outer",
            "50",
            "--pit",
            "260",
            "--flip-top",
            "24",
            "--finish",
            "--stag-w",
            "0",
            "--deadline",
            "12.0",
        ],
        [
            "--rect-dx",
            repr(dx),
            "--rect-dy",
            repr(dy),
            "--seed",
            str(seed0 + 29),
            "--loosen",
            "1.020",
            "--kick",
            "0.0090",
            "--outer",
            "55",
            "--pit",
            "260",
            "--flip-top",
            "24",
            "--finish",
            "--stag-w",
            "0",
            "--deadline",
            "12.0",
        ],
        [
            "--rect-dx",
            repr(dx),
            "--rect-dy",
            repr(dy),
            "--seed",
            str(seed0 + 53),
            "--loosen",
            "1.015",
            "--kick",
            "0.0065",
            "--sqp",
            "--outer",
            "45",
            "--pit",
            "220",
            "--flip-top",
            "16",
            "--finish",
            "--stag-w",
            "0",
            "--deadline",
            "12.0",
        ],
        [
            "--rect-dx",
            repr(dx),
            "--rect-dy",
            repr(dy),
            "--seed",
            str(seed0 + 79),
            "--round-r0",
            "0.085",
            "--round-steps",
            "8",
            "--round-maxit",
            "900",
            "--round-gtol",
            "1e-8",
            "--round-kick",
            "0.0035",
            "--hold-f",
            "1.002",
            "--loosen",
            "1.006",
            "--outer",
            "45",
            "--pit",
            "240",
            "--flip-top",
            "16",
            "--finish",
            "--stag-w",
            "0",
            "--deadline",
            "13.0",
        ],
        [
            "--rect-dx",
            repr(dx),
            "--rect-dy",
            repr(dy),
            "--seed",
            str(seed0 + 107),
            "--soft-a0",
            "0.075",
            "--soft-steps",
            "8",
            "--soft-kick",
            "0.0030",
            "--soft-compress",
            "--hold-f",
            "1.002",
            "--loosen",
            "1.006",
            "--outer",
            "45",
            "--pit",
            "240",
            "--flip-top",
            "16",
            "--finish",
            "--stag-w",
            "0",
            "--deadline",
            "13.0",
        ],
    ]

    for extra in trials:
        s_in = float(m) + RE.gadget_overhang(best_sq, w, h) + 1e-9
        rq = RE.run_fq_quench_args(best_sq, extra, s_ref=s_in, timeout_s=15.0)
        if rq is None:
            continue
        _, Q_cand, _ = rq
        if len(Q_cand) != w * h + dm:
            continue
        fc = RE.gadget_overhang(Q_cand, w, h)
        if fc < best_f - 1e-9:
            best_sq, best_f = Q_cand, fc

    return key, f0, best_f, best_sq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--max-keys-per-shard", type=int, default=40)
    ap.add_argument("--budget-sec", type=int, default=370)
    ap.add_argument("--seed-off", type=int, default=700000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cur, _, tot_score, _ = w51r.load_full_closed_frontier(args.union_npz, args.win_glob)
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    donors = []
    for n in range(5, 1001):
        K = math.isqrt(n)
        if K * K != n and cur[n][0] < K + 1 - 1e-6 and cur[n][1] is not None:
            donors.append(cur[n][1])

    G = fdp8.extract_gadgets(donors)
    dp0, PG0, PM0, PQ0 = fdp8.solve_dp(G)

    all_targets = []
    for n in range(80, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D <= 0 or D > fdp8.DMAX or K > fdp8.WMAX:
            continue
        gap_ldr = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        all_targets.append((gap_ldr / math.sqrt(n), n, K, D))
    all_targets.sort(reverse=True)

    active_keys = w53g.collect_chain_gadget_keys(
        dp0, PG0, PM0, PQ0, G, all_targets[:450]
    )
    # Also add any remaining non-trivial Pareto keys so we reach up to 640 keys globally!
    seen_k = set(active_keys)
    pareto_best = {}
    for row in G:
        gw, gh, gdm, gf = int(row[0]), int(row[1]), int(row[2]), float(row[3])
        if gw * gh + gdm >= 10 and gdm >= 1 and 0.40 < gf < 0.995:
            k = (gw, gh, gdm)
            if k not in pareto_best or gf < pareto_best[k]:
                pareto_best[k] = gf
    for k, _ in sorted(pareto_best.items(), key=lambda kv: kv[1]):
        if k not in seen_k:
            seen_k.add(k)
            active_keys.append(k)

    my_keys = active_keys[args.shard :: args.num_shards][: args.max_keys_per_shard]
    q_jobs = []
    for ki, key in enumerate(my_keys):
        rows = np.where(
            (G[:, 0].astype(int) == key[0])
            & (G[:, 1].astype(int) == key[1])
            & (G[:, 2].astype(int) == key[2])
        )[0]
        if not len(rows):
            continue
        best_r = G[rows[int(np.argmin(G[rows, 3]))]]
        sq_g, _ = fdp8.crop_gadget(
            donors[int(best_r[6])], int(best_r[7]), key[0], key[1], key[2]
        )
        if sq_g is not None:
            q_jobs.append((key, sq_g, args.seed_off + args.shard * 5003 + ki * 109))

    print(
        f"[W56-Forge s={args.shard}/{args.num_shards}] total_active={len(active_keys)} my_jobs={len(q_jobs)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    improved = []
    ex = ProcessPoolExecutor(max_workers=args.procs)
    try:
        futs = [ex.submit(quench_rect_gadget_deep, jb) for jb in q_jobs]
        rem = max(30.0, args.budget_sec - (time.time() - t0))
        try:
            for f in as_completed(futs, timeout=rem):
                key, f0, f1, sq1 = f.result()
                if f1 < f0 - 1e-7:
                    improved.append((key, f0, f1, sq1))
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex)

    improved.sort(key=lambda z: z[2] - z[1])
    print(
        f"[W56-Forge s={args.shard}] IMPROVED {len(improved)}/{len(q_jobs)} gadgets in {time.time() - t0:.1f}s!",
        flush=True,
    )
    for key, f0, f1, _ in improved[:12]:
        print(
            f"  [Forge win] {key[0]}x{key[1]}+{key[2]}: f {f0:.6f} -> {f1:.6f} (df={f1 - f0:+.6f})",
            flush=True,
        )

    pl = {
        "keys": np.array(
            [[k[0], k[1], k[2]] for k, _, _, _ in improved], dtype=np.int32
        ),
        "f_old": np.array([f0 for _, f0, _, _ in improved], dtype=np.float64),
        "f_new": np.array([f1 for _, _, f1, _ in improved], dtype=np.float64),
    }
    for idx, (_, _, _, sq1) in enumerate(improved):
        pl[f"sq_{idx}"] = np.asarray(sq1, dtype=np.float64)
    np.savez_compressed(args.out_npz, **pl)


if __name__ == "__main__":
    main()
