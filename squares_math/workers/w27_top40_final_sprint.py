#!/usr/bin/env python3
"""Wave 27 Lightning Top-40 Final Sprint (112 Cores / 28 Shards, ~28 min!).

Ultra-focused 28-minute sprint across 224 high-potential packings (8 puzzles/shard, ~3.5 min/puzzle -> 0 timeouts!)
targeting our remaining -0.000176 gap to shatter LIVE TOP 40 (< 22.085883):
  - Fresh R19/W26A basins (n=683 -0.00170, 923 -0.00029, 405 -0.00365, 298, 264, 500, 603, 499, 746, 642, 593)
  - Top 210 n=220..950 packings with f in [0.58, 0.996] ranked by lag + Fixed-D/K bumps
Uses ONLY the two highest-ROI explosive phase-transition operators (new RNG seeds 2700000+):
  1) 8-Point Dense Round-Press Sweep (r0 in 0.04, 0.06, 0.08, 0.11, 0.15, 0.20, 0.26, 0.34, L0+0.012..0.032)
  2) 6-Point Box-Expanded Multi-Center Melt (expand 0.025..0.125, rad 3.5..11.5) + Round-Press Re-Cryst
  3) Automatic follow-up 6-step kick chain + 2nd cascade round on any win + Contact LP + Decimal64 >= 1.05e-9!
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
sys.path.insert(0, DS_DIR)
sys.path.insert(0, HERE)

import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12

ANNEAL_BIN = os.environ.get("ANNEAL_BIN", "/tmp/sq_bins/anneal")


def lightning_round(n, L0, sq, rng, tmp, r_idx):
    best_s = L0
    best_a = None
    best_tag = ""
    near_best = None

    # 1) 8-Point Dense Round-Press Sweep (exact winner for n=593 -0.00610, n=692 -0.00120!)
    for r0, pad in (
        ("0.04", 0.010),
        ("0.06", 0.012),
        ("0.08", 0.014),
        ("0.11", 0.016),
        ("0.15", 0.018),
        ("0.20", 0.022),
        ("0.26", 0.026),
        ("0.34", 0.032),
    ):
        src_sq = best_a if best_a is not None else sq
        src_L = best_s if best_a is not None else L0
        r = f2d.run_fq(
            src_L + pad,
            src_sq,
            tmp,
            [
                "--loosen",
                "1.0",
                "--round-r0",
                r0,
                "--round-compress",
                "--round-press",
                "--round-steps",
                "8",
                "--round-press-outer",
                "14",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "38",
                "--pit",
                "200",
            ],
            timeout=52,
        )
        if r is not None:
            if near_best is None or r[0] < near_best[0]:
                near_best = (r[0], r[1], f"rp27_{r0}")
            if r[0] < best_s - 1e-9:
                best_s, best_a, best_tag = r[0], r[1], f"rp27_{r0}"

    # 2) 6-Point Box-Expanded Single/Dual-Center Melt (exact winner for n=642 -0.00666, n=499 -0.00229!)
    if os.path.exists(ANNEAL_BIN):
        degs = np.degrees(sq[:, 2]) % 90.0
        tilted_idx = np.flatnonzero(np.minimum(degs, 90.0 - degs) > 2.0)
        cap_r = max(3.8, 0.72 * L0)
        cfgs = [
            (3.0, min(5.8, cap_r), 0.11, 0.24, 4200, 1, 0.025, False),
            (4.8, min(8.8, cap_r), 0.15, 0.32, 4600, 1, 0.050, False),
            (5.8, min(11.0, cap_r), 0.17, 0.38, 4800, 1, 0.085, True),
            (3.8, min(7.2, cap_r), 0.14, 0.28, 3400, 2, 0.040, False),
            (5.2, min(9.5, cap_r), 0.16, 0.35, 3600, 2, 0.075, True),
            (6.0, min(11.5, cap_r), 0.18, 0.40, 3800, 2, 0.115, True),
        ]
        for rmin, rmax_rad, m0, m1, swp, n_c, expand, use_rp in cfgs:
            if rmax_rad <= rmin:
                rmin = 0.5 * rmax_rad
            try:
                pin = os.path.join(tmp, "min.txt")
                pout = os.path.join(tmp, "mout.txt")
                src_sq = best_a if best_a is not None else sq
                src_L = best_s if best_a is not None else L0
                L_in = src_L + rng.uniform(0.75 * expand, 1.25 * expand)
                f2d.write_fq_cfg(pin, L_in, src_sq)
                if os.path.exists(pout):
                    os.remove(pout)
                ok_m = True
                for ci in range(n_c):
                    if ci > 0:
                        if not os.path.exists(pout):
                            ok_m = False
                            break
                        os.replace(pout, pin)
                    pick = (
                        int(rng.choice(tilted_idx))
                        if len(tilted_idx)
                        else rng.randrange(n)
                    )
                    subprocess.run(
                        [
                            ANNEAL_BIN,
                            "melt",
                            "--in",
                            pin,
                            "--out",
                            pout,
                            "--cx",
                            repr(float(src_sq[pick, 0])),
                            "--cy",
                            repr(float(src_sq[pick, 1])),
                            "--rad",
                            f"{rng.uniform(rmin, rmax_rad):.3f}",
                            "--rmax",
                            f"{rng.uniform(m0, m1):.3f}",
                            "--sweeps",
                            str(swp),
                            "--bp",
                            "2200.0",
                            "--seed",
                            str(rng.randrange(1 << 30)),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=40,
                    )
                if ok_m and os.path.exists(pout):
                    sm, arr = f2d.read_fq_cfg_rad(pout)
                    if len(arr) == n:
                        sc_fq = (
                            [
                                "--loosen",
                                "1.0",
                                "--round-r0",
                                rng.choice(("0.12", "0.20")),
                                "--round-compress",
                                "--round-press",
                                "--round-steps",
                                "8",
                                "--round-press-outer",
                                "13",
                                "--seed",
                                str(rng.randrange(1 << 30)),
                                "--flip-top",
                                "36",
                                "--pit",
                                "190",
                            ]
                            if use_rp
                            else [
                                "--loosen",
                                "1.012",
                                "--kick",
                                "0.005",
                                "--seed",
                                str(rng.randrange(1 << 30)),
                                "--flip-top",
                                "48",
                                "--pit",
                                "320",
                            ]
                        )
                        r = f2d.run_fq(sm + 1e-4, arr, tmp, sc_fq, timeout=54)
                        if r is not None:
                            tname = f"m27_{n_c}{'r' if use_rp else 'f'}_{r_idx}"
                            if near_best is None or r[0] < near_best[0]:
                                near_best = (r[0], r[1], tname)
                            if r[0] < best_s - 1e-9:
                                best_s, best_a, best_tag = r[0], r[1], tname
            except Exception:
                pass

    # 3) 6-Step Kick Chain on win or tight near-miss (+0.008)
    seed_chain = (
        (best_s, best_a, best_tag)
        if best_a is not None
        else (
            near_best if (near_best is not None and near_best[0] < L0 + 0.008) else None
        )
    )
    if seed_chain is not None:
        ls, la, ltag = seed_chain
        for _ in range(6):
            r2 = f2d.run_fq(
                ls,
                la,
                tmp,
                [
                    "--loosen",
                    rng.choice(("1.003", "1.009", "1.018", "1.032")),
                    "--kick",
                    rng.choice(("0.002", "0.005", "0.010", "0.018")),
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "56",
                    "--pit",
                    "340",
                ],
                timeout=54,
            )
            if r2 is not None and r2[0] < ls:
                ls, la = r2
            if r2 is not None and r2[0] < best_s - 1e-9:
                best_s, best_a, best_tag = r2[0], r2[1], ltag + "+kc"

    return best_s, best_a, best_tag


def worker_sprint(job):
    n, L0, sq0, seed = job
    rng = random.Random(seed)
    cur_s, cur_a = L0, sq0
    best_tag = ""
    improved_any = False
    with tempfile.TemporaryDirectory() as tmp:
        for r_idx in range(3):
            ns, na, tag = lightning_round(n, cur_s, cur_a, rng, tmp, r_idx)
            if na is not None and ns < cur_s - 1e-9:
                gain = cur_s - ns
                cur_s, cur_a, best_tag = ns, na, tag
                improved_any = True
                if gain < 1e-6:
                    break
            else:
                break
        if improved_any and cur_s < L0 - 1e-9:
            r_fin = f2d.run_fq(
                cur_s,
                cur_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "750"],
                timeout=80,
            )
            if r_fin is not None and r_fin[0] < cur_s:
                cur_s, cur_a = r_fin[0], r_fin[1]
            cur_a = w12.lp_rot_polish(cur_a)
            ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_a)
            if ok and Lf < L0 - 1e-9:
                return n, best_tag, (Lf, xf, yf, tf)
    return n, best_tag, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--lags-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=28)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1680)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    lags = json.load(open(args.lags_json)) if os.path.exists(args.lags_json) else []
    lag_rank = {int(it["n"]): float(it.get("dS", 0.0)) for it in lags}

    priority_fresh = {
        683,
        923,
        405,
        298,
        264,
        500,
        603,
        499,
        746,
        642,
        593,
        654,
        692,
        589,
        740,
        862,
        545,
        508,
        451,
        736,
    }
    scored = []
    for n in range(180, 995):
        K = math.isqrt(n)
        D = n - K * K
        if D == 0 or cur[n][0] >= K + 1 - 1e-6:
            continue
        fn = cur[n][0] - K
        if fn < 0.56 and n not in priority_fresh:
            continue
        prio = (
            (0.00200 if n in priority_fresh else 0.0)
            + lag_rank.get(n, 0.0)
            + (0.00020 if fn >= 0.76 else 0.00008)
        )
        scored.append((prio, n))
    scored.sort(reverse=True)
    targets = [n for _, n in scored[:224]]
    my_slice = targets[args.shard :: args.num_shards]
    print(
        f"[W27-Top40Sprint s={args.shard}/{args.num_shards}] Frontier={tot_score:.10f}, "
        f"my_targets={my_slice} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0
    jobs = [
        (n, cur[n][0], cur[n][1], 2700001 + args.shard * 10000 + i * 149)
        for i, n in enumerate(my_slice)
    ]
    done = 0

    def save_ckpt():
        payload = {}
        for k_n, (Lf, xf, yf, tf) in sorted(wins.items()):
            payload[f"L_{k_n}"] = np.array([Lf], dtype=np.float64)
            payload[f"x_{k_n}"] = np.asarray(xf, dtype=np.float64)
            payload[f"y_{k_n}"] = np.asarray(yf, dtype=np.float64)
            payload[f"t_{k_n}"] = np.asarray(tf, dtype=np.float64)
        np.savez_compressed(args.out_npz, **payload)

    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_sprint, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            n, tag, res = fut.result()
            if res is not None:
                Lf, xf, yf, tf = res
                dL = Lf - cur[n][0]
                if dL < -1e-10:
                    dS = dL / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W27 SPRINT WIN!!] n={n:4d} {tag:14s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W27 s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W27 s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
