#!/usr/bin/env python3
"""Wave 22 Fixed-D Convexity-Bump Anomaly Sniper (40 Cores / 10 Shards -> 6 Puzzles/Shard!).

Mathematically detects every large-N puzzle (n >= 220) whose excess f(K, D) = L(K^2+D) - K sticks UP above its
Fixed-D interpolation 0.5*(f(K-1, D) + f(K+1, D)) — the EXACT anomaly signature that predicted n=642 (-0.00666)
and n=593 (-0.00610) before they underwent phase transitions!
For each Top-60 anomaly (6 puzzles/shard = ~10 min of 4-core power per puzzle!), unleashes:
  1) Fixed-D Neighbor Transfers: stretch from (K-1)^2+D (+1) and (K+1)^2+D (-1) across 4 orientations +
     window-peels from (K+1..K+3)^2+D + top-10 global sharpest peels + del(n+1, n+2)
  2) Dense 14-Point Round-Press Sweep (r0 in 0.03..0.34, L0+0.010..0.032) — exact operator that hit n=593 (-0.00610) & n=692 (-0.00120)!
  3) Deep Multi-Round Box-Expanded Single/Dual/Triple-Center Melt + Bandflip/Relocate + 6-Step Kick Chain + Contact LP + Decimal64!
"""

import argparse
import glob
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
import w21a_phase_transition_exmelt as w21
import stretch as st_mod

ANNEAL_BIN = os.environ.get("ANNEAL_BIN", "/tmp/sq_bins/anneal")


def detect_convexity_anomalies(cur, top_k=60):
    """Rank n >= 220 non-grid packings by Fixed-D convexity bump (f(K,D) above neighbors) / sqrt(n)."""
    f = {n: cur[n][0] - math.isqrt(n) for n in range(1, 1001)}
    anoms = []
    for n in range(220, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D == 0 or f[n] >= 1.0 - 1e-6:
            continue
        n_prev = (K - 1) * (K - 1) + D
        n_next = (K + 1) * (K + 1) + D
        bD = 0.0
        if (
            D < 2 * (K - 1) + 1
            and n_next <= 1000
            and f[n_prev] < 1.0 - 1e-6
            and f[n_next] < 1.0 - 1e-6
        ):
            bD = f[n] - 0.5 * (f[n_prev] + f[n_next])
        elif D < 2 * (K - 1) + 1 and f[n_prev] < 1.0 - 1e-6:
            bD = f[n] - (f[n_prev] - 0.011)
        if bD > 0.0015:
            anoms.append((bD / math.sqrt(n), bD, n))
    anoms.sort(reverse=True)
    return [(n, bD) for _, bD, n in anoms[:top_k]]


def harvest_fixed_d_transfers(n, L0, cur, nongrid_sorted):
    """Collect Fixed-D stretch/peel transfers + top global sharp peels + del(n+1, n+2)."""
    K = math.isqrt(n)
    D = n - K * K
    cands = []
    seen = set()

    # 1) Direct Fixed-D stretch from (K+dk)^2 + D
    for dk in (-2, -1, 1, 2):
        Ks = K + dk
        if Ks < 8 or Ks > 31:
            continue
        sn = Ks * Ks + D
        if sn < 64 or sn > 1000 or cur[sn][0] >= Ks + 1 - 1e-6:
            continue
        sq_sn = cur[sn][1]
        for oi, sym in (
            (0, f2d.SYMS[0]),
            (1, f2d.SYMS[1]),
            (2, f2d.SYMS[2]),
            (3, f2d.SYMS[3]),
        ):
            sq_o = sq_sn if oi == 0 else f2d.mapped(sq_sn, sym)[0]
            try:
                b, _ = st_mod.stretch(sq_o, -dk)
            except Exception:
                b = None
            if b is not None and len(b) == n:
                gap = w12.span(b) - L0
                sig = (sn, round(gap, 4))
                if gap < 0.18 and sig not in seen:
                    seen.add(sig)
                    cands.append((gap, b, f"stD_{sn}o{oi}"))

    # 2) Sharpest peels from (K+1..K+4)^2+D AND global m > n
    peel_cands = []
    for m in nongrid_sorted:
        if m <= n:
            continue
        Lm, sq = cur[m]
        Km = math.isqrt(m)
        Dm = m - Km * Km
        is_same_D = abs(Dm - D) <= 2
        for si, sym in enumerate(f2d.SYMS):
            sq_m, xlo, xhi, ylo, yhi = f2d.mapped(sq, sym)
            R = np.maximum(xhi, yhi)
            take = np.argpartition(R, n - 1)[:n]
            gap = float(R[take].max()) - L0
            if gap < (0.14 if is_same_D else 0.09):
                sig = (m, round(gap, 4))
                if sig not in seen:
                    seen.add(sig)
                    peel_cands.append(
                        (
                            gap - (0.015 if is_same_D else 0.0),
                            sq_m[take].copy(),
                            f"peel_{m}_s{si}",
                        )
                    )
    peel_cands.sort(key=lambda z: z[0])
    cands.extend(peel_cands[:10])

    # 3) Wall/tilted delete from n+1, n+2
    for m in (n + 1, n + 2):
        if m <= 1000 and cur[m][0] < L0 + 0.09:
            sqm = cur[m][1]
            he = 0.5 * (np.abs(np.cos(sqm[:, 2])) + np.abs(np.sin(sqm[:, 2])))
            xlo, xhi = sqm[:, 0] - he, sqm[:, 0] + he
            ylo, yhi = sqm[:, 1] - he, sqm[:, 1] + he
            wd = np.minimum(
                np.minimum(xlo - xlo.min(), xhi.max() - xhi),
                np.minimum(ylo - ylo.min(), yhi.max() - yhi),
            )
            arr_w = np.delete(sqm, np.argsort(wd)[: m - n], axis=0)
            cands.append((w12.span(arr_w) - L0, arr_w, f"delW_{m}"))

    cands.sort(key=lambda z: z[0])
    return cands[:14]


def worker_snipe_anomaly(job):
    """Deep multi-engine sniper on 1 convexity-bump anomaly puzzle (runs inside 1 worker process)."""
    n, L0, sq0, transfers, seed = job
    rng = random.Random(seed)
    cur_s, cur_a = L0, sq0
    best_tag = ""
    improved_any = False

    with tempfile.TemporaryDirectory() as tmp:
        # Phase 1: Dense 14-Point Round-Press Sweep (exact winner for n=593 -0.00610 & n=692 -0.00120!)
        near_best = None
        rp_cfgs = [
            ("0.03", 0.008),
            ("0.04", 0.010),
            ("0.05", 0.012),
            ("0.065", 0.012),
            ("0.08", 0.012),
            ("0.08", 0.022),
            ("0.10", 0.014),
            ("0.12", 0.016),
            ("0.15", 0.018),
            ("0.19", 0.020),
            ("0.23", 0.024),
            ("0.28", 0.028),
            ("0.34", 0.032),
            ("0.06", 0.025),
        ]
        for r0, pad in rp_cfgs:
            r = f2d.run_fq(
                cur_s + pad,
                cur_a,
                tmp,
                [
                    "--loosen",
                    "1.0",
                    "--round-r0",
                    r0,
                    "--round-compress",
                    "--round-press",
                    "--round-steps",
                    "9",
                    "--round-press-outer",
                    "15",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "40",
                    "--pit",
                    "220",
                ],
                timeout=62,
            )
            if r is not None:
                if near_best is None or r[0] < near_best[0]:
                    near_best = (r[0], r[1], f"rp22_{r0}")
                if r[0] < cur_s - 1e-9:
                    cur_s, cur_a, best_tag = r[0], r[1], f"rp22_{r0}"
                    improved_any = True

        # Follow up with 6-step kick chain on best rp22 hit or tight near-miss!
        seed_rp = (
            (cur_s, cur_a, best_tag)
            if improved_any
            else (
                near_best
                if (near_best is not None and near_best[0] < L0 + 0.010)
                else None
            )
        )
        if seed_rp is not None:
            ls, la, ltag = seed_rp
            for _ in range(6):
                r2 = f2d.run_fq(
                    ls,
                    la,
                    tmp,
                    [
                        "--loosen",
                        rng.choice(("1.004", "1.010", "1.020", "1.034")),
                        "--kick",
                        rng.choice(("0.002", "0.005", "0.010", "0.018")),
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "56",
                        "--pit",
                        "360",
                    ],
                    timeout=60,
                )
                if r2 is not None and r2[0] < ls:
                    ls, la = r2
                if r2 is not None and r2[0] < cur_s - 1e-9:
                    cur_s, cur_a, best_tag = r2[0], r2[1], ltag + "+kc"
                    improved_any = True

        # Phase 2: Fixed-D Neighbor Stretch/Peel/Del Transfers!
        for _, arr, ttag in transfers:
            s_in = w12.span(arr) + 1e-4
            loc = None
            for sc in (
                [
                    "--loosen",
                    "1.016",
                    "--kick",
                    "0.007",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "40",
                    "--pit",
                    "260",
                ],
                [
                    "--loosen",
                    "1.0",
                    "--round-r0",
                    rng.choice(("0.10", "0.18")),
                    "--round-compress",
                    "--round-press",
                    "--round-steps",
                    "8",
                    "--round-press-outer",
                    "13",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "32",
                    "--pit",
                    "160",
                ],
            ):
                r = f2d.run_fq(s_in, arr, tmp, sc, timeout=58)
                if r is not None:
                    if loc is None or r[0] < loc[0]:
                        loc = r
                    if r[0] < cur_s - 1e-9:
                        cur_s, cur_a, best_tag = r[0], r[1], ttag
                        improved_any = True
            if loc is not None and loc[0] < cur_s + 0.020:
                ls, la = loc
                for _ in range(5):
                    r2 = f2d.run_fq(
                        ls,
                        la,
                        tmp,
                        [
                            "--loosen",
                            rng.choice(("1.005", "1.012", "1.024")),
                            "--kick",
                            rng.choice(("0.002", "0.006", "0.014")),
                            "--seed",
                            str(rng.randrange(1 << 30)),
                            "--flip-top",
                            "52",
                            "--pit",
                            "320",
                        ],
                        timeout=58,
                    )
                    if r2 is not None and r2[0] < ls:
                        ls, la = r2
                    if r2 is not None and r2[0] < cur_s - 1e-9:
                        cur_s, cur_a, best_tag = r2[0], r2[1], ttag + "+kc"
                        improved_any = True

        # Phase 3: Deep 4-Round Phase-Transition Box-Expanded Multi-Center Melt Cascade!
        for r_idx in range(4):
            ns, na, tag = w21.run_phase_transition_round(
                n, cur_s, cur_a, rng, tmp, r_idx
            )
            if na is not None and ns < cur_s - 1e-9:
                gain = cur_s - ns
                cur_s, cur_a, best_tag = ns, na, tag
                improved_any = True
                if gain < 5e-7:
                    break
            else:
                break

        if improved_any and cur_s < L0 - 1e-9:
            r_fin = f2d.run_fq(
                cur_s,
                cur_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "750"],
                timeout=85,
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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=3550)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    nongrid_sorted = [
        m
        for m in range(20, 1001)
        if m != math.isqrt(m) ** 2 and cur[m][0] < math.isqrt(m) + 1 - 1e-6
    ]
    top60 = detect_convexity_anomalies(cur, top_k=60)
    my_slice = top60[args.shard :: args.num_shards]
    print(
        f"[W22-ConvexitySniper s={args.shard}/{args.num_shards}] Frontier={tot_score:.10f} "
        f"({len(fresh)} fresh from {n_files} files), my_targets={[(n, round(b, 4)) for n, b in my_slice]} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    jobs = []
    for i, (n, bD) in enumerate(my_slice):
        L0, sq0 = cur[n]
        transfers = harvest_fixed_d_transfers(n, L0, cur, nongrid_sorted)
        jobs.append((n, L0, sq0, transfers, 2200001 + args.shard * 10000 + i * 113))

    wins = {}
    tot_ds = 0.0
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
        futs = [ex.submit(worker_snipe_anomaly, j) for j in jobs]
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
                        f"  [W22 ANOMALY SMASH!!] n={n:4d} {tag:16s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W22 s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W22 s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
