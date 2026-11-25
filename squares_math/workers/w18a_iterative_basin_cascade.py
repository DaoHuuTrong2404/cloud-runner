#!/usr/bin/env python3
"""Wave 18A Iterative Basin-Cascade Escape Engine (40 Cores / 10 Shards).

Key empirical discovery from Wave 15A -> Wave 17B:
  Freshly-moved packings keep cascading for HUGE extra drops when re-melted!
    n=547: -0.00124 (W15A) -> ANOTHER -0.00168 (W17B)!
    n=785: -0.00045 (W16)  -> ANOTHER -0.00042 (W17B)!
    n=499: -0.00020 (W16)  -> ANOTHER -0.00022 (W17B)!
    n=582: -0.00031 (W15A) -> ANOTHER -0.00101 (W17B)!

Wave 18A automates this inside ONE worker:
  - Targets ALL freshly-moved R12/R13/W17B winners + Top-140 biggest-lag puzzles
  - Whenever a puzzle improves by > 5e-7, AUTOMATICALLY loops back (up to 4 full cascade rounds!)
    running fresh bandflip/relocate + 10 multi-scale large-radius anneal melts (including dual-center
    double-melt!) + 6 round-press radii (0.04..0.28) + 6-step kick chain + Contact LP + Decimal64!
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

ANNEAL_BIN = os.environ.get("ANNEAL_BIN", "/tmp/sq_bins/anneal")


def run_one_escape_round(n, L0, sq, rng, tmp, r_idx):
    """One deep escape round on (L0, sq). Returns (best_s, best_a, best_tag) where best_a is None if no win."""
    best_s = L0
    best_a = None
    best_tag = ""
    near_best = None

    # 1) Bandflip + Relocate (up to 4 bands)
    starts = []
    try:
        import bandflip as bf_mod

        pts_deg = np.column_stack(
            [sq[:, 0], sq[:, 1], np.degrees(sq[:, 2]) % 90.0]
        ).tolist()
        B = bf_mod.bands(pts_deg, min_size=6 if n < 160 else 8)
        for bi in range(min(4, len(B))):
            for fn_name, fn in (
                (f"fl{r_idx}", bf_mod.flip),
                (f"rl{r_idx}", bf_mod.relocate),
            ):
                out = fn(L0, pts_deg, bi, rng, B)
                if out is not None:
                    arr = np.array(out, dtype=np.float64)
                    arr[:, 2] = np.radians(arr[:, 2])
                    starts.append((fn_name, arr))
    except Exception:
        pass

    for tag_s, st_sq in starts[:6]:
        for sc in (
            [
                "--loosen",
                "1.022",
                "--kick",
                "0.009",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "44",
                "--pit",
                "280",
            ],
            [
                "--loosen",
                "1.055",
                "--kick",
                "0.026",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "44",
                "--pit",
                "250",
            ],
        ):
            r = f2d.run_fq(w12.span(st_sq) + 1e-4, st_sq, tmp, sc, timeout=54)
            if r is not None:
                if near_best is None or r[0] < near_best[0]:
                    near_best = (r[0], r[1], tag_s)
                if r[0] < best_s - 1e-9:
                    best_s, best_a, best_tag = r[0], r[1], tag_s

    # 2) 10 Multi-Scale Large-Radius Anneal Melts (including 2 Dual-Center Double-Melts!)
    if os.path.exists(ANNEAL_BIN):
        degs = np.degrees(sq[:, 2]) % 90.0
        tilted_idx = np.flatnonzero(np.minimum(degs, 90.0 - degs) > 2.0)
        he = 0.5 * (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2])))
        lo_x, hi_x = np.min(sq[:, 0] - he), np.max(sq[:, 0] + he)
        lo_y, hi_y = np.min(sq[:, 1] - he), np.max(sq[:, 1] + he)
        wall_d = np.minimum(
            np.minimum(sq[:, 0] - he - lo_x, hi_x - (sq[:, 0] + he)),
            np.minimum(sq[:, 1] - he - lo_y, hi_y - (sq[:, 1] + he)),
        )
        stress_idx = np.argsort(
            np.where(np.minimum(degs, 90.0 - degs) > 2.0, wall_d, 999.0)
        )[:24]
        cap_r = max(3.5, 0.62 * L0)
        melt_cfgs = [
            (2.0, min(4.0, cap_r), 0.08, 0.16, 3800, "stress", False),
            (3.2, min(5.8, cap_r), 0.11, 0.24, 4400, "tilt", False),
            (4.8, min(8.2, cap_r), 0.14, 0.30, 4800, "tilt", False),
            (6.0, min(10.5, cap_r), 0.17, 0.36, 5200, "any", False),
            (2.8, min(5.2, cap_r), 0.10, 0.22, 4200, "stress", False),
            (4.2, min(7.5, cap_r), 0.13, 0.28, 4600, "stress", False),
            (3.5, min(6.5, cap_r), 0.12, 0.26, 4400, "any", False),
            (5.5, min(9.5, cap_r), 0.16, 0.34, 5000, "tilt", False),
            (
                3.0,
                min(5.5, cap_r),
                0.12,
                0.24,
                3400,
                "stress",
                True,
            ),  # Dual-center double-melt!
            (
                4.5,
                min(7.5, cap_r),
                0.14,
                0.28,
                3600,
                "tilt",
                True,
            ),  # Dual-center double-melt!
        ]
        for rmin, rmax_rad, m0, m1, swp, mode_p, is_double in melt_cfgs:
            if rmax_rad <= rmin:
                rmin = 0.5 * rmax_rad
            try:
                pin = os.path.join(tmp, "min.txt")
                pout = os.path.join(tmp, "mout.txt")
                src_sq = best_a if best_a is not None else sq
                src_L = best_s if best_a is not None else L0
                f2d.write_fq_cfg(pin, src_L, src_sq)
                if os.path.exists(pout):
                    os.remove(pout)
                if mode_p == "stress" and len(stress_idx):
                    pick = int(rng.choice(stress_idx))
                elif mode_p == "tilt" and len(tilted_idx):
                    pick = int(rng.choice(tilted_idx))
                else:
                    pick = rng.randrange(n)
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
                    timeout=46,
                )
                if is_double and os.path.exists(pout):
                    # Second melt at a different tilted/stress center before quenching!
                    pin2 = os.path.join(tmp, "min2.txt")
                    pout2 = os.path.join(tmp, "mout2.txt")
                    os.replace(pout, pin2)
                    pick2 = (
                        int(rng.choice(tilted_idx))
                        if len(tilted_idx)
                        else rng.randrange(n)
                    )
                    subprocess.run(
                        [
                            ANNEAL_BIN,
                            "melt",
                            "--in",
                            pin2,
                            "--out",
                            pout2,
                            "--cx",
                            repr(float(src_sq[pick2, 0])),
                            "--cy",
                            repr(float(src_sq[pick2, 1])),
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
                        timeout=46,
                    )
                    if os.path.exists(pout2):
                        os.replace(pout2, pout)

                if os.path.exists(pout):
                    s_m, arr = f2d.read_fq_cfg_rad(pout)
                    if len(arr) == n:
                        r = f2d.run_fq(
                            s_m + 1e-4,
                            arr,
                            tmp,
                            [
                                "--loosen",
                                "1.012",
                                "--kick",
                                "0.005",
                                "--seed",
                                str(rng.randrange(1 << 30)),
                                "--flip-top",
                                "48",
                                "--pit",
                                "340",
                            ],
                            timeout=60,
                        )
                        if r is not None:
                            tname = f"dblm{r_idx}" if is_double else f"mlt{r_idx}"
                            if near_best is None or r[0] < near_best[0]:
                                near_best = (r[0], r[1], tname)
                            if r[0] < best_s - 1e-9:
                                best_s, best_a, best_tag = r[0], r[1], tname
            except Exception:
                pass

    # 3) 6-Radii Whole-Packing Round-Compress + Round-Press (including 0.28!)
    for r0 in ("0.04", "0.08", "0.12", "0.17", "0.22", "0.28"):
        src_sq = best_a if best_a is not None else sq
        src_L = best_s if best_a is not None else L0
        r = f2d.run_fq(
            src_L + 0.008,
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
                "210",
            ],
            timeout=60,
        )
        if r is not None:
            if near_best is None or r[0] < near_best[0]:
                near_best = (r[0], r[1], f"rp{r_idx}_{r0}")
            if r[0] < best_s - 1e-9:
                best_s, best_a, best_tag = r[0], r[1], f"rp{r_idx}_{r0}"

    # 4) 6-Step Kick Chain on winner OR tight near-miss (+0.007)
    seed_chain = (
        (best_s, best_a, best_tag)
        if best_a is not None
        else (
            near_best if (near_best is not None and near_best[0] < L0 + 0.007) else None
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
                    rng.choice(("1.003", "1.008", "1.016", "1.028")),
                    "--kick",
                    rng.choice(("0.001", "0.003", "0.007", "0.014")),
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
            if r2 is not None and r2[0] < best_s - 1e-9:
                best_s, best_a, best_tag = r2[0], r2[1], ltag + "+kc"

    return best_s, best_a, best_tag


def worker_iterative_cascade(job):
    n, L0, sq0, seed = job
    rng = random.Random(seed)
    cur_s, cur_a = L0, sq0
    best_tag = ""
    improved_any = False
    with tempfile.TemporaryDirectory() as tmp:
        for r_idx in range(4):
            ns, na, tag = run_one_escape_round(n, cur_s, cur_a, rng, tmp, r_idx)
            if na is not None and ns < cur_s - 1e-9:
                gain = cur_s - ns
                cur_s, cur_a = ns, na
                best_tag = tag
                improved_any = True
                # Only continue looping into another full cascade round if meaningful progress (> 5e-7)!
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
    ap.add_argument("--recent-glob", action="append", default=[])
    ap.add_argument("--lags-json", required=True)
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
    lags = json.load(open(args.lags_json)) if os.path.exists(args.lags_json) else []
    lag_rank = {int(it["n"]): float(it.get("dS", 0.0)) for it in lags}

    recent_winners = set()
    for pat in args.recent_glob:
        for fp in glob.glob(pat, recursive=True):
            try:
                sd = np.load(fp)
                for k in sd.files:
                    if k.startswith("L_"):
                        recent_winners.add(int(k[2:]))
            except Exception:
                pass

    scored = []
    for n in range(40, 1001):
        K = math.isqrt(n)
        if n == K * K or cur[n][0] >= K + 1 - 1e-6:
            continue
        f = cur[n][0] - K
        prio = (
            (0.00120 if n in recent_winners else 0.0)
            + lag_rank.get(n, 0.0)
            + (0.00010 if (n >= 450 and f >= 0.55) else 0.0)
        )
        scored.append((prio, n))
    scored.sort(reverse=True)
    targets = [n for _, n in scored[:220]]
    my_slice = targets[args.shard :: args.num_shards]
    print(
        f"[W18A-IterCascade s={args.shard}/{args.num_shards}] Frontier={tot_score:.10f} "
        f"(recent_winners={len(recent_winners)}), my_slice={len(my_slice)}/220 ({time.time() - t0:.1f}s)",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0
    jobs = [
        (n, cur[n][0], cur[n][1], 1800001 + args.shard * 10000 + i * 83)
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
        futs = [ex.submit(worker_iterative_cascade, j) for j in jobs]
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
                        f"  [W18A WIN!!] n={n:4d} {tag:14s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if done % 5 == 0:
                print(
                    f"  [W18A s={args.shard}] {done}/{len(jobs)} done, wins={len(wins)}, "
                    f"dS={tot_ds:.8f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W18A s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W18A s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
