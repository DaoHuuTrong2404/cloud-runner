#!/usr/bin/env python3
"""Wave 15A Deep Large-Radius Escape Melt + Wide-Range Round-Press + Relocate (40 Cores / 10 Shards).

Targets the top 240 highest-lag & freshly-moved puzzles on our ~22.08896 frontier (including all 32 Wave 12A
winners!) with LARGER melt radii (rad 3.0..8.5, sweeps 4500, rmax 0.12..0.32) capable of melting entire
multi-row tilted bands + wider round-press radii (r0 in 0.05, 0.08, 0.12, 0.17, 0.23) + deeper 6-step kick
chains + Contact LP + Rigid Rotation + Decimal64 >= 1.05e-9 certification!
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
sys.path.insert(0, DS_DIR)
sys.path.insert(0, HERE)

import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12

ANNEAL_BIN = os.environ.get("ANNEAL_BIN", "/tmp/sq_bins/anneal")


def worker_deep_escape(job):
    n, L0, sq, seed = job
    rng = random.Random(seed)
    best_s = L0
    best_a = None
    best_tag = ""
    near_best = None
    with tempfile.TemporaryDirectory() as tmp:
        # 1) Bandflip + Relocate (up to 4 bands, different RNG seed from W12A!)
        starts = []
        try:
            import bandflip as bf_mod

            pts_deg = np.column_stack(
                [sq[:, 0], sq[:, 1], np.degrees(sq[:, 2]) % 90.0]
            ).tolist()
            B = bf_mod.bands(pts_deg, min_size=8)
            for bi in range(min(4, len(B))):
                for fn_name, fn in (
                    ("flip2", bf_mod.flip),
                    ("reloc2", bf_mod.relocate),
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
                    "1.020",
                    "--kick",
                    "0.008",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "40",
                    "--pit",
                    "260",
                ],
                [
                    "--loosen",
                    "1.048",
                    "--kick",
                    "0.022",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "40",
                    "--pit",
                    "240",
                ],
            ):
                r = f2d.run_fq(w12.span(st_sq) + 1e-4, st_sq, tmp, sc, timeout=52)
                if r is not None:
                    if near_best is None or r[0] < near_best[0]:
                        near_best = (r[0], r[1], tag_s)
                    if r[0] < best_s - 1e-9:
                        best_s, best_a, best_tag = r[0], r[1], tag_s

        # 2) Large-Radius & High-Temperature Multi-Band Anneal Melts (8 schedules!)
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

            melt_cfgs = [
                (2.2, 3.8, 0.08, 0.16, 3600, "stress"),
                (3.5, 5.8, 0.10, 0.22, 4200, "tilt"),
                (5.0, 7.8, 0.14, 0.28, 4600, "tilt"),
                (6.0, 9.0, 0.16, 0.32, 4800, "any"),
                (3.0, 5.2, 0.10, 0.20, 4000, "stress"),
                (4.5, 7.2, 0.12, 0.26, 4400, "stress"),
                (2.0, 3.5, 0.06, 0.14, 3400, "tilt"),
                (4.0, 6.5, 0.12, 0.24, 4200, "any"),
            ]
            for rmin, rmax_rad, m0, m1, swp, mode_p in melt_cfgs:
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
                    cx, cy = float(src_sq[pick, 0]), float(src_sq[pick, 1])
                    subprocess.run(
                        [
                            ANNEAL_BIN,
                            "melt",
                            "--in",
                            pin,
                            "--out",
                            pout,
                            "--cx",
                            repr(cx),
                            "--cy",
                            repr(cy),
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
                        timeout=45,
                    )
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
                                    "320",
                                ],
                                timeout=58,
                            )
                            if r is not None:
                                if near_best is None or r[0] < near_best[0]:
                                    near_best = (r[0], r[1], "dmelt")
                                if r[0] < best_s - 1e-9:
                                    best_s, best_a, best_tag = r[0], r[1], "dmelt"
                except Exception:
                    pass

        # 3) Wide-Range Whole-Packing Round-Compress + Round-Press (5 radii!)
        for r0 in ("0.05", "0.08", "0.12", "0.17", "0.23"):
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
                    "13",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "36",
                    "--pit",
                    "200",
                ],
                timeout=58,
            )
            if r is not None:
                if near_best is None or r[0] < near_best[0]:
                    near_best = (r[0], r[1], f"drp_{r0}")
                if r[0] < best_s - 1e-9:
                    best_s, best_a, best_tag = r[0], r[1], f"drp_{r0}"

        # 4) Deep 6-step iterative follow-up kick chain on winner OR tight near-miss (+0.005)!
        seed_chain = (
            (best_s, best_a, best_tag)
            if best_a is not None
            else (
                near_best
                if (near_best is not None and near_best[0] < L0 + 0.005)
                else None
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

        if best_a is not None and best_s < L0 - 1e-9:
            r_fin = f2d.run_fq(
                best_s,
                best_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "700"],
                timeout=85,
            )
            if r_fin and r_fin[0] < best_s:
                best_s, best_a = r_fin[0], r_fin[1]
            best_a = w12.lp_rot_polish(best_a)
            ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, best_a)
            if ok and Lf < L0 - 1e-9:
                return n, best_tag, (Lf, xf, yf, tf)
    return n, best_tag, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--lags-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=3700)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    lags = json.load(open(args.lags_json)) if os.path.exists(args.lags_json) else []
    lag_rank = {int(it["n"]): float(it.get("dS", 0.0)) for it in lags}

    # Top 260 highest-priority non-grid packings n=120..1000
    scored = []
    for n in range(120, 1001):
        K = math.isqrt(n)
        if n == K * K or cur[n][0] >= K + 1 - 1e-6:
            continue
        f = cur[n][0] - K
        prio = (
            lag_rank.get(n, 0.0)
            + (0.00020 if n in fresh else 0.0)
            + (0.00008 if (n >= 500 and f >= 0.60) else 0.0)
        )
        scored.append((prio, n))
    scored.sort(reverse=True)
    targets = [n for _, n in scored[:260]]
    my_slice = targets[args.shard :: args.num_shards]
    print(
        f"[W15A-DeepEscape s={args.shard}/{args.num_shards}] Frontier={tot_score:.10f} ({len(fresh)} fresh from {n_files} files), my_slice={len(my_slice)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0
    jobs = [
        (n, cur[n][0], cur[n][1], 1500001 + args.shard * 10000 + i * 61)
        for i, n in enumerate(my_slice)
    ]
    done = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_deep_escape, j) for j in jobs]
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
                        f"  [W15A WIN] n={n:4d} {tag:14s} {cur[n][0]:.7f} -> {Lf:.7f} dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
            if done % 6 == 0:
                print(
                    f"  [W15A s={args.shard}] {done}/{len(jobs)} done, wins={len(wins)}, dS={tot_ds:.8f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W15A s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    payload = {}
    for n, (Lf, xf, yf, tf) in sorted(wins.items()):
        payload[f"L_{n}"] = np.array([Lf], dtype=np.float64)
        payload[f"x_{n}"] = np.asarray(xf, dtype=np.float64)
        payload[f"y_{n}"] = np.asarray(yf, dtype=np.float64)
        payload[f"t_{n}"] = np.asarray(tf, dtype=np.float64)
    np.savez_compressed(args.out_npz, **payload)
    print(
        f"[W15A s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
