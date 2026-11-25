#!/usr/bin/env python3
"""Wave 16 Top-20 Biggest-Gap Global Topology Sniper (80 Cores / 20 Shards -> 1 Dedicated 4-Core Runner Per Puzzle!).

Targets the EXACT Top-20 remaining dS gap puzzles holding >0.0110 total score gap:
  [234, 491, 367, 333, 738, 456, 500, 545, 730, 804, 785, 499, 371, 261, 904, 748, 962, 414, 332, 914]
Runs 5 BRAND-NEW GLOBAL-TOPOLOGY ENGINES (not local polishing of the wrong layout!):
  1) Exhaustive Window-Peel from ALL bigger packings m in (n+1..1000) x 8 symmetries + interior shifts -> fq/round-press
  2) From-Scratch Circle-to-Square Global Morph (--round-r0 0.48 --round-compress --round-press) from 160+ diverse
     random / 2-chord / opposite-corner / diagonal-band / rotated-hybrid global seeds!
  3) Full-Box Global Thermal Melt (anneal melt --rad L0) -> re-crystallize via round-press + kick-chain!
  4) Parametric band_gen / lband / lband2 tilted-band assembly at target Lb (with HiGHS threads=1 patch!) -> fq!
  5) Deep Multi-Neighbor Surgery (n-3..n-1 insert+round-press, n+1..n+4 wall-delete, stretch +-1..+-4)!
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

# 18x speedup for HiGHS MILP inside ProcessPoolExecutor!
try:
    from scipy.optimize import _milp

    _orig_hw = _milp._highs_wrapper

    def _hw_1t(*a, **kw):
        if len(a) >= 2 and isinstance(a[1], dict):
            a = (a[0], {**a[1], "threads": 1}, *a[2:])
        elif "options" in kw and isinstance(kw["options"], dict):
            kw["options"] = {**kw["options"], "threads": 1}
        return _orig_hw(*a, **kw)

    _milp._highs_wrapper = _hw_1t
except Exception:
    pass

import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12
import stretch as st_mod

ANNEAL_BIN = os.environ.get("ANNEAL_BIN", "/tmp/sq_bins/anneal")

TOP20_TARGETS = [
    (234, 15.804555),
    (491, 22.661648),
    (367, 19.667547),
    (333, 18.760014),
    (738, 27.656978),
    (456, 21.834218),
    (500, 22.835068),
    (545, 23.812764),
    (730, 27.498476),
    (804, 28.810061),
    (785, 28.498476),
    (499, 22.816226),
    (371, 19.767033),
    (261, 16.671025),
    (904, 30.548594),
    (748, 27.810880),
    (962, 31.489514),
    (414, 20.831865),
    (332, 18.731707),
    (914, 30.711721),
]


def harvest_window_peels(n, L0, cur, max_cands=90):
    """Cut tightest n-square windows from ALL bigger non-grid packings m > n across all 8 symmetries!"""
    cands = []
    seen = set()
    for m in range(n + 1, 1001):
        Lm, sq = cur[m]
        if Lm >= math.ceil(math.sqrt(m)) - 1e-6:
            continue
        for si, sym in enumerate(f2d.SYMS):
            sq_m, xlo, xhi, ylo, yhi = f2d.mapped(sq, sym)
            # Corner crop [0, R]^2 containing n squares
            R = np.maximum(xhi, yhi)
            take = np.argpartition(R, n - 1)[:n]
            L_crop = float(R[take].max())
            gap = L_crop - L0
            if gap < 0.14:
                sig = (m, round(L_crop, 5))
                if sig not in seen:
                    seen.add(sig)
                    cands.append((gap, sq_m[take].copy(), f"peel_{m}_s{si}"))
    cands.sort(key=lambda z: z[0])
    # Deduplicate by keeping diverse donor sizes m (at most 2 per m)
    out = []
    cnt_m = {}
    for gap, arr, tag in cands:
        ms = tag.split("_")[1]
        if cnt_m.get(ms, 0) < 2:
            cnt_m[ms] = cnt_m.get(ms, 0) + 1
            out.append((gap, arr, tag))
            if len(out) >= max_cands:
                break
    return out


def make_global_seed(n, Lb, kind, rng, np_rng):
    """Generate a completely fresh global starting layout inside [0, Lb+0.12]^2 for circle-to-square morph!"""
    K = math.isqrt(n)
    D = n - K * K
    S = Lb + rng.uniform(0.05, 0.22)
    if kind == "random_uniform":
        xs = np_rng.uniform(0.5, S - 0.5, n)
        ys = np_rng.uniform(0.5, S - 0.5, n)
        ts = np_rng.uniform(0.0, math.pi / 2.0, n)
        return np.column_stack([xs, ys, ts])

    elif kind == "two_corner_tilted":
        # Upright core + two opposite tilted blocks in BL and TR corners (pjmathematician #1 motif!)
        ang = math.radians(rng.uniform(14.0, 45.0))
        pts = []
        # Base jittered grid K x K, then tilt two corner blocks + add D extra in corners
        for i in range(K):
            for j in range(K):
                x = (i + 0.5) * (S / K)
                y = (j + 0.5) * (S / K)
                t = ang if (i + j < K // 2 or i + j > (3 * K) // 2) else 0.0
                pts.append((x, y, t))
        for k in range(D):
            if k % 2 == 0:
                pts.append((rng.uniform(0.5, S * 0.4), rng.uniform(0.5, S * 0.4), ang))
            else:
                pts.append(
                    (rng.uniform(S * 0.6, S - 0.5), rng.uniform(S * 0.6, S - 0.5), -ang)
                )
        return np.array(pts[:n], dtype=np.float64)

    elif kind == "double_chord_band":
        # 1 or 2 tilted diagonal/horizontal chords + upright fill (targeting f < 0.50 & mid-D!)
        ang = math.radians(rng.uniform(8.0, 45.0))
        pts = []
        n_rows_tilt = rng.choice((1, 2, 3))
        band_y = rng.uniform(0.5, S - n_rows_tilt - 0.5)
        for i in range(K):
            for j in range(K):
                x = (i + 0.5) * (S / K)
                y = (j + 0.5) * (S / K)
                t = ang if (band_y <= y <= band_y + n_rows_tilt * 1.15) else 0.0
                pts.append((x, y, t))
        for _ in range(D):
            pts.append(
                (
                    rng.uniform(0.5, S - 0.5),
                    rng.uniform(band_y, min(S - 0.5, band_y + n_rows_tilt * 1.3)),
                    ang,
                )
            )
        return np.array(pts[:n], dtype=np.float64)

    elif kind == "l_border_tilted":
        # Upright core + tilted L-shaped border along bottom & left walls
        ang1 = math.radians(rng.uniform(10.0, 45.0))
        ang2 = ang1 if rng.random() < 0.5 else -ang1
        bw = rng.choice((1, 2, 3, 4))
        pts = []
        for i in range(K):
            for j in range(K):
                x = (i + 0.5) * (S / K)
                y = (j + 0.5) * (S / K)
                if i < bw:
                    t = ang1
                elif j < bw:
                    t = ang2
                else:
                    t = 0.0
                pts.append((x, y, t))
        for k in range(D):
            if k % 2 == 0:
                pts.append(
                    (rng.uniform(0.5, bw + 1.0), rng.uniform(0.5, S - 0.5), ang1)
                )
            else:
                pts.append(
                    (rng.uniform(0.5, S - 0.5), rng.uniform(0.5, bw + 1.0), ang2)
                )
        return np.array(pts[:n], dtype=np.float64)

    else:  # rotated_global_lattice
        ang = math.radians(rng.uniform(5.0, 45.0))
        xs = np_rng.uniform(0.5, S - 0.5, n)
        ys = np_rng.uniform(0.5, S - 0.5, n)
        ts = np.where(np_rng.random(n) < 0.5, ang, 0.0)
        return np.column_stack([xs, ys, ts])


def worker_eval_seed(job):
    """Evaluate one candidate layout with aggressive round-press / melt / kick-chain + Contact LP + Decimal64."""
    mode, n, L0, Lb, arr, tag, seed = job
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    best_s = L0
    best_a = None
    local_best = None
    with tempfile.TemporaryDirectory() as tmp:
        if mode == "morph_scratch":
            sq = make_global_seed(n, Lb, tag, rng, np_rng)
            r0 = rng.choice(("0.48", "0.42", "0.35", "0.28"))
            s_in = max(w12.span(sq), Lb + 0.15)
            to = 48 if n <= 450 else 65
            r = f2d.run_fq(
                s_in,
                sq,
                tmp,
                [
                    "--loosen",
                    "1.0",
                    "--round-r0",
                    r0,
                    "--round-compress",
                    "--round-press",
                    "--round-steps",
                    "10",
                    "--round-press-outer",
                    "16",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "32",
                    "--pit",
                    "180",
                ],
                timeout=to,
            )
            if r is not None:
                local_best = r
                if r[0] < best_s - 1e-9:
                    best_s, best_a = r[0], r[1]

        elif mode == "global_melt":
            if os.path.exists(ANNEAL_BIN):
                pin = os.path.join(tmp, "gin.txt")
                pout = os.path.join(tmp, "gout.txt")
                L_loose = L0 + rng.uniform(0.08, 0.28)
                f2d.write_fq_cfg(pin, L_loose, arr)
                cx, cy = (
                    L_loose * rng.uniform(0.25, 0.75),
                    L_loose * rng.uniform(0.25, 0.75),
                )
                rad = L_loose * rng.uniform(0.45, 0.95)
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
                        f"{rad:.3f}",
                        "--rmax",
                        f"{rng.uniform(0.18, 0.40):.3f}",
                        "--sweeps",
                        "5200",
                        "--bp",
                        "2200.0",
                        "--seed",
                        str(rng.randrange(1 << 30)),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=55,
                )
                if os.path.exists(pout):
                    s_m, m_arr = f2d.read_fq_cfg_rad(pout)
                    if len(m_arr) == n:
                        r = f2d.run_fq(
                            s_m + 1e-4,
                            m_arr,
                            tmp,
                            [
                                "--loosen",
                                "1.0",
                                "--round-r0",
                                rng.choice(("0.18", "0.28", "0.38")),
                                "--round-compress",
                                "--round-press",
                                "--round-steps",
                                "9",
                                "--round-press-outer",
                                "14",
                                "--seed",
                                str(rng.randrange(1 << 30)),
                                "--flip-top",
                                "36",
                                "--pit",
                                "200",
                            ],
                            timeout=65,
                        )
                        if r is not None:
                            local_best = r
                            if r[0] < best_s - 1e-9:
                                best_s, best_a = r[0], r[1]

        elif mode == "peel_or_surg":
            s_in = w12.span(arr) + 1e-4
            to = 52 if n <= 500 else 65
            for sc in (
                [
                    "--loosen",
                    "1.015",
                    "--kick",
                    "0.006",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "36",
                    "--pit",
                    "240",
                ],
                [
                    "--loosen",
                    "1.0",
                    "--round-r0",
                    rng.choice(("0.12", "0.22")),
                    "--round-compress",
                    "--round-press",
                    "--round-steps",
                    "7",
                    "--round-press-outer",
                    "12",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "28",
                    "--pit",
                    "150",
                ],
            ):
                r = f2d.run_fq(s_in, arr, tmp, sc, timeout=to)
                if r is not None:
                    if local_best is None or r[0] < local_best[0]:
                        local_best = r
                    if r[0] < best_s - 1e-9:
                        best_s, best_a = r[0], r[1]

        # Follow-up deep kick/loosen chain if won OR promising near-miss (within +0.020!)
        if local_best is not None and local_best[0] < best_s + 0.020:
            ls, la = local_best
            for _ in range(5):
                r2 = f2d.run_fq(
                    ls,
                    la,
                    tmp,
                    [
                        "--loosen",
                        rng.choice(("1.005", "1.012", "1.024", "1.040")),
                        "--kick",
                        rng.choice(("0.002", "0.006", "0.014")),
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "56",
                        "--pit",
                        "340",
                    ],
                    timeout=60,
                )
                if r2 is not None and r2[0] < ls:
                    ls, la = r2
                if r2 is not None and r2[0] < best_s - 1e-9:
                    best_s, best_a = r2[0], r2[1]

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
                return tag, (Lf, xf, yf, tf)
    return tag, None


def worker_parametric_bands(job):
    """Try band_gen.assemble & lband.assemble_L parametric tilted-band constructions at target f!"""
    n, L0, Lb, seed = job
    K = math.isqrt(n)
    f_tgt = Lb - K
    rng = random.Random(seed)
    best_s = L0
    best_a = None
    best_tag = ""
    cands = []
    # 1) band_gen.assemble sweep
    try:
        import band_gen as bg

        for f_try in (f_tgt + 0.002, f_tgt + 0.006, f_tgt + 0.012, L0 - K - 0.002):
            if not (0.42 < f_try < 0.96):
                continue
            for N_b in range(max(1, K - 4), K + 2):
                for y0 in (0.0, 0.5, 1.0, K // 3, K // 2):
                    try:
                        sol = bg.assemble(
                            K, f_try, N_b, y0, 0.0, ec=4, phases=4, time_limit=6.0
                        )
                        if sol is not None and len(sol) >= n:
                            arr = np.array(sol[:n], dtype=np.float64)
                            if float(np.max(np.abs(arr[:, 2]))) > 2.0:
                                arr[:, 2] = np.radians(arr[:, 2])
                            cands.append((f"bg_N{N_b}_f{f_try:.3f}", arr))
                            if len(cands) >= 12:
                                break
                    except Exception:
                        continue
                if len(cands) >= 12:
                    break
            if len(cands) >= 12:
                break
    except Exception:
        pass
    # 2) lband.assemble_L sweep
    try:
        import lband as lb

        for f_try in (f_tgt + 0.003, f_tgt + 0.008, L0 - K - 0.002):
            if not (0.45 < f_try < 0.96):
                continue
            for mode in (0, 1):
                for N_b in range(max(1, K - 3), K + 2):
                    try:
                        sol = lb.assemble_L(
                            K,
                            f_try,
                            N_b,
                            0.0,
                            0.0,
                            mode,
                            ec=4,
                            phases=4,
                            time_limit=6.0,
                        )
                        if sol is not None and len(sol) >= n:
                            arr = np.array(sol[:n], dtype=np.float64)
                            if float(np.max(np.abs(arr[:, 2]))) > 2.0:
                                arr[:, 2] = np.radians(arr[:, 2])
                            cands.append((f"lband_N{N_b}_m{mode}", arr))
                            if len(cands) >= 24:
                                break
                    except Exception:
                        continue
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        for tag, arr in cands:
            r = f2d.run_fq(
                w12.span(arr) + 1e-4,
                arr,
                tmp,
                [
                    "--loosen",
                    "1.015",
                    "--kick",
                    "0.005",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "48",
                    "--pit",
                    "300",
                ],
                timeout=60,
            )
            if r is not None and r[0] < best_s - 1e-9:
                best_s, best_a, best_tag = r[0], r[1], tag
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
                return best_tag, (Lf, xf, yf, tf)
    return "param_none", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=3500)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    n, Lb = TOP20_TARGETS[args.shard % len(TOP20_TARGETS)]
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    L0, sq0 = cur[n]
    print(
        f"[W16-Sniper Shard {args.shard}/20] TARGET n={n} (cur={L0:.7f}, Lb={Lb:.6f}, gap_L={L0 - Lb:.5f}, gap_dS={(L0 - Lb) / math.sqrt(n):.6f}) | frontier={tot_score:.10f} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    jobs = []
    # 1) Exhaustive Window-Peel from ALL bigger packings m > n!
    peels = harvest_window_peels(n, L0, cur, max_cands=80)
    print(
        f"  Harvested {len(peels)} window-peel candidates (best raw gap={peels[0][0]:+.5f})"
        if peels
        else "  No window peels",
        flush=True,
    )
    for i, (gap, arr, tag) in enumerate(peels):
        jobs.append(
            ("peel_or_surg", n, L0, Lb, arr, tag, 1600001 + args.shard * 10000 + i * 17)
        )

    # 2) Multi-Neighbor Stretch (+-1..+-4) & Delete-k (n+1..n+4) & Insert-k (n-3..n-1)!
    for m in range(max(10, n - 4), min(1000, n + 5)):
        if m == n:
            continue
        Lm, sqm = cur[m]
        if m > n and Lm < L0 + 0.08:
            # Drop m-n tightest wall squares
            he = 0.5 * (np.abs(np.cos(sqm[:, 2])) + np.abs(np.sin(sqm[:, 2])))
            wd = np.minimum(
                np.minimum(
                    sqm[:, 0] - he - np.min(sqm[:, 0] - he),
                    np.max(sqm[:, 0] + he) - (sqm[:, 0] + he),
                ),
                np.minimum(
                    sqm[:, 1] - he - np.min(sqm[:, 1] - he),
                    np.max(sqm[:, 1] + he) - (sqm[:, 1] + he),
                ),
            )
            drop = np.argsort(wd)[: m - n]
            jobs.append(
                (
                    "peel_or_surg",
                    n,
                    L0,
                    Lb,
                    np.delete(sqm, drop, axis=0),
                    f"del_{m}->{n}",
                    1610001 + m * 31,
                )
            )
    # Stretch from nearby K +- 1..4
    K = math.isqrt(n)
    for dk in (-3, -2, -1, 1, 2, 3):
        Ks = K + dk
        if Ks < 8 or Ks > 31:
            continue
        for sn in range(max(64, Ks * Ks + 1), min(1000, (Ks + 1) * (Ks + 1))):
            if cur[sn][0] >= Ks + 1 - 1e-6:
                continue
            for ci in (0, 1):
                b = st_mod.stretch(cur[sn][1], -dk)[0] if ci == 0 else None
                if b is not None and len(b) == n:
                    jobs.append(
                        (
                            "peel_or_surg",
                            n,
                            L0,
                            Lb,
                            b,
                            f"st_{sn}->{n}",
                            1620001 + sn * 29,
                        )
                    )

    # 3) Full-Box Global Thermal Melts (40 diverse full-box melts!)
    for i in range(40):
        jobs.append(
            (
                "global_melt",
                n,
                L0,
                Lb,
                sq0,
                f"gmelt_{i}",
                1630001 + args.shard * 1000 + i * 23,
            )
        )

    # 4) From-Scratch Circle-to-Square Global Morphs (160 completely fresh global topologies!)
    kinds = (
        "two_corner_tilted",
        "double_chord_band",
        "l_border_tilted",
        "random_uniform",
        "rotated_global_lattice",
    )
    for i in range(160):
        jobs.append(
            (
                "morph_scratch",
                n,
                L0,
                Lb,
                None,
                kinds[i % len(kinds)],
                1640001 + args.shard * 1000 + i * 37,
            )
        )

    # Interleave so every mode runs early!
    random.Random(1600000 + args.shard).shuffle(jobs)
    print(
        f"[W16-Sniper n={n}] Total global-topology jobs={len(jobs)} + parametric band sweep on {args.procs} cores",
        flush=True,
    )

    best_win = None
    best_L = L0
    done = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_parametric_bands, (n, L0, Lb, 1690001 + args.shard))]
        futs.extend(ex.submit(worker_eval_seed, j) for j in jobs)
        for fut in as_completed(futs):
            done += 1
            try:
                tag, res = fut.result()
            except Exception:
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                if Lf < best_L - 1e-10:
                    dL = Lf - L0
                    best_L = Lf
                    best_win = res
                    print(
                        f"  [W16 SNIPER HIT!!] n={n:4d} via {tag:22s} {L0:.7f} -> {Lf:.7f} dL={dL:.7f} dS={dL / math.sqrt(n):.8f} ({done}/{len(futs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    # Checkpoint immediately on any win!
                    np.savez_compressed(
                        args.out_npz,
                        **{
                            f"L_{n}": np.array([Lf], dtype=np.float64),
                            f"x_{n}": np.asarray(xf, dtype=np.float64),
                            f"y_{n}": np.asarray(yf, dtype=np.float64),
                            f"t_{n}": np.asarray(tf, dtype=np.float64),
                        },
                    )
            if done % 25 == 0:
                print(
                    f"  [W16 n={n}] {done}/{len(futs)} done, best_L={best_L:.7f} (dL={best_L - L0:.7f}, {time.time() - t0:.0f}s)",
                    flush=True,
                )
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W16 n={n}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    if best_win is None:
        np.savez_compressed(args.out_npz)
    print(
        f"[W16 SNIPER n={n} DONE] best_L={best_L:.7f} (dL={best_L - L0:.7f}, dS={(best_L - L0) / math.sqrt(n):.8f}, {done} trials, {time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
