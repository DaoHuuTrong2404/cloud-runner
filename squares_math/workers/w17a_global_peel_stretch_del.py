#!/usr/bin/env python3
"""Wave 17A Global All-580 Multi-Mode Window-Peel + Multi-Cut Stretch + Smart Delete/Insert Engine (80 Cores / 20 Shards).

Scales Wave 16's 100%-proven winning topology-transfer operators (which hit 4/20 targets for -0.000187 dS
on just 20 puzzles: n=545 -0.00296 via peel_698, n=491 -0.00077 via del_492, n=785 -0.00048 via peel_902,
n=499 -0.00020 via st_644) across ALL ~580 non-grid packings n=50..1000 with ZERO wasted scratch compute:
  1) Exact Corner Peel + Over-1 Bottleneck-Trim Peel + Interior-Shift Peel from ALL m > n x 8 symmetries
  2) Multi-Orientation Stretch (dk in -4..+4) across rotated/mirrored arm cuts
  3) Smart Wall / Corner / Tilted-Stress Delete (n+1..n+3 -> n) + Empty-Void Insert-1 (n-1 -> n)
  4) Global sharpness-ranked priority queue -> fq + round-press + 5-step kick chain + post-win melt + LP + Decimal64!
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
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
sys.path.insert(0, DS_DIR)
sys.path.insert(0, HERE)

import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12
import stretch as st_mod

ANNEAL_BIN = os.environ.get("ANNEAL_BIN", "/tmp/sq_bins/anneal")


def harvest_peels_for_n(n, L0, cur, nongrid_sorted, max_cands=12):
    """Exact corner peels + over-1 bottleneck-trim peels + interior-shift peels from all m > n."""
    cands = []
    seen = set()
    for m in nongrid_sorted:
        if m <= n:
            continue
        Lm, sq = cur[m]
        for si, sym in enumerate(f2d.SYMS):
            sq_m, xlo, xhi, ylo, yhi = f2d.mapped(sq, sym)
            R = np.maximum(xhi, yhi)
            # (a) Exact corner crop [0, R]^2
            take = np.argpartition(R, n - 1)[:n]
            L_crop = float(R[take].max())
            gap = L_crop - L0
            if gap < 0.095:
                sig = (m, round(gap, 4), 0)
                if sig not in seen:
                    seen.add(sig)
                    cands.append((gap, sq_m[take].copy(), f"peel_{m}_s{si}"))
            # (b) Over-1 bottleneck-trim crop: take n+1, drop limiting xhi or yhi square!
            if m >= n + 1 and gap < 0.11:
                take1 = np.argpartition(R, n)[: n + 1]
                sub = sq_m[take1]
                sxhi, syhi = xhi[take1], yhi[take1]
                mx, my = float(sxhi.max()), float(syhi.max())
                drop_local = int(np.argmax(sxhi)) if mx >= my else int(np.argmax(syhi))
                sub_n = np.delete(sub, drop_local, axis=0)
                gap1 = w12.span(sub_n) - L0
                if gap1 < 0.095 and abs(gap1 - gap) > 1e-5:
                    sig1 = (m, round(gap1, 4), 1)
                    if sig1 not in seen:
                        seen.add(sig1)
                        cands.append((gap1, sub_n, f"peel1_{m}_s{si}"))
            # (c) Interior-shift crop (peel off 1-2 distorted outer rows/cols) for nearby m <= n + 200
            if si in (0, 2, 4, 6) and m <= n + 200:
                for dx, dy in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
                    keep = np.flatnonzero((xlo >= dx - 0.18) & (ylo >= dy - 0.18))
                    if len(keep) < n:
                        continue
                    Ri = np.maximum(xhi[keep] - dx, yhi[keep] - dy)
                    tk = keep[np.argpartition(Ri, n - 1)[:n]]
                    sub_i = sq_m[tk].copy()
                    gapi = w12.span(sub_i) - L0
                    if gapi < 0.085:
                        sigi = (m, round(gapi, 4), 2)
                        if sigi not in seen:
                            seen.add(sigi)
                            cands.append((gapi, sub_i, f"ipeel_{m}_s{si}"))
    cands.sort(key=lambda z: z[0])
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


def harvest_stretches_for_n(n, L0, cur, max_cands=6):
    """Multi-orientation arm stretch from K +- 1..4 shells."""
    K = math.isqrt(n)
    cands = []
    seen = set()
    for dk in (-4, -3, -2, -1, 1, 2, 3, 4):
        Ks = K + dk
        if Ks < 6 or Ks > 31:
            continue
        for sn in range(max(36, Ks * Ks + 1), min(1001, (Ks + 1) * (Ks + 1))):
            if cur[sn][0] >= Ks + 1 - 1e-6:
                continue
            sq_sn = cur[sn][1]
            # Test original + mirror_x + rot90 so pick_cuts hits distinct arm cut lines!
            for oi, sym in ((0, f2d.SYMS[0]), (1, f2d.SYMS[1]), (2, f2d.SYMS[2])):
                sq_o = sq_sn if oi == 0 else f2d.mapped(sq_sn, sym)[0]
                try:
                    b, _ = st_mod.stretch(sq_o, -dk)
                except Exception:
                    b = None
                if b is not None and len(b) == n:
                    gap = w12.span(b) - L0
                    if gap < 0.14:
                        sig = (sn, round(gap, 4))
                        if sig not in seen:
                            seen.add(sig)
                            cands.append((gap, b, f"st_{sn}o{oi}->{n}"))
    cands.sort(key=lambda z: z[0])
    return cands[:max_cands]


def harvest_del_ins_for_n(n, L0, cur, max_cands=6):
    """Wall/Corner/Tilted delete from n+1..n+3 and void insert-1 from n-1."""
    cands = []
    seen = set()
    for m in range(n + 1, min(1001, n + 4)):
        Lm, sqm = cur[m]
        if Lm >= L0 + 0.095:
            continue
        kdrop = m - n
        he = 0.5 * (np.abs(np.cos(sqm[:, 2])) + np.abs(np.sin(sqm[:, 2])))
        xlo, xhi = sqm[:, 0] - he, sqm[:, 0] + he
        ylo, yhi = sqm[:, 1] - he, sqm[:, 1] + he
        lx0, lx1, ly0, ly1 = (
            float(xlo.min()),
            float(xhi.max()),
            float(ylo.min()),
            float(yhi.max()),
        )
        wd = np.minimum(
            np.minimum(xlo - lx0, lx1 - xhi), np.minimum(ylo - ly0, ly1 - yhi)
        )
        # Variant A: closest to walls
        arr_w = np.delete(sqm, np.argsort(wd)[:kdrop], axis=0)
        gw = w12.span(arr_w) - L0
        cands.append((gw, arr_w, f"delW_{m}->{n}"))
        # Variant B: closest tilted squares to walls
        degs = np.degrees(sqm[:, 2]) % 90.0
        is_tilt = np.minimum(degs, 90.0 - degs) > 2.0
        if int(is_tilt.sum()) >= kdrop:
            score_t = np.where(is_tilt, wd, 999.0)
            arr_t = np.delete(sqm, np.argsort(score_t)[:kdrop], axis=0)
            gt = w12.span(arr_t) - L0
            sig = (m, round(gt, 4), "T")
            if sig not in seen:
                seen.add(sig)
                cands.append((gt, arr_t, f"delT_{m}->{n}"))
        # Variant C: extreme corner drop (max xhi+yhi across 2 corners)
        for ci, sym in ((0, f2d.SYMS[0]), (2, f2d.SYMS[2])):
            sm, _, sxhi, _, syhi = f2d.mapped(sqm, sym)
            arr_c = np.delete(sm, np.argsort(-(sxhi + syhi))[:kdrop], axis=0)
            gc = w12.span(arr_c) - L0
            sig = (m, round(gc, 4), ci)
            if sig not in seen:
                seen.add(sig)
                cands.append((gc, arr_c, f"delC{ci}_{m}->{n}"))

    # Void insert-1 from n-1 if n-1 is tightly packed below L0
    if n >= 50:
        Lm1, sq1 = cur[n - 1]
        if Lm1 < L0 - 0.002:
            he = 0.5 * (np.abs(np.cos(sq1[:, 2])) + np.abs(np.sin(sq1[:, 2])))
            lx0, lx1 = float((sq1[:, 0] - he).min()), float((sq1[:, 0] + he).max())
            ly0, ly1 = float((sq1[:, 1] - he).min()), float((sq1[:, 1] + he).max())
            gx = np.linspace(lx0 + 0.5, lx1 - 0.5, max(16, int((lx1 - lx0) * 3)))
            gy = np.linspace(ly0 + 0.5, ly1 - 0.5, max(16, int((ly1 - ly0) * 3)))
            XX, YY = np.meshgrid(gx, gy)
            pts = np.column_stack([XX.ravel(), YY.ravel()])
            dists, _ = cKDTree(sq1[:, :2]).query(pts, k=1)
            best_pt = pts[int(np.argmax(dists))]
            for ai, ang in ((0, 0.0), (1, math.pi / 4.0)):
                arr_i = np.vstack([sq1, np.array([[best_pt[0], best_pt[1], ang]])])
                cands.append((Lm1 - L0 + 0.03, arr_i, f"insV{ai}_{n - 1}->{n}"))

    cands.sort(key=lambda z: z[0])
    return cands[:max_cands]


def worker_eval_candidate(job):
    """Run fq + round-press + 5-step kick chain + post-win melt + LP + Decimal64 on 1 candidate."""
    n, L0, arr, tag, seed = job
    rng = random.Random(seed)
    best_s = L0
    best_a = None
    local_best = None
    with tempfile.TemporaryDirectory() as tmp:
        s_in = w12.span(arr) + 1e-4
        to = 46 if n <= 500 else 60
        is_ins = tag.startswith("insV")
        schedules = (
            (
                [
                    "--loosen",
                    "1.0",
                    "--round-r0",
                    "0.22",
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
                [
                    "--loosen",
                    "1.035",
                    "--kick",
                    "0.014",
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "40",
                    "--pit",
                    "240",
                ],
            )
            if is_ins
            else (
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
                    rng.choice(("0.11", "0.18")),
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
            )
        )
        for sc in schedules:
            r = f2d.run_fq(s_in, arr, tmp, sc, timeout=to)
            if r is not None:
                if local_best is None or r[0] < local_best[0]:
                    local_best = r
                if r[0] < best_s - 1e-9:
                    best_s, best_a = r[0], r[1]

        # 5-step deep follow-up kick chain if won OR promising near-miss (+0.018)!
        if local_best is not None and local_best[0] < best_s + 0.018:
            ls, la = local_best
            for _ in range(5):
                r2 = f2d.run_fq(
                    ls,
                    la,
                    tmp,
                    [
                        "--loosen",
                        rng.choice(("1.004", "1.010", "1.022", "1.036")),
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
                if r2 is not None and r2[0] < best_s - 1e-9:
                    best_s, best_a = r2[0], r2[1]

        # Post-win squeeze: if we beat L0, run 2 targeted melts + deep --no-alm + LP + Decimal64!
        if best_a is not None and best_s < L0 - 1e-9:
            if os.path.exists(ANNEAL_BIN):
                for rad_lo, rad_hi in ((2.5, 4.5), (4.5, 7.0)):
                    try:
                        pin = os.path.join(tmp, "wm_in.txt")
                        pout = os.path.join(tmp, "wm_out.txt")
                        f2d.write_fq_cfg(pin, best_s, best_a)
                        if os.path.exists(pout):
                            os.remove(pout)
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
                                repr(float(best_a[pick, 0])),
                                "--cy",
                                repr(float(best_a[pick, 1])),
                                "--rad",
                                f"{rng.uniform(rad_lo, rad_hi):.3f}",
                                "--rmax",
                                f"{rng.uniform(0.09, 0.22):.3f}",
                                "--sweeps",
                                "3600",
                                "--bp",
                                "2200.0",
                                "--seed",
                                str(rng.randrange(1 << 30)),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=40,
                        )
                        if os.path.exists(pout):
                            sm, marr = f2d.read_fq_cfg_rad(pout)
                            if len(marr) == n:
                                rm = f2d.run_fq(
                                    sm + 1e-4,
                                    marr,
                                    tmp,
                                    [
                                        "--loosen",
                                        "1.010",
                                        "--kick",
                                        "0.004",
                                        "--seed",
                                        str(rng.randrange(1 << 30)),
                                        "--flip-top",
                                        "48",
                                        "--pit",
                                        "300",
                                    ],
                                    timeout=55,
                                )
                                if rm is not None and rm[0] < best_s - 1e-9:
                                    best_s, best_a = rm[0], rm[1]
                    except Exception:
                        pass

            r_fin = f2d.run_fq(
                best_s,
                best_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "700"],
                timeout=80,
            )
            if r_fin is not None and r_fin[0] < best_s:
                best_s, best_a = r_fin[0], r_fin[1]
            best_a = w12.lp_rot_polish(best_a)
            ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, best_a)
            if ok and Lf < L0 - 1e-9:
                return n, tag, (Lf, xf, yf, tf)
    return n, tag, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--lags-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
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

    nongrid_sorted = [
        m
        for m in range(20, 1001)
        if m != math.isqrt(m) ** 2 and cur[m][0] < math.isqrt(m) + 1 - 1e-6
    ]
    # All non-grid packings n=50..1000 prioritized by lag + fresh + high-weight small-n
    scored = []
    for n in nongrid_sorted:
        if n < 50:
            continue
        prio = (
            lag_rank.get(n, 0.0)
            + (0.00012 if n in fresh else 0.0)
            + (0.00004 if n < 250 else 0.0)
        )
        scored.append((prio, n))
    scored.sort(reverse=True)
    targets = [n for _, n in scored]
    my_slice = targets[args.shard :: args.num_shards]
    print(
        f"[W17A-GlobalPeelStretchDel s={args.shard}/{args.num_shards}] Frontier={tot_score:.10f} "
        f"({len(fresh)} fresh from {n_files} files), my_puzzles={len(my_slice)}/{len(targets)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    # Harvest sharpness-ranked candidate queue across all my_slice puzzles!
    cand_queue = []
    for idx_n, n in enumerate(my_slice):
        L0 = cur[n][0]
        dS_w = lag_rank.get(n, 0.00005)
        peels = harvest_peels_for_n(n, L0, cur, nongrid_sorted, max_cands=12)
        stretches = harvest_stretches_for_n(n, L0, cur, max_cands=6)
        del_ins = harvest_del_ins_for_n(n, L0, cur, max_cands=6)
        for ci, (gap, arr, tag) in enumerate(peels + stretches + del_ins):
            # Lower sort key runs FIRST: tightest raw gap + highest lag_dS priority!
            rank_key = gap - 80.0 * min(dS_w, 0.0012)
            cand_queue.append(
                (
                    rank_key,
                    n,
                    L0,
                    arr,
                    tag,
                    1700001 + args.shard * 100000 + idx_n * 500 + ci * 13,
                )
            )
    cand_queue.sort(key=lambda z: z[0])
    # Cap at 220 sharpest jobs per shard (~55 jobs/core * ~55s ~ 3000s cleanly inside 3550s!)
    cand_queue = cand_queue[:220]
    jobs = [(n, L0, arr, tag, seed) for _, n, L0, arr, tag, seed in cand_queue]
    print(
        f"[W17A s={args.shard}] Harvested {len(jobs)} sharpest peel/stretch/del/ins jobs "
        f"(top gap={cand_queue[0][0]:+.4f} .. {cand_queue[-1][0]:+.4f}) in {time.time() - t0:.1f}s",
        flush=True,
    )

    wins = {}
    best_L = {n: cur[n][0] for n in my_slice}
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
        futs = [ex.submit(worker_eval_candidate, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception:
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                if Lf < best_L[n] - 1e-10:
                    prev_L = best_L[n]
                    best_L[n] = Lf
                    wins[n] = res
                    tot_ds += (Lf - prev_L) / math.sqrt(n)
                    dL0 = Lf - cur[n][0]
                    print(
                        f"  [W17A WIN!!] n={n:4d} via {tag:22s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL0:.7f} dS={dL0 / math.sqrt(n):.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if done % 20 == 0:
                print(
                    f"  [W17A s={args.shard}] {done}/{len(jobs)} done, wins={len(wins)}, "
                    f"dS={tot_ds:.8f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W17A s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W17A s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} "
        f"({done}/{len(jobs)} trials) -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
