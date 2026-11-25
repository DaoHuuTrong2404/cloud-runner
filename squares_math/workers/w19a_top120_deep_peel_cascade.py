#!/usr/bin/env python3
"""Wave 19A Deep Top-120 Biggest-Gap Peel+Stretch+Del + Post-Win Iterative Cascade (40 Cores / 10 Shards).

Focuses 3x deeper search exclusively on the Top-120 highest-lag puzzles on the 22.08785 R14 frontier
(incorporating W17A's brand-new peel winners n=500, 371, 654, 460 + all W17B winners!):
  - Top 28 sharpest multi-mode peels (exact corner, over-1 trim, over-2 trim, 4-quadrant interior shift dx,dy in 0..2)
  - Multi-orientation stretch (dk in -5..+5 across all 4 symmetries)
  - Smart wall / corner / tilted-stress delete (n+1..n+4 -> n) + void insert-1 (n-1 -> n)
  - Tight near-miss (+0.024) 5-step kick chain + automatic 2-round post-win Basin-Cascade (melt + round-press + LP)!
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


def harvest_deep_peels(n, L0, cur, nongrid_sorted, max_cands=28):
    cands = []
    seen = set()
    for m in nongrid_sorted:
        if m <= n:
            continue
        Lm, sq = cur[m]
        for si, sym in enumerate(f2d.SYMS):
            sq_m, xlo, xhi, ylo, yhi = f2d.mapped(sq, sym)
            R = np.maximum(xhi, yhi)
            # (a) Exact corner crop
            take = np.argpartition(R, n - 1)[:n]
            L_crop = float(R[take].max())
            gap = L_crop - L0
            if gap < 0.12:
                sig = (m, round(gap, 4), 0)
                if sig not in seen:
                    seen.add(sig)
                    cands.append((gap, sq_m[take].copy(), f"peel_{m}_s{si}"))
            # (b) Over-1 bottleneck-trim crop
            if m >= n + 1 and gap < 0.13:
                take1 = np.argpartition(R, n)[: n + 1]
                sub = sq_m[take1]
                sxhi, syhi = xhi[take1], yhi[take1]
                drop_local = (
                    int(np.argmax(sxhi))
                    if float(sxhi.max()) >= float(syhi.max())
                    else int(np.argmax(syhi))
                )
                sub_n = np.delete(sub, drop_local, axis=0)
                gap1 = w12.span(sub_n) - L0
                if gap1 < 0.12 and abs(gap1 - gap) > 1e-5:
                    sig1 = (m, round(gap1, 4), 1)
                    if sig1 not in seen:
                        seen.add(sig1)
                        cands.append((gap1, sub_n, f"peel1_{m}_s{si}"))
            # (c) Over-2 double-bottleneck-trim crop
            if m >= n + 2 and gap < 0.12:
                take2 = np.argpartition(R, n + 1)[: n + 2]
                sub2 = sq_m[take2]
                for _ in range(2):
                    he2 = 0.5 * (
                        np.abs(np.cos(sub2[:, 2])) + np.abs(np.sin(sub2[:, 2]))
                    )
                    sx2, sy2 = sub2[:, 0] + he2, sub2[:, 1] + he2
                    dl = (
                        int(np.argmax(sx2))
                        if float(sx2.max()) >= float(sy2.max())
                        else int(np.argmax(sy2))
                    )
                    sub2 = np.delete(sub2, dl, axis=0)
                gap2 = w12.span(sub2) - L0
                if gap2 < 0.115 and abs(gap2 - gap) > 1e-5:
                    sig2 = (m, round(gap2, 4), 2)
                    if sig2 not in seen:
                        seen.add(sig2)
                        cands.append((gap2, sub2, f"peel2_{m}_s{si}"))
            # (d) Interior-shift crop (dx, dy in 0..2) for m <= n + 280
            if si in (0, 2, 4, 6) and m <= n + 280:
                for dx, dy in (
                    (1.0, 0.0),
                    (0.0, 1.0),
                    (1.0, 1.0),
                    (2.0, 0.0),
                    (0.0, 2.0),
                ):
                    keep = np.flatnonzero((xlo >= dx - 0.18) & (ylo >= dy - 0.18))
                    if len(keep) < n:
                        continue
                    Ri = np.maximum(xhi[keep] - dx, yhi[keep] - dy)
                    tk = keep[np.argpartition(Ri, n - 1)[:n]]
                    sub_i = sq_m[tk].copy()
                    gapi = w12.span(sub_i) - L0
                    if gapi < 0.105:
                        sigi = (m, round(gapi, 4), 3)
                        if sigi not in seen:
                            seen.add(sigi)
                            cands.append((gapi, sub_i, f"ipeel_{m}_s{si}"))
    cands.sort(key=lambda z: z[0])
    out = []
    cnt_m = {}
    for gap, arr, tag in cands:
        ms = tag.split("_")[1]
        if cnt_m.get(ms, 0) < 3:
            cnt_m[ms] = cnt_m.get(ms, 0) + 1
            out.append((gap, arr, tag))
            if len(out) >= max_cands:
                break
    return out


def harvest_deep_stretches(n, L0, cur, max_cands=10):
    K = math.isqrt(n)
    cands = []
    seen = set()
    for dk in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
        Ks = K + dk
        if Ks < 6 or Ks > 31:
            continue
        for sn in range(max(36, Ks * Ks + 1), min(1001, (Ks + 1) * (Ks + 1))):
            if cur[sn][0] >= Ks + 1 - 1e-6:
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
                    if gap < 0.16:
                        sig = (sn, round(gap, 4))
                        if sig not in seen:
                            seen.add(sig)
                            cands.append((gap, b, f"st_{sn}o{oi}->{n}"))
    cands.sort(key=lambda z: z[0])
    return cands[:max_cands]


def harvest_deep_del_ins(n, L0, cur, max_cands=10):
    cands = []
    seen = set()
    for m in range(n + 1, min(1001, n + 5)):
        Lm, sqm = cur[m]
        if Lm >= L0 + 0.11:
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
        arr_w = np.delete(sqm, np.argsort(wd)[:kdrop], axis=0)
        cands.append((w12.span(arr_w) - L0, arr_w, f"delW_{m}->{n}"))
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
        for ci, sym in (
            (0, f2d.SYMS[0]),
            (2, f2d.SYMS[2]),
            (4, f2d.SYMS[4]),
            (6, f2d.SYMS[6]),
        ):
            sm, _, sxhi, _, syhi = f2d.mapped(sqm, sym)
            arr_c = np.delete(sm, np.argsort(-(sxhi + syhi))[:kdrop], axis=0)
            gc = w12.span(arr_c) - L0
            sig = (m, round(gc, 4), ci)
            if sig not in seen:
                seen.add(sig)
                cands.append((gc, arr_c, f"delC{ci}_{m}->{n}"))

    if n >= 50:
        Lm1, sq1 = cur[n - 1]
        if Lm1 < L0 - 0.001:
            he = 0.5 * (np.abs(np.cos(sq1[:, 2])) + np.abs(np.sin(sq1[:, 2])))
            lx0, lx1 = float((sq1[:, 0] - he).min()), float((sq1[:, 0] + he).max())
            ly0, ly1 = float((sq1[:, 1] - he).min()), float((sq1[:, 1] + he).max())
            gx = np.linspace(lx0 + 0.5, lx1 - 0.5, max(18, int((lx1 - lx0) * 3)))
            gy = np.linspace(ly0 + 0.5, ly1 - 0.5, max(18, int((ly1 - ly0) * 3)))
            XX, YY = np.meshgrid(gx, gy)
            pts = np.column_stack([XX.ravel(), YY.ravel()])
            dists, _ = cKDTree(sq1[:, :2]).query(pts, k=1)
            best_pt = pts[int(np.argmax(dists))]
            for ai, ang in ((0, 0.0), (1, math.pi / 4.0)):
                arr_i = np.vstack([sq1, np.array([[best_pt[0], best_pt[1], ang]])])
                cands.append((Lm1 - L0 + 0.03, arr_i, f"insV{ai}_{n - 1}->{n}"))

    cands.sort(key=lambda z: z[0])
    return cands[:max_cands]


def worker_eval_deep(job):
    n, L0, arr, tag, seed = job
    rng = random.Random(seed)
    best_s = L0
    best_a = None
    local_best = None
    with tempfile.TemporaryDirectory() as tmp:
        s_in = w12.span(arr) + 1e-4
        to = 50 if n <= 500 else 65
        schedules = (
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
                rng.choice(("0.10", "0.16", "0.22")),
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
                "170",
            ],
        )
        for sc in schedules:
            r = f2d.run_fq(s_in, arr, tmp, sc, timeout=to)
            if r is not None:
                if local_best is None or r[0] < local_best[0]:
                    local_best = r
                if r[0] < best_s - 1e-9:
                    best_s, best_a = r[0], r[1]

        # 6-step deep follow-up kick chain if won OR promising near-miss (+0.024)!
        if local_best is not None and local_best[0] < best_s + 0.024:
            ls, la = local_best
            for _ in range(6):
                r2 = f2d.run_fq(
                    ls,
                    la,
                    tmp,
                    [
                        "--loosen",
                        rng.choice(("1.004", "1.010", "1.022", "1.038")),
                        "--kick",
                        rng.choice(("0.002", "0.006", "0.015")),
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

        # Post-win 2-round mini-cascade (large-radius melt + round-press + deep --no-alm + LP!)
        if best_a is not None and best_s < L0 - 1e-9:
            for _ in range(2):
                prev = best_s
                if os.path.exists(ANNEAL_BIN):
                    for rad_lo, rad_hi in ((2.5, 4.8), (4.5, 7.8)):
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
                                    f"{rng.uniform(0.10, 0.25):.3f}",
                                    "--sweeps",
                                    "4000",
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
                                sm, marr = f2d.read_fq_cfg_rad(pout)
                                if len(marr) == n:
                                    rm = f2d.run_fq(
                                        sm + 1e-4,
                                        marr,
                                        tmp,
                                        [
                                            "--loosen",
                                            "1.012",
                                            "--kick",
                                            "0.005",
                                            "--seed",
                                            str(rng.randrange(1 << 30)),
                                            "--flip-top",
                                            "52",
                                            "--pit",
                                            "320",
                                        ],
                                        timeout=58,
                                    )
                                    if rm is not None and rm[0] < best_s - 1e-9:
                                        best_s, best_a = rm[0], rm[1]
                        except Exception:
                            pass
                for r0 in ("0.08", "0.16"):
                    rrp = f2d.run_fq(
                        best_s + 0.008,
                        best_a,
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
                    if rrp is not None and rrp[0] < best_s - 1e-9:
                        best_s, best_a = rrp[0], rrp[1]
                if prev - best_s < 1e-6:
                    break

            r_fin = f2d.run_fq(
                best_s,
                best_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "750"],
                timeout=85,
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

    nongrid_sorted = [
        m
        for m in range(20, 1001)
        if m != math.isqrt(m) ** 2 and cur[m][0] < math.isqrt(m) + 1 - 1e-6
    ]
    # Top-120 highest remaining lag puzzles
    scored = []
    for n in nongrid_sorted:
        if n < 50:
            continue
        scored.append((lag_rank.get(n, 0.0), n))
    scored.sort(reverse=True)
    targets = [n for _, n in scored[:120]]
    my_slice = targets[args.shard :: args.num_shards]
    print(
        f"[W19A-Top120DeepPeel s={args.shard}/{args.num_shards}] Frontier={tot_score:.10f} "
        f"({len(fresh)} fresh from {n_files} files), my_puzzles={len(my_slice)}/120 ({time.time() - t0:.1f}s)",
        flush=True,
    )

    cand_queue = []
    for idx_n, n in enumerate(my_slice):
        L0 = cur[n][0]
        dS_w = lag_rank.get(n, 0.0001)
        peels = harvest_deep_peels(n, L0, cur, nongrid_sorted, max_cands=28)
        stretches = harvest_deep_stretches(n, L0, cur, max_cands=10)
        del_ins = harvest_deep_del_ins(n, L0, cur, max_cands=10)
        for ci, (gap, arr, tag) in enumerate(peels + stretches + del_ins):
            rank_key = gap - 70.0 * min(dS_w, 0.0012)
            cand_queue.append(
                (
                    rank_key,
                    n,
                    L0,
                    arr,
                    tag,
                    1900001 + args.shard * 100000 + idx_n * 500 + ci * 19,
                )
            )
    cand_queue.sort(key=lambda z: z[0])
    cand_queue = cand_queue[:240]
    jobs = [(n, L0, arr, tag, seed) for _, n, L0, arr, tag, seed in cand_queue]
    print(
        f"[W19A s={args.shard}] Harvested {len(jobs)} top-120 peel/stretch/del jobs in {time.time() - t0:.1f}s",
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
        futs = [ex.submit(worker_eval_deep, j) for j in jobs]
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
                        f"  [W19A WIN!!] n={n:4d} via {tag:22s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL0:.7f} dS={dL0 / math.sqrt(n):.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if done % 20 == 0:
                print(
                    f"  [W19A s={args.shard}] {done}/{len(jobs)} done, wins={len(wins)}, "
                    f"dS={tot_ds:.8f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W19A s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W19A s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
