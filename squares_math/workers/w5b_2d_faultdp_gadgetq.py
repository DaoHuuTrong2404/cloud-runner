#!/usr/bin/env python3
"""Wave 5B Engine (Cloud Cluster): True 2D Staircase Multi-Gadget Fault-Line DP (Deotte/Chilton/JoeGun)
+ Padded-Square Rust `fq` Gadget Quench (gadgetq) + Seam-Relaxing `fq` Quench + Decimal64 >= 1.05e-9 Certification.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
sys.path.insert(0, DS_DIR)

from decimal_verifier import verify_puzzle_decimal

FQ_BIN = os.environ.get("FQ_BIN", "/tmp/sq_bins/fq")
TARGET_GAP = Decimal("1e-9")
SAFE_GAP = Decimal("1.05e-9")

WMAX, DMAX, FMAX, E = 31, 62, 0.97, 1e-9
THR = np.array([0.0, 0.3, 0.45, 0.6, 0.75, 0.9])
NB = len(THR)
SYMS = [(sx, sy, tr) for sx in (1, -1) for sy in (1, -1) for tr in (0, 1)]


def side(sq):
    he = 0.5 * (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2])))
    return float(
        max(
            (sq[:, 0] + he).max() - (sq[:, 0] - he).min(),
            (sq[:, 1] + he).max() - (sq[:, 1] - he).min(),
        )
    )


def mapped(sq, sym):
    sx, sy, tr = sym
    x, y, t = sq[:, 0] * sx, sq[:, 1] * sy, sq[:, 2] * sx * sy
    if tr:
        x, y, t = y, x, -t
    he = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    x, y = x - (x - he).min(), y - (y - he).min()
    return np.c_[x, y, t], x - he, x + he, y - he, y + he


def extract_2d_gadgets(donors):
    W = np.arange(1, WMAX + 1)
    DM = np.arange(1, DMAX + 1)
    best = {}
    for di, sq in enumerate(donors):
        n = len(sq)
        L = side(sq)
        parts = []
        for si, sym in enumerate(SYMS):
            _, xlo, xhi, ylo, yhi = mapped(sq, sym)
            for h in range(1, min(WMAX, math.ceil(L)) + 1):
                R = np.maximum(
                    np.maximum(xhi[:, None] - W[None, :], (yhi - h)[:, None]), 0.0
                )
                order = np.argsort(R, axis=0, kind="stable")
                Rs = np.take_along_axis(R, order, axis=0)
                ybr = np.minimum.accumulate(
                    np.where(xhi[order] > W[None, :] + 1e-9, ylo[order], np.inf), axis=0
                )
                xtl = np.minimum.accumulate(
                    np.where(yhi[order] > h + 1e-9, xlo[order], np.inf), axis=0
                )
                idx = W[:, None] * h + DM[None, :] - 1
                ok = (idx < n) & (W[:, None] <= math.ceil(L))
                wi, dj = np.nonzero(ok)
                k = idx[wi, dj]
                f = Rs[k, wi]
                m = f <= FMAX
                wi, dj, k, f = wi[m], dj[m], k[m], f[m]
                if len(f):
                    parts.append(
                        np.c_[
                            wi + 1,
                            np.full(len(f), h),
                            dj + 1,
                            f,
                            np.minimum(ybr[k, wi], 9.0),
                            np.minimum(xtl[k, wi], 9.0),
                            np.full(len(f), di),
                            np.full(len(f), si),
                        ]
                    )
        if not parts:
            continue
        P = np.concatenate(parts)
        qb = np.maximum(np.searchsorted(THR, P[:, 4], side="right") - 1, 0)
        qt = np.maximum(np.searchsorted(THR, P[:, 5], side="right") - 1, 0)
        key = (((P[:, 0] * 32 + P[:, 1]) * 64 + P[:, 2]) * NB + qb) * NB + qt
        o = np.lexsort((P[:, 3], key))
        first = np.r_[True, key[o][1:] != key[o][:-1]]
        for kk, row in zip(key[o][first].astype(np.int64), P[o][first]):
            if kk not in best or row[3] < best[kk][3]:
                best[kk] = row
    G = np.array(list(best.values()))
    qb = np.maximum(np.searchsorted(THR, G[:, 4], side="right") - 1, 0)
    qt = np.maximum(np.searchsorted(THR, G[:, 5], side="right") - 1, 0)
    M = np.full((WMAX + 2, WMAX + 2, DMAX + 2, NB, NB), np.inf)
    for (w, h, d, f), b, t in zip(G[:, :4], qb, qt):
        M[int(w), int(h), int(d), b, t] = min(M[int(w), int(h), int(d), b, t], f)
    for ax in (0, 1):
        M = np.minimum.accumulate(M, axis=ax)
    for ax in (2, 3, 4):
        M = np.flip(np.minimum.accumulate(np.flip(M, axis=ax), axis=ax), axis=ax)
    keep = (
        G[:, 3]
        <= M[G[:, 0].astype(int), G[:, 1].astype(int), G[:, 2].astype(int), qb, qt]
        + 1e-15
    )
    return G[keep]


def run_2d_dp(G):
    gw, gh, gd = G[:, 0].astype(int), G[:, 1].astype(int), G[:, 2].astype(int)
    gf, gx = G[:, 3], G[:, 5]
    gq = np.maximum(np.searchsorted(THR, G[:, 4], side="right") - 1, 0)
    shape = (WMAX + 1, WMAX + 1, DMAX + 1, NB)
    dp = np.full(shape, np.inf)
    PG = np.full(shape, -1, np.int32)
    PM = np.zeros(shape, np.int8)
    PQ = np.zeros(shape, np.int8)
    dp[0, 0, 0, NB - 1] = 0.0

    def relax(cw2, h0, d, q, src, g, mode):
        vals, qprev = src.min(axis=2), src.argmin(axis=2)
        tgt = dp[cw2, h0 : h0 + vals.shape[0], d : d + vals.shape[1], q]
        better = vals < tgt
        if better.any():
            tgt[better] = vals[better]
            PG[cw2, h0 : h0 + vals.shape[0], d : d + vals.shape[1], q][better] = g
            PM[cw2, h0 : h0 + vals.shape[0], d : d + vals.shape[1], q][better] = mode
            PQ[cw2, h0 : h0 + vals.shape[0], d : d + vals.shape[1], q][better] = qprev[
                better
            ]

    for cw in range(WMAX + 1):
        cur = dp[cw]
        if not np.isfinite(cur).any():
            continue
        for g in range(len(G)):
            w, h, d, f = gw[g], gh[g], gd[g], gf[g]
            if cw + w > WMAX:
                continue
            val = np.maximum(cur, f)
            ok0 = (THR[None, None, :] >= f + E) | (gx[g] >= cur + E)
            relax(
                cw + w,
                h,
                d,
                gq[g],
                np.where(ok0, val, np.inf)[: WMAX + 1 - h, : DMAX + 1 - d],
                g,
                0,
            )
            if cw == 0:
                continue
            if h + 1 <= WMAX:
                relax(cw + w, h + 1, d, gq[g], val[: WMAX - h, : DMAX + 1 - d], g, 2)
            if cw + w + 1 <= WMAX:
                relax(
                    cw + w + 1, h, d, gq[g], val[: WMAX + 1 - h, : DMAX + 1 - d], g, 1
                )
    return dp, PG, PM, PQ


def extract_chain(dp, PG, PM, PQ, G, K, D):
    sub = dp[: K + 1, : K + 1, D, :]
    if not np.isfinite(sub).any():
        return None, None
    cw, ch, q = np.unravel_index(np.argmin(sub), sub.shape)
    f, d, out = float(sub[cw, ch, q]), D, []
    while cw or ch or d:
        g, mode, qp = (
            int(PG[cw, ch, d, q]),
            int(PM[cw, ch, d, q]),
            int(PQ[cw, ch, d, q]),
        )
        out.append((g, mode))
        w, h, dd = int(G[g, 0]), int(G[g, 1]), int(G[g, 2])
        cw, ch, d, q = cw - w - (mode == 1), ch - h - (mode == 2), d - dd, qp
    return f, out[::-1]


def crop_gadget(donor, sym, w, h, dm):
    sq, xlo, xhi, ylo, yhi = mapped(donor, SYMS[sym])
    R = np.maximum(np.maximum(xhi - w, yhi - h), 0.0)
    take = np.argsort(R, kind="stable")[: w * h + dm]
    return sq[take], float(R[take].max())


def compose_packing(K, D, f, links, G, donors):
    M = len(links)
    F = f + (M + 2) * E
    a, b, gad = np.full(K, K), np.full(K, K), []
    ox, oy = 0, K
    for m, (g, mode) in enumerate(links):
        w, h, dm, di, si = (int(v) for v in G[g, [0, 1, 2, 6, 7]])
        if m:
            ox += mode == 1
            oy -= mode == 2
        oy -= h
        if mode == 1:
            a[ox - 1] = b[ox - 1] = oy + h
        a[ox : ox + w], b[ox : ox + w] = oy, oy + h
        sq, fg = crop_gadget(donors[di], si, w, h, dm)
        sq = sq.copy()
        sq[:, :2] *= 1.0 + 1e-11
        sq[:, 0] += ox + E * (1 + m)
        sq[:, 1] += oy + E * (M - m)
        gad.append(sq)
        ox += w
    a[ox:], b[ox:] = 0, 0
    cells = [(i + 0.5, j + 0.5, 0.0) for i in range(K) for j in range(a[i])]
    cells += [(i + 0.5 + F, j + 0.5 + F, 0.0) for i in range(K) for j in range(b[i], K)]
    sq = np.concatenate([np.array(cells, dtype=np.float64).reshape(-1, 3)] + gad)
    return sq


def write_fq_cfg(path, s, arr_rad):
    he = 0.5 * (np.abs(np.cos(arr_rad[:, 2])) + np.abs(np.sin(arr_rad[:, 2])))
    xs = arr_rad[:, 0] - np.min(arr_rad[:, 0] - he)
    ys = arr_rad[:, 1] - np.min(arr_rad[:, 1] - he)
    degs = np.degrees(arr_rad[:, 2]) % 90.0
    s_use = max(float(s), float(np.max(xs + he)), float(np.max(ys + he)))
    with open(path, "w") as f:
        f.write(f"{len(arr_rad)} {s_use!r}\n")
        for x, y, deg in zip(xs, ys, degs):
            f.write(f"{float(x)!r} {float(y)!r} {float(deg)!r}\n")


def read_fq_cfg_rad(path):
    with open(path) as f:
        lines = [l.split() for l in f if l.strip() and not l.startswith("#")]
    n, s = int(lines[0][0]), float(lines[0][1])
    arr = np.array(
        [
            [float(a), float(b), math.radians(float(c) % 90.0)]
            for a, b, c, *_ in lines[1 : n + 1]
        ],
        dtype=np.float64,
    )
    return s, arr


def run_fq(s, arr_rad, tmp, extra_args, timeout=75):
    a = os.path.join(tmp, "in.txt")
    b = os.path.join(tmp, "out.txt")
    write_fq_cfg(a, s, arr_rad)
    if os.path.exists(b):
        os.remove(b)
    try:
        r = subprocess.run(
            [FQ_BIN, "quench", "--in", a, "--out", b, *extra_args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not os.path.exists(b):
        return None
    lines = r.stdout.strip().splitlines()
    if not lines:
        return None
    try:
        d = json.loads(lines[-1])
    except Exception:
        return None
    mg, mw = d.get("min_gap"), d.get("min_wall")
    if mg is None or mw is None or not (mg >= -1e-12 and mw >= -1e-12):
        return None
    s2, arr2 = read_fq_cfg_rad(b)
    if len(arr2) != len(arr_rad):
        return None
    return s2, arr2


def stabilize_decimal(n, arr_rad):
    xs, ys, ts = arr_rad[:, 0].copy(), arr_rad[:, 1].copy(), arr_rad[:, 2].copy()
    hx = 0.5 * (np.abs(np.cos(ts)) + np.abs(np.sin(ts)))
    xs -= 0.5 * (float(np.min(xs - hx)) + float(np.max(xs + hx)))
    ys -= 0.5 * (float(np.min(ys - hx)) + float(np.max(ys + hx)))
    rows = [
        (Decimal(repr(float(a))), Decimal(repr(float(b))), Decimal(repr(float(c))))
        for a, b, c in zip(xs, ys, ts)
    ]
    pv = verify_puzzle_decimal(n, rows, min_required_gap=TARGET_GAP)
    if pv.valid and pv.min_separation_gap >= TARGET_GAP:
        return True, float(pv.L), xs, ys, ts
    pv0 = verify_puzzle_decimal(n, rows, min_required_gap=Decimal("-1e-11"))
    if not pv0.valid:
        return False, 1e18, xs, ys, ts
    gap0 = max(0.0, float(pv0.min_separation_gap))
    delta = max(1e-11, float(SAFE_GAP) - gap0)
    for _ in range(24):
        sc = 1.0 + delta
        xn, yn = xs * sc, ys * sc
        rows_n = [
            (Decimal(repr(float(a))), Decimal(repr(float(b))), Decimal(repr(float(c))))
            for a, b, c in zip(xn, yn, ts)
        ]
        pvn = verify_puzzle_decimal(n, rows_n, min_required_gap=TARGET_GAP)
        if pvn.valid and pvn.min_separation_gap >= TARGET_GAP:
            return True, float(pvn.L), xn, yn, ts
        delta *= 2.0
    return False, 1e18, xs, ys, ts


def quench_gadget_worker(jobs):
    """Padded-square Rust fq quench of rectangular (w, h, dm) gadgets!
    Pads short side with exact upright grid columns/rows so box becomes square [0, M+f]^2,
    runs Rust fq to shrink f -> f' < f, returning strictly improved donor packings!
    """
    out = []
    rng = random.Random(9999 + os.getpid())
    with tempfile.TemporaryDirectory() as tmp:
        for w, h, dm, f0, sq_crop in jobs:
            M = max(w, h)
            pad = []
            if w < h:
                # Pad (h - w) upright columns on right [w + f0 + 1e-6, h + f0 + 1e-6] x [0, h]
                for ci in range(h - w):
                    for rj in range(h):
                        pad.append((w + f0 + 1e-5 + ci + 0.5, rj + 0.5, 0.0))
            elif h < w:
                for ci in range(w):
                    for rj in range(w - h):
                        pad.append((ci + 0.5, h + f0 + 1e-5 + rj + 0.5, 0.0))
            full = (
                np.vstack([sq_crop, np.array(pad, dtype=np.float64)])
                if pad
                else sq_crop.copy()
            )
            s_in = M + f0 + 1e-4
            best = None
            for loosen, kick in (("1.015", "0.005"), ("1.03", "0.015")):
                r = run_fq(
                    s_in,
                    full,
                    tmp,
                    [
                        "--loosen",
                        loosen,
                        "--kick",
                        kick,
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "24",
                        "--pit",
                        "120",
                    ],
                    timeout=20,
                )
                if r is not None and r[0] < M + f0 - 1e-4:
                    if best is None or r[0] < best[0]:
                        best = r
            if best is not None:
                out.append(best[1])
    return out


def quench_composed_worker(job):
    n, s0, sq_comp, f_dp, seed = job
    rng = random.Random(seed)
    K = math.isqrt(n)
    best_s = s0
    best_a = None
    with tempfile.TemporaryDirectory() as tmp:
        # First test exact composed packing directly!
        ok0, L0, x0, y0, t0 = stabilize_decimal(n, sq_comp)
        if ok0 and L0 < best_s - 1e-10:
            best_s = L0
            best_a = np.column_stack([x0, y0, t0])
        # Then seam-quench with fq across 3 schedules!
        s_in = K + f_dp + 1e-5
        for sched in (
            ["--loosen", "1.0", "--flip-top", "32", "--pit", "200"],
            [
                "--loosen",
                "1.015",
                "--kick",
                "0.004",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "32",
                "--pit",
                "200",
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
                "200",
            ],
        ):
            r = run_fq(s_in, sq_comp, tmp, sched, timeout=55)
            if r is not None and r[0] < best_s - 1e-10:
                best_s, best_a = r[0], r[1]
        if best_a is not None and best_s < s0 - 1e-10:
            r_fin = run_fq(
                best_s,
                best_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
                timeout=80,
            )
            if r_fin is not None and r_fin[0] < best_s:
                best_s, best_a = r_fin[0], r_fin[1]
            ok, Lf, xf, yf, tf = stabilize_decimal(n, best_a)
            if ok and Lf < s0 - 1e-10:
                return n, (Lf, xf, yf, tf)
    return n, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--lags-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--quench-gadgets", type=int, default=320)
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

    print(
        f"[2D-FDP Shard {args.shard}] Extracting initial 2D staircase gadgets from {len(donors)} donors...",
        flush=True,
    )
    G = extract_2d_gadgets(donors)
    print(
        f"[2D-FDP Shard {args.shard}] Initial Pareto 2D gadgets: {len(G)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    # Select high-leverage gadgets (medium sizes w,h <= 14, dm >= 1, low f) to quench via padded-fq!
    cand_g = []
    for gi in range(len(G)):
        w, h, dm, f, ybr, xtl, di, si = G[gi]
        w, h, dm, di, si = int(w), int(h), int(dm), int(di), int(si)
        if 2 <= max(w, h) <= 16 and dm >= 1 and 0.45 <= f <= 0.94:
            # Score by surplus density dm / (w*h)
            cand_g.append((-(dm / (w * h)), f, w, h, dm, di, si))
    cand_g.sort()
    my_g = [
        g
        for idx, g in enumerate(cand_g[: args.quench_gadgets * args.num_shards])
        if idx % args.num_shards == args.shard
    ]
    q_jobs = []
    for _, f, w, h, dm, di, si in my_g:
        sq_c, _ = crop_gadget(donors[di], si, w, h, dm)
        q_jobs.append((w, h, dm, f, sq_c))

    print(
        f"[2D-FDP Shard {args.shard}] Quenching {len(q_jobs)} Pareto 2D gadgets with padded-fq on {args.procs} CPUs...",
        flush=True,
    )
    chunks = [q_jobs[i :: args.procs] for i in range(args.procs)]
    new_donors = []
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        for res in ex.map(quench_gadget_worker, chunks):
            new_donors.extend(res)
    print(
        f"[2D-FDP Shard {args.shard}] Padded-fq produced {len(new_donors)} tighter gadget-donors ({time.time() - t0:.1f}s)!",
        flush=True,
    )

    all_donors = donors + new_donors
    G2 = extract_2d_gadgets(all_donors)
    dp, PG, PM, PQ = run_2d_dp(G2)
    print(
        f"[2D-FDP Shard {args.shard}] Re-extracted {len(G2)} Pareto gadgets + solved 4D DP ({time.time() - t0:.1f}s)",
        flush=True,
    )

    lags = json.load(open(args.lags_json))
    my_ns = [
        r["n"] for idx, r in enumerate(lags) if idx % args.num_shards == args.shard
    ]
    comp_jobs = []
    for n in my_ns:
        K = math.isqrt(n)
        D = n - K * K
        if D == 0 or K > WMAX or D > DMAX:
            continue
        s0 = cur[n][0]
        f_dp, links = extract_chain(dp, PG, PM, PQ, G2, K, D)
        if f_dp is not None and K + f_dp < s0 + 0.018:
            sq_c = compose_packing(K, D, f_dp, links, G2, all_donors)
            comp_jobs.append((n, s0, sq_c, f_dp, 20261127 + 733 * n + args.shard))

    print(
        f"[2D-FDP Shard {args.shard}] Running seam-fq quench on {len(comp_jobs)} DP compositions...",
        flush=True,
    )
    improved = {}
    tot_gain = 0.0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(quench_composed_worker, j) for j in comp_jobs]
        for fut in as_completed(futs):
            n, res = fut.result()
            if res is not None:
                Lf, xf, yf, tf = res
                s0 = cur[n][0]
                gain = (s0 - Lf) / math.sqrt(n)
                tot_gain += gain
                improved[n] = (Lf, xf, yf, tf)
                print(
                    f"W5B_2DFDP_WIN n={n:4d} {s0:.9f} -> {Lf:.9f} dL={s0 - Lf:.6f} dS={gain:.6f} tot={tot_gain:.6f}",
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
        f"DONE W5B 2D-FaultDP Shard {args.shard}: {len(improved)} wins, total_gain={tot_gain:.6f} -> {args.out_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
