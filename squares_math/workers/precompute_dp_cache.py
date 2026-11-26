#!/usr/bin/env python3
"""Precompute 18-bucket 4D Staircase DP cache ONCE per wave in consolidate job!

Eliminates the 12-20 min identical startup recomputation across all 32 DP + Surgery shards
(2.5x-3x FREE throughput multiplier across w8b, w9a, and w36b!).
Uses 4-core parallel 2D gadget extraction + Numba-JIT 4D DP (~500x faster than pure Python).
"""

import argparse
import glob
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
for _d in (DS_DIR, HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import w5b_2d_faultdp_gadgetq as f2d

# Ultra-fine 18-bucket THR (adds 0.44/0.49 for K^2+1 sub-0.50 & 0.97/0.99 for A_GRID_BREAK!)
DEFAULT_THR = np.array(
    [
        0.0,
        0.30,
        0.44,
        0.49,
        0.54,
        0.58,
        0.62,
        0.66,
        0.70,
        0.74,
        0.78,
        0.82,
        0.86,
        0.89,
        0.92,
        0.95,
        0.97,
        0.99,
    ],
    dtype=np.float64,
)


def _extract_chunk(args_tuple):
    items, thr, wmax, dmax, fmax = args_tuple
    nb = len(thr)
    W = np.arange(1, wmax + 1)
    DM = np.arange(1, dmax + 1)
    best = {}
    for di, sq in items:
        n = len(sq)
        L = f2d.side(sq)
        max_wh = min(wmax, math.ceil(L))
        parts = []
        for si, sym in enumerate(f2d.SYMS):
            _, xlo, xhi, ylo, yhi = f2d.mapped(sq, sym)
            for h in range(1, max_wh + 1):
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
                m = f <= fmax
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
        qb = np.maximum(np.searchsorted(thr, P[:, 4], side="right") - 1, 0)
        qt = np.maximum(np.searchsorted(thr, P[:, 5], side="right") - 1, 0)
        key = (((P[:, 0] * 32 + P[:, 1]) * 64 + P[:, 2]) * nb + qb) * nb + qt
        o = np.lexsort((P[:, 3], key))
        first = np.r_[True, key[o][1:] != key[o][:-1]]
        for kk, row in zip(key[o][first].astype(np.int64), P[o][first]):
            if kk not in best or row[3] < best[kk][3]:
                best[kk] = row
    return best


def extract_2d_gadgets_parallel(donors, thr, wmax=31, dmax=62, fmax=0.992, procs=4):
    nb = len(thr)
    indexed = list(enumerate(donors))
    if procs <= 1 or len(indexed) < 16:
        best = _extract_chunk((indexed, thr, wmax, dmax, fmax))
    else:
        chunks = [indexed[i::procs] for i in range(procs)]
        best = {}
        with ProcessPoolExecutor(max_workers=procs) as ex:
            for part_best in ex.map(
                _extract_chunk, [(ch, thr, wmax, dmax, fmax) for ch in chunks]
            ):
                for kk, row in part_best.items():
                    if kk not in best or row[3] < best[kk][3]:
                        best[kk] = row
    G = np.array(list(best.values()), dtype=np.float64)
    qb = np.maximum(np.searchsorted(thr, G[:, 4], side="right") - 1, 0)
    qt = np.maximum(np.searchsorted(thr, G[:, 5], side="right") - 1, 0)
    M = np.full((wmax + 2, wmax + 2, dmax + 2, nb, nb), np.inf)
    for (w, h, d, f), b, t in zip(G[:, :4], qb, qt):
        wi, hi, di = int(w), int(h), int(d)
        if f < M[wi, hi, di, b, t]:
            M[wi, hi, di, b, t] = f
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


if HAS_NUMBA:

    @njit(cache=False)
    def _run_2d_dp_jit(gw, gh, gd, gf, gx, gq, thr, wmax, dmax, eps):
        nb = len(thr)
        dp = np.full((wmax + 1, wmax + 1, dmax + 1, nb), np.inf)
        PG = np.full((wmax + 1, wmax + 1, dmax + 1, nb), -1, np.int32)
        PM = np.zeros((wmax + 1, wmax + 1, dmax + 1, nb), np.int8)
        PQ = np.zeros((wmax + 1, wmax + 1, dmax + 1, nb), np.int8)
        dp[0, 0, 0, nb - 1] = 0.0
        nG = len(gw)
        max_act = (wmax + 1) * (dmax + 1)
        act_ch = np.empty(max_act, np.int32)
        act_d = np.empty(max_act, np.int32)
        act_min = np.empty(max_act, np.float64)
        act_arg = np.empty(max_act, np.int8)
        for cw in range(wmax + 1):
            n_act = 0
            for ch in range(wmax + 1):
                for d0 in range(dmax + 1):
                    mv = np.inf
                    mq = 0
                    for q in range(nb):
                        v = dp[cw, ch, d0, q]
                        if v < mv:
                            mv = v
                            mq = q
                    if mv < 1e300:
                        act_ch[n_act] = ch
                        act_d[n_act] = d0
                        act_min[n_act] = mv
                        act_arg[n_act] = mq
                        n_act += 1
            if n_act == 0:
                continue
            for g in range(nG):
                w = gw[g]
                cw2 = cw + w
                if cw2 > wmax:
                    continue
                h = gh[g]
                d = gd[g]
                f = gf[g]
                gxg = gx[g]
                qg = gq[g]
                max_ch0 = wmax - h
                max_d0 = dmax - d
                do_m2 = (cw > 0) and (h + 1 <= wmax)
                max_ch2 = wmax - h - 1
                do_m1 = (cw > 0) and (cw2 + 1 <= wmax)
                cw2_m1 = cw2 + 1
                for ia in range(n_act):
                    ch = act_ch[ia]
                    if ch > max_ch0:
                        continue
                    d0 = act_d[ia]
                    if d0 > max_d0:
                        continue
                    d2 = d0 + d
                    # Mode 0
                    best_v = np.inf
                    best_q = 0
                    for q in range(nb):
                        cq = dp[cw, ch, d0, q]
                        if cq >= 1e300:
                            continue
                        if (thr[q] >= f + eps) or (gxg >= cq + eps):
                            v = f if f > cq else cq
                            if v < best_v:
                                best_v = v
                                best_q = q
                    ch2 = ch + h
                    if best_v < dp[cw2, ch2, d2, qg]:
                        dp[cw2, ch2, d2, qg] = best_v
                        PG[cw2, ch2, d2, qg] = g
                        PM[cw2, ch2, d2, qg] = 0
                        PQ[cw2, ch2, d2, qg] = best_q
                    # Modes 1 & 2 (unconstrained q)
                    if cw > 0:
                        mv = act_min[ia]
                        v12 = f if f > mv else mv
                        q12 = act_arg[ia]
                        if do_m2 and ch <= max_ch2:
                            ch2m = ch2 + 1
                            if v12 < dp[cw2, ch2m, d2, qg]:
                                dp[cw2, ch2m, d2, qg] = v12
                                PG[cw2, ch2m, d2, qg] = g
                                PM[cw2, ch2m, d2, qg] = 2
                                PQ[cw2, ch2m, d2, qg] = q12
                        if do_m1:
                            if v12 < dp[cw2_m1, ch2, d2, qg]:
                                dp[cw2_m1, ch2, d2, qg] = v12
                                PG[cw2_m1, ch2, d2, qg] = g
                                PM[cw2_m1, ch2, d2, qg] = 1
                                PQ[cw2_m1, ch2, d2, qg] = q12
        return dp, PG, PM, PQ


def run_2d_dp_fast(G, thr, wmax=31, dmax=62, eps=1e-9):
    if HAS_NUMBA:
        gw = G[:, 0].astype(np.int32)
        gh = G[:, 1].astype(np.int32)
        gd = G[:, 2].astype(np.int32)
        gf = np.ascontiguousarray(G[:, 3], dtype=np.float64)
        gx = np.ascontiguousarray(G[:, 5], dtype=np.float64)
        gq = np.maximum(np.searchsorted(thr, G[:, 4], side="right") - 1, 0).astype(
            np.int32
        )
        return _run_2d_dp_jit(gw, gh, gd, gf, gx, gq, thr, wmax, dmax, eps)
    f2d.THR = thr
    f2d.NB = len(thr)
    f2d.WMAX = wmax
    f2d.DMAX = dmax
    return f2d.run_2d_dp(G)


def load_dp_cache(cache_path):
    """Universal ultra-fast (<1s) loader for w8b, w9a, and w36b!"""
    d = np.load(cache_path)
    thr = d["THR"].astype(np.float64)
    wmax = int(d["WMAX"][0])
    dmax = int(d["DMAX"][0])
    fmax = float(d["FMAX"][0])
    f2d.THR = thr
    f2d.NB = len(thr)
    f2d.WMAX = wmax
    f2d.DMAX = dmax
    f2d.FMAX = fmax
    G = d["G"].astype(np.float64)
    dp = d["dp"].astype(np.float64)
    PG = d["PG"].astype(np.int32)
    PM = d["PM"].astype(np.int8)
    PQ = d["PQ"].astype(np.int8)
    cur_L_arr = d["cur_L"].astype(np.float64)
    cur_flat = d["cur_flat"].astype(np.float64)
    cur_offs = d["cur_offs"].astype(np.int64)
    cur = {}
    cur_L = {}
    for n in range(1, 1001):
        s0, e0 = int(cur_offs[n - 1]), int(cur_offs[n])
        sq = cur_flat[s0:e0]
        cur_L[n] = float(cur_L_arr[n])
        cur[n] = (float(cur_L_arr[n]), sq)
    donors_flat = d["donors_flat"].astype(np.float64)
    donors_offs = d["donors_offs"].astype(np.int64)
    donors = [
        donors_flat[int(donors_offs[i]) : int(donors_offs[i + 1])]
        for i in range(len(donors_offs) - 1)
    ]
    return {
        "THR": thr,
        "G": G,
        "dp": dp,
        "PG": PG,
        "PM": PM,
        "PQ": PQ,
        "cur": cur,
        "cur_L": cur_L,
        "donors": donors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--gadget-glob", action="append", default=[])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--fmax", type=float, default=0.992)
    ap.add_argument("--out-cache", required=True)
    args = ap.parse_args()

    t0 = time.time()
    d = np.load(args.union_npz)
    x, y, t, L = d["x"], d["y"], d["t"], d["L"]
    offs = [0]
    for n in range(1, 1001):
        offs.append(offs[-1] + n)

    cur_sq = {}
    cur_L = np.zeros(1001, dtype=np.float64)
    for n in range(1, 1001):
        s0, e0 = offs[n - 1], offs[n]
        cur_L[n] = float(L[n - 1])
        cur_sq[n] = np.column_stack([x[s0:e0], y[s0:e0], t[s0:e0]]).astype(np.float64)

    n_win_upd = 0
    for pat in args.win_glob:
        for wf in sorted(glob.glob(pat, recursive=True)):
            try:
                wd = np.load(wf)
            except Exception:
                continue
            for k in wd.files:
                if k.startswith("x_"):
                    n = int(k[2:])
                    arr = np.column_stack(
                        [wd[f"x_{n}"], wd[f"y_{n}"], wd[f"t_{n}"]]
                    ).astype(np.float64)
                    if len(arr) != n:
                        continue
                    if float(np.max(np.abs(arr[:, 2]))) > 2.0:
                        arr[:, 2] = np.radians(arr[:, 2])
                    Ln = f2d.side(arr)
                    if Ln < cur_L[n] - 1e-8:
                        cur_L[n] = Ln
                        cur_sq[n] = arr
                        n_win_upd += 1

    # Backward n-1 deletion closure (if L[n] < L[n-1], dropping 1 square yields valid n-1 <= L[n]!)
    n_bwd = 0
    for n in range(1000, 1, -1):
        if cur_L[n] < cur_L[n - 1] - 1e-8:
            sub = cur_sq[n][:-1].copy()
            Lsub = f2d.side(sub)
            if Lsub < cur_L[n - 1] - 1e-8:
                cur_L[n - 1] = Lsub
                cur_sq[n - 1] = sub
                n_bwd += 1

    score = sum(float(cur_L[n]) / math.sqrt(n) - 1.0 for n in range(1, 1001))
    donors = []
    seen_sig = set()
    for n in range(1, 1001):
        if cur_L[n] < math.ceil(math.sqrt(n)) - 1e-7:
            donors.append(cur_sq[n])
            seen_sig.add((n, round(float(cur_L[n]), 6)))
    n_base = len(donors)

    n_gad = 0
    n_gfiles = 0
    for pat in args.gadget_glob:
        for gf in sorted(glob.glob(pat, recursive=True)):
            try:
                gd = np.load(gf)
            except Exception:
                continue
            n_gfiles += 1
            for k in gd.files:
                arr = gd[k]
                if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) >= 5:
                    arr64 = arr.astype(np.float64)
                    sig = (len(arr64), round(f2d.side(arr64), 6))
                    if sig not in seen_sig:
                        seen_sig.add(sig)
                        donors.append(arr64)
                        n_gad += 1

    print(
        f"[DP-CACHE] Loaded {len(donors)} donors (base={n_base}, win_upd={n_win_upd}, "
        f"gadgets={n_gad} from {n_gfiles} files, frontier_score={score:.10f}, {time.time() - t0:.1f}s)",
        flush=True,
    )

    t_ext = time.time()
    G = extract_2d_gadgets_parallel(
        donors,
        DEFAULT_THR,
        wmax=31,
        dmax=62,
        fmax=args.fmax,
        procs=args.procs,
    )
    print(
        f"[DP-CACHE] Extracted {len(G)} Pareto 2D gadgets ({len(DEFAULT_THR)}-bucket, FMAX={args.fmax}) in {time.time() - t_ext:.1f}s",
        flush=True,
    )

    t_dp = time.time()
    dp, PG, PM, PQ = run_2d_dp_fast(G, DEFAULT_THR, wmax=31, dmax=62, eps=1e-9)
    print(
        f"[DP-CACHE] Solved {len(DEFAULT_THR)}-bucket 4D-DP ({'Numba JIT' if HAS_NUMBA else 'Python'}) in {time.time() - t_dp:.2f}s!",
        flush=True,
    )

    # Compact donors to ONLY unique donors referenced by Pareto G[:, 6]!
    used_dis = sorted({int(v) for v in G[:, 6]})
    di_map = {old_i: new_i for new_i, old_i in enumerate(used_dis)}
    G_comp = G.copy()
    G_comp[:, 6] = np.array([di_map[int(v)] for v in G[:, 6]], dtype=np.float64)
    used_donors = [donors[old_i] for old_i in used_dis]
    donors_offs = np.zeros(len(used_donors) + 1, dtype=np.int64)
    for i, sq in enumerate(used_donors):
        donors_offs[i + 1] = donors_offs[i] + len(sq)
    donors_flat = (
        np.vstack(used_donors).astype(np.float64)
        if used_donors
        else np.zeros((0, 3), dtype=np.float64)
    )

    cur_offs = np.zeros(1001, dtype=np.int64)
    cur_list = []
    for n in range(1, 1001):
        cur_offs[n] = cur_offs[n - 1] + n
        cur_list.append(cur_sq[n])
    cur_flat = np.vstack(cur_list).astype(np.float64)

    np.savez(
        args.out_cache,
        THR=DEFAULT_THR,
        WMAX=np.array([31], dtype=np.int32),
        DMAX=np.array([62], dtype=np.int32),
        FMAX=np.array([args.fmax], dtype=np.float64),
        G=G_comp,
        dp=dp,
        PG=PG,
        PM=PM,
        PQ=PQ,
        cur_L=cur_L,
        cur_flat=cur_flat,
        cur_offs=cur_offs,
        donors_flat=donors_flat,
        donors_offs=donors_offs,
    )
    sz_mb = os.path.getsize(args.out_cache) / (1024 * 1024)
    print(
        f"[DP-CACHE DONE] Saved {args.out_cache} ({sz_mb:.1f} MB, {len(used_donors)}/{len(donors)} active donors) "
        f"in {time.time() - t0:.1f}s total!",
        flush=True,
    )


if __name__ == "__main__":
    main()
