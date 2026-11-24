#!/usr/bin/env python3
"""Wave 8B Mega-Pool Unified 4D Staircase DP + Diverse Chains + Deep Seam-fq + Contact LP + Rotation!
Pools ALL 1000 public packings PLUS ALL THREE WAVES of shrunk/grown gadgets:
  - Wave 6A (4,952 shrunk M<=22)
  - Wave 7A (grow dm+1 + deepq M<=26 + pair-seam)
  - Wave 8A (wide_reach M=23..31 + round2_requench + grow_v2)
Refines THR to 10 buckets, extracts up to 2 diverse (cw, ch) chains per puzzle,
and runs deep seam-fq + round-press + kick-chain + Contact LP + Rigid Rotation + Decimal64 >= 1.05e-9!
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import linprog, minimize_scalar
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
sys.path.insert(0, DS_DIR)
sys.path.insert(0, HERE)

import w5b_2d_faultdp_gadgetq as f2d

# Refine seam quantization THR from 6 to 10 buckets (eliminates false mode-0 seam rejections!)
f2d.THR = np.array([0.0, 0.25, 0.40, 0.50, 0.58, 0.65, 0.72, 0.79, 0.86, 0.93])
f2d.NB = len(f2d.THR)


def bounds(p):
    half = (np.abs(np.cos(p[:, 2])) + np.abs(np.sin(p[:, 2]))) * 0.5
    lo = (p[:, :2] - half[:, None]).min(0)
    hi = (p[:, :2] + half[:, None]).max(0)
    return half, lo, hi


def span(p):
    _, lo, hi = bounds(p)
    return float((hi - lo).max())


def wall_slack(p, unit=1e-8):
    n = len(p)
    if n < 2:
        return p
    half, low, high = bounds(p)
    length = float((high - low).max())
    if length <= math.sqrt(n) + 1e-13:
        return p
    pairs = cKDTree(p[:, :2]).query_pairs(
        math.sqrt(2.0) + 8.0 * unit, output_type="ndarray"
    )
    if len(pairs) == 0:
        return p
    i, j = pairs.T
    c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
    axes = np.stack([np.stack([c, s], axis=1), np.stack([-s, c], axis=1)], axis=1)
    normals = np.concatenate([axes[i], axes[j]], axis=1)
    delta = p[j, :2] - p[i, :2]
    projections = np.einsum("pki,pi->pk", normals, delta)
    k = np.abs(projections).argmax(axis=1)
    ix = np.arange(len(i))
    v = projections[ix, k]
    direction = normals[ix, k] * np.sign(v)[:, None]
    angle = p[i, 2] - p[j, 2]
    gap = np.abs(v) - (1.0 + np.abs(np.cos(angle)) + np.abs(np.sin(angle))) * 0.5
    active = gap < 5.0 * unit
    i, j, direction, gap = i[active], j[active], direction[active], gap[active]
    row = np.arange(len(i))
    rows, cols = [row] * 4, [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
    data = [direction[:, 0], direction[:, 1], -direction[:, 0], -direction[:, 1]]
    slack = [gap / unit - 2e-6]
    offset = len(i)
    for dim in (0, 1):
        for sign in (-1, 1):
            room = (
                (p[:, dim] - half - low[dim])
                if sign == -1
                else (low[dim] + length - half - p[:, dim])
            )
            idx = np.flatnonzero(room < 6.0 * unit)
            if len(idx) == 0:
                continue
            r = np.arange(offset, offset + len(idx))
            offset += len(idx)
            rows.append(r)
            cols.append(2 * idx + dim)
            data.append(np.full(len(idx), float(sign)))
            slack.append(room[idx] / unit)
            if sign == 1:
                rows.append(r)
                cols.append(np.full(len(idx), 2 * n))
                data.append(np.ones(len(idx)))
    if offset == 0:
        return p
    matrix = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(offset, 2 * n + 1),
    ).tocsr()
    c_obj = np.zeros(2 * n + 1)
    c_obj[-1] = -1.0
    res = linprog(
        c_obj,
        A_ub=matrix,
        b_ub=np.concatenate(slack),
        bounds=[(-1.0, 1.0)] * (2 * n) + [(0.0, 2.0)],
        method="highs",
        options={"threads": 1},
    )
    if not res.success or res.x is None or res.x[-1] < 1e-4:
        return p
    q = p.copy()
    q[:, :2] += unit * res.x[:-1].reshape(n, 2)
    return q


def rotated(p, angle):
    q = p.copy()
    center = p[:, :2].mean(axis=0)
    c, s = math.cos(angle), math.sin(angle)
    q[:, :2] = (p[:, :2] - center) @ np.array([[c, s], [-s, c]]) + center
    q[:, 2] += angle
    return q


def lp_rot_polish(sq):
    cur = sq.copy()
    for unit in (1e-5, 1e-6, 1e-7, 1e-8):
        for _ in range(3):
            q = wall_slack(cur, unit)
            if span(q) < span(cur) - 5e-11:
                cur = q
            else:
                break

    def side_fn(ang):
        return span(rotated(cur, ang))

    steps = np.geomspace(1e-11, 5e-3, 19)
    angles = np.r_[-steps[::-1], 0.0, steps]
    sides = np.array([side_fn(float(a)) for a in angles])
    k = int(sides.argmin())
    ang = float(angles[k])
    if 0 < k < len(angles) - 1:
        fit = minimize_scalar(
            side_fn,
            method="bounded",
            bounds=(float(angles[k - 1]), float(angles[k + 1])),
            options={"xatol": 1e-15},
        )
        if fit.fun < sides[k]:
            ang = float(fit.x)
    q_rot = rotated(cur, ang)
    if span(q_rot) < span(cur) - 1e-11:
        cur = q_rot
    return cur


def extract_diverse_chains(dp, PG, PM, PQ, G, K, D, max_chains=2):
    sub = dp[: K + 1, : K + 1, D, :]
    if not np.isfinite(sub).any():
        return []
    # Flatten all finite (cw, ch, q) sorted by f
    flat_f = sub.min(axis=2)  # (K+1, K+1)
    flat_q = sub.argmin(axis=2)
    ys, xs = np.nonzero(np.isfinite(flat_f))
    order = np.argsort(flat_f[ys, xs])
    best_f = float(flat_f[ys[order[0]], xs[order[0]]])
    picked = []
    used_fp = []
    for idx in order:
        cw, ch = int(ys[idx]), int(xs[idx])
        f = float(flat_f[cw, ch])
        if f > best_f + 0.004 and len(picked) >= 1:
            break
        # Require distinct footprint (L1 distance >= 3) for 2nd chain
        if any(abs(cw - pw) + abs(ch - ph) < 3 for pw, ph in used_fp):
            continue
        q = int(flat_q[cw, ch])
        cw0, ch0, d0, q0 = cw, ch, D, q
        out = []
        while cw0 or ch0 or d0:
            g = int(PG[cw0, ch0, d0, q0])
            mode = int(PM[cw0, ch0, d0, q0])
            qp = int(PQ[cw0, ch0, d0, q0])
            out.append((g, mode))
            w, h, dd = int(G[g, 0]), int(G[g, 1]), int(G[g, 2])
            cw0, ch0, d0, q0 = cw0 - w - (mode == 1), ch0 - h - (mode == 2), d0 - dd, qp
        picked.append((f, out[::-1], (cw, ch)))
        used_fp.append((cw, ch))
        if len(picked) >= max_chains:
            break
    return picked


def quench_composed_deep(job):
    n, s0, sq_comp, f_dp, seed = job
    rng = random.Random(seed)
    K = math.isqrt(n)
    best_s = s0
    best_a = None
    with tempfile.TemporaryDirectory() as tmp:
        ok0, L0, x0, y0, t0 = f2d.stabilize_decimal(n, sq_comp)
        if ok0 and L0 < best_s - 1e-10:
            best_s = L0
            best_a = np.column_stack([x0, y0, t0])
        s_in = K + f_dp + 1e-5
        local_best = None
        for sched in (
            ["--loosen", "1.0", "--flip-top", "32", "--pit", "220"],
            [
                "--loosen",
                "1.012",
                "--kick",
                "0.004",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "32",
                "--pit",
                "240",
            ],
            [
                "--loosen",
                "1.025",
                "--kick",
                "0.011",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "32",
                "--pit",
                "240",
            ],
            [
                "--loosen",
                "1.0",
                "--round-r0",
                "0.08",
                "--round-compress",
                "--round-press",
                "--round-steps",
                "5",
                "--round-press-outer",
                "8",
                "--flip-top",
                "24",
                "--pit",
                "120",
            ],
        ):
            r = f2d.run_fq(s_in, sq_comp, tmp, sched, timeout=55)
            if r is not None:
                if local_best is None or r[0] < local_best[0]:
                    local_best = r
                if r[0] < best_s - 1e-10:
                    best_s, best_a = r[0], r[1]
        if local_best is not None and local_best[0] < best_s + 0.007:
            ls, la = local_best
            for _ in range(4):
                r2 = f2d.run_fq(
                    ls,
                    la,
                    tmp,
                    [
                        "--loosen",
                        rng.choice(("1.005", "1.012", "1.022")),
                        "--kick",
                        rng.choice(("0.002", "0.006")),
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "48",
                        "--pit",
                        "300",
                    ],
                    timeout=55,
                )
                if r2 and r2[0] < ls:
                    ls, la = r2
                if r2 and r2[0] < best_s - 1e-10:
                    best_s, best_a = r2[0], r2[1]
        if best_a is not None and best_s < s0 - 1e-10:
            r_fin = f2d.run_fq(
                best_s,
                best_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
                timeout=80,
            )
            if r_fin and r_fin[0] < best_s:
                best_s, best_a = r_fin[0], r_fin[1]
            best_a = lp_rot_polish(best_a)
            ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, best_a)
            if ok and Lf < s0 - 1e-10:
                return n, (Lf, xf, yf, tf)
    return n, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--gadget-glob", action="append", required=True)
    ap.add_argument("--lags-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=12)
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

    n_pooled = 0
    n_files = 0
    seen_sig = {(len(sq), round(cur[n][0], 6)) for n, (_, sq) in cur.items()}
    for pat in args.gadget_glob:
        for gf in sorted(glob.glob(pat, recursive=True)):
            try:
                gd = np.load(gf)
            except Exception:
                continue
            n_files += 1
            for k in gd.files:
                arr = gd[k]
                if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) >= 5:
                    arr64 = arr.astype(np.float64)
                    sig = (len(arr64), round(f2d.side(arr64), 6))
                    if sig not in seen_sig:
                        seen_sig.add(sig)
                        donors.append(arr64)
                        n_pooled += 1
    print(
        f"[W8B Shard {args.shard}/{args.num_shards}] MEGA-POOL loaded {len(donors)} donors ({n_pooled} shrunk/grown gadgets from {n_files} files, {time.time() - t0:.1f}s)",
        flush=True,
    )

    G = f2d.extract_2d_gadgets(donors)
    dp, PG, PM, PQ = f2d.run_2d_dp(G)
    print(
        f"[W8B Shard {args.shard}] 10-bucket Global Pareto 2D gadgets: {len(G)}, 4D DP solved ({time.time() - t0:.1f}s)",
        flush=True,
    )

    lags = json.load(open(args.lags_json)) if os.path.exists(args.lags_json) else []
    lag_set = {int(it["n"]) for it in lags}

    cands = []
    for n in range(80, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D == 0 or D > f2d.DMAX:
            continue
        s0 = cur[n][0]
        if s0 >= K + 1 - 1e-7:
            continue
        chains = extract_diverse_chains(
            dp, PG, PM, PQ, G, K, D, max_chains=2 if n in lag_set else 1
        )
        for ci, (f_dp, links, fp) in enumerate(chains):
            gap = (K + f_dp) - s0
            lim = 0.0085 if n in lag_set else 0.004
            if gap < lim:
                sq_comp = f2d.compose_packing(K, D, f_dp, links, G, donors)
                if sq_comp is not None and len(sq_comp) == n:
                    # Priority: exact wins first, then smallest gap
                    cands.append((gap, n, s0, sq_comp, f_dp, ci))
    cands.sort(key=lambda z: (z[0], -z[1]))
    my_cands = cands[args.shard :: args.num_shards]
    print(
        f"[W8B Shard {args.shard}] Total Mega-Pool DP candidates={len(cands)} (exact<0: {sum(1 for z in cands if z[0] < -1e-7)}), my_slice={len(my_cands)}",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0
    job_args = [
        (n, s0, sq_comp, f_dp, 888001 + args.shard * 10000 + i * 31)
        for i, (gap, n, s0, sq_comp, f_dp, ci) in enumerate(my_cands)
    ]
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(quench_composed_deep, j) for j in job_args]
        for fut in as_completed(futs):
            n, res = fut.result()
            if res is not None:
                Lf, xf, yf, tf = res
                prev_L = wins[n][0] if n in wins else cur[n][0]
                if Lf < prev_L - 1e-10:
                    dL = Lf - cur[n][0]
                    dS = (Lf - prev_L) / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W8B WIN] n={n:4d} {cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f})",
                        flush=True,
                    )
            if time.time() - t0 > 4500:
                print(
                    f"[W8B Shard {args.shard}] Time limit reached cleanly; cancelling remaining futures.",
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
        f"[W8B Shard {args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
