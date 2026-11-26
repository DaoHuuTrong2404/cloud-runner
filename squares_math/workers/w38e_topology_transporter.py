#!/usr/bin/env python3
"""Wave 38E: Topology-Jumping Gap Killer (n-1 Grow + n+1/n+2 Band-Relax Removal + K^2+1 Ribbon Splice).

Targets the exact Top-96 live gap puzzles (e.g., n=234, 367, 491, 333, 738, 456, 804, 730, 261, 904, 962...)
that are stuck in suboptimal basins while their immediate neighbors (n-1, n+1, n+2) or smaller-K tilted ribbons
already possess the true optimal tilted-band topology!
  1. Engine A (n-1 -> n +1 Insertion): Expands tight n-1 packing, inserts +1 square into biggest tilted-seam
     and global voids across 4 angles, melts/compresses with Rust fq round-press -> Stock Hopper + SLP.
  2. Engine B (n+1 -> n & n+2 -> n Band-Relax Removal): Deletes 1 or 2 high-load/tilted-seam squares WITHOUT
     rejecting frozen Compactor boxes, immediately relaxes the full tilted band via Rust fq kick/round-press
     -> Stock Hopper + SLP.
  3. Engine C (Period-Extension Ribbon Splice for D in 1..4): Extends a proven K_s^2+1 tilted ribbon
     (e.g., 442/626/842) by dK rows along its lattice angle into K = K_s + dK, heals with fq round-press
     -> Stock Hopper + SLP.
All wins certified via exact Decimal64 >= 1.05e-9!
"""

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
TOOLS_DIR = os.path.abspath(os.path.join(HERE, "..", "tools"))
for _d in (DS_DIR, TOOLS_DIR, HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import sq_engine as E
import slp as SLP
import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12
import w32a_fresh_basin_fast_hopper as w32a


def _normalize(arr):
    a = arr.astype(np.float64).copy()
    he = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    a[:, 0] -= float(np.min(a[:, 0] - he))
    a[:, 1] -= float(np.min(a[:, 1] - he))
    return a


def _med_tilt_rad(arr):
    deg = np.degrees(arr[:, 2]) % 90.0
    tilt = np.minimum(deg, 90.0 - deg)
    t_arr = arr[tilt > 3.0, 2]
    if len(t_arr) == 0:
        return math.radians(24.0)
    return float(np.median(t_arr))


def _find_void_points(arr, L_box, np_rng, n_global=4, n_seam=2):
    """Find biggest voids both globally AND near the tilted-to-upright seam."""
    pts = np.column_stack(
        [
            np_rng.uniform(0.55, max(0.65, L_box - 0.55), 1200),
            np_rng.uniform(0.55, max(0.65, L_box - 0.55), 1200),
        ]
    )
    dx = pts[:, None, 0] - arr[None, :, 0]
    dy = pts[:, None, 1] - arr[None, :, 1]
    d2 = dx * dx + dy * dy
    dmin = np.sqrt(np.min(d2, axis=1))

    picked = []
    # 1. Global biggest spatially-separated voids
    for idx in np.argsort(-dmin):
        px, py = float(pts[idx, 0]), float(pts[idx, 1])
        if all(math.hypot(px - qx, py - qy) > 1.2 for qx, qy in picked):
            picked.append((px, py))
        if len(picked) >= n_global:
            break

    # 2. Tilted-seam voids (within distance 2.2 of a tilted square)
    deg = np.degrees(arr[:, 2]) % 90.0
    tilt_mask = np.minimum(deg, 90.0 - deg) > 3.0
    if np.any(tilt_mask):
        dtilt = np.sqrt(np.min(d2[:, tilt_mask], axis=1))
        seam_ok = dtilt <= 2.4
        seam_scores = np.where(seam_ok, dmin, -1.0)
        for idx in np.argsort(-seam_scores):
            if seam_scores[idx] <= 0:
                break
            px, py = float(pts[idx, 0]), float(pts[idx, 1])
            if all(math.hypot(px - qx, py - qy) > 1.0 for qx, qy in picked):
                picked.append((px, py))
            if len(picked) >= n_global + n_seam:
                break
    return picked


def _pick_removal_indices(arr, max_k=8):
    """Pick high-stress / tilted-seam squares spatially diverse across the packing."""
    P = E.to_origin(arr)
    n1 = len(P)
    tilts = np.abs(E.fold(P[:, 2]))
    tilted = np.where(tilts > 0.02)[0]
    scores = np.zeros(n1, dtype=np.float64)
    try:
        H = E.Hopper(P, seed=0)
        if H.L.ok:
            loads = np.maximum(
                np.nan_to_num(H.L.loads(), nan=0.0, posinf=0.0, neginf=0.0), 0.0
            )
            scores += loads * 10.0
    except Exception:
        pass
    # Boost tilted squares near boundary or touching upright squares
    L = float(E.side(P))
    he = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    wall_dist = np.minimum.reduce(
        [P[:, 0] - he, L - (P[:, 0] + he), P[:, 1] - he, L - (P[:, 1] + he)]
    )
    scores += np.where(wall_dist < 0.08, 0.8, 0.0)
    for i in tilted:
        scores[i] += 0.45
    # Pick spatially separated top squares so we probe diverse band regions!
    picked = []
    for idx in np.argsort(-scores):
        i = int(idx)
        if all(math.hypot(P[i, 0] - P[j, 0], P[i, 1] - P[j, 1]) > 1.8 for j in picked):
            picked.append(i)
        if len(picked) >= max_k:
            break
    if len(picked) < max_k:
        for idx in np.argsort(-scores):
            i = int(idx)
            if i not in picked:
                picked.append(i)
            if len(picked) >= max_k:
                break
    return picked


def _build_ribbon_splice(donor_arr, K_src, K_tgt, n_tgt, y_frac, dx_bias, np_rng):
    """Splice +dK rows into the middle of a K_src^2+1 tilted ribbon along its exact signed slope."""
    a = _normalize(donor_arr)
    L_src = float(f2d.side(a))
    # Orient so tilted block's longer span is along Y!
    deg = np.degrees(a[:, 2]) % 90.0
    mt = np.minimum(deg, 90.0 - deg) > 4.0
    if np.sum(mt) < 20:
        return None
    if (a[mt, 0].max() - a[mt, 0].min()) > (a[mt, 1].max() - a[mt, 1].min()):
        # Transpose (x, y, t) -> (y, x, -t)
        a = np.column_stack([a[:, 1], a[:, 0], -a[:, 2]])
        a = _normalize(a)
        deg = np.degrees(a[:, 2]) % 90.0
        mt = np.minimum(deg, 90.0 - deg) > 4.0

    dK = K_tgt - K_src
    if dK <= 0:
        return None
    # Measure exact signed slope dx/dy of the tilted ribbon!
    yt = a[mt, 1]
    xt = a[mt, 0]
    var_y = float(np.var(yt))
    slope = float(np.cov(yt, xt)[0, 1] / var_y) if var_y > 1e-6 else 0.0
    dx_exact = dK * slope
    # Round to integer upright column shift (optionally nudged by dx_bias in {-0.5, 0, +0.5})
    dx_shift = float(round(dx_exact + dx_bias))

    y_cut = y_frac * L_src
    bot = a[a[:, 1] < y_cut].copy()
    top = a[a[:, 1] >= y_cut].copy()
    top[:, 1] += float(dK)
    top[:, 0] += dx_shift

    # Clone middle slice [y_cut - dK, y_cut) shifted by (+dx_shift, +dK) so it fills [y_cut, y_cut + dK) seamlessly!
    mid = a[(a[:, 1] >= y_cut - dK) & (a[:, 1] < y_cut)].copy()
    mid[:, 1] += float(dK)
    mid[:, 0] += dx_shift

    parts = [bot, mid, top]
    comb = np.vstack(parts)
    # Shift into [0, L_tgt]^2
    L_init = float(K_tgt) + max(0.55, L_src - K_src + 0.04)
    he = 0.5 * (np.abs(np.cos(comb[:, 2])) + np.abs(np.sin(comb[:, 2])))
    comb[:, 0] -= float(np.min(comb[:, 0] - he))
    comb[:, 1] -= float(np.min(comb[:, 1] - he))
    # Keep squares inside [0, L_init]^2
    inside = (
        (comb[:, 0] - he >= -0.25)
        & (comb[:, 0] + he <= L_init + 0.25)
        & (comb[:, 1] - he >= -0.25)
        & (comb[:, 1] + he <= L_init + 0.25)
    )
    comb = comb[inside]
    if len(comb) == 0:
        return None

    # Fill empty upright lattice cells cleanly if we need more squares!
    if len(comb) < n_tgt:
        for i in range(K_tgt + 1):
            ux = i + 0.5
            if ux > L_init - 0.45:
                continue
            for j in range(K_tgt + 1):
                uy = j + 0.5
                if uy > L_init - 0.45:
                    continue
                d2 = (comb[:, 0] - ux) ** 2 + (comb[:, 1] - uy) ** 2
                if float(np.min(d2)) >= 0.88**2:
                    comb = np.vstack(
                        [comb, np.array([[ux, uy, 0.0]], dtype=np.float64)]
                    )

    # Trim excess (remove most crowded squares first) or add into biggest voids!
    while len(comb) > n_tgt:
        dx = comb[:, None, 0] - comb[None, :, 0]
        dy = comb[:, None, 1] - comb[None, :, 1]
        d2 = dx * dx + dy * dy
        np.fill_diagonal(d2, 1e9)
        worst = int(np.argmin(np.min(d2, axis=1)))
        comb = np.delete(comb, worst, axis=0)

    while len(comb) < n_tgt:
        vp = _find_void_points(comb, L_init, np_rng, n_global=2, n_seam=1)
        if not vp:
            break
        px, py = vp[0]
        comb = np.vstack([comb, np.array([[px, py, 0.0]], dtype=np.float64)])

    if len(comb) != n_tgt:
        return None
    return comb, L_init


def _fq_relax_and_polish(s_in, arr_in, tmp, rng, mode="remove", timeout=42):
    """Run Rust fq round-press / kick-quench schedules to globally relax tilted bands."""
    best = None
    if mode == "insert":
        scheds = [
            [
                "--loosen",
                "1.0",
                "--round-r0",
                "0.14",
                "--round-compress",
                "--round-press",
                "--round-steps",
                "7",
                "--round-press-outer",
                "12",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "32",
                "--pit",
                "160",
            ],
            [
                "--loosen",
                "1.0",
                "--round-r0",
                "0.20",
                "--round-compress",
                "--round-press",
                "--round-steps",
                "8",
                "--round-press-outer",
                "14",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "32",
                "--pit",
                "160",
            ],
        ]
    else:  # remove
        scheds = [
            [
                "--loosen",
                "1.018",
                "--kick",
                "0.006",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "36",
                "--pit",
                "220",
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
                "10",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "32",
                "--pit",
                "150",
            ],
        ]
    for sc in scheds:
        r = f2d.run_fq(s_in, arr_in, tmp, sc, timeout=timeout)
        if r is not None:
            if best is None or r[0] < best[0]:
                best = r
    return best


def worker_transport(job):
    (
        n,
        L0,
        arr_m1,
        L_m1,
        arr_p1,
        L_p1,
        arr_p2,
        L_p2,
        ribbon_donors,
        seed,
        hop_sec,
    ) = job
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed + 17)
    K = math.isqrt(n)
    D = n - K * K

    best_S = L0
    best_P = None
    best_tag = ""
    hopper_pool = []  # list of (s_fq, tag, arr) promising candidates for Stock Hopper!

    with tempfile.TemporaryDirectory(prefix=f"w38e_{n}_") as tmp:
        to_fq = 32 if n <= 400 else (45 if n <= 750 else 58)

        # === ENGINE A: n-1 -> n (+1 Square Insertion Topology Transporter) ===
        if arr_m1 is not None and L_m1 < L0 - 0.006:
            P_m1 = _normalize(arr_m1)
            L_exp = max(L0 + 0.04, L_m1 + 0.08)
            voids = _find_void_points(P_m1, L_exp, np_rng, n_global=4, n_seam=2)
            med_t = _med_tilt_rad(P_m1)
            ang_list = [med_t, 0.0, math.radians(45.0), -med_t]
            trials = []
            for vi, (px, py) in enumerate(voids):
                trials.append((px, py, ang_list[vi % len(ang_list)]))
                if vi < 2:
                    trials.append((px, py, ang_list[(vi + 1) % len(ang_list)]))
            for ti, (px, py, ang) in enumerate(trials):
                P_plus = np.vstack([P_m1, np.array([[px, py, ang]], dtype=np.float64)])
                s_in = max(L0 + 0.07, L_m1 + 0.11)
                r = _fq_relax_and_polish(
                    s_in, P_plus, tmp, rng, mode="insert", timeout=to_fq
                )
                if r is not None:
                    s_r, a_r = r
                    if s_r < best_S - 1e-8:
                        best_S, best_P, best_tag = s_r, a_r, f"ins_fq_{ti}"
                    if s_r < L0 + 0.009:
                        hopper_pool.append((s_r, f"ins_{ti}", a_r))

        # === ENGINE B: n+1 -> n & n+2 -> n Band-Relaxing Removal Transporter ===
        if arr_p1 is not None and L_p1 < L0 + 0.055:
            P_p1 = _normalize(arr_p1)
            idxs = _pick_removal_indices(P_p1, max_k=7)
            for idx in idxs:
                P_del = _normalize(np.delete(P_p1, idx, axis=0))
                s_in = max(L_p1 + 1e-4, float(f2d.side(P_del)) + 1e-4)
                r = _fq_relax_and_polish(
                    s_in, P_del, tmp, rng, mode="remove", timeout=to_fq
                )
                if r is not None:
                    s_r, a_r = r
                    if s_r < best_S - 1e-8:
                        best_S, best_P, best_tag = s_r, a_r, f"rem1_fq_{idx}"
                    if s_r < L0 + 0.012:
                        hopper_pool.append((s_r, f"rem1_{idx}", a_r))

        if arr_p2 is not None and L_p2 < L0 + 0.075:
            P_p2 = _normalize(arr_p2)
            idxs = _pick_removal_indices(P_p2, max_k=6)
            pairs_try = []
            for i in range(min(4, len(idxs))):
                u = idxs[i]
                # Pair with closest tilted neighbor or next high-load square
                d2 = (P_p2[:, 0] - P_p2[u, 0]) ** 2 + (P_p2[:, 1] - P_p2[u, 1]) ** 2
                d2[u] = 1e9
                v = int(np.argmin(d2))
                pair = (min(u, v), max(u, v))
                if pair not in pairs_try:
                    pairs_try.append(pair)
            for u, v in pairs_try[:4]:
                P_del2 = _normalize(np.delete(P_p2, [u, v], axis=0))
                s_in = max(L_p2 + 1e-4, float(f2d.side(P_del2)) + 1e-4)
                r = _fq_relax_and_polish(
                    s_in, P_del2, tmp, rng, mode="remove", timeout=to_fq
                )
                if r is not None:
                    s_r, a_r = r
                    if s_r < best_S - 1e-8:
                        best_S, best_P, best_tag = s_r, a_r, f"rem2_fq_{u}_{v}"
                    if s_r < L0 + 0.012:
                        hopper_pool.append((s_r, f"rem2_{u}_{v}", a_r))

        # === ENGINE C: Period-Extension Ribbon Splice for D in 1..4 ===
        if D in (1, 2, 3, 4) and ribbon_donors:
            for K_src, d_arr in ribbon_donors:
                for y_frac in (0.45, 0.55):
                    for dx_bias in (-0.4, 0.4):
                        built = _build_ribbon_splice(
                            d_arr, K_src, K, n, y_frac, dx_bias, np_rng
                        )
                        if built is None:
                            continue
                        comb, L_init = built
                        r = _fq_relax_and_polish(
                            L_init + 0.08,
                            comb,
                            tmp,
                            rng,
                            mode="insert",
                            timeout=to_fq + 15,
                        )
                        if r is not None:
                            s_r, a_r = r
                            tag_c = f"rib_{K_src}_b{dx_bias:+.1f}"
                            if s_r < best_S - 1e-8:
                                best_S, best_P, best_tag = s_r, a_r, tag_c
                            if s_r < L0 + 0.018:
                                hopper_pool.append((s_r, tag_c, a_r))

        # === PHASE 2: Deep Stock Hopper + SLP on top-3 diverse transported basins! ===
        hopper_pool.sort(key=lambda z: z[0])
        picked_hop = []
        for s_r, tag, a_r in hopper_pool:
            if all(abs(s_r - ps) > 1e-5 for ps, _, _ in picked_hop):
                picked_hop.append((s_r, tag, a_r))
            if len(picked_hop) >= 3:
                break

        for hi, (s_r, tag, a_r) in enumerate(picked_hop):
            # First run 2 quick fq kick-quenches to further relax before Hopper
            cur_s, cur_a = s_r, a_r
            for ls_v, kk_v in (("1.012", "0.004"), ("1.022", "0.009")):
                rk = f2d.run_fq(
                    cur_s + 1e-4,
                    cur_a,
                    tmp,
                    [
                        "--loosen",
                        ls_v,
                        "--kick",
                        kk_v,
                        "--seed",
                        str(rng.randrange(1 << 30)),
                        "--flip-top",
                        "40",
                        "--pit",
                        "260",
                    ],
                    timeout=to_fq,
                )
                if rk is not None and rk[0] < cur_s:
                    cur_s, cur_a = rk
                    if cur_s < best_S - 1e-8:
                        best_S, best_P, best_tag = cur_s, cur_a, f"{tag}_fq2"
            # Then unleash FAST Stock sq_engine.Hopper!
            Q, S, _ = w32a.run_stock_hopper_chain(
                w32a.safe_open(cur_a),
                seed + 70000 + hi * 4129,
                first_sec=hop_sec,
                cascade_sec=min(120, hop_sec),
                max_cascades=3,
            )
            if S < best_S - 1e-8:
                best_S, best_P, best_tag = S, Q, f"{tag}_hop"

        # === PHASE 3: Final SLP + fq + lp_rot + Decimal64 >= 1.05e-9 Certification ===
        if best_P is not None and best_S < L0 - 1e-9:
            cur_a = w32a.safe_open(best_P)
            cur_s = float(E.side(cur_a))
            try:
                xs, ys, ts, Ls = SLP.polish(
                    cur_a[:, 0],
                    cur_a[:, 1],
                    cur_a[:, 2],
                    iters=120,
                    r0=2e-3,
                    rmin=1e-10,
                    tol_sep=-1e-13,
                    time_limit=25.0,
                )
                Qs = w32a.safe_open(np.column_stack([xs, ys, ts]))
                Ss = float(E.side(Qs))
                if Ss < cur_s - 1e-10 and E.min_gap(Qs) >= -1e-12:
                    cur_a, cur_s = Qs, Ss
            except Exception:
                pass
            if os.path.exists(f2d.FQ_BIN):
                r_fin = f2d.run_fq(
                    cur_s,
                    cur_a,
                    tmp,
                    ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
                    timeout=65,
                )
                if r_fin is not None and r_fin[0] < cur_s:
                    cur_s, cur_a = r_fin[0], r_fin[1]
            try:
                cur_a = w12.lp_rot_polish(cur_a)
            except Exception:
                pass
            ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_a)
            if ok and Lf < L0 - 1e-9:
                return n, best_tag, (Lf, xf, yf, tf)

    return n, best_tag, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--lags-json", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2600)
    ap.add_argument("--hop-sec", type=int, default=100)
    ap.add_argument("--seed-off", type=int, default=380000)
    ap.add_argument("--max-targets", type=int, default=96)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    ref_L = {}
    if os.path.exists(args.frontier_json):
        for r in json.load(open(args.frontier_json)):
            nn = int(r["n"])
            ref_L[nn] = min(ref_L.get(nn, 1e9), float(r["L_front"]))
    if args.lags_json and os.path.exists(args.lags_json):
        for r in json.load(open(args.lags_json)):
            nn = int(r["n"])
            ref_L[nn] = min(ref_L.get(nn, 1e9), float(r["Lb"]))

    # Rank ALL puzzles strictly by LIVE gap (cur_L[n] - ref_L[n]) / sqrt(n)!
    # Guarantees our biggest true gaps (#1 n=234, #2 367, #3 491, #4 333, #5 738, #8 730, #12 962...) go FIRST!
    ranked = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D == 0:
            continue
        L0 = cur[n][0]
        rL = ref_L.get(n, L0)
        gap_ds = (L0 - rL) / math.sqrt(n)
        if gap_ds > 1e-6:
            ranked.append((gap_ds, n))
    ranked.sort(reverse=True)
    targets = [n for _, n in ranked[: args.max_targets]]
    my_targets = targets[args.shard :: args.num_shards]

    # Collect high-quality K^2+1 tilted-ribbon donors (K_s in 20..29) for Engine C!
    ribbon_map = {}
    for Ks in (21, 23, 25, 27, 29):
        ns = Ks * Ks + 1
        if ns in cur and cur[ns][0] < Ks + 0.530:
            ribbon_map[Ks] = cur[ns][1]

    jobs = []
    for i, n in enumerate(my_targets):
        K = math.isqrt(n)
        D = n - K * K
        L0 = cur[n][0]
        arr_m1, L_m1 = (cur[n - 1][1], cur[n - 1][0]) if (n - 1) in cur else (None, 1e9)
        arr_p1, L_p1 = (cur[n + 1][1], cur[n + 1][0]) if (n + 1) in cur else (None, 1e9)
        arr_p2, L_p2 = (cur[n + 2][1], cur[n + 2][0]) if (n + 2) in cur else (None, 1e9)
        r_donors = []
        if D in (1, 2, 3, 4) and K >= 22:
            for Ks in sorted(ribbon_map.keys(), reverse=True):
                if 1 <= (K - Ks) <= 4:
                    r_donors.append((Ks, ribbon_map[Ks]))
                    if len(r_donors) >= 2:
                        break
        jobs.append(
            (
                n,
                L0,
                arr_m1,
                L_m1,
                arr_p1,
                L_p1,
                arr_p2,
                L_p2,
                r_donors,
                args.seed_off + args.shard * 10007 + n * 53 + i,
                args.hop_sec,
            )
        )

    print(
        f"[W38E-TopoTrans s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"targets={len(my_targets)}/{len(targets)} top5={my_targets[:5]} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0

    def save_ckpt():
        pl = {}
        for kn, (Lf, xf, yf, tf) in sorted(wins.items()):
            pl[f"L_{kn}"] = np.array([Lf], dtype=np.float64)
            pl[f"x_{kn}"] = np.asarray(xf, dtype=np.float64)
            pl[f"y_{kn}"] = np.asarray(yf, dtype=np.float64)
            pl[f"t_{kn}"] = np.asarray(tf, dtype=np.float64)
        np.savez_compressed(args.out_npz, **pl)

    done = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_transport, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W38E ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                dL = Lf - cur[n][0]
                dS = dL / math.sqrt(n)
                if n not in wins or Lf < wins[n][0] - 1e-10:
                    if n in wins:
                        tot_ds -= (wins[n][0] - cur[n][0]) / math.sqrt(n)
                    wins[n] = res
                    tot_ds += dS
                    print(
                        f"  [W38E WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"(dL={dL:.7f}, dS={dS:.8f}) [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
            else:
                print(
                    f"  [W38E] n={n:4d} no gain ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                    flush=True,
                )
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W38E s={args.shard}] Graceful budget {args.budget_sec}s reached.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W38E-TopoTrans s={args.shard} DONE] {len(wins)} certified wins, "
        f"total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
