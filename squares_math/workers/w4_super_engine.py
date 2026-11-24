#!/usr/bin/env python3
"""Wave 4 v2 Super-Engine (Kaggle Private CPU / Cloud):
1) Exact & Inexact L-Cut (lcut.best_cuts + boundary strip removal -> fq)
2) Exact & Inexact L-Add (ladd.best_cuts + add_l -> fq)
3) 500x-faster fillfast-screened Structural Band Lattice Extension, Donor Transplant (incl. K^2+1),
   L-Shape Two-Wall SA (layout.py with w up to 16), & Clipped Sub-Box Strip Sweep -> fq round.15c.p5o8
4) High-throughput Multi-Move Basin Hopping (remove-k, insert-k, bandflip, crot, mirror, band_angles,
   anneal melt, harden reharden) with strict time budget + deep polish + Decimal64 >= 1.05e-9 certification.
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
import numpy as np
import scipy.optimize._milp as _milp

if hasattr(_milp, "_highs_wrapper"):
    _hw = _milp._highs_wrapper
    _milp._highs_wrapper = lambda *a: _hw(*a[:-1], dict(a[-1], threads=1))

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = None
for cand in [
    os.environ.get("SQ_DS_DIR", ""),
    "/kaggle/input/sq-wave3-union",
    "/kaggle/input/datasets/daohuutrong2404/sq-wave3-union",
    HERE,
]:
    if cand and os.path.exists(os.path.join(cand, "nov24_union.npz")):
        DS_DIR = cand
        break
if DS_DIR is None and os.path.exists("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        if "nov24_union.npz" in files:
            DS_DIR = root
            break
assert DS_DIR is not None, "Could not locate nov24_union.npz"
sys.path.insert(0, DS_DIR)

import bandflip as bf
import fillfast
import ladd
import layout as lay
import lcut
from decimal_verifier import verify_puzzle_decimal

BIN_DIR = "/tmp/sq_w4_bins"
os.makedirs(BIN_DIR, exist_ok=True)
for bname in ("fq", "anneal", "harden", "band_dp_scan", "rect_quench"):
    src = os.path.join(DS_DIR, bname)
    dst = os.path.join(BIN_DIR, bname)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)

FQ_BIN = os.environ.get("FQ_BIN", os.path.join(BIN_DIR, "fq"))
HARDEN_BIN = os.environ.get("HARDEN_BIN", os.path.join(BIN_DIR, "harden"))
ANNEAL_BIN = os.environ.get("ANNEAL_BIN", os.path.join(BIN_DIR, "anneal"))

TARGET_GAP = Decimal("1e-9")
SAFE_GAP = Decimal("1.05e-9")

# Fast evand screening schedules vs deep polish
FQ_SCREEN = ["--loosen", "1.02", "--flip-top", "0", "--pit", "16"]
FQ_ROUND_FAST = [
    "--loosen",
    "1.0",
    "--round-r0",
    "0.15",
    "--round-compress",
    "--round-press",
    "--round-steps",
    "5",
    "--round-press-outer",
    "8",
    "--flip-top",
    "0",
    "--pit",
    "12",
]
FQ_POLISH_DEEP = ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"]


def load_union():
    d = np.load(os.path.join(DS_DIR, "nov24_union.npz"))
    x, y, t, L = d["x"], d["y"], d["t"], d["L"]
    offs = [0]
    for n in range(1, 1001):
        offs.append(offs[-1] + n)
    return x, y, t, L, offs


def get_puzzle(n, x, y, t, L, offs):
    s0, e0 = offs[n - 1], offs[n]
    xs = x[s0:e0].copy()
    ys = y[s0:e0].copy()
    ts = t[s0:e0].copy()
    hx = 0.5 * (np.abs(np.cos(ts)) + np.abs(np.sin(ts)))
    xs -= np.min(xs - hx)
    ys -= np.min(ys - hx)
    degs = np.degrees(ts) % 90.0
    return float(L[n - 1]), list(zip(xs.tolist(), ys.tolist(), degs.tolist()))


def write_cfg(path, s, pts):
    with open(path, "w") as f:
        f.write(f"{len(pts)} {float(s)!r}\n")
        for x, y, deg in pts:
            f.write(f"{float(x)!r} {float(y)!r} {float(deg) % 90.0!r}\n")


def read_cfg(path):
    with open(path) as f:
        lines = [l.split() for l in f if l.strip() and not l.startswith("#")]
    n, s = int(lines[0][0]), float(lines[0][1])
    pts = [(float(a), float(b), float(c) % 90.0) for a, b, c, *_ in lines[1 : n + 1]]
    return s, pts


def run_fq(s, pts, tmp, extra_args, timeout=75):
    a = os.path.join(tmp, "in.txt")
    b = os.path.join(tmp, "out.txt")
    write_cfg(a, s, pts)
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
    s2, pts2 = read_cfg(b)
    if len(pts2) != len(pts):
        return None
    return s2, pts2


def verify_and_stabilize(n, pts):
    arr = np.array(pts, dtype=np.float64)
    xs, ys = arr[:, 0], arr[:, 1]
    ts = np.radians(arr[:, 2])
    hx = 0.5 * (np.abs(np.cos(ts)) + np.abs(np.sin(ts)))
    xs = xs - 0.5 * (float(np.min(xs - hx)) + float(np.max(xs + hx)))
    ys = ys - 0.5 * (float(np.min(ys - hx)) + float(np.max(ys + hx)))
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
    for _ in range(20):
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


def fast_fill_eval(S, T, mode="greedy", tau=0.02, time_limit=3):
    if not T:
        return []
    X = fillfast.fill(S, T, mode=mode, tau=tau, time_limit=time_limit)
    return [(float(q[0]), float(q[1]), float(q[2]) % 90.0) for q in X]


def angle_hints_for(n, gap_map, pts0):
    hints = []
    if n in gap_map:
        hints = [
            float(v)
            for v in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*°", gap_map[n]["kind"])
        ]
    arr = np.array(pts0, dtype=np.float64)
    a = arr[:, 2] % 90.0
    tilt = a[np.minimum(a, 90.0 - a) > 3.0]
    if len(tilt):
        med = float(np.median(tilt))
        m = min(med, 90.0 - med)
        if all(abs(m - h) > 1.5 for h in hints):
            hints.append(m)
    K = math.isqrt(n)
    if n - K * K <= 2 and all(abs(21.8 - h) > 1.5 for h in hints):
        hints.append(21.8)
    return hints or [24.0]


# =========================================================================
# PHASE 1: Exact & Inexact L-Cut (lcut.best_cuts + edge upright cut)
# =========================================================================
def phase_lcut(n, s0, donors, rng, tmp, t_stop):
    best_s, best_p, best_tag = s0, None, None
    M = int(math.floor(s0))
    # Sort promising larger donors by L[n0] - 1.0
    cands_n0 = []
    for delta in (0, 1, -1, 2):
        n0 = n + 2 * M + 1 + delta
        if n0 in donors:
            sd, pd = donors[n0]
            if sd - 1.0 < s0 + 0.008:
                cands_n0.append((sd - 1.0, n0, sd, pd))
    cands_n0.sort()
    for est, n0, sd, pd in cands_n0[:3]:
        if time.time() >= t_stop:
            break
        d_rem = n0 - n
        if d_rem <= 0:
            continue
        step = 0.25 if n0 > 550 else 0.16
        try:
            cuts = lcut.best_cuts(pd, sd, d_rem, step=step, m=8, sep=0.35)
        except Exception:
            cuts = []
        # Always include exact 4 corner boundary upright L-cuts (e.g. 842 -> 785!)
        edge_cuts = [
            (0.0, sd - 1.0, 0.0, 0.0),
            (0.0, sd - 1.0, sd - 1.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, sd - 1.0, 0.0),
        ]
        all_cuts = edge_cuts[:2] + cuts[:6]
        for idx, (sc, cx, cy, _) in enumerate(all_cuts):
            if time.time() >= t_stop:
                break
            C = lcut.apply_cut(pd, cx, cy, d_rem)
            s1 = max(sd - 1.0, 1.0)
            C = [
                (min(max(x, 0.5), s1 - 0.5), min(max(y, 0.5), s1 - 0.5), a % 90.0)
                for x, y, a in C
            ]
            # First quick screen, then round-press if promising
            for sched, tag in (
                (FQ_SCREEN, f"lcut_{n0}_s"),
                (FQ_ROUND_FAST, f"lcut_{n0}_r"),
            ):
                if time.time() >= t_stop:
                    break
                if tag.endswith("_r") and idx >= 3:
                    continue
                r = run_fq(s1, C, tmp, sched, timeout=50)
                if r and r[0] < best_s - 1e-10:
                    best_s, best_p, best_tag = r[0], r[1], tag
    return best_s, best_p, best_tag


# =========================================================================
# PHASE 2: Exact & Inexact L-Add (ladd.best_cuts + add_l)
# =========================================================================
def phase_ladd(n, s0, donors, rng, tmp, t_stop):
    best_s, best_p, best_tag = s0, None, None
    M = int(math.floor(s0))
    cands_n0 = []
    for delta in (0, -1, -2, 1):
        n0 = n - (2 * M - 1) + delta
        if n0 in donors:
            sd, pd = donors[n0]
            if sd + 1.0 < s0 + 0.015:
                cands_n0.append((sd + 1.0, n0, sd, pd))
    cands_n0.sort()
    for est, n0, sd, pd in cands_n0[:3]:
        if time.time() >= t_stop:
            break
        d_add = n - n0
        if d_add <= 0:
            continue
        step = 0.35 if n0 > 550 else 0.22
        try:
            cuts = ladd.best_cuts(pd, sd, step=step, m=6, sep=0.45)
        except Exception:
            continue
        for idx, (sc, cx, cy) in enumerate(cuts[:5]):
            if time.time() >= t_stop:
                break
            w = rng.uniform(1.002, 1.045)
            S_try, C = ladd.add_l(pd, sd, d_add, cx, cy, w, rng, 20.0)
            C = [
                (
                    min(max(x, 0.5), S_try - 0.5),
                    min(max(y, 0.5), S_try - 0.5),
                    a % 90.0,
                )
                for x, y, a in C
            ]
            r = run_fq(S_try, C, tmp, FQ_SCREEN, timeout=50)
            if r and r[0] < best_s - 1e-10:
                best_s, best_p, best_tag = r[0], r[1], f"ladd_{n0}"
            if idx < 2 and time.time() < t_stop:
                r2 = run_fq(S_try, C, tmp, FQ_ROUND_FAST, timeout=55)
                if r2 and r2[0] < best_s - 1e-10:
                    best_s, best_p, best_tag = r2[0], r2[1], f"ladd_{n0}_r"
    return best_s, best_p, best_tag


# =========================================================================
# PHASE 3A: fillfast-Screened Structural Band Extension & Donor Transplant
# =========================================================================
def phase_band_extend(n, best_s, best_p, extra_tilted_donors, rng, tmp, t_stop):
    wins = []
    # Collect source tilted clusters (self + any special donor like n=842 for K^2+1)
    sources = [("self", best_p)] + extra_tilted_donors
    hits = []

    for src_name, src_pts in sources:
        if time.time() >= t_stop:
            break
        arr = np.array(src_pts, dtype=np.float64)
        B = bf.bands(arr, min_size=8)
        if not B:
            continue
        # Extract all tilted squares
        a_fold = np.abs((arr[:, 2] + 45.0) % 90.0 - 45.0)
        all_tilted_idx = np.where(a_fold > 3.0)[0]
        base_T = [src_pts[i] for i in all_tilted_idx]

        # Focus on main band B[0] (and B[1] if dual-band)
        for bi in range(min(2, len(B))):
            b_idx = B[bi]
            b_set = set(b_idx.tolist())
            other_T = [src_pts[i] for i in all_tilted_idx if i not in b_set]
            med_deg = float(np.median(arr[b_idx, 2]))
            rad = math.radians(med_deg)
            c, s = math.cos(rad), math.sin(rad)
            u = np.array([c, s])
            v = np.array([-s, c])
            b_pts = arr[b_idx, :2]

            # Build multi-layer frontier around b_pts along +u, -u, +v, -v (up to 4 layers!)
            cur_set = [tuple(p) for p in b_pts]
            layers = {0: list(cur_set)}
            for lay_k in range(1, 5):
                prev_arr = np.array(cur_set, dtype=np.float64)
                new_pts = []
                for p in layers[lay_k - 1]:
                    for dv in (u, -u, v, -v):
                        cp = (p[0] + dv[0], p[1] + dv[1])
                        if (
                            np.min(
                                np.hypot(prev_arr[:, 0] - cp[0], prev_arr[:, 1] - cp[1])
                            )
                            < 0.72
                        ):
                            continue
                        if (
                            new_pts
                            and np.min(
                                [
                                    math.hypot(q[0] - cp[0], q[1] - cp[1])
                                    for q in new_pts
                                ]
                            )
                            < 0.72
                        ):
                            continue
                        new_pts.append(cp)
                if not new_pts:
                    break
                layers[lay_k] = new_pts
                cur_set.extend(new_pts)

            # Fast 10ms fillfast greedy screen of (layers, dth, dx, dy, S_try)
            for trial in range(28):
                if time.time() >= t_stop - 8:
                    break
                S_try = best_s - 0.002 if trial % 2 == 0 else best_s + 0.018
                n_lay = rng.choice((0, 1, 1, 2, 2, 3, 4))
                dth = 0.0 if trial < 6 else rng.uniform(-2.2, 2.2)
                dx = 0.0 if trial < 4 else rng.uniform(-1.2, 1.2)
                dy = 0.0 if trial < 4 else rng.uniform(-1.2, 1.2)
                deg_n = (med_deg + dth) % 90.0
                rn = math.radians(deg_n)
                hn = 0.5 * (abs(math.cos(rn)) + abs(math.sin(rn)))
                # Build rotated/shifted extended band clipped cleanly to [hn, S_try - hn]^2
                cand_pts = []
                for lk in range(min(n_lay + 1, len(layers))):
                    cand_pts.extend(layers[lk])
                # Optionally trim or keep contiguous directional growth
                cx, cy = float(np.mean(b_pts[:, 0])), float(np.mean(b_pts[:, 1]))
                rc, rs = math.cos(math.radians(dth)), math.sin(math.radians(dth))
                T_new = list(other_T)
                for px, py in cand_pts:
                    rx = cx + dx + rc * (px - cx) - rs * (py - cy)
                    ry = cy + dy + rs * (px - cx) + rc * (py - cy)
                    if (
                        hn + 1e-4 <= rx <= S_try - hn - 1e-4
                        and hn + 1e-4 <= ry <= S_try - hn - 1e-4
                    ):
                        T_new.append((rx, ry, deg_n))
                if len(T_new) < 10 or len(T_new) >= n:
                    continue
                X = fast_fill_eval(S_try, T_new, mode="greedy", tau=0.025)
                if len(X) >= n - 2:
                    # Confirm with exact fillfast at S_try and S_try + 0.025!
                    Xe = fast_fill_eval(
                        S_try, T_new, mode="exact", tau=0.02, time_limit=2
                    )
                    if len(Xe) >= n:
                        hits.append(
                            (len(Xe), S_try, Xe[:n], f"ext_{src_name}_l{n_lay}")
                        )
                    else:
                        S2 = best_s + 0.025
                        Xe2 = fast_fill_eval(
                            S2, T_new, mode="exact", tau=0.02, time_limit=2
                        )
                        if len(Xe2) >= n:
                            hits.append(
                                (len(Xe2) - 5, S2, Xe2[:n], f"ext2_{src_name}_l{n_lay}")
                            )

    # Quench top hits ONLY (zero random pad_to_n!)
    hits.sort(key=lambda z: (-z[0], z[1]))
    for _, S_h, pts_h, tag in hits[:8]:
        if time.time() >= t_stop:
            break
        r = run_fq(S_h, pts_h, tmp, FQ_ROUND_FAST, timeout=60)
        if r and r[0] < best_s - 1e-10:
            best_s, best_p = r[0], r[1]
            wins.append((tag, best_s))
    return best_s, best_p, wins


# =========================================================================
# PHASE 3B: Fast L-Shape & Variable-Width Channel SA (layout.py + fillfast!)
# =========================================================================
def mutate_wide(g, rng, max_w):
    left, bottom = [[(r[0], r[1], r[2] if len(r) > 2 else 0) for r in c] for c in g]
    ch = left if (rng.random() < 0.55 or not bottom) else bottom
    k = rng.randrange(7)
    if k <= 2 and ch:
        i = rng.randrange(len(ch))
        d = rng.gauss(0, (0.3, 1.5, 4.0)[rng.randrange(3)])
        if k == 2:
            for j in range(len(ch)):
                ch[j] = (ch[j][0], (ch[j][1] + d) % 90.0, ch[j][2])
        else:
            ch[i] = (ch[i][0], (ch[i][1] + d) % 90.0, ch[i][2])
    elif k == 3 and ch:
        i = rng.randrange(len(ch))
        if rng.random() < 0.55:
            nw = max(1, min(max_w, ch[i][0] + rng.choice((-2, -1, 1, 2))))
            ch[i] = (nw, ch[i][1], ch[i][2])
        else:
            nk = max(0.0, min(6.0, ch[i][2] + rng.gauss(0, 0.4)))
            ch[i] = (ch[i][0], ch[i][1], nk)
    elif k == 4:
        ch.append(ch[-1] if ch else (min(4, max_w), 24.0, 0))
    elif k == 5 and len(ch) > 1:
        ch.pop()
    elif k == 6 and ch:
        # Global width nudge on whole channel
        dw = rng.choice((-1, 1))
        for j in range(len(ch)):
            ch[j] = (max(1, min(max_w, ch[j][0] + dw)), ch[j][1], ch[j][2])
    return (left, bottom)


def phase_lshape_sa(n, best_s, best_p, hints, rng, tmp, t_stop):
    wins = []
    K = math.isqrt(n)
    max_w = max(6, min(15, K // 2))
    S_tgt = best_s - 0.002

    # 2 restarts: (1) L-shape two-wall at tight S, (2) single deep wall channel at slight slack S
    for restart in range(2):
        if time.time() >= t_stop - 8:
            break
        S_tgt = best_s - 0.002 if restart == 0 else best_s + 0.016
        th1 = hints[0] if hints else 24.0
        th2 = hints[1] if len(hints) >= 2 else (90.0 - th1 if restart == 0 else th1)
        w1 = max(2, min(max_w, K // 3 if (n - K * K <= 2) else K // 5))
        n1 = max(4, K // 2)
        if restart == 0:
            g = ([(w1, th1, 0)] * n1, [(max(2, w1 // 2), th2, 0)] * max(3, n1 // 2))
        else:
            g = ([(w1, 90.0 - th1, 0)] * (K - 2), [])

        def eval_g(gg):
            T = lay.build(S_tgt, *gg)
            if len(T) < 6 or len(T) >= n:
                return -1.0, T, []
            X0 = fast_fill_eval(S_tgt, T, mode="greedy", tau=0.0)
            X2 = fast_fill_eval(S_tgt, T, mode="greedy", tau=0.025)
            return len(X0) + 0.35 * max(0, len(X2) - len(X0)), T, X2

        f, T_cur, X_cur = eval_g(g)
        best_g_f = f
        best_hit = None
        if len(X_cur) >= n:
            best_hit = (S_tgt, X_cur[:n], T_cur)

        t_sa_end = min(t_stop - 8, time.time() + 14.0)
        it = 0
        while time.time() < t_sa_end and it < 85:
            it += 1
            g2 = mutate_wide(g, rng, max_w)
            f2, T2, X2 = eval_g(g2)
            if f2 >= f or (f2 > 0 and rng.random() < math.exp((f2 - f) / 0.8)):
                g, f, T_cur, X_cur = g2, f2, T2, X2
            if f2 > best_g_f:
                best_g_f = f2
                if len(X2) >= n - 1:
                    Xe = fast_fill_eval(S_tgt, T2, mode="exact", tau=0.02, time_limit=2)
                    if len(Xe) >= n:
                        best_hit = (S_tgt, Xe[:n], T2)
                    else:
                        Xe2 = fast_fill_eval(
                            best_s + 0.025, T2, mode="exact", tau=0.02, time_limit=2
                        )
                        if len(Xe2) >= n:
                            best_hit = (best_s + 0.025, Xe2[:n], T2)
            if best_hit is not None and len(best_hit[1]) >= n and best_hit[0] < best_s:
                break

        if best_hit is not None and time.time() < t_stop:
            S_h, pts_h, _ = best_hit
            r = run_fq(S_h, pts_h, tmp, FQ_ROUND_FAST, timeout=65)
            if r and r[0] < best_s - 1e-10:
                best_s, best_p = r[0], r[1]
                wins.append((f"lshape_sa_{restart}", best_s))
    return best_s, best_p, wins


# =========================================================================
# PHASE 4: Fast Multi-Move Basin Hopping (remove-k, bandflip, crot, mirror,
#          band_angles, anneal melt, harden reharden)
# =========================================================================
def phase_basin_hop(n, best_s, best_p, donors, rng, tmp, t_stop):
    wins = []
    # Quick remove-k from n+1..n+3
    for k in (1, 2, 3):
        if time.time() >= t_stop or (n + k) not in donors:
            break
        s_d, p_d = donors[n + k]
        if s_d > best_s + 0.09:
            continue
        arr = np.array(p_d, dtype=np.float64)
        dist_wall = np.minimum.reduce(
            [arr[:, 0], s_d - arr[:, 0], arr[:, 1], s_d - arr[:, 1]]
        )
        pool = np.argsort(dist_wall)[: min(n + k, 24)].tolist()
        for rep in range(3):
            if time.time() >= t_stop:
                break
            drop = set(rng.sample(pool if rep < 2 else range(n + k), k))
            cand = [p_d[i] for i in range(n + k) if i not in drop]
            r = run_fq(s_d, cand, tmp, FQ_SCREEN, timeout=45)
            if r and r[0] < best_s - 1e-10:
                best_s, best_p = r[0], r[1]
                wins.append((f"rem_{k}", best_s))

    # Multi-move hopping until t_stop
    it = 0
    while time.time() < t_stop:
        it += 1
        arr = np.array(best_p, dtype=np.float64)
        m = rng.random()
        cand = None
        tag = "kick"
        if m < 0.22:
            # Local cluster rotation (crot)
            c0 = np.array(
                [rng.uniform(0.5, best_s - 0.5), rng.uniform(0.5, best_s - 0.5)]
            )
            rad = rng.uniform(1.3, 3.2)
            phi = (
                90.0
                if rng.random() < 0.15
                else rng.choice([-1, 1]) * rng.uniform(3.0, 42.0)
            )
            idx = np.hypot(arr[:, 0] - c0[0], arr[:, 1] - c0[1]) < rad
            c, sn = math.cos(math.radians(phi)), math.sin(math.radians(phi))
            out = arr.copy()
            d = out[idx, :2] - c0
            out[idx, 0] = c0[0] + c * d[:, 0] - sn * d[:, 1]
            out[idx, 1] = c0[1] + sn * d[:, 0] + c * d[:, 1]
            out[idx, 2] += phi
            cand, tag = [tuple(row) for row in out], "crot"
        elif m < 0.42:
            # Bandflip surgery
            B = bf.bands(arr, min_size=10)
            if B:
                cand, _ = bf.flip(best_s, best_p, B[rng.randrange(len(B))], rng, B)
                tag = "bandflip"
        elif m < 0.62:
            # Mirror local patch
            c0 = (rng.uniform(0.5, best_s - 0.5), rng.uniform(0.5, best_s - 0.5))
            rad = rng.uniform(1.5, 3.2)
            kind = rng.choice(("h", "v", "d"))
            out = []
            for x, y, a in best_p:
                if math.hypot(x - c0[0], y - c0[1]) < rad:
                    dx, dy = x - c0[0], y - c0[1]
                    if kind == "h":
                        out.append((x, c0[1] - dy, -a))
                    elif kind == "v":
                        out.append((c0[0] - dx, y, -a))
                    else:
                        out.append((c0[0] + dy, c0[1] + dx, 90.0 - a))
                else:
                    out.append((x, y, a))
            cand, tag = out, "mirror"
        elif m < 0.78:
            # Coherent band shift/retilt
            B = bf.bands(arr, min_size=8)
            if B:
                bi = B[rng.randrange(len(B))]
                cx, cy = float(np.mean(arr[bi, 0])), float(np.mean(arr[bi, 1]))
                dth = rng.choice([-1, 1]) * rng.uniform(0.5, 3.0)
                dx, dy = rng.gauss(0, 0.08), rng.gauss(0, 0.08)
                c, sn = math.cos(math.radians(dth)), math.sin(math.radians(dth))
                out = arr.copy()
                d = out[bi, :2] - np.array([cx, cy])
                out[bi, 0] = cx + dx + c * d[:, 0] - sn * d[:, 1]
                out[bi, 1] = cy + dy + sn * d[:, 0] + c * d[:, 1]
                out[bi, 2] += dth
                cand, tag = [tuple(row) for row in out], "band_shift"
        elif m < 0.90 and os.path.exists(ANNEAL_BIN):
            # Regional hard-particle NPT melt
            a_f = os.path.join(tmp, "min.txt")
            b_f = os.path.join(tmp, "mout.txt")
            write_cfg(a_f, best_s, best_p)
            if os.path.exists(b_f):
                os.remove(b_f)
            cx, cy = best_p[rng.randrange(n)][:2]
            rem_t = max(5.0, min(40.0, t_stop - time.time()))
            try:
                rm = subprocess.run(
                    [
                        ANNEAL_BIN,
                        "melt",
                        "--in",
                        a_f,
                        "--out",
                        b_f,
                        "--cx",
                        repr(float(cx)),
                        "--cy",
                        repr(float(cy)),
                        "--rad",
                        f"{rng.uniform(1.8, 3.4):.3f}",
                        "--rmax",
                        f"{rng.uniform(0.06, 0.14):.3f}",
                        "--sweeps",
                        "2000",
                        "--bp",
                        "2000.0",
                        "--seed",
                        str(rng.randrange(1 << 30)),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=rem_t,
                )
                if rm.returncode == 0 and os.path.exists(b_f):
                    _, cand = read_cfg(b_f)
                    tag = "melt"
            except Exception:
                pass
        elif os.path.exists(HARDEN_BIN):
            # Single reharden shot (strictly time-bounded!)
            a_f = os.path.join(tmp, "hin.txt")
            b_f = os.path.join(tmp, "hout.txt")
            write_cfg(a_f, best_s, best_p)
            if os.path.exists(b_f):
                os.remove(b_f)
            rem_t = max(5.0, min(45.0, t_stop - time.time()))
            try:
                rh = subprocess.run(
                    [
                        HARDEN_BIN,
                        "reharden",
                        "--in",
                        a_f,
                        "--out",
                        b_f,
                        "--r0",
                        rng.choice(("0.08", "0.12")),
                        "--noise",
                        rng.choice(("0.004", "0.007")),
                        "--hold-f",
                        rng.choice(("1.0", "1.008")),
                        "--pre",
                        "1200",
                        "--hold",
                        "3500",
                        "--release",
                        "2500",
                        "--settle",
                        "1500",
                        "--seed",
                        str(rng.randrange(1 << 30)),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=rem_t,
                )
                if rh.returncode == 0 and os.path.exists(b_f):
                    _, cand = read_cfg(b_f)
                    tag = "harden"
            except Exception:
                pass

        if cand is None:
            sig = rng.choice((0.008, 0.025, 0.06))
            c_idx = rng.randrange(n)
            cx, cy = best_p[c_idx][:2]
            rad = rng.uniform(1.8, 3.8)
            cand = [
                (
                    x + rng.gauss(0, sig),
                    y + rng.gauss(0, sig),
                    a + rng.gauss(0, 20 * sig),
                )
                if (x - cx) ** 2 + (y - cy) ** 2 < rad * rad
                else (x, y, a)
                for x, y, a in best_p
            ]
            tag = "lkick"

        if time.time() >= t_stop:
            break
        sched = (
            FQ_ROUND_FAST
            if tag in ("bandflip", "band_shift") and rng.random() < 0.4
            else FQ_SCREEN
        )
        r = run_fq(
            best_s,
            cand,
            tmp,
            sched,
            timeout=max(8, int(min(45, t_stop - time.time() + 2))),
        )
        if r and r[0] < best_s - 1e-10:
            best_s, best_p = r[0], r[1]
            wins.append((tag, best_s))
    return best_s, best_p, wins


def solve_one(job):
    n, s0, pts0, donors, extra_tilted, gap_map, budget_min, seed = job
    rng = random.Random(seed)
    hints = angle_hints_for(n, gap_map, pts0)
    best_s, best_p = s0, pts0
    t_start = time.time()
    total_sec = budget_min * 60.0
    t_end = t_start + total_sec
    wins = []

    with tempfile.TemporaryDirectory() as tmp:
        # 1) Exact & inexact L-cut (25% budget)
        s1, p1, tag1 = phase_lcut(
            n, best_s, donors, rng, tmp, t_start + 0.25 * total_sec
        )
        if p1 is not None and s1 < best_s - 1e-10:
            best_s, best_p = s1, p1
            wins.append((tag1, best_s))

        # 2) Exact & inexact L-add (15% budget)
        s2, p2, tag2 = phase_ladd(
            n, best_s, donors, rng, tmp, t_start + 0.40 * total_sec
        )
        if p2 is not None and s2 < best_s - 1e-10:
            best_s, best_p = s2, p2
            wins.append((tag2, best_s))

        # 3A) fillfast-screened structural band extension & donor transplant (20% budget)
        best_s, best_p, w3a = phase_band_extend(
            n, best_s, best_p, extra_tilted, rng, tmp, t_start + 0.60 * total_sec
        )
        wins.extend(w3a)

        # 3B) Fast L-shape & wide-channel SA with fillfast (18% budget)
        best_s, best_p, w3b = phase_lshape_sa(
            n, best_s, best_p, hints, rng, tmp, t_start + 0.78 * total_sec
        )
        wins.extend(w3b)

        # 4) Fast multi-move basin hopping with remaining 22% budget
        best_s, best_p, w4 = phase_basin_hop(n, best_s, best_p, donors, rng, tmp, t_end)
        wins.extend(w4)

        # Final deep polish + exact Decimal64 >= 1.05e-9 stabilization
        if best_s < s0 - 1e-10:
            r_fin = run_fq(best_s, best_p, tmp, FQ_POLISH_DEEP, timeout=90)
            if r_fin and r_fin[0] < best_s:
                best_s, best_p = r_fin[0], r_fin[1]
            ok, Lf, xf, yf, tf = verify_and_stabilize(n, best_p)
            if ok and Lf < s0 - 1e-10:
                return n, (Lf, xf, yf, tf), wins
    return n, None, wins


def save_ckpt(path, improved):
    payload = {}
    for n, (Lf, xf, yf, tf) in improved.items():
        payload[f"L_{n}"] = np.array([Lf], dtype=np.float64)
        payload[f"x_{n}"] = xf
        payload[f"y_{n}"] = yf
        payload[f"t_{n}"] = tf
    np.savez_compressed(path, **payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=3)
    ap.add_argument("--max-targets", type=int, default=150)
    ap.add_argument("--minutes-total", type=float, default=150.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out-npz", default="/kaggle/working/w4_super_wins.npz")
    args = ap.parse_args()

    x, y, t, L, offs = load_union()
    lags = json.load(open(os.path.join(DS_DIR, "lag_targets.json")))
    gaps_raw = json.load(open(os.path.join(DS_DIR, "gaps.json")))
    gap_map = {g["n"]: g for g in gaps_raw}

    all_targets = [r["n"] for r in lags[: args.max_targets]]
    my_targets = [
        n for idx, n in enumerate(all_targets) if idx % args.num_shards == args.shard
    ]
    per_min = max(1.5, (args.minutes_total * args.procs) / max(1, len(my_targets)))
    print(
        f"W4v2 SuperEngine Shard {args.shard}/{args.num_shards}: {len(my_targets)} targets, {per_min:.2f}m/target on {args.procs} CPUs -> {my_targets[:12]}...",
        flush=True,
    )

    # Pre-cache puzzles needed as donors
    p_cache = {}

    def get_p(k):
        if k not in p_cache:
            p_cache[k] = get_puzzle(k, x, y, t, L, offs)
        return p_cache[k]

    p842 = get_p(842)[1]
    jobs = []
    for n in my_targets:
        s0, pts0 = get_p(n)
        M = int(math.floor(s0))
        donors = {}
        # Relevant L-cut, L-add, and remove-k donors only (lightweight pickle!)
        for dn in (
            [n + k for k in (1, 2, 3)]
            + [n + 2 * M + 1 + d for d in (-1, 0, 1, 2)]
            + [n - (2 * M - 1) + d for d in (-2, -1, 0, 1)]
        ):
            if 10 <= dn <= 1000:
                donors[dn] = get_p(dn)
        extra_tilted = []
        K = math.isqrt(n)
        if n - K * K == 1 and K >= 25 and n != 842:
            extra_tilted.append(("d842", p842))
        jobs.append(
            (
                n,
                s0,
                pts0,
                donors,
                extra_tilted,
                gap_map,
                per_min,
                20261124 + 997 * n + args.shard,
            )
        )

    improved = {}
    tot_gain = 0.0
    done_cnt = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(solve_one, j) for j in jobs]
        for fut in as_completed(futs):
            done_cnt += 1
            n, res, wins = fut.result()
            s0 = float(L[n - 1])
            if res is not None:
                Lf, xf, yf, tf = res
                gain = (s0 - Lf) / math.sqrt(n)
                tot_gain += gain
                improved[n] = (Lf, xf, yf, tf)
                print(
                    f"W4V2_WIN [{done_cnt}/{len(jobs)}] n={n:4d} {s0:.9f} -> {Lf:.9f} dL={s0 - Lf:.6f} dS={gain:.6f} tot={tot_gain:.6f} via={[w[0] for w in wins]}",
                    flush=True,
                )
            else:
                print(
                    f"W4V2_DRY [{done_cnt}/{len(jobs)}] n={n:4d} s0={s0:.9f}",
                    flush=True,
                )
            if done_cnt % 8 == 0 and improved:
                save_ckpt(args.out_npz, improved)

    save_ckpt(args.out_npz, improved)
    print(
        f"DONE W4v2 Shard {args.shard}: {len(improved)} wins, total_gain={tot_gain:.6f} -> {args.out_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
