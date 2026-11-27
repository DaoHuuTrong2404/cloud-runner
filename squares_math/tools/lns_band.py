"""Band-aware Large-Neighbourhood Search (band_angles, reseed, PIN fixed-angle L-BFGS + squeeze2).

Adapted cleanly from public MIT/Apache-2.0 tools (squeeze2, lbfgs, quench, sa) with NumPy 2.x compatibility.
"""

import math
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import squeeze2 as sq
import sa


KINDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def knn(x, y, i, k):
    d = (x - x[i]) ** 2 + (y - y[i]) ** 2
    return np.argsort(d)[: k + 1]


def band_angles(x, y, t, rng, L):
    """Band-wide smooth / alternating angle wave across all tilted squares."""
    tm = np.abs(((t + np.pi / 4) % (np.pi / 2)) - np.pi / 4)
    T = np.where(tm > 0.02)[0]
    x = x.copy()
    y = y.copy()
    t = t.copy()
    if len(T) < 3:
        return x, y, t, 7, []
    P = np.stack([x[T], y[T]], 1)
    P0 = P - P.mean(0)
    _, V = np.linalg.eigh(np.cov(P0.T))
    u = P0 @ V[:, 1]
    amp = math.radians(rng.uniform(0.5, 6.0))
    mode = int(rng.integers(4))
    if mode == 0:
        per = rng.uniform(2.0, 20.0)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        d = amp * np.sin(2.0 * np.pi * u / per + ph)
    elif mode == 1:
        d = amp * (u - u.mean()) / max(float(np.ptp(u)), 1e-9) * 2.0
    elif mode == 2:
        v = P0 @ V[:, 0]
        d = amp * np.sign(v - np.median(v))
    else:
        d = rng.normal(0.0, amp, len(T))
    t[T] += d
    x[T] += rng.normal(0.0, 0.02, len(T))
    y[T] += rng.normal(0.0, 0.02, len(T))
    x = np.clip(x, 0.5, L - 0.5)
    y = np.clip(y, 0.5, L - 0.5)
    return x, y, t, 7, []


def reseed(x, y, t, rng, L, band=False):
    """Remove squares in a strip/region and refill with a tilted lattice at angle theta."""
    n = len(x)
    tm = np.abs(((t + np.pi / 4) % (np.pi / 2)) - np.pi / 4)
    tilted = np.where(tm > 0.02)[0]
    if band and len(tilted) >= 4:
        c = int(tilted[rng.integers(len(tilted))])
        P = np.stack([x[tilted], y[tilted]], 1)
        d0 = np.hypot(P[:, 0] - x[c], P[:, 1] - y[c])
        loc = P[d0 < 4.0]
        if len(loc) >= 3:
            C = np.cov((loc - loc.mean(0)).T)
            _, V = np.linalg.eigh(C)
            u = V[:, 1]
        else:
            u = np.array([1.0, 1.0]) / math.sqrt(2.0)
        vperp = np.array([-u[1], u[0]])
        half_len = rng.uniform(1.5, 5.0)
        half_w = rng.uniform(0.9, 2.2)
        rel = np.stack([x - x[c], y - y[c]], 1)
        a1 = rel @ u
        a2 = rel @ vperp
        sel = np.where((np.abs(a1) < half_len) & (np.abs(a2) < half_w))[0]
        theta = (
            float(np.median(t[tilted[d0 < 4.0]])) if np.any(d0 < 4.0) else float(t[c])
        ) + rng.normal(0.0, 0.15)
    else:
        c = int(rng.integers(n))
        R = rng.uniform(1.2, 3.0)
        sel = np.where(np.hypot(x - x[c], y - y[c]) < R)[0]
        theta = (
            rng.uniform(0.0, np.pi / 2.0)
            if rng.random() < 0.6
            else float(t[c]) + rng.normal(0.0, 0.3)
        )
    m = len(sel)
    if m < 2:
        return x.copy(), y.copy(), t.copy(), 5 + int(band), []
    e1 = np.array([math.cos(theta), math.sin(theta)])
    e2 = np.array([-math.sin(theta), math.cos(theta)])
    off = rng.uniform(0.0, 1.0, 2)
    gx, gy = np.meshgrid(np.arange(-8, 9), np.arange(-8, 9))
    pts = (gx.reshape(-1, 1) + off[0]) * e1 + (gy.reshape(-1, 1) + off[1]) * e2
    mx, my = float(x[sel].mean()), float(y[sel].mean())
    dist = np.hypot(pts[:, 0], pts[:, 1])
    pick = pts[np.argsort(dist)[:m]]
    x = x.copy()
    y = y.copy()
    t = t.copy()
    x[sel] = mx + pick[:, 0]
    y[sel] = my + pick[:, 1]
    t[sel] = theta + rng.normal(0.0, 0.01, m)
    x = np.clip(x, 0.5, L - 0.5)
    y = np.clip(y, 0.5, L - 0.5)
    return x, y, t, 5 + int(band), []


def perturb(x, y, t, rng, L):
    n = len(x)
    x = x.copy()
    y = y.copy()
    t = t.copy()
    # Bias heavily toward macro band moves (6=band_reseed, 7=band_angles, 8/9=PIN angle) + cluster rotations
    macro_kinds = [6, 6, 7, 7, 8, 9, 1, 2, 4, 5]
    kind = int(macro_kinds[rng.integers(len(macro_kinds))])
    i = int(rng.integers(n))
    if kind == 7:
        return band_angles(x, y, t, rng, L)
    if kind in (8, 9):
        pins = [i] if kind == 8 else [int(z) for z in knn(x, y, i, 1)[:2]]
        for j in pins:
            t[j] += rng.choice([-1.0, 1.0]) * math.radians(rng.uniform(8.0, 32.0))
        return x, y, t, kind, pins
    if kind in (5, 6):
        return reseed(x, y, t, rng, L, band=(kind == 6))
    if kind == 1:
        cl = knn(x, y, i, int(rng.integers(4, 16)))
        t[cl] += rng.normal(0.0, 0.15, len(cl))
        x[cl] += rng.normal(0.0, 0.08, len(cl))
        y[cl] += rng.normal(0.0, 0.08, len(cl))
    elif kind == 2:
        cl = knn(x, y, i, int(rng.integers(3, 12)))
        a = rng.normal(0.0, 0.12)
        cx, cy = float(x[cl].mean()), float(y[cl].mean())
        dx, dy = x[cl] - cx, y[cl] - cy
        x[cl] = cx + dx * math.cos(a) - dy * math.sin(a)
        y[cl] = cy + dx * math.sin(a) + dy * math.cos(a)
        t[cl] += a
    else:
        cl = knn(x, y, i, int(rng.integers(2, 6)))
        j = int(cl[rng.integers(1, len(cl))])
        t[i] = t[j] + rng.normal(0.0, 0.05)
        x[cl] += rng.normal(0.0, 0.03, len(cl))
        y[cl] += rng.normal(0.0, 0.03, len(cl))
    x = np.clip(x, 0.5, L - 0.5)
    y = np.clip(y, 0.5, L - 0.5)
    return x, y, t, kind, []


def lns_burst(P_in, budget_s, seed, near_tol=0.0045):
    """Run fast Numba L-BFGS macro-LNS for `budget_s`.
    Returns `(best_P, best_S, near_cands, trials)` where `near_cands` are distinct squeezed basins within `near_tol` of `L0`!
    """
    rng = np.random.default_rng(int(seed))
    x, y, t = sq.normalize(
        np.asarray(P_in[:, 0], float),
        np.asarray(P_in[:, 1], float),
        np.asarray(P_in[:, 2], float),
    )
    n = len(x)
    Lbest = float(sq.side(x, y, t))
    L0 = Lbest
    best_P = np.column_stack([x, y, t])
    near_cands = []
    t_end = time.time() + float(budget_s)
    trials = 0

    while time.time() < t_end:
        px, py, pt, kind, pins = perturb(x, y, t, rng, Lbest)
        big = kind in (5, 6, 7, 8, 9)
        gk = (
            (5e-3 if kind == 7 else (rng.uniform(1e-2, 4.5e-2) if kind >= 8 else 2e-2))
            if big
            else 1.5e-3
        )
        Sg = Lbest * (1.0 + gk)
        if kind >= 8 and pins:
            fx_ = np.zeros(n, np.int64)
            fx_[pins] = 2
            ok, rx, ry, rt, _ = sq.relax_at(px, py, pt, Sg, 7000, fixed=fx_)
        else:
            ok, rx, ry, rt, _ = sq.relax_at(px, py, pt, Sg, 7000 if big else 3500)
        trials += 1
        if not ok:
            continue
        S, x2, y2, t2 = sq.squeeze(
            rx, ry, rt, rel0=gk * 0.5, rel_min=3e-7, max_iter=3500, max_tries=45
        )
        if S < L0 + near_tol:
            Lf, fx, fy, ft, good = sq.finalize(x2, y2, t2, tol=2e-12)
            if good:
                cand_P = np.column_stack([fx, fy, ft])
                # Check if genuinely distinct geometry (angle diff > 0.04 on >=1 square)
                ang_diff = np.max(
                    np.abs(((ft - t + math.pi / 4) % (math.pi / 2)) - math.pi / 4)
                )
                if Lf < Lbest - 1e-9:
                    S2, x3, y3, t3 = sq.squeeze(
                        fx,
                        fy,
                        ft,
                        rel0=1e-6,
                        rel_min=3e-10,
                        max_iter=15000,
                        max_tries=55,
                    )
                    L2, fx2, fy2, ft2, good2 = sq.finalize(x3, y3, t3, tol=2e-12)
                    if good2 and L2 < Lbest - 1e-10:
                        x, y, t, Lbest = fx2, fy2, ft2, float(L2)
                        best_P = np.column_stack([x, y, t])
                        near_cands.append(
                            (Lbest - L0, f"lns_k{kind}_win", best_P.copy())
                        )
                elif ang_diff > 0.035 and len(near_cands) < 6:
                    if all(abs(Lf - (L0 + g)) > 5e-6 for g, _, _ in near_cands):
                        near_cands.append((float(Lf - L0), f"lns_k{kind}", cand_P))
    near_cands.sort(key=lambda z: z[0])
    return best_P, Lbest, near_cands, trials
