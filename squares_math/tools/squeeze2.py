"""Fast squeeze: bisection-like shrink with numba L-BFGS relaxations."""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import lbfgs as _lb
import sa


def side(x, y, t):
    h = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    return max((x + h).max() - (x - h).min(), (y + h).max() - (y - h).min())


def normalize(x, y, t):
    h = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    return x - (x - h).min(), y - (y - h).min(), t


def relax_at(x, y, t, S, max_iter=20000, tolE=1e-26, fixed=None):
    n = len(x)
    x, y, t = normalize(x, y, t)
    L = side(x, y, t)
    f = S / L
    v = np.concatenate([x * f, y * f, t]).astype(np.float64)
    if fixed is None:
        fixed = np.zeros(n, np.int64)
    E, it = _lb.lbfgs(v, S, 1.0, max_iter, tolE, fixed)
    return E < tolE, v[:n].copy(), v[n:2 * n].copy(), v[2 * n:].copy(), E


def squeeze(x, y, t, rel0=1e-3, rel_min=3e-10, max_iter=20000, max_tries=70, fixed=None):
    """Shrink/relax from a feasible-ish packing. Returns (S, x, y, t) with E(S) ~ 0."""
    x, y, t = normalize(np.asarray(x, float), np.asarray(y, float), np.asarray(t, float))
    S = side(x, y, t)
    # make sure start is feasible at its own side (relax overlaps first)
    ok, x1, y1, t1, E = relax_at(x, y, t, S, max_iter, fixed=fixed)
    if not ok:
        # inflate until feasible
        for k in range(30):
            S *= 1.0 + 1e-4 * 2 ** k
            ok, x1, y1, t1, E = relax_at(x, y, t, S, max_iter, fixed=fixed)
            if ok:
                break
        if not ok:
            return float('inf'), x, y, t
    x, y, t = x1, y1, t1
    rel = rel0
    for _ in range(max_tries):
        if rel < rel_min:
            break
        St = S * (1 - rel)
        ok, nx, ny, nt, E = relax_at(x, y, t, St, max_iter, fixed=fixed)
        if ok:
            x, y, t, S = nx, ny, nt, St
            rel *= 2.0
        else:
            rel *= 0.3
    return S, x, y, t


def finalize(x, y, t, tol=2e-12):
    """remove residual overlaps (tiny) by scaling centres; returns (L, x, y, t, ok)"""
    x, y, t = normalize(x, y, t)
    cx, cy = x.mean(), y.mean()
    for _ in range(80):
        ms = sa.min_sep(x, y, t)
        if ms >= tol:
            x, y, t = normalize(x, y, t)
            return side(x, y, t), x, y, t, True
        eps = (tol - ms) * 1.2 + 1e-15
        x = cx + (x - cx) * (1 + eps); y = cy + (y - cy) * (1 + eps)
    return float('inf'), x, y, t, False
