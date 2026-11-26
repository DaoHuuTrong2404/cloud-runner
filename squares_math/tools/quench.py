"""Penalty energy + analytic gradient, and L-BFGS quench at fixed box S."""
import math
import numpy as np
import numba as nb
from scipy.optimize import minimize

CAP = 24


@nb.njit(cache=True)
def energy_grad(v, S, wb):
    n = v.shape[0] // 3
    x = v[0:n]; y = v[n:2 * n]; t = v[2 * n:3 * n]
    g = np.zeros(3 * n)
    c = np.cos(t); s = np.sin(t)
    cs = 1.42
    G = int(S / cs) + 3
    cnt = np.zeros((G, G), np.int32)
    items = np.full((G, G, CAP), -1, np.int32)
    gx_ = np.empty(n, np.int32); gy_ = np.empty(n, np.int32)
    for i in range(n):
        gx = min(max(int(x[i] / cs) + 1, 0), G - 1); gy = min(max(int(y[i] / cs) + 1, 0), G - 1)
        if cnt[gx, gy] < CAP:
            items[gx, gy, cnt[gx, gy]] = i; cnt[gx, gy] += 1
        gx_[i] = gx; gy_[i] = gy
    E = 0.0
    for i in range(n):
        # boundary
        ac = abs(c[i]); as_ = abs(s[i])
        h = 0.5 * (ac + as_)
        dh = 0.5 * (-np.sign(c[i]) * s[i] + np.sign(s[i]) * c[i])
        for dim in range(2):
            p = x[i] if dim == 0 else y[i]
            lo = h - p
            if lo > 0:
                E += wb * lo * lo
                g[dim * n + i] += -2 * wb * lo
                g[2 * n + i] += 2 * wb * lo * dh
            hi = p + h - S
            if hi > 0:
                E += wb * hi * hi
                g[dim * n + i] += 2 * wb * hi
                g[2 * n + i] += 2 * wb * hi * dh
        # pairs
        for ax_ in range(max(gx_[i] - 1, 0), min(gx_[i] + 2, G)):
            for ay_ in range(max(gy_[i] - 1, 0), min(gy_[i] + 2, G)):
                for q in range(cnt[ax_, ay_]):
                    j = items[ax_, ay_, q]
                    if j <= i:
                        continue
                    dx = x[j] - x[i]; dy = y[j] - y[i]
                    if dx * dx + dy * dy >= 2.0:
                        continue
                    best = 1e300; bk = -1
                    for k in range(4):
                        if k == 0:
                            ax = c[i]; ay = s[i]
                        elif k == 1:
                            ax = -s[i]; ay = c[i]
                        elif k == 2:
                            ax = c[j]; ay = s[j]
                        else:
                            ax = -s[j]; ay = c[j]
                        ri = 0.5 * (abs(ax * c[i] + ay * s[i]) + abs(-ax * s[i] + ay * c[i]))
                        rj = 0.5 * (abs(ax * c[j] + ay * s[j]) + abs(-ax * s[j] + ay * c[j]))
                        o = ri + rj - abs(ax * dx + ay * dy)
                        if o <= 0.0:
                            best = 0.0; bk = -1
                            break
                        if o < best:
                            best = o; bk = k
                    if bk < 0:
                        continue
                    o = best
                    E += o * o
                    w = 2 * o
                    # owner of axis
                    if bk < 2:
                        own = i; oth = j
                    else:
                        own = j; oth = i
                    to = t[own]; co = c[own]; so = s[own]
                    if bk % 2 == 0:
                        ax = co; ay = so; dax = -so; day = co
                    else:
                        ax = -so; ay = co; dax = -co; day = -so
                    proj = ax * dx + ay * dy
                    sg = 1.0 if proj >= 0 else -1.0
                    # d o / d centres: o = r - |a.d|
                    g[j] += w * (-sg * ax); g[n + j] += w * (-sg * ay)
                    g[i] += w * (sg * ax); g[n + i] += w * (sg * ay)
                    # other square radius r_oth = 0.5(|a.u_oth| + |a.v_oth|), u_oth=(c,s), v_oth=(-s,c)
                    cot = c[oth]; sot = s[oth]
                    p1 = ax * cot + ay * sot
                    p2 = -ax * sot + ay * cot
                    s1 = np.sign(p1); s2 = np.sign(p2)
                    # wrt theta_oth
                    dp1_dto = -ax * sot + ay * cot
                    dp2_dto = -ax * cot - ay * sot
                    g[2 * n + oth] += w * 0.5 * (s1 * dp1_dto + s2 * dp2_dto)
                    # wrt theta_own (through axis a)
                    dp1_da = dax * cot + day * sot
                    dp2_da = -dax * sot + day * cot
                    dproj = dax * dx + day * dy
                    g[2 * n + own] += w * (0.5 * (s1 * dp1_da + s2 * dp2_da) - sg * dproj)
    return E, g


def quench(x, y, t, S, wb=1.0, maxiter=3000, gtol=1e-18):
    n = len(x)
    v0 = np.concatenate([x, y, t]).astype(np.float64)
    r = minimize(lambda v: energy_grad(v, S, wb), v0, jac=True, method='L-BFGS-B',
                 options=dict(maxiter=maxiter, maxcor=30, ftol=1e-30, gtol=gtol, maxls=50))
    v = r.x
    return v[:n].copy(), v[n:2 * n].copy(), v[2 * n:].copy(), r.fun
