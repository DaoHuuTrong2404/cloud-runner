"""numba L-BFGS minimisation of overlap energy at fixed box S."""
import math
import numpy as np
import numba as nb
from quench import energy_grad


@nb.njit(cache=True)
def lbfgs(v, S, wb, max_iter, tolE, fixed, m=12):
    N = v.shape[0]; n = N // 3
    sk = np.zeros((m, N)); yk = np.zeros((m, N)); rho = np.zeros(m)
    E, g = energy_grad(v, S, wb)
    for i in range(n):
        if fixed[i] == 1:
            g[i] = 0.0; g[n + i] = 0.0; g[2 * n + i] = 0.0
        elif fixed[i] == 2:
            g[2 * n + i] = 0.0
    k = 0
    alpha_hist = np.zeros(m)
    stall = 0
    for it in range(max_iter):
        if E < tolE:
            return E, it
        # two-loop
        q = g.copy()
        cnt = min(k, m)
        for jj in range(cnt):
            j = (k - 1 - jj) % m
            a = rho[j] * np.dot(sk[j], q)
            alpha_hist[j] = a
            q -= a * yk[j]
        if cnt > 0:
            j = (k - 1) % m
            gamma = np.dot(sk[j], yk[j]) / max(np.dot(yk[j], yk[j]), 1e-300)
        else:
            gamma = 1e-3 / max(math.sqrt(np.dot(g, g)), 1e-300)
        r = gamma * q
        for jj in range(cnt - 1, -1, -1):
            j = (k - 1 - jj) % m
            b = rho[j] * np.dot(yk[j], r)
            r += sk[j] * (alpha_hist[j] - b)
        d = -r
        gd = np.dot(g, d)
        if gd >= 0:
            d = -g * 1e-3 / max(math.sqrt(np.dot(g, g)), 1e-300)
            gd = np.dot(g, d)
            k = 0
        # backtracking Armijo
        step = 1.0
        ok = False
        for ls in range(40):
            vn = v + step * d
            En, gn = energy_grad(vn, S, wb)
            if En <= E + 1e-4 * step * gd:
                ok = True
                break
            step *= 0.5
        if not ok:
            stall += 1
            k = 0
            if stall > 3:
                return E, it
            continue
        stall = 0
        for i in range(n):
            if fixed[i] == 1:
                gn[i] = 0.0; gn[n + i] = 0.0; gn[2 * n + i] = 0.0
            elif fixed[i] == 2:
                gn[2 * n + i] = 0.0
        s_ = vn - v; y_ = gn - g
        sy = np.dot(s_, y_)
        if sy > 1e-300:
            j = k % m
            sk[j] = s_; yk[j] = y_; rho[j] = 1.0 / sy
            k += 1
        relE = (E - En) / max(E, 1e-300)
        v[:] = vn; g = gn; E = En
        if relE < 1e-10 and it > 50:
            return E, it
    return E, max_iter
