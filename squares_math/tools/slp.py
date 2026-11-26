"""SLP squeeze: minimise box side S by sequential LPs on linearised SAT contacts.

state: arrays x, y, t (centres, angles). Box is [0,S]^2.
Each LP: variables (dx_i, dy_i, dt_i) for all i, plus S.
 - pair (i,j) near contact: separation along its current best SAT axis >= margin (linearised;
   |.| kinks split into both branches)
 - box: x_i - h_i >= 0, x_i + h_i <= S (h = half extent, kinks split)
 - trust region |dx|,|dy| <= r, |dt| <= r
"""
import math
import numpy as np
import numba as nb
import scipy.sparse as sp
from scipy.optimize import linprog


@nb.njit(cache=True)
def _near_pairs(x, y, rad):
    n = x.shape[0]
    cs = rad
    xmin = x.min(); ymin = y.min()
    G = int(max(x.max() - xmin, y.max() - ymin) / cs) + 2
    head = np.full((G, G), -1, np.int64)
    nxt = np.full(n, -1, np.int64)
    gx = np.empty(n, np.int64); gy = np.empty(n, np.int64)
    for i in range(n):
        a = int((x[i] - xmin) / cs); b = int((y[i] - ymin) / cs)
        gx[i] = a; gy[i] = b
        nxt[i] = head[a, b]; head[a, b] = i
    out = np.empty((n * 12, 2), np.int64)
    m = 0
    r2 = rad * rad
    for i in range(n):
        for a in range(max(gx[i] - 1, 0), min(gx[i] + 2, G)):
            for b in range(max(gy[i] - 1, 0), min(gy[i] + 2, G)):
                j = head[a, b]
                while j >= 0:
                    if j > i:
                        dx = x[j] - x[i]; dy = y[j] - y[i]
                        if dx * dx + dy * dy < r2:
                            if m >= out.shape[0]:
                                out2 = np.empty((out.shape[0] * 2, 2), np.int64)
                                out2[:m] = out[:m]
                                out = out2
                            out[m, 0] = i; out[m, 1] = j; m += 1
                    j = nxt[j]
    return out[:m]


@nb.njit(cache=True)
def _pair_rows(x, y, t, pairs, gate):
    """For each pair, best axis linearisation. Returns rows: (i, j, g0, coef[6]) possibly duplicated for kinks.
    coef order: dxi, dyi, dti, dxj, dyj, dtj"""
    P = pairs.shape[0]
    rows_ij = np.empty((P * 4, 2), np.int64)
    rows_g = np.empty(P * 4)
    rows_c = np.empty((P * 4, 6))
    m = 0
    for p in range(P):
        i = pairs[p, 0]; j = pairs[p, 1]
        ci = math.cos(t[i]); si = math.sin(t[i]); cj = math.cos(t[j]); sj = math.sin(t[j])
        dx = x[j] - x[i]; dy = y[j] - y[i]
        best = -1e300; bk = 0
        for k in range(4):
            if k == 0:
                ax = ci; ay = si
            elif k == 1:
                ax = -si; ay = ci
            elif k == 2:
                ax = cj; ay = sj
            else:
                ax = -sj; ay = cj
            ri = 0.5 * (abs(ax * ci + ay * si) + abs(-ax * si + ay * ci))
            rj = 0.5 * (abs(ax * cj + ay * sj) + abs(-ax * sj + ay * cj))
            g = abs(ax * dx + ay * dy) - ri - rj
            if g > best:
                best = g; bk = k
        if best > gate:
            continue
        if bk < 2:
            own = i; oth = j; co = ci; so = si; cot = cj; sot = sj
        else:
            own = j; oth = i; co = cj; so = sj; cot = ci; sot = si
        if bk % 2 == 0:
            ax = co; ay = so; dax = -so; day = co
        else:
            ax = -so; ay = co; dax = -co; day = -so
        proj = ax * dx + ay * dy
        sg = 1.0 if proj >= 0 else -1.0
        p1 = ax * cot + ay * sot
        p2 = -ax * sot + ay * cot
        dp1_dto = -ax * sot + ay * cot
        dp2_dto = -ax * cot - ay * sot
        dp1_da = dax * cot + day * sot
        dp2_da = -dax * sot + day * cot
        dproj = dax * dx + day * dy
        # g = sg*proj - 0.5 - 0.5(|p1| + |p2|)   (own radius is 0.5 on its own axis)
        # kink branches
        nb1 = 2 if abs(p1) < 1e-9 else 1
        nb2 = 2 if abs(p2) < 1e-9 else 1
        for b1 in range(nb1):
            s1 = (1.0 if p1 >= 0 else -1.0) if nb1 == 1 else (1.0 if b1 == 0 else -1.0)
            for b2 in range(nb2):
                s2 = (1.0 if p2 >= 0 else -1.0) if nb2 == 1 else (1.0 if b2 == 0 else -1.0)
                g0 = sg * proj - 0.5 - 0.5 * (s1 * p1 + s2 * p2)
                # derivatives
                # wrt centres: proj = a.(c_j - c_i)
                dcx_j = sg * ax; dcy_j = sg * ay
                dt_oth = -0.5 * (s1 * dp1_dto + s2 * dp2_dto)
                dt_own = sg * dproj - 0.5 * (s1 * dp1_da + s2 * dp2_da)
                rows_ij[m, 0] = i; rows_ij[m, 1] = j
                rows_g[m] = g0
                rows_c[m, 0] = -dcx_j; rows_c[m, 1] = -dcy_j
                rows_c[m, 3] = dcx_j; rows_c[m, 4] = dcy_j
                if own == i:
                    rows_c[m, 2] = dt_own; rows_c[m, 5] = dt_oth
                else:
                    rows_c[m, 2] = dt_oth; rows_c[m, 5] = dt_own
                m += 1
    return rows_ij[:m], rows_g[:m], rows_c[:m]


def side(x, y, t):
    h = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    return max((x + h).max() - (x - h).min(), (y + h).max() - (y - h).min())


def normalize(x, y, t):
    h = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    return x - (x - h).min(), y - (y - h).min(), t


@nb.njit(cache=True)
def min_sep_pairs(x, y, t, pairs):
    worst = 1e300
    for p in range(pairs.shape[0]):
        i = pairs[p, 0]; j = pairs[p, 1]
        ci = math.cos(t[i]); si = math.sin(t[i]); cj = math.cos(t[j]); sj = math.sin(t[j])
        dx = x[j] - x[i]; dy = y[j] - y[i]
        best = -1e300
        for k in range(4):
            if k == 0:
                ax = ci; ay = si
            elif k == 1:
                ax = -si; ay = ci
            elif k == 2:
                ax = cj; ay = sj
            else:
                ax = -sj; ay = cj
            ri = 0.5 * (abs(ax * ci + ay * si) + abs(-ax * si + ay * ci))
            rj = 0.5 * (abs(ax * cj + ay * sj) + abs(-ax * sj + ay * cj))
            g = abs(ax * dx + ay * dy) - ri - rj
            if g > best:
                best = g
        if best < worst:
            worst = best
    return worst


def lp_step(x, y, t, r, margin=0.0, fixed=None, gate=None, rt=None):
    if rt is None:
        rt = r
    n = len(x)
    if gate is None:
        gate = 2.2 * r + 1.6 * rt + 1e-9
    pairs = _near_pairs(x, y, math.sqrt(2) + gate + 1e-9)
    ij, g0, C = _pair_rows(x, y, t, pairs, gate)
    nv = 3 * n + 1
    iS = 3 * n
    rows = []; cols = []; vals = []; rhs = []
    m = 0
    # pair rows:  g0 + C.d >= margin   ->  -C.d <= g0 - margin
    for k in range(len(g0)):
        i, j = ij[k]
        idx = [i, n + i, 2 * n + i, j, n + j, 2 * n + j]
        for q in range(6):
            if C[k, q] != 0.0:
                rows.append(m); cols.append(idx[q]); vals.append(-C[k, q])
        rhs.append(g0[k] - margin); m += 1
    # box rows
    c = np.cos(t); s = np.sin(t)
    for i in range(n):
        branches = []
        # h = 0.5(|c|+|s|); dh/dt = 0.5(sgn(c)*(-s) + sgn(s)*c)
        sc_opts = [np.sign(c[i])] if abs(c[i]) > 1e-9 else [1.0, -1.0]
        ss_opts = [np.sign(s[i])] if abs(s[i]) > 1e-9 else [1.0, -1.0]
        for a in sc_opts:
            for b in ss_opts:
                h0 = 0.5 * (a * c[i] + b * s[i])
                dh = 0.5 * (-a * s[i] + b * c[i])
                branches.append((h0, dh))
        for (h0, dh) in branches:
            for dim, p in ((0, x[i]), (1, y[i])):
                if p - h0 < 2 * r + rt + 1e-9:
                    # p + dp - h0 - dh*dt >= 0  ->  -dp + dh*dt <= p - h0
                    rows += [m, m]; cols += [dim * n + i, 2 * n + i]; vals += [-1.0, dh]; rhs.append(p - h0); m += 1
                # p + dp + h0 + dh dt <= S  -> dp + dh dt - S <= -(p+h0)
                rows += [m, m, m]; cols += [dim * n + i, 2 * n + i, iS]; vals += [1.0, dh, -1.0]; rhs.append(-(p + h0)); m += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(m, nv))
    b = np.array(rhs)
    lb = np.full(nv, -r); ub = np.full(nv, r)
    lb[2 * n:3 * n] = -rt; ub[2 * n:3 * n] = rt
    lb[iS] = 0; ub[iS] = None
    if fixed is not None:
        for i in np.where(fixed)[0]:
            lb[i] = ub[i] = 0; lb[n + i] = ub[n + i] = 0; lb[2 * n + i] = ub[2 * n + i] = 0
    cvec = np.zeros(nv); cvec[iS] = 1.0
    res = linprog(cvec, A_ub=A, b_ub=b, bounds=list(zip(lb, ub)), method='highs')
    if res.status != 0:
        return None
    d = res.x
    return x + d[:n], y + d[n:2 * n], t + d[2 * n:3 * n], res.x[iS], pairs


def polish(x, y, t, iters=200, r0=1e-2, rmin=1e-10, tol_sep=-1e-13, verbose=False, fixed=None, time_limit=None, kappa=0.3):
    """SLP to local optimum with backtracking. Keeps state feasible (min_sep >= tol_sep). Returns (x,y,t,L)."""
    import time
    t_end = time.time() + time_limit if time_limit else None
    x, y, t = normalize(np.asarray(x, float).copy(), np.asarray(y, float).copy(), np.asarray(t, float).copy())
    L = side(x, y, t)
    r = r0
    stall = 0
    for it in range(iters):
        if t_end and time.time() > t_end:
            break
        rt = r * kappa
        out = lp_step(x, y, t, r, margin=2.0 * rt * rt + 1.0 * r * rt, fixed=fixed, rt=rt)
        if out is None:
            r *= 0.5
            if r < rmin: break
            continue
        nx, ny, nt, S, pairs = out
        dxv, dyv, dtv = nx - x, ny - y, nt - t
        acc = False
        for alpha in (1.0, 0.5, 0.25, 0.1):
            cx, cy, ct = normalize(x + alpha * dxv, y + alpha * dyv, t + alpha * dtv)
            pairs2 = _near_pairs(cx, cy, math.sqrt(2) + 1e-6)
            ms = min_sep_pairs(cx, cy, ct, pairs2) if len(pairs2) else 1.0
            L2 = side(cx, cy, ct)
            if ms >= tol_sep and L2 < L - 1e-15:
                acc = True
                break
        if acc:
            gain = L - L2
            x, y, t, L = cx, cy, ct, L2
            if verbose:
                print(f'  it{it} r={r:.1e} a={alpha} L={L:.12f} gain={gain:.2e} ms={ms:.1e}', flush=True)
            if alpha == 1.0:
                r = min(r * 2.0, 0.1)
            elif alpha <= 0.25:
                r *= 0.5
            stall = stall + 1 if gain < 1e-12 else 0
            if stall > 5:
                break
        else:
            r *= 0.3
            if r < rmin:
                break
    return x, y, t, L
