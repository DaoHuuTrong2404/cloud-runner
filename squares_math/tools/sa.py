"""Simulated-annealing squeeze optimizer for unit squares in a square box.

Energy = sum_{pairs} pen_ij^2 + sum_i bnd_i^2
  pen_ij : SAT penetration depth (0 if separated)
  bnd_i  : how far square i sticks out of the box [0,S]^2 (x and y summed)
Moves  : single-square translate / rotate (Metropolis), cell-list neighbours.
"""
import math
import numpy as np
import numba as nb

CAP = 24  # max squares per cell


@nb.njit(cache=True, inline='always')
def pair_pen(xi, yi, ci, si, xj, yj, cj, sj):
    dx = xj - xi; dy = yj - yi
    if dx * dx + dy * dy >= 2.0:
        return 0.0
    best = 1e300
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
        o = ri + rj - abs(ax * dx + ay * dy)
        if o <= 0.0:
            return 0.0
        if o < best:
            best = o
    return best


@nb.njit(cache=True, inline='always')
def bnd_pen(x, y, c, s, S):
    h = 0.5 * (abs(c) + abs(s))
    v = 0.0
    if x - h < 0.0:
        v += h - x
    if x + h > S:
        v += x + h - S
    if y - h < 0.0:
        v += h - y
    if y + h > S:
        v += y + h - S
    return v


@nb.njit(cache=True)
def build_cells(x, y, S, cs):
    G = int(S / cs) + 1
    cnt = np.zeros((G, G), np.int32)
    items = np.full((G, G, CAP), -1, np.int32)
    cell = np.zeros((x.shape[0], 2), np.int32)
    for i in range(x.shape[0]):
        gx = min(max(int(x[i] / cs), 0), G - 1); gy = min(max(int(y[i] / cs), 0), G - 1)
        items[gx, gy, cnt[gx, gy]] = i; cnt[gx, gy] += 1
        cell[i, 0] = gx; cell[i, 1] = gy
    return G, cnt, items, cell


@nb.njit(cache=True)
def local_energy(i, xi, yi, ci, si, x, y, c, s, S, G, cnt, items, cs, wb):
    gx = min(max(int(xi / cs), 0), G - 1); gy = min(max(int(yi / cs), 0), G - 1)
    e = 0.0
    for ax in range(max(gx - 1, 0), min(gx + 2, G)):
        for ay in range(max(gy - 1, 0), min(gy + 2, G)):
            for q in range(cnt[ax, ay]):
                j = items[ax, ay, q]
                if j == i:
                    continue
                p = pair_pen(xi, yi, ci, si, x[j], y[j], c[j], s[j])
                e += p * p
    b = bnd_pen(xi, yi, ci, si, S)
    return e + wb * b * b


@nb.njit(cache=True)
def total_energy(x, y, t, S, wb):
    n = x.shape[0]
    c = np.cos(t); s = np.sin(t)
    cs = 1.42
    G, cnt, items, cell = build_cells(x, y, S, cs)
    e = 0.0
    for i in range(n):
        e += 0.5 * (local_energy(i, x[i], y[i], c[i], s[i], x, y, c, s, S, G, cnt, items, cs, wb) - wb * bnd_pen(x[i], y[i], c[i], s[i], S) ** 2)
        b = bnd_pen(x[i], y[i], c[i], s[i], S)
        e += wb * b * b
    return e


@nb.njit(cache=True)
def anneal(x, y, t, S, n_steps, T0, T1, step0, rot_frac, seed, wb, fixed):
    """In-place SA. fixed[i]=1 -> square i never moves. Returns final energy, best-zero flag."""
    np.random.seed(seed)
    n = x.shape[0]
    c = np.cos(t); s = np.sin(t)
    cs = 1.42
    G, cnt, items, cell = build_cells(x, y, S, cs)
    E = 0.0
    for i in range(n):
        b = bnd_pen(x[i], y[i], c[i], s[i], S)
        E += wb * b * b
        gx = cell[i, 0]; gy = cell[i, 1]
        for ax in range(max(gx - 1, 0), min(gx + 2, G)):
            for ay in range(max(gy - 1, 0), min(gy + 2, G)):
                for q in range(cnt[ax, ay]):
                    j = items[ax, ay, q]
                    if j > i:
                        p = pair_pen(x[i], y[i], c[i], s[i], x[j], y[j], c[j], s[j])
                        E += p * p
    step = step0
    acc = 0; tried = 0
    lT0 = math.log(T0); lT1 = math.log(T1)
    movable = np.where(fixed == 0)[0]
    nm = movable.shape[0]
    for it in range(n_steps):
        T = math.exp(lT0 + (lT1 - lT0) * it / n_steps)
        i = movable[np.random.randint(nm)]
        e_old = local_energy(i, x[i], y[i], c[i], s[i], x, y, c, s, S, G, cnt, items, cs, wb)
        r = np.random.random()
        nx = x[i]; ny = y[i]; nt = t[i]
        if r < rot_frac:
            nt = t[i] + np.random.normal() * step * 0.7
        elif r < rot_frac + 0.1:
            nx = x[i] + np.random.normal() * step
            ny = y[i] + np.random.normal() * step
            nt = t[i] + np.random.normal() * step * 0.7
        else:
            nx = x[i] + np.random.normal() * step
            ny = y[i] + np.random.normal() * step
        nc = math.cos(nt); ns = math.sin(nt)
        e_new = local_energy(i, nx, ny, nc, ns, x, y, c, s, S, G, cnt, items, cs, wb)
        dE = e_new - e_old
        tried += 1
        if dE <= 0.0 or np.random.random() < math.exp(-dE / T):
            acc += 1
            E += dE
            # update cell
            gx = min(max(int(nx / cs), 0), G - 1); gy = min(max(int(ny / cs), 0), G - 1)
            ox = cell[i, 0]; oy = cell[i, 1]
            if gx != ox or gy != oy:
                if cnt[gx, gy] < CAP:
                    for q in range(cnt[ox, oy]):
                        if items[ox, oy, q] == i:
                            cnt[ox, oy] -= 1
                            items[ox, oy, q] = items[ox, oy, cnt[ox, oy]]
                            break
                    items[gx, gy, cnt[gx, gy]] = i; cnt[gx, gy] += 1
                    cell[i, 0] = gx; cell[i, 1] = gy
                else:
                    E -= dE; acc -= 1
                    continue
            x[i] = nx; y[i] = ny; t[i] = nt; c[i] = nc; s[i] = ns
        if tried == 2000:
            ra = acc / tried
            if ra > 0.4:
                step *= 1.3
            elif ra < 0.2:
                step *= 0.7
            if step > 0.5:
                step = 0.5
            if step < 1e-7:
                step = 1e-7
            acc = 0; tried = 0
    return E, step


@nb.njit(cache=True)
def min_sep(x, y, t):
    """min SAT separation over close pairs (negative = overlap)"""
    n = x.shape[0]
    c = np.cos(t); s = np.sin(t)
    worst = 1e300
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]; dy = y[j] - y[i]
            if dx * dx + dy * dy >= 2.0:
                continue
            best = -1e300
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
                g = abs(ax * dx + ay * dy) - ri - rj
                if g > best:
                    best = g
            if best < worst:
                worst = best
    return worst


def side(x, y, t):
    h = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    return max((x + h).max() - (x - h).min(), (y + h).max() - (y - h).min())


def normalize(x, y, t):
    """translate so bounding box starts at 0"""
    h = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    return x - (x - h).min(), y - (y - h).min(), t
