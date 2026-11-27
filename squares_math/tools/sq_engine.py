# generated from the notebook source: definitions only

import math, time, glob, os

import numpy as np

import pandas as pd

from numba import njit

import highspy

from scipy.optimize import brentq


def load_highspy():
    # HiGHS ships with the attached dataset, either as a wheel or as an unpacked wheel folder;
    # with Internet on, pip can also fetch it
    try:
        import highspy

        return highspy
    except ImportError:
        pass
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    roots = [d for d in ("/kaggle/input", "input") if os.path.isdir(d)]
    wheels = [
        w
        for r in roots
        for w in glob.glob(f"{r}/**/highspy-*{tag}*.whl", recursive=True)
    ]
    if wheels:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-index", wheels[0]],
            check=False,
        )
    else:
        for r in roots:
            for d in glob.glob(f"{r}/**/highspy-*{tag}*", recursive=True):
                if os.path.isdir(os.path.join(d, "highspy")):
                    sys.path.insert(0, d)
                    break
    try:
        import highspy

        return highspy
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "highspy"], check=False
        )
        import highspy

        return highspy


INPUT_DIRS = ["/kaggle/input", "input"]

N_MAX = 1000

HALF_PI = math.pi / 2

INF = highspy.kHighsInf


def find_input(name):
    for d in INPUT_DIRS:
        hits = sorted(glob.glob(f"{d}/**/{name}", recursive=True))
        if hits:
            return hits[0]
    return None


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


@njit
def pair_gap(xi, yi, ti, xj, yj, tj):
    # G_ij of Theorem 6
    dx = xj - xi
    dy = yj - yi
    d = tj - ti
    r = 0.5 * (abs(math.cos(d)) + abs(math.sin(d)))
    ci = math.cos(ti)
    si = math.sin(ti)
    cj = math.cos(tj)
    sj = math.sin(tj)
    best = max(
        abs(dx * ci + dy * si),
        abs(-dx * si + dy * ci),
        abs(dx * cj + dy * sj),
        abs(-dx * sj + dy * cj),
    )
    return best - 0.5 - r


@njit
def corners(P):
    n = P.shape[0]
    C = np.empty((n, 4, 2))
    for i in range(n):
        c = math.cos(P[i, 2])
        s = math.sin(P[i, 2])
        hx = 0.5 * (c - s)
        hy = 0.5 * (s + c)
        C[i, 0, 0] = P[i, 0] + hx
        C[i, 0, 1] = P[i, 1] + hy
        C[i, 1, 0] = P[i, 0] - hy
        C[i, 1, 1] = P[i, 1] + hx
        C[i, 2, 0] = P[i, 0] - hx
        C[i, 2, 1] = P[i, 1] - hy
        C[i, 3, 0] = P[i, 0] + hy
        C[i, 3, 1] = P[i, 1] - hx
    return C


@njit
def extent(P):
    # lowest and highest coordinate reached by any square, x and y together (Lemma 5)
    lo = 1e18
    hi = -1e18
    for i in range(P.shape[0]):
        w = 0.5 * (abs(math.cos(P[i, 2])) + abs(math.sin(P[i, 2])))
        for a in range(2):
            if P[i, a] - w < lo:
                lo = P[i, a] - w
            if P[i, a] + w > hi:
                hi = P[i, a] + w
    return lo, hi


@njit
def side(P):
    xl = 1e18
    xh = -1e18
    yl = 1e18
    yh = -1e18
    for i in range(P.shape[0]):
        w = 0.5 * (abs(math.cos(P[i, 2])) + abs(math.sin(P[i, 2])))
        xl = min(xl, P[i, 0] - w)
        xh = max(xh, P[i, 0] + w)
        yl = min(yl, P[i, 1] - w)
        yh = max(yh, P[i, 1] + w)
    return max(xh - xl, yh - yl)


@njit
def neighbour_pairs(P, cut):
    # all index pairs with centre distance below cut, by a uniform cell list
    n = P.shape[0]
    xs = P[:, 0]
    ys = P[:, 1]
    x0 = xs.min()
    y0 = ys.min()
    nx = int((xs.max() - x0) / cut) + 1
    ny = int((ys.max() - y0) / cut) + 1
    head = -np.ones(nx * ny, np.int64)
    nxt = -np.ones(n, np.int64)
    for i in range(n):
        c = int((xs[i] - x0) / cut) + nx * int((ys[i] - y0) / cut)
        nxt[i] = head[c]
        head[c] = i
    cap = n * 12 + 16
    out = np.empty((cap, 2), np.int64)
    k = 0
    c2 = cut * cut
    for i in range(n):
        ci = int((xs[i] - x0) / cut)
        cj = int((ys[i] - y0) / cut)
        for a in range(max(0, ci - 1), min(nx, ci + 2)):
            for b in range(max(0, cj - 1), min(ny, cj + 2)):
                j = head[a + nx * b]
                while j >= 0:
                    if j > i:
                        dx = xs[j] - xs[i]
                        dy = ys[j] - ys[i]
                        if dx * dx + dy * dy < c2:
                            if k == cap:
                                cap *= 2
                                bigger = np.empty((cap, 2), np.int64)
                                bigger[:k] = out[:k]
                                out = bigger
                            out[k, 0] = i
                            out[k, 1] = j
                            k += 1
                    j = nxt[j]
    return out[:k].copy()


@njit
def min_gap_over(P, pairs):
    m = 1e9
    for p in range(pairs.shape[0]):
        i = pairs[p, 0]
        j = pairs[p, 1]
        dx = P[j, 0] - P[i, 0]
        dy = P[j, 1] - P[i, 1]
        if dx * dx + dy * dy < 2.0000001:
            m = min(m, pair_gap(P[i, 0], P[i, 1], P[i, 2], P[j, 0], P[j, 1], P[j, 2]))
    return m


def min_gap(P):
    P = np.ascontiguousarray(P, dtype=np.float64)
    return 1e9 if len(P) < 2 else min_gap_over(P, neighbour_pairs(P, 1.4143))


def to_origin(P):
    P = np.array(P, dtype=np.float64)
    w = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    P[:, 0] -= (P[:, 0] - w).min()
    P[:, 1] -= (P[:, 1] - w).min()
    return P


def fold(t):
    # a square is unchanged by quarter turns: reduce an angle to (-45, 45] degrees
    return np.asarray(t) - HALF_PI * np.round(np.asarray(t) / HALF_PI)


def draw(ax, P, title=None, lw=0.35, box=True, window=None, colors=None):
    P = to_origin(P)
    C = corners(P)
    S = side(P)
    if colors is None:
        t = np.degrees(np.abs(fold(P[:, 2])))
        colors = [
            "#ebeae5" if v < 1e-4 else BLUES(0.25 + 0.75 * min(1.0, v / 45.0))
            for v in t
        ]
    ax.add_collection(
        PolyCollection(C, facecolors=colors, edgecolors="#3b3a37", linewidths=lw)
    )
    if box:
        ax.add_patch(Rectangle((0, 0), S, S, fill=False, ec=ORANGE, lw=1.4))
    if window is None:
        window = (-0.02 * S, 1.02 * S, -0.02 * S, 1.02 * S)
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)


def grid_packing(n):
    k = math.isqrt(n - 1) + 1
    return np.array([[i % k + 0.5, i // k + 0.5, 0.0] for i in range(n)])


class Portfolio:
    def __init__(self):
        self.side = {n: float(math.isqrt(n - 1) + 1) for n in range(1, N_MAX + 1)}
        self.P = {n: grid_packing(n) for n in range(1, N_MAX + 1)}
        self.src = {n: "grid" for n in range(1, N_MAX + 1)}

    def offer(self, P, src, tol=-1e-9):
        P = np.ascontiguousarray(P, dtype=np.float64)
        n = P.shape[0]
        if (
            not (1 <= n <= N_MAX)
            or P.ndim != 2
            or P.shape[1] != 3
            or not np.isfinite(P).all()
        ):
            return False
        s = side(P)
        if s >= self.side[n] - 1e-12 or s < math.sqrt(n) - 1e-9 or min_gap(P) < tol:
            return False
        self.side[n] = s
        self.P[n] = to_origin(P)
        self.src[n] = src
        return True

    def score(self):
        return sum(self.side[n] / math.sqrt(n) - 1 for n in range(1, N_MAX + 1))

    def sides(self):
        return np.array([self.side[n] for n in range(1, N_MAX + 1)])

    def copy(self):
        q = Portfolio()
        q.side = dict(self.side)
        q.P = dict(self.P)
        q.src = dict(self.src)
        return q


def read_submission(path):
    df = pd.read_csv(path, dtype={"x": str, "y": str, "angle": str})
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ("x", "y", "angle"):
        df[c] = df[c].str.lstrip("s").astype(np.float64)
    out = {}
    for n, g in df.groupby("n", sort=True):
        if 1 <= int(n) <= N_MAX and len(g) == int(n):
            out[int(n)] = np.ascontiguousarray(
                g.sort_values("square_id")[["x", "y", "angle"]].to_numpy(np.float64)
            )
    return out


def submission_files():
    found = []
    for d in INPUT_DIRS:
        for p in sorted(glob.glob(f"{d}/**/*.csv", recursive=True)):
            try:
                head = (
                    open(p, errors="ignore")
                    .readline()
                    .replace('"', "")
                    .replace(" ", "")
                    .lower()
                )
            except OSError:
                continue
            if head.startswith("id,n,square_id,x,y,angle"):
                found.append(p)
    return found


@njit
def _find(par, i):
    while par[i] != i:
        par[i] = par[par[i]]
        i = par[i]
    return i


@njit
def _weld(P, pairs, tol):
    n = P.shape[0]
    par = np.arange(n)
    for p in range(pairs.shape[0]):
        i = pairs[p, 0]
        j = pairs[p, 1]
        d = P[j, 2] - P[i, 2]
        d = d - HALF_PI * round(d / HALF_PI)
        if abs(d) > tol:
            continue
        c = math.cos(P[i, 2])
        s = math.sin(P[i, 2])
        dx = P[j, 0] - P[i, 0]
        dy = P[j, 1] - P[i, 1]
        u = dx * c + dy * s
        v = -dx * s + dy * c
        if (abs(abs(u) - 1.0) < tol and abs(v) < tol) or (
            abs(abs(v) - 1.0) < tol and abs(u) < tol
        ):
            a = _find(par, i)
            b = _find(par, j)
            if a != b:
                par[a] = b
    out = np.empty(n, np.int64)
    for i in range(n):
        out[i] = _find(par, i)
    return out


def find_bodies(P, tol=1e-3, singles=None):
    # body of every square; kind 2 = translations only (upright blocks), 3 = translations and a rotation
    P = np.ascontiguousarray(P, dtype=np.float64)
    n = P.shape[0]
    root = _weld(P, neighbour_pairs(P, 1.2), tol) if n > 1 else np.zeros(n, np.int64)
    if singles is not None and len(singles):
        root = root.copy()
        root[np.asarray(singles)] = n + np.arange(len(singles))
    _, body = np.unique(root, return_inverse=True)
    size = np.bincount(body)
    upright = np.zeros(len(size), bool)
    np.logical_or.at(upright, body, np.abs(fold(P[:, 2])) < 1e-9)
    kind = np.where(upright & (size > 1), 2, 3).astype(np.int64)
    start = np.zeros(len(size), np.int64)
    start[1:] = np.cumsum(kind)[:-1]
    return body.astype(np.int64), kind, start, size


def body_centres(P, body, size):
    c = np.zeros((len(size), 2))
    np.add.at(c[:, 0], body, P[:, 0])
    np.add.at(c[:, 1], body, P[:, 1])
    return c / size[:, None]


def side_triangles(P):
    # Proposition 9 bar by bar: every bar wastes tan(theta) at its two short ends
    body, kind, start, size = find_bodies(P)
    tilt = np.zeros(len(size))
    tilt[body] = fold(P[:, 2])
    bar = (np.abs(tilt) > 0.02) & (size >= 3)
    return float(np.sum(np.tan(np.abs(tilt[bar])))), int(bar.sum())


@njit
def empty_area(P, S, res):
    # empty area of every unit cell, by rasterising at `res` samples per unit
    m = int(S * res) + 1
    g = np.zeros((m, m), np.uint8)
    for i in range(P.shape[0]):
        c = math.cos(P[i, 2])
        s = math.sin(P[i, 2])
        for a in range(
            int(max(0, (P[i, 0] - 0.72) * res)),
            int(min(m - 1, (P[i, 0] + 0.72) * res)) + 1,
        ):
            for b in range(
                int(max(0, (P[i, 1] - 0.72) * res)),
                int(min(m - 1, (P[i, 1] + 0.72) * res)) + 1,
            ):
                dx = (a + 0.5) / res - P[i, 0]
                dy = (b + 0.5) / res - P[i, 1]
                if abs(dx * c + dy * s) <= 0.5 and abs(-dx * s + dy * c) <= 0.5:
                    g[a, b] = 1
    k = int(math.ceil(S))
    lim = int(S * res)
    V = np.zeros((k, k))
    for a in range(lim):
        for b in range(lim):
            if g[a, b] == 0:
                V[a // res, b // res] += 1.0 / (res * res)
    return V


def band_tilts(P):
    # the two most common tilts among tilted squares, in radians (vertical band first if recognisable)
    t = np.abs(fold(P[:, 2]))
    t = t[t > 0.03]
    vals, counts = np.unique(np.round(t, 3), return_counts=True)
    return vals[np.argsort(-counts)[:2]]


def theta_of(b, eps):
    f = lambda t: b * (1 - math.cos(t)) - math.sin(t) - eps
    return brentq(f, 2 * math.atan(1 / b) + 1e-9, 1.4)


def portfolio_table(pf):
    rec = []
    for n in range(325, N_MAX + 1):
        m = math.isqrt(n)
        if pf.src[n] == "grid" or m * m == n:
            continue
        S = pf.side[n]
        t = np.degrees(np.abs(fold(pf.P[n][:, 2])))
        t = t[t > 1.0]
        rec.append(
            dict(
                n=n,
                m=m,
                j=n - m * m,
                delta=S - m,
                eps=m + 1 - S,
                W=S * S - n,
                tilt=float(np.median(t)) if len(t) else np.nan,
            )
        )
    A = pd.DataFrame(rec)
    A["W_over_m"] = A.W / A.m
    return A


@njit
def cap_count(c, s, W, y):
    tot = 0
    i = 0
    while i < 200:
        a = max(0.0, s + (i + 1 - y) * c / s)
        m = math.floor(W - a + 1e-9)
        if m <= 0 and a > 0:
            break
        if m > 0:
            tot += m
        i += 1
    return tot


@njit
def band_best(b, th, L, nsplit):
    c = math.cos(th)
    s = math.sin(th)
    W = b * c + s
    h = b * s + c
    if L <= h:
        return -1, 0.0
    R = math.floor((L - h) * c + 1e-9) + 1
    F = L - (R - 1) / c - h
    best = -1
    yb = 0.0
    for q in range(nsplit + 1):
        y0 = F * q / nsplit
        cnt = b * R + cap_count(c, s, W, y0) + cap_count(c, s, W, F - y0)
        if cnt > best:
            best = cnt
            yb = y0
    return best, yb


@njit
def scan_frames(K, nth, bmax):
    out = np.full(K * K + 1, 1e9)
    par = np.zeros((K * K + 1, 6))
    for bV in range(2, bmax + 1):
        for iv in range(1, nth):
            tv = 0.9 * iv / nth
            S = K - bV + bV * math.cos(tv) + math.sin(tv)
            if S >= K - 1e-6 or S <= K - 1:
                continue
            cv, yv = band_best(bV, tv, S, 16)
            if cv < 0:
                continue
            for bH in range(2, bmax + 1):
                for ih in range(1, nth):
                    th = 0.9 * ih / nth
                    if K - bH + bH * math.cos(th) + math.sin(th) > S:
                        continue
                    ch, yh = band_best(bH, th, float(K - bV), 16)
                    if ch < 0:
                        continue
                    n = (K - bV) * (K - bH) + cv + ch
                    if n <= K * K and S < out[n]:
                        out[n] = S
                        par[n, 0] = bV
                        par[n, 1] = tv
                        par[n, 2] = yv
                        par[n, 3] = bH
                        par[n, 4] = th
                        par[n, 5] = yh
    return out, par


def band_squares(b, th, L, y0):
    c = math.cos(th)
    s = math.sin(th)
    W = b * c + s
    h = b * s + c
    R = math.floor((L - h) * c + 1e-9) + 1
    F = L - (R - 1) / c - h
    sq = [
        (s + (i + 0.5) * c - 0.5 * s, y0 + j / c + (i + 0.5) * s + 0.5 * c, th)
        for j in range(R)
        for i in range(b)
    ]
    for y, far in ((y0, False), (F - y0, True)):
        i = 0
        while i < 200:
            a = max(0.0, s + (i + 1 - y) * c / s)
            m = math.floor(W - a + 1e-9)
            if m <= 0 and a > 0:
                break
            for q in range(max(m, 0)):
                X, Y = W - m + q + 0.5, i + 0.5
                sq.append((W - X, L - Y, 0.0) if far else (X, Y, 0.0))
            i += 1
    return sq


def build_frame(K, bV, tV, yV, bH, tH, yH):
    bV = int(bV)
    bH = int(bH)
    S = K - bV + bV * math.cos(tV) + math.sin(tV)
    sq = [(i + 0.5, j + 0.5, 0.0) for i in range(K - bV) for j in range(K - bH)]
    sq += [(K - bV + X, Y, t) for X, Y, t in band_squares(bV, tV, S, yV)]
    sq += [(Y, K - bH + X, -t) for X, Y, t in band_squares(bH, tH, float(K - bV), yH)]
    return np.array(sq), S


@njit
def move_bodies(P, body, ref, start, kind, x):
    Q = P.copy()
    for i in range(P.shape[0]):
        A = body[i]
        s = start[A]
        if kind[A] == 3:
            ph = x[s + 2]
            rx = P[i, 0] - ref[A, 0]
            ry = P[i, 1] - ref[A, 1]
            c = math.cos(ph)
            sn = math.sin(ph)
            Q[i, 0] = ref[A, 0] + x[s] + c * rx - sn * ry
            Q[i, 1] = ref[A, 1] + x[s + 1] + sn * rx + c * ry
            Q[i, 2] = P[i, 2] + ph
        else:
            Q[i, 0] = P[i, 0] + x[s]
            Q[i, 1] = P[i, 1] + x[s + 1]
    return Q


@njit
def _branches(c, s):
    # slope of the active branch of rho, slope of the next one, and how far below it sits (Lemma 11)
    s1 = 1.0 if c >= 0 else -1.0
    s2 = 1.0 if s >= 0 else -1.0
    da = 0.5 * (-s1 * s + s2 * c)
    if abs(s) <= abs(c):
        return da, 0.5 * (-s1 * s - s2 * c), abs(s)
    return da, 0.5 * (s1 * s + s2 * c), abs(c)


@njit
def lp_rows(P, body, ref, start, kind, pairs, walls, nv, r, S, fixed):
    # Rows  A z <= b  in z = (p, q, sigma), body motion x = p - q.
    # Two rows per candidate contact or wall, one per branch of rho.  With fixed=True a row is kept
    # for every candidate (zeros, b = inf when out of reach) so the sparsity pattern never changes.
    KINK = 0.12
    npair = pairs.shape[0]
    nwall = walls.shape[0]
    maxrow = 2 * npair + 4 * nwall
    rstart = np.empty(maxrow + 1, np.int32)
    rhs = np.empty(maxrow)
    ri = np.empty(maxrow, np.int64)
    rj = np.empty(maxrow, np.int64)
    idx = np.empty(maxrow * 13, np.int32)
    val = np.empty(maxrow * 13)
    cut = 6.0 * r + 1e-6
    k = 0
    row = 0
    for p in range(npair):
        i = pairs[p, 0]
        j = pairs[p, 1]
        dx = P[j, 0] - P[i, 0]
        dy = P[j, 1] - P[i, 1]
        ti = P[i, 2]
        tj = P[j, 2]
        ci = math.cos(ti)
        si = math.sin(ti)
        cj = math.cos(tj)
        sj = math.sin(tj)
        d = tj - ti
        cd = math.cos(d)
        sd = math.sin(d)
        rho = 0.5 * (abs(cd) + abs(sd))
        best = -1e9
        kb = 0
        pb = 0.0
        for q in range(4):
            if q == 0:
                ax = ci
                ay = si
            elif q == 1:
                ax = -si
                ay = ci
            elif q == 2:
                ax = cj
                ay = sj
            else:
                ax = -sj
                ay = cj
            pr = dx * ax + dy * ay
            g = abs(pr) - 0.5 - rho
            if g > best:
                best = g
                kb = q
                pb = pr
        reach = best <= cut
        if (not reach) and (not fixed):
            continue
        if kb == 0:
            ax = ci
            ay = si
            apx = -si
            apy = ci
        elif kb == 1:
            ax = -si
            ay = ci
            apx = -ci
            apy = -si
        elif kb == 2:
            ax = cj
            ay = sj
            apx = -sj
            apy = cj
        else:
            ax = -sj
            ay = cj
            apx = -cj
            apy = -sj
        sg = 1.0 if pb >= 0 else -1.0
        gth = sg * (dx * apx + dy * apy)  # effect of turning the axis with its owner
        da, db, slack = _branches(cd, sd)
        for ver in range(2):
            on = reach and (ver == 0 or slack < KINK)
            if (not on) and (not fixed):
                continue
            drho = da if ver == 0 else db
            rstart[row] = k
            for which in range(2):
                sq = i if which == 0 else j
                B = body[sq]
                sgn = 1.0 if which == 0 else -1.0
                s0 = start[B]
                c0 = sgn * sg * ax if on else 0.0
                c1 = sgn * sg * ay if on else 0.0
                idx[k] = s0
                val[k] = c0
                k += 1
                idx[k] = s0 + nv
                val[k] = -c0
                k += 1
                idx[k] = s0 + 1
                val[k] = c1
                k += 1
                idx[k] = s0 + 1 + nv
                val[k] = -c1
                k += 1
                if kind[B] == 3:
                    rx = P[sq, 0] - ref[B, 0]
                    ry = P[sq, 1] - ref[B, 1]
                    v = sgn * (sg * ax * (-ry) + sg * ay * rx - drho)
                    if (which == 0 and kb < 2) or (which == 1 and kb >= 2):
                        v -= gth
                    if not on:
                        v = 0.0
                    idx[k] = s0 + 2
                    val[k] = v
                    k += 1
                    idx[k] = s0 + 2 + nv
                    val[k] = -v
                    k += 1
            rhs[row] = (best if ver == 0 else best + slack) if on else 1e30
            ri[row] = i
            rj[row] = j
            row += 1
    for w in range(nwall):
        i = walls[w, 0]
        a = walls[w, 1]
        A = body[i]
        t = P[i, 2]
        c = math.cos(t)
        s = math.sin(t)
        hw = 0.5 * (abs(c) + abs(s))
        da, db, slack = _branches(c, s)
        x = P[i, a]
        for far in range(2):
            reach = (x + hw >= S - 14.0 * r) if far == 1 else (x - hw <= cut)
            if (not reach) and (not fixed):
                continue
            for ver in range(2):
                on = reach and (ver == 0 or slack < KINK)
                if (not on) and (not fixed):
                    continue
                dw = da if ver == 0 else db
                rstart[row] = k
                s0 = start[A]
                sgn = 1.0 if far == 1 else -1.0
                c0 = (sgn if a == 0 else 0.0) if on else 0.0
                c1 = (sgn if a == 1 else 0.0) if on else 0.0
                idx[k] = s0
                val[k] = c0
                k += 1
                idx[k] = s0 + nv
                val[k] = -c0
                k += 1
                idx[k] = s0 + 1
                val[k] = c1
                k += 1
                idx[k] = s0 + 1 + nv
                val[k] = -c1
                k += 1
                if kind[A] == 3:
                    rx = P[i, 0] - ref[A, 0]
                    ry = P[i, 1] - ref[A, 1]
                    if far == 1:
                        lever = -ry if a == 0 else rx
                    else:
                        lever = ry if a == 0 else -rx
                    v = (lever + dw) if on else 0.0
                    idx[k] = s0 + 2
                    val[k] = v
                    k += 1
                    idx[k] = s0 + 2 + nv
                    val[k] = -v
                    k += 1
                if far == 1:
                    idx[k] = 2 * nv
                    val[k] = -1.0 if on else 0.0
                    k += 1
                    b = -(x + hw)
                else:
                    b = x - hw
                rhs[row] = (b if ver == 0 else b + slack) if on else 1e30
                ri[row] = i
                rj[row] = -1 - a - 2 * far
                row += 1
    rstart[row] = k
    return (
        rstart[: row + 1].copy(),
        idx[:k].copy(),
        val[:k].copy(),
        rhs[:row].copy(),
        ri[:row].copy(),
        rj[:row].copy(),
    )


def new_solver():
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("primal_feasibility_tolerance", 1e-10)
    h.setOptionValue("dual_feasibility_tolerance", 1e-10)
    h.setOptionValue("presolve", "off")
    h.setOptionValue("time_limit", 4.0)
    return h


class Compactor:
    MU = 1e-4  # price of movement
    REACH = 0.14  # candidate contacts are collected this far beyond touching

    def __init__(self, P, tol=1e-3, singles=None, r=0.02):
        self.tol = tol
        self.r = r
        self.free = new_solver()
        self.fix = new_solver()
        self.nsolve = 0
        self.build(P, singles)

    def build(self, P, singles=None):
        P = to_origin(P)
        self.P = P
        self.singles = singles
        self.body, self.kind, self.start, self.size = find_bodies(P, self.tol, singles)
        nb = len(self.kind)
        nv = int(self.kind.sum())
        self.nv = nv
        cand = (
            neighbour_pairs(P, 1.4143 + 2 * self.REACH + 0.05)
            if len(P) > 1
            else np.zeros((0, 2), np.int64)
        )
        self.near = cand
        self.pairs = np.ascontiguousarray(
            cand[self.body[cand[:, 0]] != self.body[cand[:, 1]]]
        )
        S = extent(P)[1]
        w = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
        m = self.REACH + 0.25
        near_wall = (P[:, :2] - w[:, None] < m) | (P[:, :2] + w[:, None] > S - m)
        self.walls = np.array(
            [(q, a) for a in (0, 1) for q in np.where(near_wall[:, a])[0]], np.int64
        ).reshape(-1, 2)
        ref = body_centres(P, self.body, self.size)
        self.rad = np.zeros(nb)
        np.maximum.at(
            self.rad,
            self.body,
            np.hypot(P[:, 0] - ref[self.body, 0], P[:, 1] - ref[self.body, 1]),
        )
        self.trust = np.empty(nv)
        self.trust_fixed = np.empty(nv)
        weight = np.ones(nv)
        for b in range(nb):
            s = self.start[b]
            self.trust[s] = self.trust[s + 1] = self.r
            self.trust_fixed[s] = self.trust_fixed[s + 1] = 2 * self.r
            if self.kind[b] == 3:
                self.trust[s + 2] = self.r / max(1.0, self.rad[b])
                self.trust_fixed[s + 2] = 0.0
                weight[s + 2] = max(1.0, self.rad[b])
        self.cost = np.concatenate([self.MU * weight, self.MU * weight, [1.0]])
        self.basis = {id(self.free): None, id(self.fix): None}
        self.anchor = P[:, :2].copy()

    def stale(self, P):
        return bool(np.abs(P[:, :2] - self.anchor).max() > 0.85 * self.REACH)

    def _solve(self, h, P, trust, S, width, duals=False):
        ref = body_centres(P, self.body, self.size)
        st, idx, val, rhs, ri, rj = lp_rows(
            P,
            self.body,
            ref,
            self.start,
            self.kind,
            self.pairs,
            self.walls,
            self.nv,
            self.r,
            S,
            True,
        )
        nv = self.nv
        m = highspy.HighsLp()
        m.num_col_ = 2 * nv + 1
        m.num_row_ = len(rhs)
        m.col_cost_ = self.cost
        m.col_lower_ = np.concatenate([np.zeros(2 * nv), [S - width]])
        m.col_upper_ = np.concatenate([trust, trust, [S + width]])
        m.row_lower_ = np.full(len(rhs), -INF)
        m.row_upper_ = np.minimum(rhs, INF)
        m.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        m.a_matrix_.start_ = st
        m.a_matrix_.index_ = idx
        m.a_matrix_.value_ = val
        self.nsolve += 1
        key = id(h)
        for attempt in range(2):
            h.passModel(m)
            warm = attempt == 0 and self.basis[key] is not None
            if warm:
                h.setBasis(self.basis[key])
            h.setOptionValue("simplex_iteration_limit", 3000 if warm else 2000000)
            h.setOptionValue("time_limit", 1.0 if warm else 60.0)
            h.run()
            if h.getModelStatus() == highspy.HighsModelStatus.kOptimal:
                break
            h.clearSolver()
            self.basis[key] = (
                None  # a stale basis can stall the simplex: one cold retry
            )
        else:
            return None
        self.basis[key] = h.getBasis()
        sol = h.getSolution()
        z = np.array(sol.col_value)
        return (
            z[:nv] - z[nv : 2 * nv],
            z[2 * nv],
            ref,
            (-np.array(sol.row_dual) if duals else None),
            ri,
            rj,
        )

    def direction(self, P, S, scale=1.0, duals=False):
        return self._solve(
            self.free, P, self.trust * scale, S, 8 * self.r * scale, duals
        )

    def repair(self, Q, scale=1.0):
        S = extent(Q)[1]
        out = self._solve(self.fix, Q, self.trust_fixed * scale, S, 16 * self.r * scale)
        if out is None:
            return None
        Q2 = move_bodies(Q, self.body, out[2], self.start, self.kind, out[0])
        lo, hi = extent(Q2)
        return Q2, hi, max(-min_gap_over(Q2, self.near), -lo)

    def improve(self, P, S, alpha, scale, tol=1e-11):
        d = self.direction(P, S, scale)
        if d is None:
            return None
        x, pred, ref = d[0], d[1], d[2]
        if S - pred < tol:
            return "stationary"
        while alpha > 2e-3:
            out = self.repair(
                move_bodies(P, self.body, ref, self.start, self.kind, alpha * x), scale
            )
            if out is not None and out[2] < 2e-9 and out[1] < S - tol:
                return out[0], out[1], alpha
            alpha *= 0.4
        return None

    def compact(self, P=None, max_iter=400, patience=6, trace=None):
        P = self.P if P is None else to_origin(P)
        if P.shape[0] != self.P.shape[0]:
            self.build(P)
        elif self.stale(P):
            self.build(P, self.singles)
        S = extent(P)[1]
        alpha = 1.0
        scale = 1.0
        fails = 0
        for it in range(max_iter):
            out = self.improve(P, S, alpha, scale)
            if isinstance(out, str):
                break
            if out is None:
                if alpha < 1.0:
                    alpha = 1.0
                else:
                    scale *= 0.2
                    fails += 1
                    if fails >= patience:
                        break
                continue
            P, S, a = out
            if trace is not None:
                trace.append(S)
            alpha = min(1.0, a * 2.5)
            scale = min(1.0, scale * 2.0)
            if self.stale(P):
                self.build(P, self.singles)
        self.P = P
        lo, hi = extent(P)
        return P, S, max(-min_gap_over(P, self.near), -lo)


@njit
def _energy(X, pairs, S, G, free):
    # sum of squared overlap depths over pairs with a free member, plus wall terms; gradient in G
    n = X.shape[0] // 3
    E = 0.0
    for i in range(3 * n):
        G[i] = 0.0
    for p in range(pairs.shape[0]):
        i = pairs[p, 0]
        j = pairs[p, 1]
        dx = X[3 * j] - X[3 * i]
        dy = X[3 * j + 1] - X[3 * i + 1]
        if dx * dx + dy * dy > 2.0:
            continue
        ti = X[3 * i + 2]
        tj = X[3 * j + 2]
        ci = math.cos(ti)
        si = math.sin(ti)
        cj = math.cos(tj)
        sj = math.sin(tj)
        d = tj - ti
        cd = math.cos(d)
        sd = math.sin(d)
        r = 0.5 * (abs(cd) + abs(sd))
        best = -1e9
        kb = 0
        pb = 0.0
        for k in range(4):
            if k == 0:
                ax = ci
                ay = si
            elif k == 1:
                ax = -si
                ay = ci
            elif k == 2:
                ax = cj
                ay = sj
            else:
                ax = -sj
                ay = cj
            pr = dx * ax + dy * ay
            g = abs(pr) - 0.5 - r
            if g > best:
                best = g
                kb = k
                pb = pr
        if best >= 0:
            continue
        v = -best
        E += v * v
        if kb == 0:
            ax = ci
            ay = si
            apx = -si
            apy = ci
        elif kb == 1:
            ax = -si
            ay = ci
            apx = -ci
            apy = -si
        elif kb == 2:
            ax = cj
            ay = sj
            apx = -sj
            apy = cj
        else:
            ax = -sj
            ay = cj
            apx = -cj
            apy = -sj
        sg = 1.0 if pb >= 0 else -1.0
        drd = 0.5 * (-sd * (1.0 if cd >= 0 else -1.0) + cd * (1.0 if sd >= 0 else -1.0))
        f = -2 * v
        G[3 * j] += f * sg * ax
        G[3 * j + 1] += f * sg * ay
        G[3 * i] -= f * sg * ax
        G[3 * i + 1] -= f * sg * ay
        gth = sg * (dx * apx + dy * apy)
        if kb < 2:
            G[3 * i + 2] += f * gth
        else:
            G[3 * j + 2] += f * gth
        G[3 * j + 2] -= f * drd
        G[3 * i + 2] += f * drd
    for i in range(n):
        if not free[i]:
            continue
        t = X[3 * i + 2]
        c = math.cos(t)
        s = math.sin(t)
        w = 0.5 * (abs(c) + abs(s))
        dw = 0.5 * (-s * (1.0 if c >= 0 else -1.0) + c * (1.0 if s >= 0 else -1.0))
        for a in range(2):
            x = X[3 * i + a]
            v = w - x
            if v > 0:
                E += v * v
                G[3 * i + a] -= 2 * v
                G[3 * i + 2] += 2 * v * dw
            v = x + w - S
            if v > 0:
                E += v * v
                G[3 * i + a] += 2 * v
                G[3 * i + 2] += 2 * v * dw
    for i in range(n):
        if not free[i]:
            G[3 * i] = 0.0
            G[3 * i + 1] = 0.0
            G[3 * i + 2] = 0.0
    return E


@njit
def _pairs_with_free(P, free, cut):
    n = P.shape[0]
    out = np.empty((n * 40 + 8, 2), np.int64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not (free[i] or free[j]):
                continue
            dx = P[j, 0] - P[i, 0]
            dy = P[j, 1] - P[i, 1]
            if dx * dx + dy * dy < cut * cut and k < out.shape[0]:
                out[k, 0] = i
                out[k, 1] = j
                k += 1
    return out[:k].copy()


@njit
def relax(P, free, S, maxiter=400, etol=1e-19, m=8):
    # L-BFGS with backtracking, compiled end to end.  Frozen squares never move, and pairs of
    # frozen squares are ignored, so a jammed surrounding cannot keep the energy from reaching zero.
    n = P.shape[0]
    pairs = _pairs_with_free(P, free, 2.3)
    N = 3 * n
    X = P.copy().reshape(-1)
    G = np.zeros(N)
    Gn = np.zeros(N)
    Xn = np.zeros(N)
    D = np.zeros(N)
    Sh = np.zeros((m, N))
    Yh = np.zeros((m, N))
    rho = np.zeros(m)
    al = np.zeros(m)
    E = _energy(X, pairs, S, G, free)
    hist = 0
    head = 0
    for it in range(maxiter):
        if E < etol:
            break
        for k in range(N):
            D[k] = -G[k]
        for q in range(hist):
            idx = (head - 1 - q) % m
            a = 0.0
            for k in range(N):
                a += Sh[idx, k] * D[k]
            a *= rho[idx]
            al[idx] = a
            for k in range(N):
                D[k] -= a * Yh[idx, k]
        if hist > 0:
            idx = (head - 1) % m
            yy = 0.0
            sy = 0.0
            for k in range(N):
                yy += Yh[idx, k] * Yh[idx, k]
                sy += Sh[idx, k] * Yh[idx, k]
            if yy > 0:
                for k in range(N):
                    D[k] *= sy / yy
        else:
            for k in range(N):
                D[k] *= 0.5
        for q in range(hist - 1, -1, -1):
            idx = (head - 1 - q) % m
            b = 0.0
            for k in range(N):
                b += Yh[idx, k] * D[k]
            b *= rho[idx]
            for k in range(N):
                D[k] += Sh[idx, k] * (al[idx] - b)
        gd = 0.0
        for k in range(N):
            gd += G[k] * D[k]
        if gd >= 0:
            for k in range(N):
                D[k] = -G[k]
            gd = 0.0
            for k in range(N):
                gd -= G[k] * G[k]
            hist = 0
            if gd == 0:
                break
        dm = 0.0
        for k in range(N):
            dm = max(dm, abs(D[k]))
        t = 1.0 if dm <= 0.2 else 0.2 / dm
        ok = False
        for ls in range(25):
            for k in range(N):
                Xn[k] = X[k] + t * D[k]
            En = _energy(Xn, pairs, S, Gn, free)
            if En <= E + 1e-4 * t * gd:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        sy = 0.0
        for k in range(N):
            Sh[head, k] = Xn[k] - X[k]
            Yh[head, k] = Gn[k] - G[k]
            sy += Sh[head, k] * Yh[head, k]
        if sy > 1e-30:
            rho[head] = 1.0 / sy
            head = (head + 1) % m
            hist = min(hist + 1, m)
        for k in range(N):
            X[k] = Xn[k]
            G[k] = Gn[k]
        E = En
    return X.reshape(n, 3).copy(), E


def from_scratch(n, starts, seed):
    rng = np.random.default_rng(seed)
    k = math.isqrt(n - 1) + 1
    S0 = k - 0.05
    best = (float(k), grid_packing(n))
    curve = []
    for _ in range(starts):
        P = np.column_stack(
            [
                rng.random(n) * (S0 - 1) + 0.5,
                rng.random(n) * (S0 - 1) + 0.5,
                rng.uniform(-0.8, 0.8, n),
            ]
        )
        Q, E = relax(P, np.ones(n, np.bool_), S0, 600, 1e-22)
        if E < 1e-18:
            Q, S, ov = Compactor(Q).compact()
            if ov < 2e-9 and S < best[0]:
                best = (S, Q)
        curve.append(best[0])
    return best, curve


def skeleton(P):
    # the min-side LP without the price on movement (see Section 5.3), so the dual is in exact balance
    C = Compactor(P)
    S = extent(C.P)[1]
    C.cost = np.zeros_like(C.cost)
    C.cost[-1] = 1.0
    x, pred, ref, lam, ri, rj = C.direction(C.P, S, duals=True)
    f = [(int(ri[q]), int(rj[q]), float(lam[q])) for q in np.where(lam > 1e-9)[0]]
    return C.P, f, S - pred


@njit
def _pairs_of(P, moved, is_moved, reach):
    n = P.shape[0]
    out = np.empty((moved.shape[0] * 48 + 8, 2), np.int64)
    k = 0
    for a in range(moved.shape[0]):
        i = moved[a]
        for j in range(n):
            if j == i or (is_moved[j] and j < i):
                continue
            dx = P[j, 0] - P[i, 0]
            dy = P[j, 1] - P[i, 1]
            if dx * dx + dy * dy < reach * reach and k < out.shape[0]:
                out[k, 0] = i
                out[k, 1] = j
                k += 1
    return out[:k].copy()


class LocalLP:
    # the min-side LP of one jammed packing, kept loaded; evaluate() swaps only the rows of moved squares
    def __init__(self, C):
        self.C = C
        P = C.P
        nv = C.nv
        self.P = P
        self.S = extent(P)[1]
        self.nv = nv
        n = P.shape[0]
        self.ref = body_centres(P, C.body, C.size)
        every = np.array([(q, a) for a in (0, 1) for q in range(n)], np.int64)
        st, idx, val, rhs, ri, rj = lp_rows(
            P, C.body, self.ref, C.start, C.kind, C.pairs, every, nv, C.r, self.S, False
        )
        self.nrow = len(rhs)
        self.rhs = rhs
        self.ri = ri
        self.rj = rj
        rows_of = [[] for _ in range(n)]
        for q in range(self.nrow):
            rows_of[ri[q]].append(q)
            if rj[q] >= 0:
                rows_of[rj[q]].append(q)
        self.rows_of = [np.array(v, np.int32) for v in rows_of]
        h = new_solver()
        m = highspy.HighsLp()
        m.num_col_ = 2 * nv + 1
        m.num_row_ = self.nrow
        m.col_cost_ = C.cost
        m.col_lower_ = np.concatenate([np.zeros(2 * nv), [self.S - 8 * C.r]])
        m.col_upper_ = np.concatenate([C.trust, C.trust, [self.S + 8 * C.r]])
        m.row_lower_ = np.full(self.nrow, -INF)
        m.row_upper_ = rhs
        m.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        m.a_matrix_.start_ = st
        m.a_matrix_.index_ = idx
        m.a_matrix_.value_ = val
        h.passModel(m)
        h.run()
        self.h = h
        self.ok = h.getModelStatus() == highspy.HighsModelStatus.kOptimal
        self.base = h.getBasis() if self.ok else None
        sol = h.getSolution()
        self.lam = -np.array(sol.row_dual) if self.ok else np.zeros(self.nrow)
        self.is_moved = np.zeros(n, np.bool_)

    def loads(self):
        load = np.zeros(self.P.shape[0])
        for q in np.where(self.lam > 1e-9)[0]:
            load[self.ri[q]] += self.lam[q]
            if self.rj[q] >= 0:
                load[self.rj[q]] += self.lam[q]
        return load

    def evaluate(self, Q, moved):
        C = self.C
        h = self.h
        nv = self.nv
        moved = np.asarray(moved, np.int64)
        old = np.unique(np.concatenate([self.rows_of[i] for i in moved])).astype(
            np.int32
        )
        self.is_moved[:] = False
        self.is_moved[moved] = True
        pairs = _pairs_of(Q, moved, self.is_moved, 1.4143 + 6 * C.r + 1e-3)
        pairs = pairs[C.body[pairs[:, 0]] != C.body[pairs[:, 1]]]
        walls = np.array([(q, a) for a in (0, 1) for q in moved], np.int64)
        st, idx, val, rhs, _, _ = lp_rows(
            Q, C.body, self.ref, C.start, C.kind, pairs, walls, nv, C.r, self.S, False
        )
        k = len(rhs)
        if len(old):
            h.changeRowsBounds(
                len(old), old, np.full(len(old), -INF), np.full(len(old), INF)
            )
        if k:
            h.addRows(k, np.full(k, -INF), rhs, len(idx), st[:-1], idx, val)
        h.setOptionValue("simplex_iteration_limit", 4000)
        h.run()
        sig = (
            h.getSolution().col_value[2 * nv]
            if h.getModelStatus() == highspy.HighsModelStatus.kOptimal
            else None
        )
        if k:
            h.deleteRows(k, np.arange(self.nrow, self.nrow + k, dtype=np.int32))
        if len(old):
            h.changeRowsBounds(len(old), old, np.full(len(old), -INF), self.rhs[old])
        h.setBasis(self.base)
        return sig


class Hopper:
    def __init__(self, P, seed=0, tol=1e-3, wide=False):
        self.rng = np.random.default_rng(seed)
        self.tol = tol
        self.wide = wide
        self.stats = dict(trials=0, fit=0, promising=0, wins=0)
        self.clock = dict(relax=0.0, local_lp=0.0, compact=0.0, reset=0.0)
        self.history = []
        self.t0 = time.time()
        self.reset(to_origin(P))

    def reset(self, P):
        t = time.perf_counter()
        body, kind, start, size = find_bodies(P, self.tol)
        # loose: squares not welded into a bar or the grid, plus their neighbours in bars of up to 12 squares,
        # so that the last squares of a bar at a band end can be re-packed too
        core = size[body] <= 2
        free_ = core.copy()
        for i, j in neighbour_pairs(np.ascontiguousarray(P), 1.45):
            if core[i] and size[body[j]] <= 12:
                free_[j] = True
            if core[j] and size[body[i]] <= 12:
                free_[i] = True
        loose = np.where(free_)[0]
        self.C = Compactor(P, self.tol, singles=loose)
        self.P = self.C.P
        self.S = extent(self.P)[1]
        self.loose = np.zeros(len(P), bool)
        self.loose[loose] = True
        self.L = LocalLP(self.C)
        self.w = (
            np.maximum(
                np.nan_to_num(self.L.loads(), nan=0.0, posinf=0.0, neginf=0.0), 0.0
            )
            * self.loose
        )
        tilts = fold(self.P[:, 2])
        tilts = tilts[np.abs(tilts) > 0.02]
        self.palette = np.concatenate([[0.0], np.unique(np.round(tilts, 3))[:8]])
        self.cen = body_centres(self.P, self.C.body, self.C.size)
        self.C.direction(self.P, self.S)
        self.C.repair(self.P)  # bases of the jammed state
        self.warm = dict(self.C.basis)
        self.history.append((time.time() - self.t0, self.S))
        self.clock["reset"] += time.perf_counter() - t

    def applicable(self):
        return bool(self.L.ok and np.isfinite(self.w).all() and self.w.sum() > 0)

    def trial(self):
        rng = self.rng
        P = self.P
        S = self.S
        C = self.C
        self.stats["trials"] += 1
        i0 = rng.choice(len(P), p=self.w / self.w.sum())
        c0 = P[i0, :2] + rng.normal(0, 0.4, 2)
        is_wide = self.wide and (rng.random() < 0.5)
        rho = rng.uniform(1.8, 3.6) if is_wide else rng.uniform(1.0, 2.3)
        max_cl = 18 if is_wide else 9
        d = np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1])
        cluster = np.where((d < rho) & self.loose)[0]
        if len(cluster) == 0 or len(cluster) > max_cl:
            return False
        near = np.where(d < rho + 1.5)[0]
        ring = np.where((d >= rho + 1.5) & (d < rho + 3.9))[0]
        sub = np.concatenate([near, ring])
        free = np.zeros(len(sub), np.bool_)
        free[: len(near)] = self.loose[near] | (d[near] < rho)
        Q = P.copy()
        room = rng.random() < 0.6
        if room:
            eta = rng.uniform(0.03, 0.12)
            axis = int(rng.integers(2))
            up = self.cen[C.body, axis] > c0[axis] + rng.uniform(-1.0, 1.0)
            Q[up, axis] += eta
            if min_gap_over(Q, C.near) < -1e-9:
                return False
        for i in cluster:
            Q[i, :2] += rng.normal(0, 0.12 if rng.random() < 0.5 else 0.3, 2)
            u = rng.random()
            if u < 0.35:
                pass
            elif u < 0.6:
                Q[i, 2] = rng.choice(self.palette)
            elif u < 0.8:
                Q[i, 2] += rng.normal(0, 0.05)
            else:
                Q[i, 2] = rng.uniform(-0.78, 0.78)
        t = time.perf_counter()
        rel_it = 360 if is_wide else 250
        R, E = relax(Q[sub], free, S + (eta if room else 0.0) + 1e-9, rel_it, 1e-16)
        if E <= 1e-14 and room:
            Q[near] = R[: len(near)]
            Q[up, axis] -= eta
            R, E = relax(Q[sub], free, S + 1e-9, rel_it, 1e-16)
        self.clock["relax"] += time.perf_counter() - t
        if E > 1e-14:
            return False
        Q[near] = R[: len(near)]
        shift = np.abs(Q[near] - P[near]).max(axis=1)
        if shift.max() < 1e-3:
            return False
        Q[near[shift <= 1e-9]] = P[near[shift <= 1e-9]]
        self.stats["fit"] += 1
        t = time.perf_counter()
        pred = self.L.evaluate(Q, near[shift > 1e-9])
        self.clock["local_lp"] += time.perf_counter() - t
        if pred is None or pred > S - 2e-8:
            return False
        self.stats["promising"] += 1
        t = time.perf_counter()
        C.basis = dict(self.warm)
        Q2, S2, ov = C.compact(Q, max_iter=60, patience=1)
        self.clock["compact"] += time.perf_counter() - t
        if ov > 2e-9 or S2 > S - 1e-8:
            if not np.array_equal(C.anchor, self.P[:, :2]):
                C.build(self.P, np.where(self.loose)[0])
            C.P = self.P
            return False
        self.stats["wins"] += 1
        self.reset(Q2)
        return True

    def run(self, seconds):
        # applicable() is re-checked after every trial: a win re-computes the force weights, and a packing
        # can come out of it with no loose square left carrying force
        t = time.time()
        while time.time() - t < seconds and self.applicable():
            self.trial()
        return self.P, self.S


def time_scipy_lp(P):
    # the plain formulation: every square its own body, one cold solve through scipy
    P = to_origin(P)
    n = len(P)
    C = Compactor(P, singles=np.arange(n))
    S = extent(P)[1]
    ref = body_centres(P, C.body, C.size)
    st, idx, val, rhs, _, _ = lp_rows(
        P, C.body, ref, C.start, C.kind, C.pairs, C.walls, C.nv, C.r, S, False
    )
    A = csr_matrix((val, idx, st), shape=(len(rhs), 2 * C.nv + 1))
    cost = np.zeros(2 * C.nv + 1)
    cost[-1] = 1.0
    bounds = [(0, t) for t in C.trust] * 2 + [(S - 8 * C.r, S + 8 * C.r)]
    t = time.perf_counter()
    linprog(cost, A_ub=A, b_ub=rhs, bounds=bounds, method="highs")
    return time.perf_counter() - t


def stake(n):
    # score separating our packing from the best side known for this puzzle (0 without the table)
    return (current.side[n] - lb[n]) / math.sqrt(n) if n in lb else 0.0


def live_job(args):
    n, P, seconds, seed = args
    seconds = min(seconds, deadline - time.time() - 2)
    if seconds < 3:
        return n, None, None, None
    H = Hopper(P, seed=seed)
    Q, S = H.run(seconds)
    return n, Q, S, H.stats


def band_summary(P, tol=1e-3):
    body, kind, start, size = find_bodies(P, tol)
    tilt = np.zeros(len(size))
    tilt[body] = fold(P[:, 2])
    return body, size, body_centres(P, body, size), tilt


def find_cut(P, axis, tol=1e-3):
    # a level, on a line of the upright grid, where a cut across the band stacked along `axis`
    # meets only equal bars of one band and upright squares; prefers levels far from loose squares
    P = to_origin(P)
    S = side(P)
    body, size, cen, tilt = band_summary(P, tol)
    other = 1 - axis
    upright = np.abs(fold(P[:, 2])) < 0.02
    grid_body = np.argmax(np.where(np.abs(tilt) < 1e-9, size, 0))
    is_bar = (np.abs(tilt) > 0.02) & (size >= 3)
    loose = (size[body] == 1) & ~upright
    best = None
    for y in np.unique(np.round(P[body == grid_body, axis] + 0.5, 6)):
        if y < 3 or y > S - 3:
            continue
        strip = np.where(np.abs(P[:, axis] - y) < 1.75)[0]
        tl = strip[~upright[strip]]
        if len(tl) == 0 or not np.all(is_bar[body[tl]]):
            continue
        bars = np.unique(body[tl])
        if len(bars) < 2 or np.ptp(tilt[bars]) > 2e-3 or np.ptp(size[bars]) != 0:
            continue
        one = np.where(body == bars[0])[0]
        if np.ptp(P[one, other]) < np.ptp(P[one, axis]):
            continue
        above = bars[cen[bars, axis] > y]
        if len(above) == 0:
            continue
        dist = np.min(np.abs(P[loose, axis] - y)) if loose.any() else 9.0
        if best is None or dist > best[0]:
            th = float(np.mean(tilt[bars]))
            n_bars = int(
                np.sum(is_bar & (np.abs(tilt - th) < 2e-3) & (size == size[bars[0]]))
            )
            best = (
                dist,
                float(y),
                th,
                np.where(body == above[np.argmin(cen[above, axis])])[0],
                n_bars,
            )
    return best


def change_length(P, axis, add, gap=2e-4):
    P = to_origin(P)
    cut = find_cut(P, axis)
    if cut is None:
        return None
    _, y, th, bar, n_bars = cut
    body, size, cen, tilt = band_summary(P)
    in_band = ((np.abs(tilt) > 0.02) & (size >= 3) & (np.abs(tilt - th) < 2e-3))[body]
    above = np.where(in_band, cen[body, axis] > y, P[:, axis] > y)
    row = np.where((~in_band) & (P[:, axis] > y - 1) & (P[:, axis] <= y))[0]
    Q = P.copy()
    info = dict(b=len(bar), tilt=th, bars=n_bars)
    if add:
        Q[above, axis] += 1.0 / math.cos(th) + 2 * gap
        new_bar = P[bar].copy()
        new_bar[:, axis] += gap
        new_row = P[row].copy()
        new_row[:, axis] += 1.0 + gap
        return np.vstack([Q, new_bar, new_row]), info
    Q[above, axis] -= 1.0
    keep = np.ones(len(P), bool)
    keep[bar] = False
    keep[row] = False
    return Q[keep], info


def transplant(P, up):
    a = change_length(P, 1, up)
    if a is None:
        return None
    b = change_length(a[0], 0, up)
    if b is None or min_gap(b[0]) < -1e-9:
        return None
    Q, S, ov = Compactor(b[0]).compact()
    if ov > 2e-9:
        return None
    return Q, S, a[1], b[1]


def retilt_cost(info):
    th = abs(info["tilt"])
    kappa = info["b"] * math.sin(th) - math.cos(th)
    return (
        (1 / math.cos(th) - 1)
        * kappa
        / (info["bars"] / math.cos(th) * math.tan(th) + kappa)
    )


def transplant_job(args):
    n, up = args
    try:
        out = transplant(current.P[n], up)
    except Exception:
        out = None
    return n, up, out


def l_border(P):
    P = to_origin(P)
    S = side(P)
    k = int(math.floor(S + 1e-9))
    add = (
        [(S + 0.5, j + 0.5, 0.0) for j in range(k)]
        + [(i + 0.5, S + 0.5, 0.0) for i in range(k)]
        + [(S + 0.5, S + 0.5, 0.0)]
    )
    return np.vstack([P, np.array(add)])


def close(pf):
    count = 0
    changed = True
    while changed:
        changed = False
        for n in range(N_MAX, 1, -1):
            if pf.side[n] < pf.side[n - 1] - 1e-12 and pf.offer(
                pf.P[n][:-1], pf.src[n] + " (one square deleted)"
            ):
                changed = True
                count += 1
        for n in range(1, N_MAX + 1):
            if pf.src[n] == "grid":
                continue
            Q = l_border(pf.P[n])
            if len(Q) <= N_MAX and pf.offer(Q, pf.src[n] + " (L-border)"):
                changed = True
                count += 1
    return count


@njit
def min_gap_strict(P):
    # smallest gap, not counting exact contacts of upright squares on half-integer coordinates
    n = P.shape[0]
    m = 1e9
    for i in range(n):
        hi = (P[i, 0] * 2.0 == math.floor(P[i, 0] * 2.0)) and (
            P[i, 1] * 2.0 == math.floor(P[i, 1] * 2.0)
        )
        for j in range(i + 1, n):
            dx = P[j, 0] - P[i, 0]
            dy = P[j, 1] - P[i, 1]
            if dx * dx + dy * dy < 2.0000001:
                hj = (P[j, 0] * 2.0 == math.floor(P[j, 0] * 2.0)) and (
                    P[j, 1] * 2.0 == math.floor(P[j, 1] * 2.0)
                )
                if P[i, 2] == 0.0 and P[j, 2] == 0.0 and hi and hj:
                    g = max(abs(dx), abs(dy)) - 1.0
                    if g >= 0:
                        continue
                else:
                    g = pair_gap(P[i, 0], P[i, 1], P[i, 2], P[j, 0], P[j, 1], P[j, 2])
                m = min(m, g)
    return m


def open_contacts(P, target=3e-12):
    P = np.array(P, dtype=np.float64)
    a = fold(P[:, 2])
    a[np.abs(a) < 1e-15] = 0.0
    P[:, 2] = a
    C = corners(P)
    P[:, 0] -= C[:, :, 0].min()
    P[:, 1] -= C[:, :, 1].min()
    for _ in range(60):
        g = min_gap_strict(P)
        if g >= target:
            return P
        lam = (target - g) * 1.5 + 1e-13
        P[:, 0] *= 1 + lam
        P[:, 1] *= 1 + lam
    raise RuntimeError("could not open contacts")


def check_file(path, **kw):
    df = pd.read_csv(path, **kw)
    total = 0.0
    bad = []
    gmin = 1e9
    for n, g in df.groupby("n", sort=True):
        n = int(n)
        P = np.ascontiguousarray(g[["x", "y", "angle"]].to_numpy(np.float64))
        k = math.isqrt(n - 1) + 1
        s = side(P)
        gp = min_gap_strict(P)
        if len(P) != n or gp < 1e-12 or s > k + 1e-12:
            bad.append(n)
            total += k / math.sqrt(n) - 1
        else:
            total += s / math.sqrt(n) - 1
            if gp < 1e8:
                gmin = min(gmin, gp)
    return total, bad, gmin


def gap_50(a, b):
    xi, yi, ti = (mpmath.mpf(v) for v in a)
    xj, yj, tj = (mpmath.mpf(v) for v in b)
    d = tj - ti
    r = (abs(mpmath.cos(d)) + abs(mpmath.sin(d))) / 2
    return (
        max(
            abs((xj - xi) * mpmath.cos(t) + (yj - yi) * mpmath.sin(t))
            for t in (ti, ti + mpmath.pi / 2, tj, tj + mpmath.pi / 2)
        )
        - mpmath.mpf(1) / 2
        - r
    )
