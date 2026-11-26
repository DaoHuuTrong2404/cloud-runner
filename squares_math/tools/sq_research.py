"""Research tools for the notebook "Growing packings": inflation of a packing by k rows and columns
import highspy
(grid cut and band-aware diagonal cut), and the multi-lattice re-fill of a window.  Uses the engine module
from the first notebook (sq_engine)."""
import sys, math
import numpy as np
from numba import njit
from scipy.spatial import cKDTree
import sq_engine as E
import highspy


# ---- from inflate2.py ----

@njit(cache=True)
def clash_pairs(P, pairs, src, tol):
    # pairs that overlap and involve at least one copy (src >= 0); copies of the same source never clash
    out = np.empty((pairs.shape[0], 2), np.int64); k = 0
    for p in range(pairs.shape[0]):
        i = pairs[p, 0]; j = pairs[p, 1]
        if src[i] < 0 and src[j] < 0: continue
        if src[i] == src[j]: continue
        dx = P[j, 0] - P[i, 0]; dy = P[j, 1] - P[i, 1]
        if dx * dx + dy * dy >= 2.0000001: continue
        if E.pair_gap(P[i, 0], P[i, 1], P[i, 2], P[j, 0], P[j, 1], P[j, 2]) < tol:
            out[k, 0] = i; out[k, 1] = j; k += 1
    return out[:k]

def resolve(R, src, tol=0.0):
    pairs = E.neighbour_pairs(np.ascontiguousarray(R), 1.4143)
    cp = clash_pairs(np.ascontiguousarray(R), pairs, src, tol)
    alive = np.ones(len(R), bool)
    if len(cp) == 0: return alive
    # a copy that clashes with an original must go
    for i, j in cp:
        if src[i] < 0: alive[j] = False
        elif src[j] < 0: alive[i] = False
    # copy-copy clashes: greedy by degree
    cc = [(i, j) for i, j in cp if src[i] >= 0 and src[j] >= 0 and alive[i] and alive[j]]
    while cc:
        deg = {}
        for i, j in cc:
            deg[i] = deg.get(i, 0) + 1; deg[j] = deg.get(j, 0) + 1
        v = max(deg, key=deg.get); alive[v] = False
        cc = [(i, j) for i, j in cc if alive[i] and alive[j]]
    return alive

def inflate_at(P, k, x0, y0, tol=-1e-12):
    x = P[:, 0]; y = P[:, 1]
    right = x > x0; up = y > y0
    Q = P.copy(); Q[right, 0] += k; Q[up, 1] += k
    blocks = []
    def add(mask, sx, sy):
        if mask.any():
            C = P[mask].copy(); C[:, 0] += sx; C[:, 1] += sy; blocks.append(C)
    rowL = (y > y0 - k) & ~up                 # the layer just below the horizontal cut
    colL = (x > x0 - k) & ~right              # the layer just left of the vertical cut
    add(rowL & ~right, 0, k)                  # (1) horizontal strip, left part
    add(rowL & right, k, k)                   # (2) horizontal strip, right part
    add(colL & ~up, k, 0)                     # (3) vertical strip, lower part
    add(colL & up, k, k)                      # (4) vertical strip, upper part
    add(rowL & colL, k, k)                    # (5) crossing block from the lower-left
    rowU = (y <= y0 + k) & up; colR = (x <= x0 + k) & right
    add(rowL & colR, 0, k)                    # (6) crossing block from the lower-right (diagonal band, falling)
    add(rowU & colL, k, 0)                    # (7) crossing block from the upper-left  (diagonal band, falling)
    add(rowU & colR, 0, 0)                    # (8) crossing block from the upper-right
    R = np.vstack([Q] + blocks)
    src = np.full(len(R), -1, np.int64); o = len(Q)
    for b, C in enumerate(blocks):
        src[o:o + len(C)] = b; o += len(C)
    alive = resolve(R, src, tol)
    return R[alive]

def levels(P, axis, S):
    up = np.abs(E.fold(P[:, 2])) < 1e-9
    lv = np.concatenate([P[up, axis] + 0.5, np.arange(0.75, S - 0.5, 0.25)])
    lv = np.unique(np.round(lv, 6))
    return lv[(lv > 0.6) & (lv < S - 0.6)]

def best_inflation(P, k, tol=-1e-12, max_levels=24, rng=None):
    P = E.to_origin(P); S = E.side(P)
    lx = levels(P, 0, S); ly = levels(P, 1, S)
    if rng is not None:
        if len(lx) > max_levels: lx = np.sort(rng.choice(lx, max_levels, replace=False))
        if len(ly) > max_levels: ly = np.sort(rng.choice(ly, max_levels, replace=False))
    best = (-1, None, None)
    for x0 in lx:
        for y0 in ly:
            R = inflate_at(P, k, x0, y0, tol)
            if len(R) > best[0]: best = (len(R), (x0, y0), R)
    return best[2], S + k, best[1]

def band_points(P, S, step=0.25, offsets=(-1.0, -0.5, 0.0, 0.5, 1.0)):
    """Cross points on the fitted centre line of the tilted squares (for diagonal bands)."""
    t = np.abs(E.fold(P[:, 2])); tl = t > np.radians(5)
    if tl.sum() < 8: return []
    X = P[tl, 0]; Y = P[tl, 1]
    out = []
    # fit y on x and x on y; keep the better-conditioned one
    if np.ptp(X) >= np.ptp(Y):
        b, a = np.polyfit(X, Y, 1)
        for x0 in np.arange(max(0.6, X.min()), min(S - 0.6, X.max()), step):
            for o in offsets:
                y0 = a + b * x0 + o
                if 0.6 < y0 < S - 0.6: out.append((x0, y0))
    else:
        b, a = np.polyfit(Y, X, 1)
        for y0 in np.arange(max(0.6, Y.min()), min(S - 0.6, Y.max()), step):
            for o in offsets:
                x0 = a + b * y0 + o
                if 0.6 < x0 < S - 0.6: out.append((x0, y0))
    return out

def best_inflation2(P, k, tol=-1e-12, rng=None, n_random=150):
    P = E.to_origin(P); S = E.side(P)
    pts = band_points(P, S)
    lx = levels(P, 0, S); ly = levels(P, 1, S)
    rng = rng or np.random.default_rng(0)
    pts += [(rng.choice(lx), rng.choice(ly)) for _ in range(n_random)]
    best = (-1, None, None)
    for x0, y0 in pts:
        R = inflate_at(P, k, x0, y0, tol)
        if len(R) > best[0]: best = (len(R), (x0, y0), R)
    return best[2], S + k, best[1]


# ---- from diag.py ----

def _clash_free_add(R, src, cand, tol):
    """Add candidate squares one batch at a time, keeping those that clash with nothing kept so far."""
    keep = []
    cur = R.copy()
    order = np.argsort(cand[:, 3]) if cand.shape[1] > 3 else np.arange(len(cand))
    C = cand[order, :3]
    # greedy in batches using a cell list
    from scipy.spatial import cKDTree
    T = cKDTree(cur[:, :2])
    added = []
    for c in C:
        idx = T.query_ball_point(c[:2], 1.4143)
        ok = True
        for i in idx:
            if E.pair_gap(cur[i, 0], cur[i, 1], cur[i, 2], c[0], c[1], c[2]) < tol: ok = False; break
        if ok:
            for a in added:
                if (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2 < 2.0000001 and E.pair_gap(a[0], a[1], a[2], c[0], c[1], c[2]) < tol:
                    ok = False; break
        if ok: added.append(c)
    return np.array(added) if added else np.zeros((0, 3))

def inflate_diag(P, k, s0, tol=-1e-12, mirror=False):
    P = E.to_origin(P); S = E.side(P)
    if mirror:
        P = P.copy(); P[:, 0] = S - P[:, 0]; P[:, 2] = -P[:, 2]
    t = np.abs(E.fold(P[:, 2])); band = t > np.radians(3)
    up = ~band
    c = P[band, :2].mean(axis=0)
    d = np.array([1.0, -1.0]) / math.sqrt(2); nrm = np.array([1.0, 1.0]) / math.sqrt(2)
    s = (P[:, :2] - c) @ d; h = (P[:, :2] - c) @ nrm
    # signed distance of upright squares from the band: use nearest band square in the normal direction
    # local side of the band: compare with band squares at similar position along the band
    sb = s[band]; hb = h[band]
    A = np.zeros(len(P), bool)
    for i in np.where(up)[0]:
        near = np.abs(sb - s[i]) < 1.5
        if not near.any(): near = np.argsort(np.abs(sb - s[i]))[:3]
        A[i] = h[i] > np.mean(hb[near])
    B = up & ~A
    UL = band & (s < s0); LR = band & (s >= s0)
    Q = P.copy()
    Q[A, 0] += k; Q[A, 1] += k
    Q[UL, 1] += k
    Q[LR, 0] += k
    # band segment: the part of LR within k*sqrt(2) of the split, shifted by (0, k)
    seg = LR & (s < s0 + k * math.sqrt(2))
    Cb = P[seg].copy(); Cb[:, 1] += k
    R = np.vstack([Q, Cb])
    src = np.full(len(R), -1, np.int64); src[len(Q):] = 0
    pairs = E.neighbour_pairs(np.ascontiguousarray(R), 1.4143)
    cp = clash_pairs(np.ascontiguousarray(R), pairs, src, tol)
    alive = np.ones(len(R), bool)
    for i, j in cp:
        if src[i] >= 0: alive[i] = False
        if src[j] >= 0: alive[j] = False
    R = R[alive]
    # upright fill on the grids of A (shifted) and B, each only on its own side of the (inflated) band
    S2 = S + k
    bandR = np.abs(E.fold(R[:, 2])) > np.radians(3)
    cR = R[bandR, :2].mean(axis=0)
    sR = (R[bandR, :2] - cR) @ d; hR = (R[bandR, :2] - cR) @ nrm
    order = np.argsort(sR); sR = sR[order]; hR = hR[order]
    def side_of(XY):
        sc = (XY - cR) @ d; hc = (XY - cR) @ nrm
        hb = np.interp(sc, sR, hR)
        return hc - hb, sc
    adds = []
    for mask, shift, sign in ((A, k, 1.0), (B, 0, -1.0)):
        if not mask.any(): continue
        ox = np.median((P[mask, 0] - 0.5) % 1.0); oy = np.median((P[mask, 1] - 0.5) % 1.0)
        xs = np.arange(0.5 + (ox % 1.0), S2 - 0.5 + 1e-9, 1.0); ys = np.arange(0.5 + (oy % 1.0), S2 - 0.5 + 1e-9, 1.0)
        X, Y = np.meshgrid(xs, ys); XY = np.stack([X.ravel(), Y.ravel()], axis=1)
        dh, sc = side_of(XY)
        ok = (sign * dh > 0.4) & (np.abs(dh) < k + 3)
        cand = np.concatenate([XY[ok], np.zeros((ok.sum(), 1))], axis=1)
        # nearest to the band first
        cand = cand[np.argsort(np.abs(dh[ok]))]
        add = _clash_free_add(np.vstack([R] + adds) if adds else R, None, cand, tol)
        if len(add): adds.append(add)
    if adds: R = np.vstack([R] + adds)
    if mirror:
        R = R.copy(); R[:, 0] = S2 - R[:, 0]; R[:, 2] = -R[:, 2]
    return R, S2

def band_dir(P):
    t = np.abs(E.fold(P[:, 2])); band = t > np.radians(3)
    if band.sum() < 6: return None
    X = P[band, :2] - P[band, :2].mean(axis=0)
    w, V = np.linalg.eigh(X.T @ X)
    v = V[:, 1]
    return 'fall' if v[0] * v[1] < 0 else 'rise'

def best_diag(P, k, tol=-1e-12, steps=40):
    P = E.to_origin(P); S = E.side(P)
    dr = band_dir(P)
    if dr is None: return None, S + k, None
    mirror = dr == 'rise'
    Pm = P.copy()
    if mirror: Pm[:, 0] = S - Pm[:, 0]; Pm[:, 2] = -Pm[:, 2]
    t = np.abs(E.fold(Pm[:, 2])); band = t > np.radians(3)
    c = Pm[band, :2].mean(axis=0); d = np.array([1.0, -1.0]) / math.sqrt(2)
    sb = (Pm[band, :2] - c) @ d
    best = (-1, None, None)
    for s0 in np.linspace(sb.min() + 0.5, sb.max() - 0.5, steps):
        R, S2 = inflate_diag(P, k, s0, tol, mirror)
        if len(R) <= best[0]: continue
        if E.min_gap(R) < -1e-9 or E.side(R) > S2 + 1e-9: continue
        best = (len(R), (mirror, float(s0)), R)
    return best[2], S + k, best[1]


# ---- from refill.py ----

@njit(cache=True)
def gaps_to(C, F, pairs):
    out = np.empty(pairs.shape[0])
    for p in range(pairs.shape[0]):
        i = pairs[p, 0]; j = pairs[p, 1]
        out[p] = E.pair_gap(C[i, 0], C[i, 1], C[i, 2], F[j, 0], F[j, 1], F[j, 2])
    return out

def wall_ok(C, S, tol=1e-12):
    w = 0.5 * (np.abs(np.cos(C[:, 2])) + np.abs(np.sin(C[:, 2])))
    return (C[:, 0] - w >= -tol) & (C[:, 1] - w >= -tol) & (C[:, 0] + w <= S + tol) & (C[:, 1] + w <= S + tol)

def lattice_vectors(P, idx):
    """Empirical neighbour displacements between squares of (nearly) the same tilt."""
    T = cKDTree(P[idx, :2]); vec = []
    for a, b in T.query_pairs(1.25):
        i, j = idx[a], idx[b]
        if abs(E.fold(np.array([P[i, 2] - P[j, 2]]))[0]) > 2e-3: continue
        d = P[j, :2] - P[i, :2]; vec.append(d); vec.append(-d)
    if not vec: return np.zeros((0, 2))
    V = np.round(np.array(vec) / 0.02) * 0.02
    V, cnt = np.unique(V, axis=0, return_counts=True)
    order = np.argsort(-cnt)[:8]
    # use the mean of the raw vectors in each cluster for accuracy
    raw = np.array(vec); out = []
    for k in order:
        m = np.all(np.abs(np.round(raw / 0.02) * 0.02 - V[k]) < 1e-9, axis=1)
        out.append(raw[m].mean(axis=0))
    return np.array(out)

def candidates(P, S, x0, y0, w, h, gens=5, margin=3.0):
    cx = P[:, 0]; cy = P[:, 1]
    inwin = (cx > x0) & (cx < x0 + w) & (cy > y0) & (cy < y0 + h)
    near = (cx > x0 - margin) & (cx < x0 + w + margin) & (cy > y0 - margin) & (cy < y0 + h + margin)
    t = E.fold(P[:, 2])
    cands = [P[inwin]]
    # group nearby squares by tilt (rounded), grow each group's lattice into the window
    groups = {}
    for i in np.where(near)[0]:
        groups.setdefault(round(float(t[i]), 3), []).append(i)
    for key, idx in groups.items():
        idx = np.array(idx)
        if abs(key) < 1e-6: V = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float)
        else:
            V = lattice_vectors(P, np.where(np.abs(t - key) < 2e-3)[0])
            if len(V) == 0: continue
        front = P[idx].copy(); seen = set()
        for g in range(gens):
            new = (front[:, None, :2] + V[None, :, :]).reshape(-1, 2)
            new = np.concatenate([new, np.full((len(new), 1), P[idx[0], 2])], axis=1)
            keep = (new[:, 0] > x0 - 0.6) & (new[:, 0] < x0 + w + 0.6) & (new[:, 1] > y0 - 0.6) & (new[:, 1] < y0 + h + 0.6)
            new = new[keep]
            if len(new) == 0: break
            keys = [tuple(np.round(r / np.array([0.01, 0.01, 1e-3]))) for r in new]
            sel = [i for i, k in enumerate(keys) if k not in seen]
            for i in sel: seen.add(keys[i])
            front = new[sel]
            if len(front) == 0: break
            cands.append(front)
    C = np.vstack(cands)
    # merge near-duplicates (same tilt, centres within 0.01), keeping originals first
    key = np.round(C / np.array([0.01, 0.01, 1e-3]))
    C = C[np.sort(np.unique(key, axis=0, return_index=True)[1])]
    C = C[wall_ok(C, S)]
    fixed = P[~inwin]
    # drop candidates clashing with fixed squares
    T = cKDTree(fixed[:, :2]); pr = []
    for i, js in enumerate(T.query_ball_point(C[:, :2], 1.4143)):
        for j in js: pr.append((i, j))
    bad = np.zeros(len(C), bool)
    if pr:
        pr = np.array(pr); g = gaps_to(np.ascontiguousarray(C), np.ascontiguousarray(fixed), pr)
        bad[pr[g < -1e-12, 0]] = True
    return C[~bad], inwin

def max_set(C, time_limit=20.0, start=None):
    """Largest clash-free subset of the candidates C, as a 0-1 program solved by HiGHS.
    `start` (boolean mask, e.g. the squares already in the window) is passed as a feasible incumbent."""
    n = len(C)
    T = cKDTree(C[:, :2]); pr = np.array(sorted(T.query_pairs(1.4143)), dtype=np.int64).reshape(-1, 2)
    if len(pr):
        g = gaps_to(np.ascontiguousarray(C), np.ascontiguousarray(C), pr)
        pr = pr[g < -1e-12]
    h = highspy.Highs(); h.setOptionValue('output_flag', False); h.setOptionValue('time_limit', time_limit)
    h.setOptionValue('mip_rel_gap', 0.0)
    h.addCols(n, -np.ones(n), np.zeros(n), np.ones(n), 0, np.array([], np.int32), np.array([], np.int32), np.array([]))
    h.changeColsIntegrality(n, np.arange(n, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * n))
    if len(pr):
        m = len(pr)
        starts = np.arange(0, 2 * m, 2, dtype=np.int32); index = pr.reshape(-1).astype(np.int32)
        h.addRows(m, -np.inf * np.ones(m), np.ones(m), 2 * m, starts, index, np.ones(2 * m))
    if start is not None:
        sol = highspy.HighsSolution(); sol.col_value = [1.0 if v else 0.0 for v in start]; sol.value_valid = True
        h.setSolution(sol)
    h.run()
    x = np.array(h.getSolution().col_value)
    if len(x) != n or not np.all(np.isfinite(x)):
        return (np.asarray(start, bool) if start is not None else np.zeros(n, bool)), -1, h.getModelStatus()
    sel = x > 0.5
    if start is not None and sel.sum() < np.sum(start): sel = np.asarray(start, bool)
    return sel, int(sel.sum()), h.getModelStatus()

def refill(P, S, x0, y0, w, h):
    C, inwin = candidates(P, S, x0, y0, w, h)
    sel, val, st = max_set(C)
    Q = np.vstack([P[~inwin], C[sel]])
    return Q, int(inwin.sum()), val, len(C), str(st)
