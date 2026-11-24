"""L-band assembly with the MILP split into independent regions. Fixed arm interiors cut the free candidates into
separate groups (top-left end, junction, bottom-right end); each group is its own small MILP, solved to optimality
when it can be, instead of one time-limited MILP over all of them. Optional rich junction candidates: uprights on the
top-wall and right-wall rows at fine offsets, so the corner block can follow the slanted ends of both arms.

usage (diagnostic): python3 lband2.py K f N mode ph_top ph_right [--tl 30] [--rich]
"""
import argparse, math, os, sys, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from band_gen import conflicts, solve_theta, wall_candidates
from lband import columns, place


def build(K, f, N, ph_top, ph_right, mode, ec=4, phases=6, jm=3.0, rich=0.0, tol=1e-7):
    """Fixed squares (arm interiors + block) and free candidates, as in lband.assemble_L. rich > 0 adds uprights at x
    steps of `rich` on the top-wall rows and at y steps of `rich` on the right-wall columns inside the junction."""
    Wb = K + f
    th = solve_theta(N, N - 1, f)
    if th is None:
        return None
    a = K - N + 1
    core, cand = [], []
    for p in range(phases):
        sl = p / phases
        top = place(columns(th, N, ph_top, Wb, sl), 'top', a, Wb)
        rgt = place(columns(th, N, ph_right, Wb, sl), mode, a, Wb)
        ktop = sorted({q[0] for q in top})
        for k, x, y, t in top:
            inner = ktop[0] + ec <= k and x < a - jm
            if inner and p == 0:
                core.append((x, y, t))
            elif not inner:
                cand.append((x, y, t))
        for k, x, y, t in rgt:
            if y > Wb + 1e-9:
                continue
            inner = y > 1.0 + ec and y < a - jm
            if inner and p == 0:
                core.append((x, y, t))
            elif not inner:
                cand.append((x, y, t))
    nt = len(cand)
    up = [wall_candidates(Wb, Wb)]
    if rich > 0:
        lo = a - jm - 1.0
        xs = np.arange(lo + 0.5, Wb - 0.5 + 1e-9, rich)
        rows = [Wb - 0.5 - j for j in range(int(Wb - lo) + 1) if Wb - 0.5 - j >= lo + 0.5]
        up.append(np.array([(x, y, 0.0) for x in xs for y in rows]))
        up.append(np.array([(x, y, 0.0) for y in xs for x in rows]))
    core = np.array(core).reshape(-1, 3)
    block = np.array([(i + .5, j + .5, 0.0) for i in range(a) for j in range(a)])
    cand = np.vstack([np.array(cand).reshape(-1, 3)] + up)
    # drop duplicate candidates (rich rows repeat lattice points)
    key = np.round(cand, 7)
    _, first = np.unique(key, axis=0, return_index=True)
    first = np.sort(first)
    tilted = first < nt
    cand = cand[first]
    nt = int(tilted.sum())
    order = np.argsort(~tilted, kind='stable')
    cand = cand[order]
    fixed = np.vstack([core, block])
    allsq = np.vstack([fixed, cand])
    I, J = conflicts(allsq, margin=tol)
    nf = len(fixed)
    if np.any((I < nf) & (J < nf)):
        return None
    bad = np.zeros(len(allsq), bool)
    bad[J[(I < nf) & (J >= nf)]] = True
    bad[I[(J < nf) & (I >= nf)]] = True
    keep = ~bad[nf:]
    nt = int(keep[:nt].sum())
    return fixed, cand[keep], nt


def point_cliques(sub, step=0.2, half=0.5 - 1e-9):
    """Sets of candidates (any tilt) that all contain one grid point: each is a clique of the conflict graph, and
    together they make the LP bound close to the area bound (pairwise rows alone let every square sit at 1/2)."""
    c, s = np.cos(sub[:, 2]), np.sin(sub[:, 2])
    r = half * (np.abs(c) + np.abs(s))
    x0 = np.floor((sub[:, 0] - r).min() / step) * step
    y0 = np.floor((sub[:, 1] - r).min() / step) * step
    ny = int(np.ceil(((sub[:, 1] + r).max() - y0) / step)) + 1
    pts, mem = [], []
    for i in range(len(sub)):
        gx = np.arange(np.ceil((sub[i, 0] - r[i] - x0) / step), np.floor((sub[i, 0] + r[i] - x0) / step) + 1)
        gy = np.arange(np.ceil((sub[i, 1] - r[i] - y0) / step), np.floor((sub[i, 1] + r[i] - y0) / step) + 1)
        GX, GY = np.meshgrid(gx, gy, indexing='ij')
        dx, dy = x0 + GX.ravel() * step - sub[i, 0], y0 + GY.ravel() * step - sub[i, 1]
        inside = (np.abs(c[i] * dx + s[i] * dy) < half) & (np.abs(-s[i] * dx + c[i] * dy) < half)
        key = (GX.ravel() * ny + GY.ravel())[inside].astype(np.int64)
        pts.append(key)
        mem.append(np.full(len(key), i))
    if not pts:
        return []
    pts, mem = np.concatenate(pts), np.concatenate(mem)
    o = np.lexsort((mem, pts))
    pts, mem = pts[o], mem[o]
    cut = np.flatnonzero(np.diff(pts)) + 1
    seen, out = set(), []
    for g in np.split(mem, cut):
        if len(g) > 1:
            t = tuple(g)
            if t not in seen:
                seen.add(t)
                out.append(g)
    return out


def solve_group(cand, idx, nt, time_limit, tol=1e-7):
    """Max independent set over candidates idx: pairwise conflict rows plus point cliques over all candidates (points
    tested against squares shrunk by tol, so the cliques stay valid when overlaps up to tol are allowed)."""
    sub = cand[idx]
    m = len(sub)
    if m == 1:
        return idx, 'trivial', 0.0
    I, J = conflicts(sub, margin=tol)
    rows = list(np.repeat(np.arange(len(I)), 2))
    cols = list(np.stack([I, J], 1).ravel()) if len(I) else []
    nrow = len(I)
    for g in point_cliques(sub, half=0.5 - max(tol, 1e-9)):
        rows += [nrow] * len(g)
        cols += list(g)
        nrow += 1
    if nrow == 0:
        return idx, 'free', 0.0
    A = coo_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols))), shape=(nrow, m))
    res = milp(c=-np.ones(m), constraints=[LinearConstraint(A, -np.inf, 1)], integrality=np.ones(m),
               bounds=Bounds(0, 1), options={'time_limit': time_limit, 'disp': False})
    if res.x is None:
        return idx[:0], f'fail{res.status}', float('nan')
    gap = getattr(res, 'mip_gap', float('nan'))
    return idx[res.x > 0.5], ('opt' if res.status == 0 else f'tl{res.status}'), gap


def assemble_split(K, f, N, ph_top, ph_right, mode, ec=4, phases=6, jm=3.0, rich=0.0, time_limit=120.0, log=None, tol=1e-7):
    """tol: overlap depth tolerated between candidates (and against the fixed squares); > 1e-7 gives near-feasible
    assemblies for fq quench to resolve."""
    b = build(K, f, N, ph_top, ph_right, mode, ec, phases, jm, rich, tol)
    if b is None:
        return 0, None
    fixed, cand, nt = b
    I, J = conflicts(cand, margin=tol)
    m = len(cand)
    G = coo_matrix((np.ones(len(I)), (I, J)), shape=(m, m))
    ncomp, lab = connected_components(G, directed=False)
    chosen = []
    for c in range(ncomp):
        idx = np.where(lab == c)[0]
        t0 = time.time()
        sel, st, gap = solve_group(cand, idx, nt, time_limit, tol)
        chosen.append(sel)
        if log is not None and len(idx) > 1:
            cx, cy = cand[idx, 0].mean(), cand[idx, 1].mean()
            log.append(f'group at ({cx:5.1f},{cy:5.1f}) cand {len(idx):5d} chose {len(sel):4d} {st} gap {gap:.3g} {time.time() - t0:.1f}s')
    sel = np.concatenate(chosen) if chosen else np.zeros(0, int)
    sol = np.vstack([fixed, cand[sel]])
    return len(sol), sol


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('K', type=int)
    ap.add_argument('f', type=float)
    ap.add_argument('N', type=int)
    ap.add_argument('mode')
    ap.add_argument('pt', type=float)
    ap.add_argument('pr', type=float)
    ap.add_argument('--tl', type=float, default=120.0)
    ap.add_argument('--rich', type=float, default=0.0)
    ap.add_argument('--jm', type=float, default=3.0)
    ap.add_argument('--save', default='')
    ap.add_argument('--nomono', action='store_true')
    a = ap.parse_args()
    import lband as lb
    t0 = time.time()
    if not a.nomono:
        c0, _ = lb.assemble_L(a.K, a.f, a.N, a.pt, a.pr, a.mode, jm=a.jm, time_limit=30.0)
        print(f'monolithic 30s: {c0}  {time.time() - t0:.0f}s')
    t0 = time.time()
    log = []
    c1, sol = assemble_split(a.K, a.f, a.N, a.pt, a.pr, a.mode, jm=a.jm, rich=a.rich, time_limit=a.tl, log=log)
    print(f'split (rich {a.rich}): {c1}  {time.time() - t0:.0f}s')
    print('\n'.join(log))
    if a.save:
        np.savetxt(a.save, sol, delimiter=',', header='x,y,angle', comments='')