"""Single-band (Erdos-Graham "fault") constructions for n = K^2 + D, then fq quench on the ends.

Band geometry, exact-angle rule and the MILP assembly are ported from lostdecimal27's public Kaggle notebook
"Squares in Squares: bands, gadgets, exact angles" (Apache 2.0). Added here: the search for the smallest overhang f
at which the assembly reaches n, trimming to exactly n, and the fq quench of the ends (slides and small angle changes
that no lattice grid contains), with every result checked by an exact separating-axis test.

usage: python3 band_gen.py K D [D ...] --out DIR [--N 6-14] [--fq FQ] [--quench 20] [--workers 8]
"""
import argparse, math, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import brentq, milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
import scipy.optimize._milp as _milp

# HiGHS starts a pool of (cores + 1) / 2 threads per process; with many worker processes on the CPU their idle spinning
# starves the solving thread (one-core test: 50 s -> 2.7 s for the same 994-square L). milp() drops a 'threads' option,
# so it is injected into the wrapper that milp() calls.
if hasattr(_milp, '_highs_wrapper'):
    _hw = _milp._highs_wrapper
    _milp._highs_wrapper = lambda *a: _hw(*a[:-1], dict(a[-1], threads=1))

OFF =np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])


def corners(sol):
    sol = np.asarray(sol, float)
    c, s = np.cos(sol[:, 2]), np.sin(sol[:, 2])
    cs = np.empty((len(sol), 4, 2))
    cs[:, :, 0] = sol[:, None, 0] + OFF[None, :, 0] * c[:, None] - OFF[None, :, 1] * s[:, None]
    cs[:, :, 1] = sol[:, None, 1] + OFF[None, :, 0] * s[:, None] + OFF[None, :, 1] * c[:, None]
    return cs


def side_L(sol):
    cs = corners(sol).reshape(-1, 2)
    return max(np.ptp(cs[:, 0]), np.ptp(cs[:, 1]))


def separation(sol, I, J):
    """SAT separation of pairs (I, J): > 0 apart, <= 0 overlapping by that depth."""
    cs = corners(sol)
    ci, si, cj, sj = np.cos(sol[I, 2]), np.sin(sol[I, 2]), np.cos(sol[J, 2]), np.sin(sol[J, 2])
    axes = np.stack([np.stack([ci, si], -1), np.stack([-si, ci], -1), np.stack([cj, sj], -1), np.stack([-sj, cj], -1)], 1)
    pi, pj = np.einsum('mkd,mad->mak', cs[I], axes), np.einsum('mkd,mad->mak', cs[J], axes)
    return (np.maximum(pi.min(-1), pj.min(-1)) - np.minimum(pi.max(-1), pj.max(-1))).max(-1)


def close_pairs(sol, pad=1e-9):
    """Pairs of unit squares whose centres are within sqrt 2 (the only pairs that can touch or overlap)."""
    from scipy.spatial import cKDTree
    sol = np.asarray(sol, float)
    if len(sol) < 2:
        return np.empty(0, int), np.empty(0, int)
    p = cKDTree(sol[:, :2]).query_pairs(math.sqrt(2) + 1e-6 + pad, output_type='ndarray')
    return p[:, 0], p[:, 1]


def conflicts(sol, margin=1e-7):
    I, J = close_pairs(sol)
    if len(I) == 0:
        return I, J
    m = separation(sol, I, J) < -margin
    return I[m], J[m]


def valid(sol, L=None, tol=1e-9):
    sol = np.asarray(sol, float)
    I, J = close_pairs(sol)
    if len(I) and separation(sol, I, J).min() < -tol:
        return False
    if L is not None:
        cs = corners(sol).reshape(-1, 2)
        return cs.min() > -tol and cs.max() < L + tol
    return True


def solve_theta(N, M, f):
    g = lambda t: N * math.cos(t) + math.sin(t) - (M + f)
    lo = math.atan2(1.0, N) + 1e-9
    return None if g(lo) < 0 else brentq(g, lo, math.pi / 2 - 1e-9, xtol=1e-15)


def hband(theta, y0, N, x0, Wb, Hb, slide=0.0):
    c, s = math.cos(theta), math.sin(theta)
    hx = 0.5 * (c + s)
    out = []
    for k in range(-5, int(Wb * c) + 7):
        for j in range(N):
            jj = j + slide
            x, y = x0 + k / c - jj * s, y0 + hx + jj * c
            if x - hx < -1e-9 or x + hx > Wb + 1e-9 or y - hx < -1e-9 or y + hx > Hb + 1e-9:
                continue
            out.append((k, j, x, y, theta))
    return out


def wall_candidates(Wb, Hb):
    xs = sorted({round(.5 + i, 9) for i in range(int(Wb) + 1) if .5 + i <= Wb - .5 + 1e-9} |
                {round(Wb - .5 - i, 9) for i in range(int(Wb) + 1) if .5 + i <= Wb - .5 + 1e-9})
    ys = sorted({round(.5 + i, 9) for i in range(int(Hb) + 1) if .5 + i <= Hb - .5 + 1e-9} |
                {round(Hb - .5 - i, 9) for i in range(int(Hb) + 1) if .5 + i <= Hb - .5 + 1e-9})
    return np.array([(x, y, 0.0) for x in xs for y in ys])


def end_candidates(th, y0, N, f, Wb, x_lo, x_hi, angles=(3, 6, 10), xsteps=8, nslide=3):
    """Extra end candidates outside the core's x-range [x_lo, x_hi]: columns of N or N-1 squares at angles th + delta
    (they are shorter than the band, so they may slide inside it), and uprights inside the band height aligned to the
    band's bottom or top edge at fine x offsets."""
    H = N - 1 + f
    tilt, up = [], []
    for d in (0.0,) + tuple(angles):
        t = th + math.radians(d)
        c, s = math.cos(t), math.sin(t)
        hx = 0.5 * (c + s)
        for m in (N, N - 1):
            colh = m * c + s
            if m < 1 or colh > H + 1e-9:
                continue
            for sl in np.linspace(0.0, H - colh, nslide) if H - colh > 1e-6 else (0.0,):
                for x0 in np.arange(xsteps) / (xsteps * c):
                    for k in range(-5, int(Wb * c) + 7):
                        for j in range(m):
                            x, y = x0 + k / c - j * s, y0 + sl + hx + j * c
                            if x_lo + 1 < x < x_hi - 1:
                                continue
                            if x - hx < -1e-9 or x + hx > Wb + 1e-9 or y - hx < y0 - 1e-9 or y + hx > y0 + H + 1e-9:
                                continue
                            tilt.append((x, y, t))
    for r in range(N - 1):
        for y in {y0 + .5 + r, y0 + H - .5 - r}:
            for x in np.arange(0.5, Wb - 0.5 + 1e-9, 1.0 / xsteps):
                if x_lo + 1 < x < x_hi - 1:
                    continue
                up.append((x, y, 0.0))
    return np.array(tilt).reshape(-1, 3), np.array(up).reshape(-1, 3)


def slid_columns(th, y0, N, x0, Wb, f):
    """Band columns k with their squares slid individually: a column's squares sit at slide parameters jj in the exact
    feasible interval [a, b] (box walls and band height), flush to a, flush to b, and on the lattice (jj integer).
    Columns stay 1 apart along u whatever their slides, so only squares of the same column can conflict.
    Returns (full columns as one array, partial-column candidates as one array)."""
    c, s = math.cos(th), math.sin(th)
    hx = 0.5 * (c + s)
    full, part = [], []
    for k in range(-5, int(Wb * c) + 7):
        xc = x0 + k / c
        a = max(0.0, (xc + hx - Wb) / s)
        b = min(N - 1.0, (xc - hx) / s)
        if b < a - 1e-12:
            continue
        cnt = int(math.floor(b - a + 1e-9)) + 1
        if cnt >= N:
            full += [(xc - j * s, y0 + hx + j * c, th) for j in range(N)]
            continue
        starts = {a, b - (cnt - 1)}
        lo_int = math.ceil(a - 1e-9)
        if lo_int <= b + 1e-9:
            starts.add(float(lo_int))
        for st in starts:
            m = int(math.floor(b - st + 1e-9)) + 1
            part += [(xc - (st + j) * s, y0 + hx + (st + j) * c, th) for j in range(m)]
    return np.array(full).reshape(-1, 3), np.array(part).reshape(-1, 3)


def assemble2(K, f, N, y0, phase, time_limit=60.0):
    """Band with every partial end column slid individually (exact, no phase grid), uprights from the four wall
    lattices; a MILP picks the partial-column squares and uprights. Returns (count, packing)."""
    Wb = Hb = K + f
    th = solve_theta(N, N - 1, f)
    if th is None:
        return 0, None
    core, part = slid_columns(th, y0, N, phase, Wb, f)
    if not len(core):
        return 0, None
    nt = len(part)
    cand = np.vstack([part, wall_candidates(Wb, Hb)])
    allsq = np.vstack([core, cand])
    I, J = conflicts(allsq)
    bad = np.zeros(len(allsq), bool)
    ic, jc = I < len(core), J < len(core)
    bad[J[ic & ~jc]] = True
    bad[I[~ic & jc]] = True
    keep = ~bad[len(core):]
    cand = cand[keep]
    nt = int(keep[:nt].sum())
    m = len(cand)
    I, J = conflicts(cand)
    rows = list(np.repeat(np.arange(len(I)), 2))
    cols = list(np.stack([I, J], 1).ravel()) if len(I) else []
    nrow = len(I)
    up = np.arange(nt, m)
    for shift in (0.0, 0.5):
        cell = np.floor(cand[up, 0] + shift).astype(int) * 10000 + np.floor(cand[up, 1] + shift).astype(int)
        for cc in np.unique(cell):
            mem = up[cell == cc]
            if len(mem) > 1:
                rows += [nrow] * len(mem)
                cols += list(mem)
                nrow += 1
    A = coo_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols))), shape=(nrow, m))
    res = milp(c=-np.ones(m), constraints=[LinearConstraint(A, -np.inf, 1)], integrality=np.ones(m),
               bounds=Bounds(0, 1), options={'time_limit': time_limit, 'disp': False})
    if res.x is None:
        return 0, None
    sol = np.vstack([core, cand[res.x > 0.5]])
    return len(sol), sol


def assemble(K, f, N, y0, phase, ec=5, phases=8, time_limit=60.0, rich=False):
    """One horizontal band of N-square columns with its bottom at height y0; interior columns fixed, end columns
    (all slide phases) and upright candidates on the four wall lattices chosen by a MILP. Returns (count, packing).
    rich=True adds end_candidates (angle changes, shorter sliding columns, in-band uprights at fine offsets)."""
    Wb = Hb = K + f
    th = solve_theta(N, N - 1, f)
    if th is None:
        return 0, None
    lat = hband(th, y0, N, phase, Wb, Hb)
    if not lat:
        return 0, None
    ks = sorted({q[0] for q in lat})
    inner = lambda k: ks[0] + ec <= k <= ks[-1] - ec
    core = np.array([(x, y, t) for k, j, x, y, t in lat if inner(k)]).reshape(-1, 3)
    cand = [(x, y, t) for k, j, x, y, t in lat if not inner(k)]
    for p in range(1, phases):
        cand += [(x, y, t) for k, j, x, y, t in hband(th, y0, N, phase, Wb, Hb, slide=p / phases) if not inner(k)]
    cand = np.array(cand).reshape(-1, 3)
    extra_up = np.empty((0, 3))
    if rich and len(core):
        xc = corners(core)[:, :, 0]
        et, extra_up = end_candidates(th, y0, N, f, Wb, xc.min(), xc.max())
        cand = np.vstack([cand, et])
    nt = len(cand)
    cand = np.vstack([cand, wall_candidates(Wb, Hb), extra_up])
    if len(core):
        allsq = np.vstack([core, cand])
        I, J = conflicts(allsq)
        bad = np.zeros(len(allsq), bool)
        ic, jc = I < len(core), J < len(core)
        bad[J[ic & ~jc]] = True
        bad[I[~ic & jc]] = True
        keep = ~bad[len(core):]
        cand = cand[keep]
        nt = int(keep[:nt].sum())
    m = len(cand)
    I, J = conflicts(cand)
    rows, cols, nrow = [], [], 0
    if len(I):
        rows += list(np.repeat(np.arange(len(I)), 2))
        cols += list(np.stack([I, J], 1).ravel())
        nrow = len(I)
    up = np.arange(nt, m)
    for shift in (0.0, 0.5):  # unit-cell cliques: uprights whose centres share a cell overlap
        cell = np.floor(cand[up, 0] + shift).astype(int) * 10000 + np.floor(cand[up, 1] + shift).astype(int)
        for c in np.unique(cell):
            mem = up[cell == c]
            if len(mem) > 1:
                rows += [nrow] * len(mem)
                cols += list(mem)
                nrow += 1
    A = coo_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols))), shape=(nrow, m))
    res = milp(c=-np.ones(m), constraints=[LinearConstraint(A, -np.inf, 1)], integrality=np.ones(m),
               bounds=Bounds(0, 1), options={'time_limit': time_limit, 'disp': False})
    if res.x is None:
        return 0, None
    x = res.x > 0.5
    sol = np.vstack([core, cand[x]])
    return len(sol), sol


def trim(sol, n, band_lo, band_hi):
    """Drop the extra squares: uprights nearest to the band ends first (they block the ends' adjustments)."""
    if len(sol) <= n:
        return sol
    up = np.where(np.abs(np.remainder(sol[:, 2] + math.pi / 4, math.pi / 2) - math.pi / 4) < 1e-9)[0]
    mid = (band_lo + band_hi) / 2
    order = up[np.argsort(np.abs(sol[up, 1] - mid) + 0.01 * np.minimum(sol[up, 0], sol[:, 0].max() - sol[up, 0]))]
    drop = set(order[:len(sol) - n].tolist())
    return sol[[i for i in range(len(sol)) if i not in drop]]


def quench(sol, fq, seconds, work):
    """fq quench --finish on a padded box; returns (L, packing) of the result if it verifies, else (None, None)."""
    cs = corners(sol).reshape(-1, 2)
    mn = cs.min(0)
    L = side_L(sol)
    a, b = os.path.join(work, 'a.cfg'), os.path.join(work, 'b.cfg')
    with open(a, 'w') as fh:
        fh.write(f'{len(sol)} {L + 1e-9:.17f}\n')
        for x, y, t in sol:
            fh.write(f'{x - mn[0]:.17f} {y - mn[1]:.17f} {math.degrees(t) % 90.0:.15f}\n')
    if os.path.exists(b):
        os.remove(b)
    try:
        subprocess.run([fq, 'quench', '--in', a, '--out', b, '--finish', '--deadline', str(seconds)],
                       capture_output=True, timeout=seconds + 30)
    except subprocess.TimeoutExpired:
        pass
    if not os.path.exists(b):
        return None, None
    lines = [l.split() for l in open(b) if l.strip() and not l.startswith('#')]
    q = np.array([[float(t[0]), float(t[1]), math.radians(float(t[2]))] for t in lines[1:1 + len(sol)]])
    if len(q) != len(sol):
        return None, None
    Lq = side_L(q)
    cs = corners(q).reshape(-1, 2)
    q[:, :2] -= cs.min(0)
    return (Lq, q) if valid(q, Lq) else (None, None)


def best_for(args):
    """Smallest f (step df) at which some (N, y0, phase) assembly reaches n, then quench it."""
    K, D, Ns, f_lo, f_hi, df, fq, qsec, out = args
    n = K * K + D
    work = os.path.join(out, f'work_{n}')
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    log = []
    best = None
    f = f_lo
    while f <= f_hi + 1e-12 and best is None:
        for N in Ns:
            if N - 1 > K:
                continue
            for y0 in sorted({0, (K - N + 1) // 2}):
                for phase in np.linspace(0, 1, 4, endpoint=False):
                    cnt, sol = assemble(K, f, N, float(y0), float(phase))
                    if cnt >= n and (best is None or cnt > best[1]):
                        best = (f, cnt, N, y0, phase, sol)
        log.append(f'f={f:.4f} best_count={best[1] if best else "-"}')
        f += df
    if best is None:
        return n, None, None, None, f'no assembly reached n by f={f_hi}', time.time() - t0
    f, cnt, N, y0, phase, sol = best
    th = solve_theta(N, N - 1, f)
    sol = trim(sol, n, y0, y0 + N - 1 + f)
    L0 = side_L(sol)
    ok0 = valid(sol, L0)
    Lq, q = quench(sol, fq, qsec, work) if fq else (None, None)
    tag = f'N={N} y0={y0} phase={phase:.2f} theta={math.degrees(th):.3f} count={cnt} valid={ok0}'
    final = (Lq, q) if Lq is not None and Lq < L0 else (L0, sol)
    np.savetxt(os.path.join(out, f'band_{n}.csv'), final[1], delimiter=',', header='x,y,angle', comments='')
    return n, L0, Lq, final[0], tag, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('K', type=int)
    ap.add_argument('D', type=int, nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--N', default='6-14')
    ap.add_argument('--flo', type=float, default=None, help='first f to try (default: leader f - 0.01 from --ref)')
    ap.add_argument('--fhi', type=float, default=1.0)
    ap.add_argument('--df', type=float, default=0.004)
    ap.add_argument('--ref', default=None, help='gap_table.csv for f_pub / f_best per n')
    ap.add_argument('--fq', default='./fq')
    ap.add_argument('--quench', type=float, default=20.0)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.N.split('-'))
    ref = {}
    if a.ref:
        import csv
        ref = {int(r['n']): (float(r['f_pub']), float(r['f_best'])) for r in csv.DictReader(open(a.ref, encoding='utf-8'))}
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for D in a.D:
        n = a.K * a.K + D
        fp, fb = ref.get(n, (1.0, 0.4))
        flo = a.flo if a.flo is not None else max(0.3, fb - 0.01)
        jobs.append((a.K, D, list(range(lo, hi + 1)), flo, min(a.fhi, fp + 0.02), a.df, a.fq, a.quench, a.out))
    with ProcessPoolExecutor(a.workers) as ex, open(os.path.join(a.out, 'log.txt'), 'a') as lg:
        for n, L0, Lq, Lf, tag, sec in ex.map(best_for, jobs):
            fp, fb = ref.get(n, (float('nan'), float('nan')))
            K = math.isqrt(n)
            line = (f'n={n} f_assembled={(L0 or 0) - K:.6f} f_quenched={(Lq or 0) - K:.6f} f_final={(Lf or 0) - K:.6f} '
                    f'f_pub={fp:.6f} f_best={fb:.6f} {tag} {sec:.0f}s')
            print(line, flush=True)
            lg.write(line + '\n')


if __name__ == '__main__':
    main()