"""L-shaped band (the shape of the public n = 303 packing): an upright a x a block in the bottom-left corner (a = K - N + 1) and one
band of N-square columns wrapping it along the top wall and down the right wall. Interior band squares are fixed; the
two band ends (top-left and bottom-right corners) and the junction (top-right H x H corner) are chosen by a MILP from
the lattice squares of both arms at several slide phases plus uprights on the four wall lattices.

usage: python3 lband.py K D [D ...] --out DIR [--N 6-14] [--ref gap_table.csv] [--quench 30] [--workers 8]
"""
import argparse, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from band_gen import corners, conflicts, valid, side_L, solve_theta, wall_candidates, quench


def columns(theta, N, x0, length, slide=0.0):
    """Band along the bottom of a frame: columns of N squares stacked along (-sin, cos), columns 1/cos apart along x,
    occupying y in [0, N cos + sin]. Returns rows (k, x, y, angle) for squares whose bounding box is in x in [0, length]."""
    c, s = math.cos(theta), math.sin(theta)
    hx = 0.5 * (c + s)
    out = []
    for k in range(-6, int(length * c) + 8):
        for j in range(N):
            jj = j + slide
            x, y = x0 + k / c - jj * s, hx + jj * c
            if x - hx < -1e-9 or x + hx > length + 1e-9 or y - hx < -1e-9:
                continue
            out.append((k, x, y, theta))
    return out


def place(sq, mode, a, Wb):
    """Map frame squares (x along the band, y across it from the wall) onto the box. mode 'top': band along the top
    wall; 'right_rot' / 'right_ref': band along the right wall, rotated (same tilt) or reflected (mirrored tilt)."""
    out = []
    for k, x, y, t in sq:
        if mode == 'top':
            out.append((k, x, Wb - y, -t))
        elif mode == 'right_rot':
            out.append((k, Wb - y, x, t))
        else:
            out.append((k, Wb - y, Wb - x, -t))
    return out


def assemble_L(K, f, N, ph_top, ph_right, mode, ec=4, phases=6, time_limit=90.0, jm=1.5, extra=()):
    """jm: the arms' fixed interiors stop jm short of the corner region, so the MILP may cut the arms back and put an
    upright block in the corner (as in the public n = 303 packing: 20 uprights where the two mirrored arms meet)."""
    Wb = K + f
    th = solve_theta(N, N - 1, f)
    if th is None:
        return 0, None
    a = K - N + 1
    H = N - 1 + f
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
    for d in extra:  # end/junction candidates at steeper tilts: columns of N or N-1 squares sliding inside the band
        t = th + math.radians(d)
        ct, st = math.cos(t), math.sin(t)
        for m_sq in (N, N - 1):
            colh = m_sq * ct + st
            if m_sq < 1 or colh > H + 1e-9:
                continue
            for sl in np.linspace(0.0, (H - colh) / ct, 3):
                for x0 in (0.0, 0.5 / ct):
                    fr = columns(t, m_sq, x0, Wb, sl)
                    for k, x, y, tt in place(fr, 'top', a, Wb):
                        if x < 2 * H or x >= a - jm:
                            cand.append((x, y, tt))
                    for k, x, y, tt in place(fr, mode, a, Wb):
                        if y <= Wb + 1e-9 and (y < 2 * H or y >= a - jm):
                            cand.append((x, y, tt))
    core = np.array(core).reshape(-1, 3)
    block = np.array([(i + .5, j + .5, 0.0) for i in range(a) for j in range(a)])
    nt = len(cand)
    cand = np.vstack([np.array(cand).reshape(-1, 3), wall_candidates(Wb, Wb)])
    fixed = np.vstack([core, block])
    allsq = np.vstack([fixed, cand])
    I, J = conflicts(allsq)
    nf = len(fixed)
    if np.any((I < nf) & (J < nf)):
        return 0, None  # the fixed interior must be overlap-free
    bad = np.zeros(len(allsq), bool)
    bad[J[(I < nf) & (J >= nf)]] = True
    bad[I[(J < nf) & (I >= nf)]] = True
    keep = ~bad[nf:]
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
    sol = np.vstack([fixed, cand[res.x > 0.5]])
    return len(sol), sol


def search(args):
    """Smallest f (step df) at which some (N, phases, mode) L assembly reaches n; then quench."""
    K, D, Ns, f_lo, f_hi, df, fq, qsec, out = args
    n = K * K + D
    work = os.path.join(out, f'work_{n}')
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    f = f_lo
    best = None
    while f <= f_hi + 1e-12 and best is None:
        for N in Ns:
            for mode in ('right_rot', 'right_ref'):
                for pt in (0.0, 0.5):
                    for pr in (0.0, 0.5):
                        cnt, sol = assemble_L(K, f, N, pt, pr, mode)
                        if cnt >= n and (best is None or cnt > best[1]):
                            best = (f, cnt, N, mode, pt, pr, sol)
        f += df
    if best is None:
        return n, None, None, f'none up to f={f_hi}', time.time() - t0
    f, cnt, N, mode, pt, pr, sol = best
    keep = np.argsort(-np.abs(sol[:, 0] - sol[:, 1]))[:n] if len(sol) > n else np.arange(len(sol))
    sol = sol[np.sort(keep)]
    L0 = side_L(sol)
    Lq, q = quench(sol, fq, qsec, work) if fq else (None, None)
    final = q if Lq is not None and Lq < L0 else sol
    np.savetxt(os.path.join(out, f'lband_{n}.csv'), final, delimiter=',', header='x,y,angle', comments='')
    tag = f'N={N} mode={mode} ph=({pt},{pr}) count={cnt} valid={valid(sol, L0)}'
    return n, L0, Lq, tag, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('K', type=int)
    ap.add_argument('D', type=int, nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--N', default='6-14')
    ap.add_argument('--ref', default=None)
    ap.add_argument('--df', type=float, default=0.005)
    ap.add_argument('--fq', default='./fq')
    ap.add_argument('--quench', type=float, default=30.0)
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
        fp, fb = ref.get(n, (1.0, 0.45))
        jobs.append((a.K, D, list(range(lo, hi + 1)), max(0.3, fb - 0.03), min(1.0, fp + 0.03), a.df, a.fq, a.quench, a.out))
    with ProcessPoolExecutor(a.workers) as ex, open(os.path.join(a.out, 'log.txt'), 'a') as lg:
        for n, L0, Lq, tag, sec in ex.map(search, jobs):
            fp, fb = ref.get(n, (float('nan'), float('nan')))
            K = math.isqrt(n)
            line = (f'n={n} f_asm={(L0 or 0) - K:.6f} f_q={(Lq or 0) - K:.6f} f_pub={fp:.6f} f_best={fb:.6f} {tag} {sec:.0f}s')
            print(line, flush=True)
            lg.write(line + '\n')


if __name__ == '__main__':
    main()