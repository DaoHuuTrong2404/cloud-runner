#!/usr/bin/env python3
"""Constructive channel layouts (contact-preserving by construction) + exact axis fill.

Genome = (left, bottom): each a list of rows (w, theta_deg).
Left channel: row r = w unit squares along u = (cos t, sin t); its leftmost point touches x = 0; row 0 is placed as high as
possible (touching the top wall), each later row as high as possible without overlapping anything already placed
("slide down from the top until contact").  Bottom channel: the same construction in transposed coordinates
(x <-> y), with the left channel as an obstacle, so its columns stand on the bottom wall and stack leftward from the right wall.
A row that cannot be placed inside the box ends its channel.
Score = |tilted| + exact axis fill (gen.mis_fill2).

  layout.py --s S --n N --seed k --time T --out prefix [--init 3x6@27/3x6@62]
"""
import argparse, math, random, time
from gen import pen, wall_viol, mis_fill2, write, corners


def row_squares(w, t, k=0):
    """Row = w + k squares along u anchored at x = 0, minus its first k (a lattice segment set back from the wall)."""
    tr = math.radians(t)
    ki = int(math.floor(k)); kf = k - ki
    full = [(j * math.cos(tr), j * math.sin(tr), t % 90) for j in range(w + ki)]
    pts = [p for q in full for p in corners(*q)]
    mx = min(p[0] for p in pts)
    sq = [(x - mx + kf * math.cos(tr), y + kf * math.sin(tr), a) for x, y, a in full[ki:]]
    pts = [p for q in sq for p in corners(*q)]
    return sq, max(p[1] for p in pts) - min(p[1] for p in pts), min(p[1] for p in pts)


def place_channel(s, rows, obstacles):
    """Rows anchored at x = 0, slid down from the top until contact.  Returns placed squares."""
    placed = []
    for row in rows:
        w, t = row[0], row[1]
        k = row[2] if len(row) > 2 else 0
        base, height, ymin0 = row_squares(w, t, k)
        if max(x for x, _, _ in base) + 0.5 * (abs(math.cos(math.radians(t))) + abs(math.sin(math.radians(t)))) > s + 1e-9:
            break
        hi = s - (ymin0 + height)          # offset putting the row's top point on the top wall
        lo = -ymin0                         # offset putting its bottom point on the bottom wall
        if hi < lo:
            break
        obs = obstacles + placed

        def ok(dy):
            for x, y, a in base:
                q = (x, y + dy, a)
                if any(pen(q, p) > 1e-12 for p in obs if abs(p[0] - q[0]) < 1.5 and abs(p[1] - q[1]) < 1.5):
                    return False
            return True
        if ok(hi):
            dy = hi
        else:
            # find the highest free offset below hi: scan down, then bisect
            step = 0.05
            d = hi
            while d >= lo and not ok(d):
                d -= step
            if d < lo:
                break
            a, b = d, min(d + step, hi)
            for _ in range(45):
                m = 0.5 * (a + b)
                if ok(m):
                    a = m
                else:
                    b = m
            dy = a
        placed += [(x, y + dy, a_) for x, y, a_ in base]
    return placed


def transpose(sq):
    return [(y, x, (90 - a) % 90) for x, y, a in sq]


def build(s, left, bottom):
    L = place_channel(s, left, [])
    B = transpose(place_channel(s, bottom, transpose(L)))
    T = [q for q in L + B if wall_viol(q, s) <= 1e-9]
    return T


def score(s, g, bonus=0.25):
    T = build(s, *g)
    ax = mis_fill2(s, T, 1e-6)
    f = len(T) + len(ax)
    if bonus:
        f += bonus * max(0, len(mis_fill2(s, T, 0.03)) - len(ax))
    return f, T, ax


def mutate(g, rng):
    left, bottom = [[(r[0], r[1], r[2] if len(r) > 2 else 0) for r in c] for c in g]
    ch = left if rng.random() < 0.5 else bottom
    k = rng.randrange(6)
    if k <= 2 and ch:                                  # angle of one row, or of all rows
        i = rng.randrange(len(ch))
        d = rng.gauss(0, [0.3, 1.5, 4][rng.randrange(3)])
        if k == 2:
            for j in range(len(ch)):
                ch[j] = (ch[j][0], ch[j][1] + d, ch[j][2])
        else:
            ch[i] = (ch[i][0], ch[i][1] + d, ch[i][2])
    elif k == 3 and ch:                                # width or set-back of one row
        i = rng.randrange(len(ch))
        if rng.random() < 0.5:
            ch[i] = (max(1, min(6, ch[i][0] + rng.choice((-1, 1)))), ch[i][1], ch[i][2])
        else:
            if rng.random() < 0.5:
                ch[i] = (ch[i][0], ch[i][1], max(0, min(4, round(ch[i][2]) + rng.choice((-1, 1)))))
            else:
                ch[i] = (ch[i][0], ch[i][1], max(0.0, min(4.0, ch[i][2] + rng.gauss(0, 0.15))))
    elif k == 4:                                       # add a row (copy of the last)
        ch.append(ch[-1] if ch else (3, 27.0, 0))
    elif ch:                                           # remove the last row
        ch.pop()
    return (left, bottom)


def parse(spec):
    """'3x6@27/3x6@62' -> left 6 rows of 3 at 27 deg, bottom 6 rows of 3 at 62 deg (bottom angles given in the
    transposed frame: 62 there is 28 deg in the picture)."""
    out = []
    for part in spec.split('/'):
        w, rest = part.split('x'); L, t = rest.split('@')
        out.append([(int(w), float(t))] * int(L))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s', type=float, required=True); ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--time', type=float, default=600)
    ap.add_argument('--temp', type=float, default=0.5); ap.add_argument('--init', default='3x6@27/3x6@27')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    g = parse(a.init)
    f, T, ax = score(a.s, g)
    best = f
    t0 = time.time(); it = 0
    print(f'init f={f:.2f} |T|={len(T)} need {a.n}', flush=True)
    while time.time() - t0 < a.time:
        it += 1
        g2 = mutate(g, rng)
        f2, T2, ax2 = score(a.s, g2)
        if f2 >= f or rng.random() < math.exp((f2 - f) / a.temp):
            g, f, T, ax = g2, f2, T2, ax2
        if f > best:
            best = f
            write(f'{a.out}.best.txt', a.s, T + ax)
            print(f'it={it} f={f:.2f} |T|={len(T)} left={[(r[0], round(r[1], 2), r[2]) for r in g[0]]} '
                  f'bottom={[(r[0], round(r[1], 2), r[2]) for r in g[1]]} t={time.time() - t0:.0f}s', flush=True)
        if len(T) + len(ax) >= a.n:
            write(f'{a.out}.hit.txt', a.s, (T + ax)[:a.n])
            print(f'HIT it={it} count={len(T) + len(ax)}', flush=True)
            break
    print(f'done best={best:.2f} its={it}', flush=True)


if __name__ == '__main__' and __import__('sys').argv[1:2] != ['hyb']:
    main()


def count_at(s, g):
    T = build(s, *g)
    return len(T) + len(mis_fill2(s, T, 1e-6)), T


def s_star(g, n, lo, hi, tol=1e-5):
    """Smallest side (bisection, assuming count is monotone in s) at which genome g reaches n squares; hi if never."""
    if count_at(hi, g)[0] < n:
        return float('inf')
    while hi - lo > tol:
        m = 0.5 * (lo + hi)
        if count_at(m, g)[0] >= n:
            hi = m
        else:
            lo = m
    return hi


def main_sstar():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True); ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--time', type=float, default=600)
    ap.add_argument('--temp', type=float, default=0.003); ap.add_argument('--init', default='3x6@27/3x6@27')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lo, hi = a.k - 0.3, a.k + 0.02
    g = parse(a.init)
    f = s_star(g, a.n, lo, hi)
    best = f
    t0 = time.time(); it = 0
    print(f'init s*={f:.6f}', flush=True)
    while time.time() - t0 < a.time:
        it += 1
        g2 = mutate(g, rng)
        f2 = s_star(g2, a.n, lo, hi)
        if f2 <= f or (f2 < float('inf') and rng.random() < math.exp(-(f2 - f) / a.temp)):
            g, f = g2, f2
        if f < best - 1e-7:
            best = f
            T = build(f, *g)
            write(f'{a.out}.best.txt', f, T + mis_fill2(f, T, 1e-6))
            print(f'it={it} s*={f:.6f} left={[(r[0], round(r[1], 2), r[2]) for r in g[0]]} '
                  f'bottom={[(r[0], round(r[1], 2), r[2]) for r in g[1]]} t={time.time() - t0:.0f}s', flush=True)
    print(f'done best s*={best:.6f} its={it}', flush=True)


# ---------- forced side ("full line" / grid lock) ----------
# Lemma (11-20).  Unit square i: centre (x_i, y_i), signed tilt th_i in (-45, 45] deg (counter-clockwise), c_i = cos th_i,
# s_i = sin |th_i|, T_i = tan th_i, h_i = (c_i - s_i)/2, e_i = (c_i + s_i)/2 (half its x extent).  A horizontal line at
# height y_i + t with |t| <= h_i crosses the two near-vertical edges: chord of length 1/c_i centred at x_i - t T_i.
# Two squares with disjoint interiors and |y_j - y_i| <= h_i + h_j share an interval of such heights; on each of them the
# two chords are disjoint and keep one order (j right of i), so
#     x_j - x_i >= a_ij = max over y in [lo, hi] of (1/c_i + 1/c_j)/2 - d T_j + y (T_j - T_i),
#     d = y_j - y_i, lo = max(-h_i, d - h_j), hi = min(h_i, d + h_j)      (linear in y: the max is at lo or hi).
# A chain i_1 < ... < i_m of such links in a box [0, S] gives S >= e_1 + sum a + e_m (the forced side).  With every
# pair and wall penetration (SAT depth) <= p <= LINK_MARGIN the true side is >= forced - p (2 + (m - 1)(1 + max|T|)).
# Exactly axis-aligned squares: h = 1/2, a = 1, the old rule (a row or a staggered chain of m forces m).  Tilted ones
# advance less (a parallel staircase at 0.5 deg: 0.991 per square), and with |d| > h_i + h_j nothing is forced.
TILT_MAX = 1.0       # deg: squares this close to an axis enter the chain search (bounds work; soundness does not need it)
LINK_MARGIN = 1e-8   # a link needs |d| <= h_i + h_j - LINK_MARGIN: sound for penetration up to 1e-8 (feasible: 1e-9)
FORCE_EPS = 1e-7     # full_lines: forced side >= k - FORCE_EPS (a jammed row at penetration 1e-9 is ~3e-8 below k)


def _chain_axis(u, v, th, margin=LINK_MARGIN):
    """Best chain along u (lines parallel to u): (forced extent, chain as indices into u).  th: signed tilt in rad."""
    import numpy as np
    N = len(u)
    if N == 0:
        return 0.0, []
    o = np.argsort(u, kind='stable')
    u, v, th = u[o], v[o], th[o]
    c, s, T = np.cos(th), np.sin(np.abs(th)), np.tan(th)
    h, e, ic = (c - s) / 2, (c + s) / 2, 1 / c
    # candidate pairs (h_i + h_j <= 1, so only |v_j - v_i| <= 1 can link; v-sorted windows), i before j in u order
    w = np.argsort(v, kind='stable'); vw = v[w]
    lo, hi = np.searchsorted(vw, v - 1.0, 'left'), np.searchsorted(vw, v + 1.0, 'right')
    cnt = hi - lo
    I = np.repeat(np.arange(N), cnt)
    J = w[np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)]
    keep = (J > I) & (u[J] > u[I])
    I, J = I[keep], J[keep]
    hI, hJ, TI, TJ = h[I], h[J], T[I], T[J]
    d = v[J] - v[I]
    sl = TJ - TI
    A = (ic[I] + ic[J]) / 2 - d * TJ + np.maximum(np.maximum(-hI, d - hJ) * sl, np.minimum(hI, d + hJ) * sl)
    ok = (np.abs(d) <= hI + hJ - margin) & (A > 0)
    I, J, A = I[ok], J[ok], A[ok]
    q = np.argsort(J, kind='stable'); I, J, A = I[q].tolist(), J[q].tolist(), A[q].tolist()
    G = e.tolist(); par = [-1] * N
    for t in range(len(J)):                                           # J ascending, I < J: G[I] is final
        g = G[I[t]] + A[t]
        if g > G[J[t]]:
            G[J[t]] = g; par[J[t]] = I[t]
    tot = np.asarray(G) + e
    j = int(np.argmax(tot)); F = float(tot[j])
    chain = []
    while j >= 0:
        chain.append(int(o[j])); j = par[j]
    return F, chain[::-1]


def forced_chain(sq, c, tilt_max=TILT_MAX, margin=LINK_MARGIN):
    """Forced container extent along coordinate c (0: x, 1: y) from the best chain of near-axis squares (lemma above),
    and the chain as indices into sq (x, y, deg).  Any packing with penetration <= p <= margin has extent along c at
    least forced - p (2 + (m - 1)(1 + tan tilt_max)), m = chain length."""
    import numpy as np
    P = np.asarray(sq, float).reshape(-1, 3)
    a = np.mod(P[:, 2], 90.0); th = np.radians(np.where(a > 45.0, a - 90.0, a))
    idx = np.nonzero(np.abs(th) < math.radians(tilt_max))[0]
    if c == 0:
        F, ch = _chain_axis(P[idx, 0], P[idx, 1], th[idx], margin)
    else:                                                             # transpose: x <-> y reverses the tilt
        F, ch = _chain_axis(P[idx, 1], P[idx, 0], -th[idx], margin)
    return F, [int(idx[i]) for i in ch]


def forced_sides(sq, tilt_max=TILT_MAX, margin=LINK_MARGIN):
    """(forced extent along x, along y): the lower bounds on the container that the near-axis chains prove."""
    return tuple(forced_chain(sq, c, tilt_max, margin)[0] for c in (0, 1))


def forced_side(sq, **kw):
    """max of forced_sides: a lower bound on the side of a square container (up to the tolerance in forced_chain)."""
    return max(forced_sides(sq, **kw))


def grid_locked(sq, k, eps=FORCE_EPS, **kw):
    """Number of axes (0, 1 or 2) along which forced_chain proves extent >= k - eps."""
    return sum(F >= k - eps for F in forced_sides(sq, **kw))


def full_lines(path, k, eps=FORCE_EPS):
    """Number of axes (0, 1 or 2) along which a chain of near-axis squares forces the side to >= k - eps (forced_chain;
    rows and staggered chains are special cases).  > 0 reads "grid-locked at k".  Sound: a packing with penetration
    <= 1e-9 and side < k - eps - 4e-8 (any n <= 1000) is never flagged; an exact row of k jammed at that penetration is."""
    L = [l.split() for l in open(path) if l.strip() and not l.startswith('#')]
    n = int(L[0][0])
    sq = [(float(x), float(y), float(a)) for x, y, a, *_ in L[1:n + 1]]
    return grid_locked(sq, k, eps)


def s_hyb(g, n, k, tmp, step=0.15, span=0.8):
    """Build at the first side (scanning up from k - 0.1) where the count reaches n, then squeeze with packer."""
    import subprocess, os
    s = k - 0.1
    while s < k + span:
        c, T = count_at(s, g)
        if c >= n:
            ax = mis_fill2(s, T, 1e-6)
            a = os.path.join(tmp, 'h.txt'); b = os.path.join(tmp, 'hs.txt')
            write(a, s, (T + ax)[:n])
            r = subprocess.run([os.path.join(os.path.dirname(os.path.abspath(__file__)), 'target/release/packer'),
                                'relax', '--in', a, '--squeeze-pen', '--out', b], capture_output=True, text=True)
            try:
                sq_s = float(r.stdout.split(' s=')[1].split()[0])
            except Exception:
                return float('inf'), None
            if full_lines(b, k) > 0:
                return float('inf'), None
            return sq_s, b
        s += step
    return float('inf'), None


def main_hyb():
    import tempfile, shutil
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True); ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--time', type=float, default=600)
    ap.add_argument('--temp', type=float, default=0.005); ap.add_argument('--init', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tmp = tempfile.mkdtemp()
    g = eval(a.init) if a.init.startswith('(') else parse(a.init)
    f, p = s_hyb(g, a.n, a.k, tmp)
    best = f
    if p: shutil.copy(p, f'{a.out}.best.txt')
    t0 = time.time(); it = 0
    print(f'init s={f:.6f}', flush=True)
    while time.time() - t0 < a.time:
        it += 1
        g2 = mutate(g, rng)
        f2, p2 = s_hyb(g2, a.n, a.k, tmp)
        if f2 < best - 1e-7 and p2:
            best = f2
            shutil.copy(p2, f'{a.out}.best.txt')
            print(f'it={it} s={f2:.6f} left={[(r[0], round(r[1], 2), round(r[2], 2)) for r in g2[0]]} '
                  f'bottom={[(r[0], round(r[1], 2), round(r[2], 2)) for r in g2[1]]} t={time.time() - t0:.0f}s', flush=True)
        if f2 <= f or (f2 < float('inf') and rng.random() < math.exp(-(f2 - f) / a.temp)):
            g, f = g2, f2
    print(f'done best s={best:.6f} its={it}', flush=True)


if __name__ == '__main__' and len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == 'hyb':
    __import__('sys').argv.pop(1)
    main_hyb()
