"""8-Symmetry Fault-Line Gadget Extractor & 4D Composer (`fault_dp_8sym.py`).

Pure NumPy port of Deotte/goodpjw 8-dihedral-symmetry crop extractor + 3-mode (touch, buffer-X, buffer-Y)
4D fault-line DP over our full 558-donor closed frontier!
"""

import math
import numpy as np

WMAX, DMAX, FMAX, EPS = 31, 62, 0.985, 1e-8
THR = np.array([0.0, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95])
NB = len(THR)
SYMS = [(sx, sy, tr) for sx in (1, -1) for sy in (1, -1) for tr in (0, 1)]


def mapped(sq, sym):
    sx, sy, tr = sym
    x = sq[:, 0] * sx
    y = sq[:, 1] * sy
    t = sq[:, 2] * (sx * sy)
    if tr:
        x, y, t = y, x, -t
    he = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    x = x - (x - he).min()
    y = y - (y - he).min()
    return np.column_stack([x, y, t]), x - he, x + he, y - he, y + he


def side_np(sq):
    he = 0.5 * (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2])))
    return float(
        max(
            (sq[:, 0] + he).max() - (sq[:, 0] - he).min(),
            (sq[:, 1] + he).max() - (sq[:, 1] - he).min(),
        )
    )


def extract_gadgets(donors):
    """Extract non-dominated (w, h, dm, f, ybr, xtl, donor_idx, sym_idx) gadgets across all 8 symmetries."""
    W = np.arange(1, WMAX + 1)
    DM = np.arange(1, DMAX + 1)
    best = {}
    for di, sq in enumerate(donors):
        n = len(sq)
        L = side_np(sq)
        parts = []
        max_wh = min(WMAX, int(math.ceil(L)))
        for si, sym in enumerate(SYMS):
            _, xlo, xhi, ylo, yhi = mapped(sq, sym)
            for h in range(1, max_wh + 1):
                R = np.maximum(
                    np.maximum(xhi[:, None] - W[None, :], (yhi - h)[:, None]), 0.0
                )
                order = np.argsort(R, axis=0, kind="stable")
                Rs = np.take_along_axis(R, order, axis=0)
                ybr = np.minimum.accumulate(
                    np.where(xhi[order] > W[None, :] + 1e-9, ylo[order], np.inf), axis=0
                )
                xtl = np.minimum.accumulate(
                    np.where(yhi[order] > h + 1e-9, xlo[order], np.inf), axis=0
                )
                idx = W[:, None] * h + DM[None, :] - 1
                ok = (idx < n) & (W[:, None] <= max_wh)
                wi, dj = np.nonzero(ok)
                k = idx[wi, dj]
                f = Rs[k, wi]
                m = f <= FMAX
                if not np.any(m):
                    continue
                wi, dj, k, f = wi[m], dj[m], k[m], f[m]
                parts.append(
                    np.column_stack(
                        [
                            wi + 1,
                            np.full(len(f), h),
                            dj + 1,
                            f,
                            np.minimum(ybr[k, wi], 9.0),
                            np.minimum(xtl[k, wi], 9.0),
                            np.full(len(f), di),
                            np.full(len(f), si),
                        ]
                    )
                )
        if not parts:
            continue
        P = np.concatenate(parts, axis=0)
        qb = np.maximum(np.searchsorted(THR, P[:, 4], side="right") - 1, 0)
        qt = np.maximum(np.searchsorted(THR, P[:, 5], side="right") - 1, 0)
        key = (((P[:, 0] * 32 + P[:, 1]) * 64 + P[:, 2]) * NB + qb) * NB + qt
        o = np.lexsort((P[:, 3], key))
        first = np.r_[True, key[o][1:] != key[o][:-1]]
        for kk, row in zip(key[o][first].astype(np.int64), P[o][first]):
            if kk not in best or row[3] < best[kk][3]:
                best[kk] = row

    if not best:
        return np.zeros((0, 8))
    G = np.array(list(best.values()), dtype=np.float64)
    qb = np.maximum(np.searchsorted(THR, G[:, 4], side="right") - 1, 0)
    qt = np.maximum(np.searchsorted(THR, G[:, 5], side="right") - 1, 0)
    M = np.full((WMAX + 2, WMAX + 2, DMAX + 2, NB, NB), np.inf)
    for (w, h, d, f), b, t in zip(G[:, :4], qb, qt):
        M[int(w), int(h), int(d), b, t] = min(M[int(w), int(h), int(d), b, t], f)
    for ax in (0, 1):
        M = np.minimum.accumulate(M, axis=ax)
    for ax in (2, 3, 4):
        M = np.flip(np.minimum.accumulate(np.flip(M, axis=ax), axis=ax), axis=ax)
    keep = (
        G[:, 3]
        <= M[G[:, 0].astype(int), G[:, 1].astype(int), G[:, 2].astype(int), qb, qt]
        + 1e-15
    )
    return G[keep]


def solve_dp(G):
    gw, gh, gd = G[:, 0].astype(int), G[:, 1].astype(int), G[:, 2].astype(int)
    gf, gx = G[:, 3], G[:, 5]
    gq = np.maximum(np.searchsorted(THR, G[:, 4], side="right") - 1, 0)
    shape = (WMAX + 1, WMAX + 1, DMAX + 1, NB)
    dp = np.full(shape, np.inf)
    PG = np.full(shape, -1, np.int32)
    PM = np.zeros(shape, np.int8)
    PQ = np.zeros(shape, np.int8)
    dp[0, 0, 0, NB - 1] = 0.0

    def relax(cw2, h0, d, q, src, g, mode):
        vals, qprev = src.min(axis=2), src.argmin(axis=2)
        tgt = dp[cw2, h0 : h0 + vals.shape[0], d : d + vals.shape[1], q]
        better = vals < tgt
        if better.any():
            tgt[better] = vals[better]
            PG[cw2, h0 : h0 + vals.shape[0], d : d + vals.shape[1], q][better] = g
            PM[cw2, h0 : h0 + vals.shape[0], d : d + vals.shape[1], q][better] = mode
            PQ[cw2, h0 : h0 + vals.shape[0], d : d + vals.shape[1], q][better] = qprev[
                better
            ]

    for cw in range(WMAX + 1):
        cur = dp[cw]
        if not np.isfinite(cur).any():
            continue
        for g in range(len(G)):
            w, h, d, f = gw[g], gh[g], gd[g], gf[g]
            if cw + w > WMAX or h > WMAX or d > DMAX:
                continue
            val = np.maximum(cur, f)
            ok0 = (THR[None, None, :] >= f + EPS) | (gx[g] >= cur + EPS)
            relax(
                cw + w,
                h,
                d,
                gq[g],
                np.where(ok0, val, np.inf)[: WMAX + 1 - h, : DMAX + 1 - d],
                g,
                0,
            )
            if cw == 0:
                continue
            if h + 1 <= WMAX:
                relax(cw + w, h + 1, d, gq[g], val[: WMAX - h, : DMAX + 1 - d], g, 2)
            if cw + w + 1 <= WMAX:
                relax(
                    cw + w + 1, h, d, gq[g], val[: WMAX + 1 - h, : DMAX + 1 - d], g, 1
                )
    return dp, PG, PM, PQ


def crop_gadget(donor, sym_idx, w, h, dm):
    sq, _, xhi, _, yhi = mapped(donor, SYMS[sym_idx])
    R = np.maximum(np.maximum(xhi - w, yhi - h), 0.0)
    take = np.argsort(R, kind="stable")[: w * h + dm]
    return sq[take], float(R[take].max())


def trace_and_compose(dp, PG, PM, PQ, G, donors, K, D, top_m=2):
    """Return up to `top_m` distinct multi-gadget chains `(L_est, tag, P_composed)` for `n = K^2 + D`!"""
    if K > WMAX or D <= 0 or D > DMAX:
        return []
    sub = dp[: K + 1, : K + 1, D, :]
    if not np.isfinite(sub).any():
        return []

    flat_idx = np.argsort(sub.ravel())
    results = []
    seen_sig = set()
    for fi in flat_idx[:48]:
        cw, ch, q = np.unravel_index(int(fi), sub.shape)
        f_val = float(sub[cw, ch, q])
        if not math.isfinite(f_val):
            break
        # Trace back links
        t_cw, t_ch, t_d, t_q = int(cw), int(ch), int(D), int(q)
        links = []
        ok_trace = True
        while t_cw or t_ch or t_d:
            g = int(PG[t_cw, t_ch, t_d, t_q])
            if g < 0:
                ok_trace = False
                break
            mode = int(PM[t_cw, t_ch, t_d, t_q])
            qp = int(PQ[t_cw, t_ch, t_d, t_q])
            links.append((g, mode))
            w, h, dd = int(G[g, 0]), int(G[g, 1]), int(G[g, 2])
            t_cw -= w + (1 if mode == 1 else 0)
            t_ch -= h + (1 if mode == 2 else 0)
            t_d -= dd
            t_q = qp
        if not ok_trace or not links:
            continue
        links = links[::-1]
        # Skip trivial single full-box crop (len(links)==1 and w==K and h==K) — we want multi-gadget or smaller-K compositions!
        if (
            len(links) == 1
            and int(G[links[0][0], 0]) == K
            and int(G[links[0][0], 1]) == K
        ):
            continue
        sig = tuple((int(G[g, 0]), int(G[g, 1]), int(G[g, 2]), m) for g, m in links)
        if sig in seen_sig:
            continue
        seen_sig.add(sig)

        try:
            M = len(links)
            F = f_val + (M + 2) * EPS
            a = np.full(K, K, dtype=int)
            b = np.full(K, K, dtype=int)
            gad = []
            ox, oy = 0, K
            for m_i, (g, mode) in enumerate(links):
                w, h, dm, di, si = (int(v) for v in G[g, [0, 1, 2, 6, 7]])
                if m_i > 0:
                    ox += 1 if mode == 1 else 0
                    oy -= 1 if mode == 2 else 0
                oy -= h
                if mode == 1:
                    a[ox - 1] = b[ox - 1] = oy + h
                a[ox : ox + w] = oy
                b[ox : ox + w] = oy + h
                sq_g, _ = crop_gadget(donors[di], si, w, h, dm)
                sq_g = sq_g.copy()
                sq_g[:, :2] *= 1.0 + 1e-11
                sq_g[:, 0] += ox + EPS * (1 + m_i)
                sq_g[:, 1] += oy + EPS * (M - m_i)
                gad.append(sq_g)
                ox += w
            a[ox:] = 0
            b[ox:] = 0
            if oy < 0 or ox > K or not (np.all(a <= b) and np.all(a[1:] <= b[:-1])):
                continue
            cells = [(i + 0.5, j + 0.5, 0.0) for i in range(K) for j in range(a[i])]
            cells += [
                (i + 0.5 + F, j + 0.5 + F, 0.0)
                for i in range(K)
                for j in range(b[i], K)
            ]
            parts = (
                [np.array(cells, dtype=np.float64).reshape(-1, 3)] if cells else []
            ) + gad
            P_comp = np.concatenate(parts, axis=0)
            if len(P_comp) == K * K + D:
                results.append((float(K + F), f"fdp8_m{M}", P_comp))
                if len(results) >= top_m:
                    break
        except Exception:
            continue
    return results


def trace_and_compose_v2(dp, PG, PM, PQ, G, donors, donor_n, K, D, top_m=6):
    """Upgraded composer that keeps single-link `(w=K, h=K)` crops whenever they come from a forged
    gadget (`donor_n[di] < 0`) or a different-size puzzle (`donor_n[di] != K*K + D`), and deduplicates
    by full `(w, h, dm, mode, di)` chain signature across top 96 states!
    """
    if K > WMAX or D <= 0 or D > DMAX:
        return []
    sub = dp[: K + 1, : K + 1, D, :]
    if not np.isfinite(sub).any():
        return []

    n_target = K * K + D
    flat_idx = np.argsort(sub.ravel())
    results = []
    seen_sig = set()
    for fi in flat_idx[:96]:
        cw, ch, q = np.unravel_index(int(fi), sub.shape)
        f_val = float(sub[cw, ch, q])
        if not math.isfinite(f_val):
            break
        t_cw, t_ch, t_d, t_q = int(cw), int(ch), int(D), int(q)
        links = []
        ok_trace = True
        while t_cw or t_ch or t_d:
            g = int(PG[t_cw, t_ch, t_d, t_q])
            if g < 0:
                ok_trace = False
                break
            mode = int(PM[t_cw, t_ch, t_d, t_q])
            qp = int(PQ[t_cw, t_ch, t_d, t_q])
            links.append((g, mode))
            w, h, dd = int(G[g, 0]), int(G[g, 1]), int(G[g, 2])
            t_cw -= w + (1 if mode == 1 else 0)
            t_ch -= h + (1 if mode == 2 else 0)
            t_d -= dd
            t_q = qp
        if not ok_trace or not links:
            continue
        links = links[::-1]
        if (
            len(links) == 1
            and int(G[links[0][0], 0]) == K
            and int(G[links[0][0], 1]) == K
        ):
            di0 = int(G[links[0][0], 6])
            if (
                donor_n is not None
                and di0 < len(donor_n)
                and int(donor_n[di0]) == n_target
            ):
                continue
        sig = tuple(
            (int(G[g, 0]), int(G[g, 1]), int(G[g, 2]), m, int(G[g, 6]))
            for g, m in links
        )
        if sig in seen_sig:
            continue
        seen_sig.add(sig)

        try:
            M = len(links)
            F = f_val + (M + 2) * EPS
            a = np.full(K, K, dtype=int)
            b = np.full(K, K, dtype=int)
            gad = []
            ox, oy = 0, K
            for m_i, (g, mode) in enumerate(links):
                w, h, dm, di, si = (int(v) for v in G[g, [0, 1, 2, 6, 7]])
                if m_i > 0:
                    ox += 1 if mode == 1 else 0
                    oy -= 1 if mode == 2 else 0
                oy -= h
                if mode == 1:
                    a[ox - 1] = b[ox - 1] = oy + h
                a[ox : ox + w] = oy
                b[ox : ox + w] = oy + h
                sq_g, _ = crop_gadget(donors[di], si, w, h, dm)
                sq_g = sq_g.copy()
                sq_g[:, :2] *= 1.0 + 1e-11
                sq_g[:, 0] += ox + EPS * (1 + m_i)
                sq_g[:, 1] += oy + EPS * (M - m_i)
                gad.append(sq_g)
                ox += w
            a[ox:] = 0
            b[ox:] = 0
            if oy < 0 or ox > K or not (np.all(a <= b) and np.all(a[1:] <= b[:-1])):
                continue
            cells = [(i + 0.5, j + 0.5, 0.0) for i in range(K) for j in range(a[i])]
            cells += [
                (i + 0.5 + F, j + 0.5 + F, 0.0)
                for i in range(K)
                for j in range(b[i], K)
            ]
            parts = (
                [np.array(cells, dtype=np.float64).reshape(-1, 3)] if cells else []
            ) + gad
            P_comp = np.concatenate(parts, axis=0)
            if len(P_comp) == n_target:
                results.append((float(K + F), f"fdp8v2_m{M}", P_comp))
                if len(results) >= top_m:
                    break
        except Exception:
            continue
    return results
