#!/usr/bin/env python3
"""Wave 51R: Exact Trigonometric Residual Contact Continuation + Downward Window-Peel Closure!

Applies two high-precision discoveries across ALL ~558 non-grid puzzles in our R32+W50 frontier:
  1. Full Bidirectional Closure (`load_full_closed_frontier`):
     Closes upward L-borders (`n -> n + 2*floor(L) + 1`), single boundary deletions (`n+1 -> n`),
     AND downward 4-corner Window-Peels (`n -> n - 2*floor(L-1) - 1` + general sub-box `[x0, x0+W]x[y0, y0+W]`
     when `m` squares lie strictly inside with zero overlap — immediately recovering `442/485/530`
     from `577` and cascading any new `K^2+1/+2` improvements both UP and DOWN!).
  2. Residual Contact Continuation (`biased_contact_lp` + Octagon Trust Region + `wall_slack` LP):
     Unlike standard SLP (`slp.py`) which adds a pessimistic quadratic blanket margin `2*rt^2 + r*rt`
     to every row, Residual Contact Continuation solves the contact LP, evaluates the EXACT nonlinear
     trigonometric SAT/wall residual `nonlinear(x + step)` on the proposed step, tightens ONLY the
     optimistic rows (`bias = 0.35*bias + 0.65*max(remainder, 0)`), and replays `corrections=4` times
     across a multi-scale radius schedule `(1e-3, 3.5e-4, 1.875e-4, 6e-5, 2e-5, 6e-6)` with both
     box and octagon (`|dx|+|dy| <= sqrt(2)*r`) trust regions + `SLP` + `fq` + `lp_rot` + `Decimal64 >= 1.05e-9`!
"""

import argparse
import math
import os
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
TOOLS_DIR = os.path.abspath(os.path.join(HERE, "..", "tools"))
for _d in (DS_DIR, TOOLS_DIR, HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import sq_engine as E
import slp as SLP
import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12
import w32a_fresh_basin_fast_hopper as w32a


def load_full_closed_frontier(union_npz, win_globs):
    """Load frontier + close upward L-borders, boundary delete-1, AND downward 4-corner Window-Peel!"""
    cur, fresh, _, n_files = w12.load_frontier_with_wins(union_npz, win_globs)
    for _ in range(4):
        moved = 0
        # 1) Downward 4-corner Window-Peel: peel 1, 2, or 3 unit L-borders from any packing!
        for n in range(1000, 30, -1):
            Ln, sq = cur[n]
            if sq is None:
                continue
            h = 0.5 * (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2])))
            x0, x1 = float((sq[:, 0] - h).min()), float((sq[:, 0] + h).max())
            y0, y1 = float((sq[:, 1] - h).min()), float((sq[:, 1] + h).max())
            for p_layers in (1, 2):
                W_sub = Ln - p_layers
                if W_sub < 4.5:
                    continue
                for cx0 in (x0, x1 - W_sub):
                    for cy0 in (y0, y1 - W_sub):
                        inside = (
                            ((sq[:, 0] - h) >= cx0 - 1e-4)
                            & ((sq[:, 0] + h) <= cx0 + W_sub + 1e-4)
                            & ((sq[:, 1] - h) >= cy0 - 1e-4)
                            & ((sq[:, 1] + h) <= cy0 + W_sub + 1e-4)
                        )
                        m = int(inside.sum())
                        if 10 <= m < n:
                            sub = sq[inside].copy()
                            L_sub = w12.span(sub)
                            if L_sub < cur[m][0] - 1e-8:
                                cur[m] = (L_sub, sub)
                                fresh.add(m)
                                moved += 1
        # 2) Downward n+1 -> n boundary square deletion
        for n in range(999, 10, -1):
            Lp1, sq = cur[n + 1]
            if Lp1 < cur[n][0] + 0.015:
                h = 0.5 * (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2])))
                x0, x1 = (sq[:, 0] - h).min(), (sq[:, 0] + h).max()
                y0, y1 = (sq[:, 1] - h).min(), (sq[:, 1] + h).max()
                on_wall = np.where(
                    ((sq[:, 0] - h) < x0 + 1e-4)
                    | ((sq[:, 0] + h) > x1 - 1e-4)
                    | ((sq[:, 1] - h) < y0 + 1e-4)
                    | ((sq[:, 1] + h) > y1 - 1e-4)
                )[0]
                cands = list(on_wall[:16]) + [len(sq) - 1]
                for idx in cands:
                    sub = np.delete(sq, idx, axis=0)
                    Ls = w12.span(sub)
                    if Ls < cur[n][0] - 1e-8:
                        cur[n] = (Ls, sub)
                        fresh.add(n)
                        moved += 1
        # 3) Upward L-border extension: n -> n + 2*floor(L) + 1
        for n in range(10, 1000):
            Ln, sq = cur[n]
            k = int(math.floor(Ln))
            n2 = n + 2 * k + 1
            if n2 <= 1000 and Ln + 1.0 + 1e-7 < cur[n2][0] - 1e-7:
                he = 0.5 * (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2])))
                xs = sq[:, 0] - np.min(sq[:, 0] - he)
                ys = sq[:, 1] - np.min(sq[:, 1] - he)
                L_act = max(float(np.max(xs + he)), float(np.max(ys + he)))
                L_new = L_act + 1.0 + 1e-7
                extra = []
                for i in range(k):
                    extra.append((L_new - 0.5, i + 0.5, 0.0))
                    extra.append((i + 0.5, L_new - 0.5, 0.0))
                extra.append((L_new - 0.5, L_new - 0.5, 0.0))
                sq2 = np.vstack(
                    [
                        np.column_stack([xs, ys, sq[:, 2]]),
                        np.array(extra, dtype=np.float64),
                    ]
                )
                cur[n2] = (w12.span(sq2), sq2)
                fresh.add(n2)
                moved += 1
        if not moved:
            break
    tot = sum(cur[n][0] / math.sqrt(n) - 1.0 for n in range(1, 1001))
    return cur, fresh, tot, n_files


def _bounds(p):
    half = 0.5 * (np.abs(np.cos(p[:, 2])) + np.abs(np.sin(p[:, 2])))
    lo = (p[:, :2] - half[:, None]).min(0)
    hi = (p[:, :2] + half[:, None]).max(0)
    return half, lo, hi


def _span(p):
    _, lo, hi = _bounds(p)
    return float((hi - lo).max())


def _normalize(p):
    q = np.array(p, dtype=np.float64, copy=True)
    half, lo, _ = _bounds(q)
    q[:, :2] -= lo
    return q


def _pair_data(p, radius=1.55):
    pairs = cKDTree(p[:, :2]).query_pairs(radius, output_type="ndarray")
    if not len(pairs):
        return pairs, np.empty((0, 4)), np.empty(0)
    i, j = pairs.T
    d = p[j, :2] - p[i, :2]
    a, b = p[i, 2], p[j, 2]
    projections = np.column_stack(
        (
            d[:, 0] * np.cos(a) + d[:, 1] * np.sin(a),
            -d[:, 0] * np.sin(a) + d[:, 1] * np.cos(a),
            d[:, 0] * np.cos(b) + d[:, 1] * np.sin(b),
            -d[:, 0] * np.sin(b) + d[:, 1] * np.cos(b),
        )
    )
    support = 0.5 * (1.0 + np.abs(np.cos(a - b)) + np.abs(np.sin(a - b)))
    return pairs, projections, support


def _wall_slack(p, unit=1e-8, clearance=8e-14, repair=False):
    n = len(p)
    if n < 2:
        return p
    half, low, high = _bounds(p)
    length = float((high - low).max())
    if length <= math.sqrt(n) + 1e-13:
        return p
    pairs = cKDTree(p[:, :2]).query_pairs(
        math.sqrt(2.0) + 8.0 * unit, output_type="ndarray"
    )
    if not len(pairs):
        return p
    i, j = pairs.T
    c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
    axes = np.stack([np.stack([c, s], axis=1), np.stack([-s, c], axis=1)], axis=1)
    normals = np.concatenate([axes[i], axes[j]], axis=1)
    delta = p[j, :2] - p[i, :2]
    projections = np.einsum("pki,pi->pk", normals, delta)
    k = np.abs(projections).argmax(axis=1)
    ix = np.arange(len(i))
    v = projections[ix, k]
    direction = normals[ix, k] * np.where(v >= 0.0, 1.0, -1.0)[:, None]
    angle = p[i, 2] - p[j, 2]
    gap = np.abs(v) - 0.5 * (1.0 + np.abs(np.cos(angle)) + np.abs(np.sin(angle)))
    active = gap < 5.0 * unit
    i, j, direction, gap = i[active], j[active], direction[active], gap[active]
    row = np.arange(len(i))
    rows, cols = [row] * 4, [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
    data = [direction[:, 0], direction[:, 1], -direction[:, 0], -direction[:, 1]]
    slack = [(gap - clearance) / unit]
    offset = len(i)
    for dim in (0, 1):
        for sign in (-1, 1):
            room = (
                (p[:, dim] - half - low[dim])
                if sign == -1
                else (low[dim] + length - half - p[:, dim])
            )
            idx = np.flatnonzero(room < 6.0 * unit)
            r = np.arange(offset, offset + len(idx))
            offset += len(idx)
            rows.append(r)
            cols.append(2 * idx + dim)
            data.append(np.full(len(idx), float(sign)))
            slack.append(room[idx] / unit)
            if sign == 1:
                rows.append(r)
                cols.append(np.full(len(idx), 2 * n))
                data.append(np.ones(len(idx)))
    matrix = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(offset, 2 * n + 1),
    ).tocsr()
    c_obj = np.zeros(2 * n + 1)
    c_obj[-1] = -1.0
    side_bounds = (-2.0, 2.0) if repair else (0.0, 2.0)
    res = linprog(
        c_obj,
        A_ub=matrix,
        b_ub=np.concatenate(slack),
        bounds=[(-1.0, 1.0)] * (2 * n) + [side_bounds],
        method="highs",
        options={"threads": 1, "time_limit": 2.0},
    )
    if not res.success or (not repair and res.x[-1] < 1e-4):
        return p
    q = p.copy()
    q[:, :2] += unit * res.x[:-1].reshape(n, 2)
    return q


def _repair_scale(p, clearance=8e-14):
    p = _normalize(p)
    _, projection, support = _pair_data(p)
    if len(support):
        ratio = float(
            np.max((support + clearance) / np.maximum(np.abs(projection).max(1), 1e-30))
        )
        if ratio > 1.005:
            return None
        p[:, :2] *= max(1.0, ratio)
    return _normalize(p)


def _repair_local(p):
    _, projections, supports = _pair_data(p)
    worst = float(np.max(supports - np.abs(projections).max(1), initial=0.0))
    if worst < 8e-14:
        return p
    unit = max(1e-10, 8.0 * worst)
    if unit <= 1e-3:
        q = _wall_slack(p, unit=unit, clearance=8e-14, repair=True)
        _, projections, supports = _pair_data(q)
        worst2 = float(np.max(supports - np.abs(projections).max(1), initial=0.0))
        if worst2 <= 0.0:
            return _normalize(q)
    return None


def _wrap(x):
    return (x + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0


def _build_residual_model(p, step, angle_ratio=0.4):
    p = _normalize(p)
    n = len(p)
    L = _span(p)
    a = p[:, 2]
    c, s = np.cos(a), np.sin(a)
    h = 0.5 * (np.abs(c) + np.abs(s))
    edges = cKDTree(p[:, :2]).query_pairs(
        math.sqrt(2.0) + 2.0 * math.sqrt(2.0) * step + 1e-4, output_type="ndarray"
    )
    rows, cols, vals, rhs = [], [], [], []
    offset = 0
    exact_rows = []

    def add(cc, vv, bb):
        nonlocal offset
        bb = np.asarray(bb, dtype=np.float64)
        count = len(bb)
        if count == 0:
            return
        rows.append(np.repeat(np.arange(offset, offset + count), cc.shape[1]))
        cols.append(cc.ravel())
        vals.append(vv.ravel())
        rhs.append(bb)
        offset += count

    if len(edges):
        i, j = edges.T
        d = p[j, :2] - p[i, :2]
        ux = np.column_stack((c[i], -s[i], c[j], -s[j]))
        uy = np.column_stack((s[i], c[i], s[j], c[j]))
        projection = d[:, 0, None] * ux + d[:, 1, None] * uy
        chosen = np.abs(projection).argmax(1)
        rr = np.arange(len(i))
        sg = np.where(projection[rr, chosen] >= 0.0, 1.0, -1.0)
        vx, vy = sg * ux[rr, chosen], sg * uy[rr, chosen]
        dist = np.abs(projection[rr, chosen])
        own = chosen < 2
        rotation = -d[:, 0] * vy + d[:, 1] * vx
        delta = a[j] - a[i]
        cd, sd = np.cos(delta), np.sin(delta)
        rad = 0.5 * (1.0 + np.abs(cd) + np.abs(sd))
        dr = 0.5 * (-sd * np.sign(cd) + cd * np.sign(sd))
        columns = np.column_stack(
            (3 * i, 3 * i + 1, 3 * j, 3 * j + 1, 3 * i + 2, 3 * j + 2)
        )
        values = np.column_stack(
            (
                vx,
                vy,
                -vx,
                -vy,
                -dr - np.where(own, rotation, 0.0),
                dr - np.where(own, 0.0, rotation),
            )
        )

        def contact_rows(z, q, length):
            if not np.any(z):
                return np.empty(0, dtype=np.float64)
            rel = q[j[z], :2] - q[i[z], :2]
            angles = q[:, 2]
            axis = np.where(own[z], angles[i[z]], angles[j[z]]) + (chosen[z] % 2) * (
                math.pi / 2.0
            )
            dangle = angles[j[z]] - angles[i[z]]
            extent = 0.5 * (1.0 + np.abs(np.cos(dangle)) + np.abs(np.sin(dangle)))
            proj = sg[z] * (np.cos(axis) * rel[:, 0] + np.sin(axis) * rel[:, 1])
            return extent + 6e-14 - proj

        near = np.abs(_wrap(delta)) <= 2.0 * angle_ratio * step + 1e-8
        far = ~near
        if np.any(far):
            add(columns[far], values[far], dist[far] - rad[far] - 6e-14)
            exact_rows.append(lambda q, length, fm=far: contact_rows(fm, q, length))
        if np.any(near):
            xi = _wrap(delta[near])
            for sign in (-1.0, 1.0):
                slope = 0.5 * (-np.sin(xi) + sign)
                base = 0.5 * (1.0 + np.cos(xi) + sign * xi)
                v = values[near].copy()
                v[:, 4] = -slope - np.where(own[near], rotation[near], 0.0)
                v[:, 5] = slope - np.where(own[near], 0.0, rotation[near])
                add(columns[near], v, dist[near] - base - 6e-14)
                exact_rows.append(
                    lambda q, length, nm=near: contact_rows(nm, q, length)
                )

    ids = np.arange(n)
    dh = 0.5 * (-s * np.sign(c) + c * np.sign(s))
    near_wall = np.abs(_wrap(a)) <= angle_ratio * step + 1e-8
    for dim in (0, 1):
        for upper in (False, True):
            groups = []
            if np.any(~near_wall):
                groups.append((ids[~near_wall], dh[~near_wall], h[~near_wall]))
            if np.any(near_wall):
                xi = _wrap(a[near_wall])
                for sign in (-1.0, 1.0):
                    groups.append(
                        (
                            ids[near_wall],
                            0.5 * (-np.sin(xi) + sign),
                            0.5 * (np.cos(xi) + sign * xi),
                        )
                    )
            for z, derivative, extent in groups:
                cc = np.column_stack((3 * z + dim, 3 * z + 2, np.full(len(z), 3 * n)))
                vv = np.column_stack(
                    (
                        np.full(len(z), 1.0 if upper else -1.0),
                        derivative,
                        np.full(len(z), -1.0 if upper else 0.0),
                    )
                )
                bb = L - p[z, dim] - extent if upper else p[z, dim] - extent
                add(cc, vv, bb)

                def wall_rows(q, length, zz=z, dm=dim, up=upper):
                    ang = q[zz, 2]
                    ext = 0.5 * (np.abs(np.cos(ang)) + np.abs(np.sin(ang)))
                    return (q[zz, dm] + ext - length) if up else (ext - q[zz, dm])

                exact_rows.append(wall_rows)

    A = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(offset, 3 * n + 1),
    ).tocsr()
    b = np.concatenate(rhs) / step
    bounds = np.vstack(
        [
            np.column_stack(
                [
                    -np.tile([1.0, 1.0, angle_ratio], n),
                    np.tile([1.0, 1.0, angle_ratio], n),
                ]
            ),
            np.array([[-1.0, 1.0]]),
        ]
    )
    objective = np.zeros(3 * n + 1)
    objective[-1] = 1.0

    def exact(move):
        q = p + step * move[:-1].reshape(-1, 3)
        length = L + step * move[-1]
        return np.concatenate([compute(q, length) for compute in exact_rows])

    return dict(c=objective, A=A, b=b, bounds=bounds, exact=exact)


def _diagonals(n):
    ids = np.arange(n)
    rows = np.tile(np.repeat(ids, 2), 4) + np.repeat(n * np.arange(4), 2 * n)
    cols = np.tile(np.column_stack((3 * ids, 3 * ids + 1)).ravel(), 4)
    coeff = np.repeat(
        np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]), n, axis=0
    )
    return coo_matrix((coeff.ravel(), (rows, cols)), shape=(4 * n, 3 * n + 1)).tocsr()


def _best_step(p, move):
    best, best_side, best_frac = p, _span(p), 0.0
    trials = []
    for frac in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        trial = p + frac * move
        sd = _span(trial)
        if sd >= best_side - 1e-12:
            continue
        trials.append((sd, frac, trial))
        expanded = _repair_scale(trial, clearance=8e-14)
        if expanded is not None:
            esd = _span(expanded)
            if esd < best_side - 1e-12:
                best, best_side, best_frac = expanded, esd, frac
    for sd, frac, trial in sorted(trials, key=lambda r: r[0])[:2]:
        if sd >= best_side - 1e-12:
            break
        loc = _repair_local(trial)
        if loc is not None:
            lsd = _span(loc)
            if lsd < best_side - 1e-12:
                best, best_side, best_frac = loc, lsd, frac
    return best, best_frac


def biased_contact_lp(p, radius, octagon=False, angle_ratio=0.4, corrections=3):
    p = _normalize(p)
    model = _build_residual_model(p, radius, angle_ratio=angle_ratio)
    A, b, limits = model["A"], model["b"], model["bounds"]
    active = np.abs(A) @ limits[:, 1] >= b - max(1e-8, radius * 0.02)
    A_act, b_act = A[active], b[active]
    if octagon:
        diag = _diagonals(len(p))
        matrix = vstack((A_act, diag), format="csr")
        extra = np.full(diag.shape[0], math.sqrt(2.0))
    else:
        matrix = A_act
        extra = np.empty(0)
    bias = np.zeros_like(b_act)
    best = p
    for _ in range(corrections + 1):
        rhs = np.r_[b_act - bias, extra]
        fit = linprog(
            model["c"],
            A_ub=matrix,
            b_ub=rhs,
            bounds=list(map(tuple, limits)),
            method="highs-ds",
            options={
                "threads": 1,
                "time_limit": 4.0,
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
        if not fit.success or not np.isfinite(fit.fun):
            break
        if (
            max(
                float(np.max(matrix @ fit.x - rhs, initial=0.0)),
                float(np.max(np.abs(fit.x) - limits[:, 1])),
            )
            > 2e-8
        ):
            break
        predicted = -radius * float(fit.fun)
        if predicted <= 1e-12:
            break
        nonlinear = model["exact"](fit.x)[active] / radius
        remainder = nonlinear - (A_act @ fit.x - b_act)
        bias = 0.35 * bias + 0.65 * np.maximum(remainder, 0.0)
        q, _ = _best_step(p, fit.x[:-1].reshape(-1, 3) * radius)
        if _span(q) < _span(best) - 1e-12:
            best = q
    return best


def residual_continuation_polish(P_in, budget_s=28.0):
    """Multi-scale Residual Contact Continuation + Octagon + Wall-Slack + SLP!"""
    t0 = time.time()
    best_p = _normalize(w32a.safe_open(P_in))
    best_s = _span(best_p)

    # Initial fast SLP pass with two angular trust scales
    for kap in (0.30, 0.55):
        if time.time() - t0 > budget_s * 0.45:
            break
        try:
            xs, ys, ts, _ = SLP.polish(
                best_p[:, 0],
                best_p[:, 1],
                best_p[:, 2],
                iters=90,
                r0=3e-3,
                rmin=1e-10,
                tol_sep=-1e-13,
                time_limit=min(8.0, budget_s * 0.25),
                kappa=kap,
            )
            q_slp = _normalize(w32a.safe_open(np.column_stack([xs, ys, ts])))
            s_slp = _span(q_slp)
            if s_slp < best_s - 1e-11 and E.min_gap(q_slp) >= -1e-12:
                best_p, best_s = q_slp, s_slp
        except Exception:
            pass

    # Residual Contact Continuation cascade across radii and box/octagon trust regions
    radii = (1.2e-3, 3.5e-4, 1.875e-4, 6e-5, 2e-5, 6e-6)
    for rnd in range(2):
        improved_rnd = False
        for rad in radii:
            if time.time() - t0 > budget_s:
                break
            for oct_flag in (
                (False, True) if rad in (3.5e-4, 1.875e-4, 6e-5) else (False,)
            ):
                if time.time() - t0 > budget_s:
                    break
                try:
                    q = biased_contact_lp(
                        best_p, rad, octagon=oct_flag, angle_ratio=0.42, corrections=3
                    )
                    sq_q = _span(q)
                    if sq_q < best_s - 1e-12:
                        q_op = _normalize(w32a.safe_open(q))
                        s_op = _span(q_op)
                        if s_op < best_s - 1e-12 and E.min_gap(q_op) >= -1e-12:
                            best_p, best_s = q_op, s_op
                            improved_rnd = True
                except Exception:
                    pass
        if not improved_rnd:
            break

    # Final wall-slack tightening
    try:
        qw = _wall_slack(best_p, unit=1e-8, clearance=8e-14, repair=False)
        qw = _normalize(w32a.safe_open(qw))
        if _span(qw) < best_s - 1e-12 and E.min_gap(qw) >= -1e-12:
            best_p, best_s = qw, _span(qw)
    except Exception:
        pass

    return best_p, best_s


def _worker_puzzle(args_tuple):
    n, P_cur, L_orig, per_puz_sec, deadline = args_tuple
    rem = deadline - time.time() - 6.0
    if rem < 3.0:
        return n, None
    bud = min(float(per_puz_sec), rem * 0.7)
    # Check if window-peel / closure in P_cur already beat L_orig BEFORE optimization!
    cur_p = w32a.safe_open(P_cur)
    cur_s = float(E.side(cur_p))

    qp, sp = residual_continuation_polish(cur_p, budget_s=bud)
    if sp < cur_s - 1e-12:
        cur_p, cur_s = qp, sp

    # If improved vs L_orig (either from downward Window-Peel OR Residual Continuation!), run fq + lp_rot + Decimal64!
    if cur_s < L_orig - 5e-10:
        if (
            os.path.exists(f2d.FQ_BIN)
            and (deadline - time.time()) > 8.0
            and (L_orig - cur_s) > 1e-7
        ):
            with tempfile.TemporaryDirectory(prefix=f"w51r_{n}_") as tmp:
                r_fin = f2d.run_fq(
                    cur_s,
                    cur_p,
                    tmp,
                    ["--loosen", "1.0", "--no-alm", "--flip-top", "48", "--pit", "400"],
                    timeout=min(25, max(8, int(deadline - time.time() - 4))),
                )
                if r_fin is not None and r_fin[0] < cur_s:
                    cur_s, cur_p = r_fin[0], r_fin[1]
        try:
            cur_p = w12.lp_rot_polish(cur_p)
        except Exception:
            pass
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_p)
        if ok and Lf < L_orig - 5e-10:
            return n, (float(Lf), xf, yf, tf)
    return n, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2250)
    ap.add_argument("--per-puzzle-sec", type=float, default=26.0)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    raw_cur, _, raw_score, _ = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    cur, fresh, tot_score, _ = load_full_closed_frontier(args.union_npz, args.win_glob)

    # Collect ALL non-grid puzzles n in 5..1000!
    non_grid = []
    for n in range(5, 1001):
        K = math.isqrt(n)
        if K * K == n:
            continue
        if cur[n][0] >= K + 1 - 1e-6:
            continue
        # Prioritize larger n & recent/window-peeled puzzles first
        peeled_gain = max(0.0, raw_cur[n][0] - cur[n][0])
        non_grid.append((peeled_gain, n))
    non_grid.sort(reverse=True)
    all_ns = [n for _, n in non_grid]
    my_slice = all_ns[args.shard :: args.num_shards]
    deadline = t0 + args.budget_sec

    print(
        f"[W51R-ResidualCont s={args.shard}/{args.num_shards}] raw={raw_score:.10f} -> closed={tot_score:.10f} "
        f"my_slice={len(my_slice)} non-grid puzzles ({time.time() - t0:.1f}s)",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0

    def save_ckpt():
        pl = {}
        for kn, (Lf, xf, yf, tf) in sorted(wins.items()):
            pl[f"L_{kn}"] = np.array([Lf], dtype=np.float64)
            pl[f"x_{kn}"] = np.asarray(xf, dtype=np.float64)
            pl[f"y_{kn}"] = np.asarray(yf, dtype=np.float64)
            pl[f"t_{kn}"] = np.asarray(tf, dtype=np.float64)
        np.savez_compressed(args.out_npz, **pl)

    ex = ProcessPoolExecutor(max_workers=args.procs)
    try:
        # Process in clean batches of 16 puzzles so checkpointing is frequent and timeouts cannot stall
        batch_sz = 16
        for b0 in range(0, len(my_slice), batch_sz):
            if time.time() > deadline - 15.0:
                break
            batch = my_slice[b0 : b0 + batch_sz]
            futs = [
                ex.submit(
                    _worker_puzzle,
                    (n, cur[n][1], raw_cur[n][0], args.per_puzzle_sec, deadline),
                )
                for n in batch
            ]
            try:
                for f in as_completed(
                    futs, timeout=max(30.0, args.per_puzzle_sec * 6.0 + 45.0)
                ):
                    n, res = f.result()
                    if res is not None:
                        Lf = res[0]
                        dL = Lf - raw_cur[n][0]
                        dS = dL / math.sqrt(n)
                        wins[n] = res
                        tot_ds += dS
                        if dS < -1e-8:
                            print(
                                f"  [W51R WIN!!] n={n:4d} {raw_cur[n][0]:.9f} -> {Lf:.9f} (dL={dL:.9f}, dS={dS:.9f})",
                                flush=True,
                            )
            except TimeoutError:
                pass
            save_ckpt()
            print(
                f"  [W51R prog s={args.shard}] {min(b0 + batch_sz, len(my_slice))}/{len(my_slice)} done | wins={len(wins)} | dS={tot_ds:.9f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    except Exception as exc:
        print(f"[W51R s={args.shard} EXC] {exc}", flush=True)

    save_ckpt()
    print(
        f"[W51R s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.9f}",
        flush=True,
    )
    for proc in list(getattr(ex, "_processes", {}).values()):
        try:
            proc.terminate()
        except Exception:
            pass
    os._exit(0)


if __name__ == "__main__":
    main()
