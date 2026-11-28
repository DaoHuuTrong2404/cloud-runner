#!/usr/bin/env python3
"""Wave 76 Shared Engine (`kink_polish_and_strip_lifts_w76.py`):
1. `hull_rotate_exact(P_in)`: Exact convex-hull support interval + width/height crossing
   whole-packing global rotation (`lightningv08` / `malyshev_atlas` Apache-2.0 algorithm).
2. `kink_contact_lp_polish(P_in, rounds=45, seed=0, max_seconds=18.0)`: Kink-aware free-angle
   trust-region sequential sparse LP (`prvsiyan` / `lightningv08`), adding BOTH `-0.5` and `+0.5`
   one-sided subgradients at every `0/90 deg` relative-angle pair kink and wall-alignment kink!
3. `sweep_strip_and_l_lifts_for_n(n, cur)`: Bidirectional `sq_engine` structural lifts:
   - `E.transplant(up=False)` (`n + 2K + 1 -> n`) & `E.transplant(up=True)` (`n - 2(K-1) - 1 -> n`),
   - `E.change_length(axis=0/1, add=False)` (`n + K`, `n + K + 1 -> n` single-strip peel + `Compactor`),
   - `E.change_length(axis=0/1, add=True)` (`n - K`, `n - K + 1 -> n` forward single-strip lift + `Compactor`)!
"""

import math
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull, cKDTree

import bidir_transfer_w70 as btr
import rust_engines as RE
import sq_engine as E


def hull_rotate_exact(P_in):
    """Exact candidate enumeration for the smallest enclosing axis-aligned square of a rigid layout."""
    p = RE.normalize_to_origin(P_in)
    if len(p) <= 1:
        return RE.side_exact(p), p
    s0 = RE.side_exact(p)
    c = np.cos(p[:, 2]) * 0.5
    s = np.sin(p[:, 2]) * 0.5
    verts = np.concatenate(
        [
            p[:, :2] + np.column_stack((sx * c - sy * s, sx * s + sy * c))
            for sx, sy in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
        ],
        axis=0,
    )
    try:
        hull = verts[ConvexHull(verts).vertices]
    except Exception:
        return s0, p
    edges = np.roll(hull, -1, axis=0) - hull
    cuts = np.unique(
        np.r_[
            0.0,
            np.mod(0.5 * math.pi - np.arctan2(edges[:, 1], edges[:, 0]), 0.5 * math.pi),
            0.5 * math.pi,
        ]
    )
    cands = list(cuts)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mid = 0.5 * (lo + hi)
        co, si = math.cos(mid), math.sin(mid)
        xx = hull[:, 0] * co - hull[:, 1] * si
        yy = hull[:, 0] * si + hull[:, 1] * co
        dx = hull[int(xx.argmax())] - hull[int(xx.argmin())]
        dy = hull[int(yy.argmax())] - hull[int(yy.argmin())]
        A, B = dx[0], -dx[1]
        C, D = dy[1], dy[0]
        for a in (
            math.atan2(-(A - C), B - D),
            math.atan2(B, A),
            math.atan2(D, C),
        ):
            for k_pi in (-1, 0, 1):
                t = a + k_pi * math.pi
                if lo < t < hi:
                    cands.append(t)
    aa = np.array(cands, dtype=np.float64)
    co = np.cos(aa)
    si = np.sin(aa)
    xx = hull[:, 0, None] * co - hull[:, 1, None] * si
    yy = hull[:, 0, None] * si + hull[:, 1, None] * co
    sizes = np.maximum(np.ptp(xx, axis=0), np.ptp(yy, axis=0))
    best_i = int(sizes.argmin())
    if float(sizes[best_i]) >= s0 - 1e-10:
        return s0, p
    a_opt = float(aa[best_i])
    ca, sa = math.cos(a_opt), math.sin(a_opt)
    q = p.copy()
    q[:, 0] = p[:, 0] * ca - p[:, 1] * sa
    q[:, 1] = p[:, 0] * sa + p[:, 1] * ca
    q[:, 2] = p[:, 2] + a_opt
    s_rep, q_rep = btr.exact_repair_sat(q)
    if q_rep is not None and s_rep < s0 - 1e-10:
        return s_rep, q_rep
    return s0, p


def kink_contact_lp_polish(P_in, rounds=45, seed=0, max_seconds=18.0):
    """Kink-aware free-angle sequential sparse LP with exact nonlinear SAT admission."""
    s_cur, p = hull_rotate_exact(P_in)
    n = len(p)
    if n <= 1:
        return s_cur, p
    rng = np.random.default_rng(int(seed))
    t0 = time.perf_counter()
    step = 0.025

    for _it in range(rounds):
        if time.perf_counter() - t0 > max_seconds:
            break
        p = RE.normalize_to_origin(p)
        L = RE.side_exact(p)
        theta = p[:, 2]
        c = np.cos(theta)
        s = np.sin(theta)
        h = 0.5 * (np.abs(c) + np.abs(s))

        edges = cKDTree(p[:, :2]).query_pairs(1.75, output_type="ndarray")
        if len(edges) == 0:
            break
        i, j = edges.T
        d = p[j, :2] - p[i, :2]
        ux = np.column_stack((c[i], -s[i], c[j], -s[j]))
        uy = np.column_stack((s[i], c[i], s[j], c[j]))
        proj = d[:, 0, None] * ux + d[:, 1, None] * uy
        ranks = np.abs(proj)
        chosen = (ranks + rng.uniform(-2e-12, 2e-12, ranks.shape)).argmax(axis=1)
        rr = np.arange(len(i))
        sgn = np.sign(proj[rr, chosen])
        sgn[sgn == 0.0] = 1.0
        vx = sgn * ux[rr, chosen]
        vy = sgn * uy[rr, chosen]
        rotation = -d[:, 0] * vy + d[:, 1] * vx

        delta = theta[j] - theta[i]
        cd = np.cos(delta)
        sd = np.sin(delta)
        rad = 0.5 * (1.0 + np.abs(cd) + np.abs(sd))
        dr = 0.5 * (-sd * np.sign(cd) + cd * np.sign(sd))
        dist = np.abs(proj[rr, chosen])
        own = chosen < 2

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

        rows = []
        cols = []
        vals = []
        bounds = []
        offset = 0

        def add_block(cc, vv, bb):
            nonlocal offset
            bb = np.asarray(bb, dtype=np.float64)
            cnt = len(bb)
            if cnt == 0:
                return
            rows.append(
                np.repeat(np.arange(offset, offset + cnt, dtype=np.int32), cc.shape[1])
            )
            cols.append(cc.ravel().astype(np.int32))
            vals.append(vv.ravel().astype(np.float64))
            bounds.append(bb)
            offset += cnt

        add_block(columns, values, dist - rad - 4e-11)

        # Both one-sided radius subgradients at relative right-angle kinks!
        kink = np.flatnonzero(np.minimum(np.abs(cd), np.abs(sd)) < 1e-8)
        if len(kink) > 0:
            for slope in (-0.5, 0.5):
                vv = values[kink].copy()
                vv[:, 4] = -slope - np.where(own[kink], rotation[kink], 0.0)
                vv[:, 5] = slope - np.where(own[kink], 0.0, rotation[kink])
                add_block(columns[kink], vv, dist[kink] - rad[kink] - 4e-11)

        k_all = np.arange(n, dtype=np.int32)
        dh = 0.5 * (-s * np.sign(c) + c * np.sign(s))
        aligned = np.minimum(np.abs(c), np.abs(s)) < 1e-8
        extra = k_all[aligned]
        for dim in (0, 1):
            for upper in (False, True):
                for ids, deriv in (
                    (k_all, dh),
                    (extra, np.full(len(extra), -0.5)),
                    (extra, np.full(len(extra), 0.5)),
                ):
                    if len(ids) == 0:
                        continue
                    cc = np.column_stack(
                        (3 * ids + dim, 3 * ids + 2, np.full(len(ids), 3 * n))
                    )
                    vv = np.column_stack(
                        (
                            np.full(len(ids), 1.0 if upper else -1.0),
                            deriv,
                            np.full(len(ids), -1.0 if upper else 0.0),
                        )
                    )
                    bb = L - p[ids, dim] - h[ids] if upper else p[ids, dim] - h[ids]
                    add_block(cc, vv, bb)

        k3 = np.arange(3 * n, dtype=np.int32)
        u3 = 3 * n + 1 + k3
        for sgn_l1 in (-1.0, 1.0):
            add_block(
                np.column_stack((k3, u3)),
                np.tile([sgn_l1, -1.0], (3 * n, 1)),
                np.zeros(3 * n),
            )

        A = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(offset, 6 * n + 1),
        ).tocsr()
        rhs = np.concatenate(bounds)
        obj = np.zeros(6 * n + 1, dtype=np.float64)
        obj[3 * n] = 1.0
        obj[3 * n + 1 :] = 2e-6
        trust = np.tile([step, step, step * 0.4], n)
        limits = list(zip(-trust, trust)) + [(-step, step)] + [(0.0, None)] * (3 * n)

        try:
            fit = linprog(
                obj,
                A_ub=A,
                b_ub=rhs,
                bounds=limits,
                method="highs",
                options={
                    "primal_feasibility_tolerance": 1e-10,
                    "dual_feasibility_tolerance": 1e-9,
                    "time_limit": 2.0,
                },
            )
        except Exception:
            break
        if not fit.success or fit.x is None:
            step *= 0.4
            if step < 2e-7:
                break
            continue

        move = fit.x[: 3 * n].reshape(n, 3)
        winner = None
        w_side = L
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            s_tr, q_tr = btr.exact_repair_sat(p + alpha * move)
            if q_tr is not None and s_tr < w_side - 2e-10:
                w_side = s_tr
                winner = q_tr
        if winner is None:
            step *= 0.4
            if step < 2e-7:
                break
        else:
            p = winner
            s_cur = w_side
            step = min(0.06, step * 1.3)

    s_fin, p_fin = hull_rotate_exact(p)
    return s_fin, p_fin


def sweep_strip_and_l_lifts_for_n(n, cur):
    """Generate all valid `sq_engine` L-band & Single-Strip forward/reverse lifts landing on `n`!"""
    K = math.isqrt(n)
    r_k = int(round(math.sqrt(n)))
    cands = []

    # 1) Reverse L-Lift (`n + 2*r_k + 1 -> n`) & Forward L-Lift (`n - 2*m - 1 -> n`)
    for src_n, up_flag in (
        (n + 2 * r_k + 1, False),
        (n + 2 * K + 1, False),
        (n - 2 * max(1, r_k - 1) - 1, True),
        (n - 2 * max(1, K - 1) - 1, True),
    ):
        if 4 <= src_n <= 1000 and cur[src_n][1] is not None:
            try:
                P_src = E.to_origin(cur[src_n][1].copy())
                out = E.transplant(P_src, up=up_flag)
                if out is not None and len(out[0]) == n:
                    s_r, q_r = btr.exact_repair_sat(out[0])
                    if q_r is not None and math.isfinite(s_r):
                        tag = f"llift_{'up' if up_flag else 'dn'}_{src_n}"
                        cands.append((s_r, tag, q_r))
            except Exception:
                pass

    # 2) Reverse & Forward Single-Strip Lifts (`n +- K`, `n +- (K+1)` via `E.change_length` + `Compactor`)!
    seen_src = set()
    for src_n, add_flag in (
        (n + K, False),
        (n + K + 1, False),
        (n + r_k, False),
        (n - K, True),
        (n - K + 1, True),
        (n - r_k, True),
    ):
        if (
            (src_n, add_flag) in seen_src
            or not (4 <= src_n <= 1000)
            or cur[src_n][1] is None
        ):
            continue
        seen_src.add((src_n, add_flag))
        for ax in (0, 1):
            try:
                P_src = E.to_origin(cur[src_n][1].copy())
                ch = E.change_length(P_src, axis=ax, add=add_flag)
                if ch is not None and len(ch[0]) == n:
                    Q_st = E.to_origin(ch[0])
                    Q_c, S_c, ov = E.Compactor(Q_st).compact(
                        Q_st, max_iter=45, patience=2
                    )
                    if ov <= 5e-8:
                        s_r, q_r = btr.exact_repair_sat(Q_c)
                        if q_r is not None and math.isfinite(s_r):
                            tag = f"strip_{'add' if add_flag else 'del'}a{ax}_{src_n}"
                            cands.append((s_r, tag, q_r))
            except Exception:
                pass

    cands.sort(key=lambda z: z[0])
    return cands
