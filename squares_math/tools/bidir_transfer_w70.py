#!/usr/bin/env python3
"""Bidirectional Neighbour Transfer & 4-Corner Soft L-Peel Engine (`bidir_transfer_w70.py`):
1) `insert_holes_candidates`: Vectorized SAT-clearance Voronoi/lattice hole insertion (`n - 1 -> n`)
   + `target-hole insertion` at centres occupied by `cur[n]` farthest from `cur[n - 1]`.
2) `delete_one_candidates` + `compact_fixed_angle_lp`: Boundary (`edge < 0.028`) + sparse-interior
   (`cKDTree(k=5)`) `n + 1 -> n` deletions screened with sparse HiGHS fixed-angle LP + Rust `fq`.
3) `soft_corner_lpeel_candidates`: 4-corner `(K + c)^2 + D -> K^2 + D` sub-box cropping (`BL, BR, TL, TR`)
   healed via compiled Rust `harden reharden` + `fq quench --finish`.
"""

import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

import rust_engines as RE


def exact_repair_sat(P_in, margin=2.5e-12):
    """Ensure strict f64 pairwise SAT feasibility by tiny uniform dilation about `(0.5, 0.5)` if needed."""
    P = RE.normalize_to_origin(P_in)
    n = len(P)
    if n <= 1:
        return RE.side_exact(P), P
    c = np.cos(P[:, 2])
    s = np.sin(P[:, 2])
    tree = cKDTree(P[:, :2])
    pairs = tree.query_pairs(1.42, output_type="ndarray")
    if len(pairs) > 0:
        i = pairs[:, 0]
        j = pairs[:, 1]
        dx = P[j, 0] - P[i, 0]
        dy = P[j, 1] - P[i, 1]
        p1 = np.abs(dx * c[i] + dy * s[i])
        p2 = np.abs(-dx * s[i] + dy * c[i])
        p3 = np.abs(dx * c[j] + dy * s[j])
        p4 = np.abs(-dx * s[j] + dy * c[j])
        rad_ij = 0.5 * (
            1.0 + np.abs(np.cos(P[i, 2] - P[j, 2])) + np.abs(np.sin(P[i, 2] - P[j, 2]))
        )
        sep = np.maximum.reduce([p1 - rad_ij, p2 - rad_ij, p3 - rad_ij, p4 - rad_ij])
        min_sep = float(sep.min())
        if min_sep < margin:
            pen = margin - min_sep
            if pen > 0.02:
                return math.inf, None
            # Uniform dilation by (1 + 2.2 * pen) clears small linear LP boundary approximations
            dil = 1.0 + 2.5 * max(pen, 1e-12)
            P[:, :2] *= dil
            P = RE.normalize_to_origin(P)
            # Re-verify
            dx = P[j, 0] - P[i, 0]
            dy = P[j, 1] - P[i, 1]
            sep2 = np.maximum.reduce(
                [
                    np.abs(dx * c[i] + dy * s[i]) - rad_ij,
                    np.abs(-dx * s[i] + dy * c[i]) - rad_ij,
                    np.abs(dx * c[j] + dy * s[j]) - rad_ij,
                    np.abs(-dx * s[j] + dy * c[j]) - rad_ij,
                ]
            )
            if float(sep2.min()) < 0.0:
                return math.inf, None
    return RE.side_exact(P), P


def compact_fixed_angle_lp(P_in, seed=0, trust=0.26, time_limit=2.2):
    """Fast sparse HiGHS fixed-angle centre compaction (`~0.03-0.12s` for `n <= 1000`)."""
    P = RE.normalize_to_origin(P_in)
    n = len(P)
    L0 = RE.side_exact(P)
    if n <= 2:
        return L0, P
    rng = np.random.default_rng(int(seed))
    theta = P[:, 2]
    c = np.cos(theta)
    s = np.sin(theta)
    pairs = cKDTree(P[:, :2]).query_pairs(
        1.416 + 2.0 * trust * math.sqrt(2.0), output_type="ndarray"
    )
    if len(pairs) == 0:
        return L0, P
    i = pairs[:, 0]
    j = pairs[:, 1]
    d = P[j, :2] - P[i, :2]
    ux = np.column_stack([c[i], -s[i], c[j], -s[j]])
    uy = np.column_stack([s[i], c[i], s[j], c[j]])
    z = d[:, 0:1] * ux + d[:, 1:2] * uy
    k = (np.abs(z) + rng.uniform(-1e-11, 1e-11, z.shape)).argmax(axis=1)
    rr = np.arange(len(i))
    sgn = np.sign(z[rr, k])
    sgn[sgn == 0.0] = 1.0
    vx = sgn * ux[rr, k]
    vy = sgn * uy[rr, k]
    rad = (
        0.5
        * (
            1.0
            + np.abs(np.cos(theta[i] - theta[j]))
            + np.abs(np.sin(theta[i] - theta[j]))
        )
        + 6e-11
    )

    m_p = len(i)
    pair_rows = np.repeat(np.arange(m_p, dtype=np.int32), 4)
    pair_cols = np.column_stack([2 * i, 2 * i + 1, 2 * j, 2 * j + 1]).ravel()
    pair_vals = np.column_stack([vx, vy, -vx, -vy]).ravel()

    h = 0.5 * (np.abs(c) + np.abs(s))
    wall_cnt = 2 * n
    wall_rows = np.repeat(m_p + np.arange(wall_cnt, dtype=np.int32), 2)
    wall_cols = np.empty(2 * wall_cnt, dtype=np.int32)
    wall_cols[0::2] = np.arange(2 * n, dtype=np.int32)
    wall_cols[1::2] = 2 * n
    wall_vals = np.tile(np.array([1.0, -1.0], dtype=np.float64), wall_cnt)
    wall_rhs = -np.repeat(h, 2)

    rows = np.concatenate([pair_rows, wall_rows])
    cols = np.concatenate([pair_cols, wall_cols])
    vals = np.concatenate([pair_vals, wall_vals])
    rhs = np.concatenate([-rad, wall_rhs])

    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 2 * n + 1)).tocsr()
    obj = np.zeros(2 * n + 1, dtype=np.float64)
    obj[-1] = 1.0
    bounds = [
        (max(float(h[idx]), float(P[idx, dim] - trust)), float(P[idx, dim] + trust))
        for idx in range(n)
        for dim in (0, 1)
    ] + [(math.sqrt(n), L0 + 2e-5)]
    try:
        fit = linprog(
            obj,
            A_ub=A,
            b_ub=rhs,
            bounds=bounds,
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-10,
                "dual_feasibility_tolerance": 1e-9,
                "time_limit": float(time_limit),
            },
        )
    except Exception:
        return L0, P
    if not fit.success or fit.x is None:
        return L0, P
    Q = P.copy()
    Q[:, :2] = fit.x[:-1].reshape(n, 2)
    S_rep, Q_rep = exact_repair_sat(Q)
    if Q_rep is not None and S_rep < L0:
        return S_rep, Q_rep
    return L0, P


def insert_holes_candidates(P_nm1, P_n_own=None, seed=1, n_holes=4, n_target=3):
    """Generate `n - 1 -> n` hole-insertion + target-hole candidates (`malyshevdanil` `run_gpu.py` lines 22-62)."""
    p = RE.normalize_to_origin(P_nm1)
    L = RE.side_exact(p)
    rng = np.random.default_rng(int(seed))
    pos_rand = rng.uniform(0.5, max(0.55, L - 0.5), (1024, 2))
    ng = max(3, min(65, int(L * 2.0) + 1))
    grid = np.linspace(0.5, max(0.55, L - 0.5), ng)
    gx, gy = np.meshgrid(grid, grid)
    positions = np.vstack([pos_rand, np.column_stack([gx.ravel(), gy.ravel()])])
    nearest = cKDTree(p[:, :2]).query(positions)[0]
    positions = positions[np.argsort(-nearest)[:128]]

    tilts = np.minimum(
        np.mod(p[:, 2], 0.5 * math.pi), 0.5 * math.pi - np.mod(p[:, 2], 0.5 * math.pi)
    )
    til_idx = np.where(tilts > 0.012)[0]
    ang_til = (
        float(np.median(np.mod(p[til_idx, 2], 0.5 * math.pi)))
        if len(til_idx) > 0
        else float(p[int(rng.integers(len(p))), 2])
    )

    c = np.cos(p[:, 2])
    s = np.sin(p[:, 2])
    choices = []
    for angle in (0.0, 0.25 * math.pi, ang_til):
        ca = math.cos(angle)
        sa = math.sin(angle)
        d = positions[:, None, :] - p[None, :, :2]
        proj = np.maximum.reduce(
            [
                np.abs(d[:, :, 0] * c + d[:, :, 1] * s),
                np.abs(-d[:, :, 0] * s + d[:, :, 1] * c),
                np.abs(d[:, :, 0] * ca + d[:, :, 1] * sa),
                np.abs(-d[:, :, 0] * sa + d[:, :, 1] * ca),
            ]
        )
        rad = 0.5 * (
            1.0 + np.abs(np.cos(p[:, 2] - angle)) + np.abs(np.sin(p[:, 2] - angle))
        )
        gap = (proj - rad).min(axis=1)
        h = 0.5 * (abs(ca) + abs(sa))
        wall_gap = np.minimum(positions - h, L - positions - h).min(axis=1)
        gap = np.minimum(gap, wall_gap)
        for k in np.argsort(-gap)[:n_holes]:
            choices.append(
                (
                    float(gap[k]),
                    float(positions[k, 0]),
                    float(positions[k, 1]),
                    float(angle),
                )
            )

    choices.sort(key=lambda z: -z[0])
    picked_pts = []
    out = []
    for gval, px, py, ang in choices:
        if all((px - ox) ** 2 + (py - oy) ** 2 > 0.36 for ox, oy in picked_pts):
            picked_pts.append((px, py))
            out.append(
                (
                    f"hole_g{gval:+.3f}",
                    np.vstack([p, np.array([[px, py, ang]], dtype=np.float64)]),
                )
            )
        if len(out) >= n_holes:
            break

    if P_n_own is not None and len(P_n_own) == len(p) + 1:
        own = RE.normalize_to_origin(P_n_own)
        dist = cKDTree(p[:, :2]).query(own[:, :2])[0]
        for idx in np.argsort(-dist)[:n_target]:
            out.append((f"thole_{int(idx)}", np.vstack([p, own[idx : idx + 1]])))
    return out


def delete_one_candidates(P_np1, max_boundary=10, max_sparse=6):
    """Generate boundary + sparse-interior `n + 1 -> n` single-square deletions (`fixed_transfer.py`)."""
    a = RE.normalize_to_origin(P_np1)
    h = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    lo = (a[:, :2] - h[:, None]).min(axis=0)
    hi = (a[:, :2] + h[:, None]).max(axis=0)
    edge = np.minimum(a[:, :2] - h[:, None] - lo, hi - (a[:, :2] + h[:, None])).min(
        axis=1
    )
    b_ids = list(np.argsort(edge)[:max_boundary])
    k_nn = min(5, len(a))
    d = cKDTree(a[:, :2]).query(a[:, :2], k=k_nn)[0]
    s_ids = list(np.argsort(-d[:, 1:].mean(axis=1))[:max_sparse]) if k_nn > 1 else []
    ids = list(dict.fromkeys(s_ids + b_ids))
    return [(f"del1_{int(i)}", np.delete(a, int(i), axis=0)) for i in ids]


def soft_corner_lpeel_candidates(P_big, n_target):
    """Crop `n_target` squares into each of the 4 sub-corners (`BL, BR, TL, TR`) of `P_big` (`len > n_target`)."""
    P = RE.normalize_to_origin(P_big)
    if len(P) <= n_target:
        return []
    L = RE.side_exact(P)
    K_big = math.isqrt(len(P))
    K_tgt = math.isqrt(n_target)
    sub_w = max(float(K_tgt) + 0.5, L - float(max(1, K_big - K_tgt)))
    he = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    xlo = P[:, 0] - he
    xhi = P[:, 0] + he
    ylo = P[:, 1] - he
    yhi = P[:, 1] + he

    corners = [
        ("lpeel_BL", np.maximum(np.maximum(xhi - sub_w, yhi - sub_w), 0.0)),
        ("lpeel_BR", np.maximum(np.maximum((L - sub_w) - xlo, yhi - sub_w), 0.0)),
        ("lpeel_TL", np.maximum(np.maximum(xhi - sub_w, (L - sub_w) - ylo), 0.0)),
        ("lpeel_TR", np.maximum(np.maximum((L - sub_w) - xlo, (L - sub_w) - ylo), 0.0)),
    ]
    out = []
    seen_sets = set()
    for ctag, r_over in corners:
        keep = np.argsort(r_over, kind="stable")[:n_target]
        key = tuple(sorted(int(x) for x in keep))
        if key not in seen_sets:
            seen_sets.add(key)
            Q_sub = RE.normalize_to_origin(P[keep])
            out.append((ctag, Q_sub))
    return out


def eval_bidir_seed_job(job):
    """Evaluate one `('ins' | 'del_lp' | 'lpeel', tag, n, L0, P_cand, seed)` candidate in a worker process."""
    mode, tag, n, L0, P_cand, seed = job
    try:
        if mode == "ins":
            rq = RE.run_fq_quench_args(
                P_cand,
                [
                    "--seed",
                    str(int(seed)),
                    "--round-r0",
                    "0.085",
                    "--round-steps",
                    "12",
                    "--round-press",
                    "--round-press-outer",
                    "6",
                    "--round-kick",
                    "0.004",
                    "--round-kick-peak",
                    "0.05",
                    "--loosen",
                    "1.0035",
                    "--outer",
                    "42",
                    "--pit",
                    "250",
                    "--flip-top",
                    "18",
                    "--finish",
                    "--stag-w",
                    "0",
                    "--deadline",
                    "8.5",
                ],
                s_ref=L0,
                timeout_s=11.0,
            )
            if rq is None:
                return mode, tag, n, math.inf, None
            return mode, tag, n, float(rq[0]), rq[1]

        if mode == "lpeel":
            rh = RE.run_harden_reharden(
                P_cand,
                s_ref=L0,
                seed=int(seed),
                r0=0.14,
                stiff0=0.32,
                pre=1800,
                hold=6800,
                release=3200,
                settle=2400,
                hold_f=1.0006,
                noise=0.0045,
                timeout_s=10.5,
            )
            P_mid = rh[1] if rh is not None else P_cand
            rq = RE.run_fq_quench_args(
                P_mid,
                [
                    "--seed",
                    str(int(seed) + 11),
                    "--loosen",
                    "1.003",
                    "--outer",
                    "40",
                    "--pit",
                    "260",
                    "--flip-top",
                    "18",
                    "--finish",
                    "--stag-w",
                    "0",
                    "--deadline",
                    "8.0",
                ],
                s_ref=L0,
                timeout_s=10.5,
            )
            if rq is None:
                return mode, tag, n, math.inf, None
            return mode, tag, n, float(rq[0]), rq[1]

        if mode == "del_lp":
            # P_cand is a list of `(dtag, P_del)` for a single puzzle `n`: screen all via HiGHS LP, then `fq` top 2!
            best_lp = []
            for di, (dtag, P_del) in enumerate(P_cand):
                S_lp, Q_lp = compact_fixed_angle_lp(
                    P_del, seed=int(seed) + di, trust=0.26, time_limit=1.6
                )
                if Q_lp is not None and math.isfinite(S_lp):
                    best_lp.append((S_lp, dtag, Q_lp))
            best_lp.sort(key=lambda z: z[0])
            best_S = math.inf
            best_Q = None
            best_tag = tag
            for S_lp, dtag, Q_lp in best_lp[:2]:
                if S_lp < best_S:
                    best_S, best_Q, best_tag = S_lp, Q_lp, dtag + "_lp"
                if S_lp < L0 + 0.0065:
                    rq = RE.run_fq_quench_args(
                        Q_lp,
                        [
                            "--seed",
                            str(int(seed) + 31),
                            "--loosen",
                            "1.0015",
                            "--outer",
                            "34",
                            "--pit",
                            "260",
                            "--flip-top",
                            "18",
                            "--finish",
                            "--stag-w",
                            "0",
                            "--deadline",
                            "6.5",
                        ],
                        s_ref=min(L0, S_lp),
                        timeout_s=8.5,
                    )
                    if rq is not None and rq[0] < best_S:
                        best_S, best_Q, best_tag = float(rq[0]), rq[1], dtag + "_lp_fq"
            return mode, best_tag, n, best_S, best_Q

        if mode == "defsnap":
            rh = RE.run_harden_reharden(
                P_cand,
                s_ref=L0,
                seed=int(seed),
                r0=0.15,
                stiff0=0.30,
                pre=2000,
                hold=7200,
                release=3400,
                settle=2600,
                hold_f=1.0008,
                noise=0.0045,
                timeout_s=11.0,
            )
            P_mid = rh[1] if rh is not None else P_cand
            rq = RE.run_fq_quench_args(
                P_mid,
                [
                    "--seed",
                    str(int(seed) + 19),
                    "--round-r0",
                    "0.075",
                    "--round-steps",
                    "10",
                    "--round-press",
                    "--round-press-outer",
                    "5",
                    "--loosen",
                    "1.0035",
                    "--outer",
                    "42",
                    "--pit",
                    "260",
                    "--flip-top",
                    "18",
                    "--finish",
                    "--stag-w",
                    "0",
                    "--deadline",
                    "8.5",
                ],
                s_ref=L0,
                timeout_s=11.0,
            )
            if rq is None:
                return mode, tag, n, math.inf, None
            return mode, tag, n, float(rq[0]), rq[1]
    except Exception:
        return mode, tag, n, math.inf, None
    return mode, tag, n, math.inf, None


def defect_snap_candidates(P_in):
    """Build Defect-Snapped & Band-Coalesced seeds replicating the `n=333` (`-0.007765`) unjamming mechanism!"""
    P = RE.normalize_to_origin(P_in)
    L = RE.side_exact(P)
    t_mod = np.mod(P[:, 2], 0.5 * math.pi)
    fold = np.minimum(t_mod, 0.5 * math.pi - t_mod)
    out = []
    for deg_cut in (5.8, 9.5, 12.5):
        mask = (fold > math.radians(0.15)) & (fold < math.radians(deg_cut))
        cnt = int(mask.sum())
        if cnt > 0:
            Q = P.copy()
            Q[mask, 2] = 0.0
            gx = np.clip(np.round(Q[mask, 0] - 0.5) + 0.5, 0.5, max(0.55, L - 0.5))
            gy = np.clip(np.round(Q[mask, 1] - 0.5) + 0.5, 0.5, max(0.55, L - 0.5))
            Q[mask, 0] = 0.80 * Q[mask, 0] + 0.20 * gx
            Q[mask, 1] = 0.80 * Q[mask, 1] + 0.20 * gy
            out.append((f"snap_{deg_cut:.1f}d_c{cnt}", Q))

    # Also build a band-coalesced seed snapping scattered tilted angles within 8 deg of median tilt
    hi_mask = fold >= math.radians(10.5)
    if int(hi_mask.sum()) >= 6:
        med_t = float(np.median(t_mod[hi_mask]))
        da = np.minimum(
            np.mod(t_mod - med_t, 0.5 * math.pi),
            0.5 * math.pi - np.mod(t_mod - med_t, 0.5 * math.pi),
        )
        c_mask = hi_mask & (da > math.radians(0.8)) & (da < math.radians(8.5))
        if int(c_mask.sum()) > 0:
            Q2 = P.copy()
            Q2[c_mask, 2] = med_t
            out.append((f"coal_{int(c_mask.sum())}", Q2))
    return out
