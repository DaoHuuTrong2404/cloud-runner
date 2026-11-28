#!/usr/bin/env python3
"""Wave 79A (`w79a_large_trust_fixed_lp_dualwall_unlock_gav8.py`):
80-Core (`20` Shards x `2` Puzzles = `40` Wall-Locked & Neighbor-Step Puzzles!)
Iterative Large-Trust (`0.28` x `8` = `2.24` Units!) Fixed-Angle Sparse HiGHS LP Compaction
+ Dual-Wall Orthogonal Teleport Unlock (`cur[n+1] -> n` & `cur[n+2] -> n`) + `santa_ga` v8!

Why this breaks the wall-lock barrier where `sq_engine` small-trust (`0.015`) LP returned `0.00000`:
  1. When `theta` is fixed, `(x_j - x_i) v_x + (y_j - y_i) v_y >= rad(theta_i, theta_j)` is 100%
     EXACT linear geometry for arbitrary translation `trust = 0.28` (`0` Taylor angle error!).
     Iterating 7-8 rounds of sparse HiGHS `linprog` slides entire 25-square rows/columns by `1.0+`
     full unit into a deleted square's hole in `~0.08s`!
  2. Dual-Wall Orthogonal Teleport: when deleting square `i` from `cur[n + 1]` unlocks X (`x_span < y_span - 0.004`)
     while Y stays locked at `L_{n+1}` (or vice versa), we teleport a tight orthogonal-wall square `j`
     into the deepest Voronoi pocket created by `x_span`'s collapse and re-run Large-Trust Fixed-LP,
     simultaneously unlocking BOTH X and Y spans!
  3. Orthogonal `(i_x, i_y)` Delete-2 on `cur[n + 2]`: pairs X-wall/band squares with Y-wall/band
     squares so both spans unlock on step 1, then feeds the tightest unlocked cavities into
     Multi-Round `WideHopper` + 6-Island `santa_ga` v8 + `kink_contact_lp_polish`!
"""

import argparse
import json
import math
import os
import sys
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
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
TOOLS_DIR = os.path.abspath(os.path.join(HERE, "..", "tools"))
for _d in (DS_DIR, TOOLS_DIR, HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import kink_polish_and_strip_lifts_w76 as kps
import rust_engines as RE
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w63k_1puzzle_per_shard_killshot as w63k
import w73a_exhaustive_nearband_lp_compactor_hopper as w73a


def spans_xy(P):
    c = np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2]))
    h = 0.5 * c
    xlo, xhi = float((P[:, 0] - h).min()), float((P[:, 0] + h).max())
    ylo, yhi = float((P[:, 1] - h).min()), float((P[:, 1] + h).max())
    return xhi - xlo, yhi - ylo, xlo, ylo


def fixed_angle_lp_step(P_in, seed=0, trust=0.28):
    """Single exact fixed-angle sparse HiGHS LP step (`2n + 1` variables) with large translation trust!"""
    P = RE.normalize_to_origin(P_in)
    n = len(P)
    theta = P[:, 2]
    c = np.cos(theta)
    s = np.sin(theta)
    h = 0.5 * (np.abs(c) + np.abs(s))
    L = float(max((P[:, 0] + h).max(), (P[:, 1] + h).max()))
    rng = np.random.default_rng(seed)

    pairs = cKDTree(P[:, :2]).query_pairs(
        1.42 + 2.0 * trust * math.sqrt(2.0), output_type="ndarray"
    )
    rows = []
    cols = []
    vals = []
    rhs = []
    if len(pairs) > 0:
        i = pairs[:, 0]
        j = pairs[:, 1]
        d = P[j, :2] - P[i, :2]
        ux = np.column_stack([c[i], -s[i], c[j], -s[j]])
        uy = np.column_stack([s[i], c[i], s[j], c[j]])
        z = d[:, 0:1] * ux + d[:, 1:2] * uy
        k = (np.abs(z) + rng.uniform(-1e-11, 1e-11, size=z.shape)).argmax(axis=1)
        rr = np.arange(len(i))
        sgn = np.where(z[rr, k] >= 0.0, 1.0, -1.0)
        vx = sgn * ux[rr, k]
        vy = sgn * uy[rr, k]
        rad = (
            0.5
            * (
                1.0
                + np.abs(np.cos(theta[i] - theta[j]))
                + np.abs(np.sin(theta[i] - theta[j]))
            )
            + 5e-11
        )
        cc = np.column_stack([2 * i, 2 * i + 1, 2 * j, 2 * j + 1])
        vv = np.column_stack([vx, vy, -vx, -vy])
        rows.extend(np.repeat(rr, 4).tolist())
        cols.extend(cc.ravel().tolist())
        vals.extend(vv.ravel().tolist())
        rhs.extend((-rad).tolist())

    base_r = len(rhs)
    for a in range(n):
        ha = float(h[a])
        r0 = base_r + 2 * a
        rows.extend([r0, r0, r0 + 1, r0 + 1])
        cols.extend([2 * a, 2 * n, 2 * a + 1, 2 * n])
        vals.extend([1.0, -1.0, 1.0, -1.0])
        rhs.extend([-ha, -ha])

    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 2 * n + 1)).tocsr()
    obj = np.zeros(2 * n + 1, dtype=np.float64)
    obj[-1] = 1.0
    bounds = [
        (max(float(h[k_i]), float(P[k_i, dim]) - trust), float(P[k_i, dim]) + trust)
        for k_i in range(n)
        for dim in (0, 1)
    ] + [(math.sqrt(n), L + 1e-5)]

    res = linprog(
        obj,
        A_ub=A,
        b_ub=np.asarray(rhs, dtype=np.float64),
        bounds=bounds,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-9,
            "time_limit": 2.5,
        },
    )
    if not res.success or res.x is None:
        return L, P, False
    Q = P.copy()
    Q[:, :2] = res.x[:-1].reshape(n, 2)
    Q = RE.normalize_to_origin(Q)
    return RE.side_exact(Q), Q, True


def iter_fixed_angle_lp_compact(P_in, rounds=7, trust=0.28, seed=0):
    Q = RE.normalize_to_origin(P_in)
    S = RE.side_exact(Q)
    for r in range(rounds):
        tr = trust if r < rounds - 2 else 0.12
        Sn, Qn, ok = fixed_angle_lp_step(Q, seed=seed + 37 * r, trust=tr)
        if not ok:
            break
        imp = S - Sn
        Q, S = Qn, Sn
        if r >= 2 and imp < 1e-9:
            break
    return S, Q


def try_dual_wall_teleport(Q_in, seed=0):
    """If one axis shrank (`sx < sy - 0.003` or `sy < sx - 0.003`) while the orthogonal axis
    is still wall-locked, teleport one square touching the locked wall into the deepest interior
    Voronoi hole and re-compact!
    """
    Q = RE.normalize_to_origin(Q_in)
    sx, sy, _, _ = spans_xy(Q)
    if abs(sx - sy) < 0.003:
        return None
    n = len(Q)
    h = 0.5 * (np.abs(np.cos(Q[:, 2])) + np.abs(np.sin(Q[:, 2])))
    rng = np.random.default_rng(seed)
    if sy > sx:
        # Y is locked -> pick a square touching top or bottom Y boundary!
        wall_ids = np.where((Q[:, 1] - h < 0.015) | (sy - (Q[:, 1] + h) < 0.015))[0]
    else:
        # X is locked -> pick a square touching left or right X boundary!
        wall_ids = np.where((Q[:, 0] - h < 0.015) | (sx - (Q[:, 0] + h) < 0.015))[0]
    if len(wall_ids) == 0:
        return None

    L_box = max(sx, sy)
    probes = rng.uniform(0.52, max(0.55, L_box - 0.52), size=(512, 2))
    best_S = L_box
    best_Q = None
    picks = rng.choice(wall_ids, size=min(3, len(wall_ids)), replace=False)
    for j_id in picks:
        rem = np.delete(Q, int(j_id), axis=0)
        d2 = np.min(
            (probes[:, None, 0] - rem[None, :, 0]) ** 2
            + (probes[:, None, 1] - rem[None, :, 1]) ** 2,
            axis=1,
        )
        top_idx = int(np.argmax(d2))
        if d2[top_idx] < 0.82:
            continue
        px, py = float(probes[top_idx, 0]), float(probes[top_idx, 1])
        near = int(np.argmin((rem[:, 0] - px) ** 2 + (rem[:, 1] - py) ** 2))
        Qt = np.vstack([rem, [px, py, rem[near, 2]]])
        # Use quick `sq_engine` or Rust `fq` quench if slight overlap, then large-trust fixed LP!
        rq = RE.run_fq_quench_args(
            Qt,
            [
                "--loosen",
                "1.004",
                "--outer",
                "28",
                "--pit",
                "220",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "2.5",
            ],
            s_ref=L_box,
            timeout_s=4.0,
        )
        if rq is not None:
            St, Q_cmp = iter_fixed_angle_lp_compact(
                rq[1], rounds=6, trust=0.26, seed=seed + int(j_id)
            )
            if St < best_S - 1e-9:
                best_S, best_Q = St, Q_cmp
    return (best_S, best_Q) if best_Q is not None else None


def eval_large_trust_cavity_job(job):
    P_src, del_tuple, tag, seed = job
    try:
        Qd = np.delete(P_src, list(del_tuple), axis=0)
        S1, Q1 = iter_fixed_angle_lp_compact(Qd, rounds=7, trust=0.28, seed=seed)
        tp = try_dual_wall_teleport(Q1, seed=seed + 91)
        if tp is not None and tp[0] < S1 - 1e-9:
            S1, Q1 = tp[0], tp[1]
            tag = tag + "_tp"
        # Follow with 12-iter rotating+sliding `sq_engine` compaction!
        S2, Q2 = w51d.compact_deleted(Q1, max_iter=18, patience=2)
        if S2 < S1:
            return tag, float(S2), Q2
        return tag, float(S1), Q1
    except Exception:
        return tag, math.inf, None


def gen_orthogonal_del2_pairs(P2, max_pairs=24, seed=0):
    P = RE.normalize_to_origin(P2)
    h = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    sx, sy, _, _ = spans_xy(P)
    x_wall = np.where((P[:, 0] - h < 0.018) | (sx - (P[:, 0] + h) < 0.018))[0]
    y_wall = np.where((P[:, 1] - h < 0.018) | (sy - (P[:, 1] + h) < 0.018))[0]
    nb = set(w73a.select_nearband_indices(P, max_cands=90))
    xw = [int(i) for i in x_wall if int(i) in nb] or [int(i) for i in x_wall]
    yw = [int(i) for i in y_wall if int(i) in nb] or [int(i) for i in y_wall]
    rng = np.random.default_rng(seed)
    pairs = []
    seen = set()
    if xw and yw:
        for _ in range(max_pairs * 4):
            if len(pairs) >= max_pairs:
                break
            u = int(xw[rng.integers(0, len(xw))])
            v = int(yw[rng.integers(0, len(yw))])
            if u == v:
                continue
            pr = (min(u, v), max(u, v))
            if pr not in seen:
                seen.add(pr)
                pairs.append(pr)
    return pairs


def select_40_dualwall_targets(cur, front_L, num_targets=40):
    ranked = []
    for n in range(95, 998):
        K = math.isqrt(n)
        if K * K == n or cur[n][1] is None or cur[n + 1][1] is None:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        step1 = cur[n + 1][0] - cur[n][0]
        step2 = (
            (cur[n + 2][0] - cur[n][0])
            if (n + 2 <= 1000 and cur[n + 2][1] is not None)
            else 1.0
        )
        if gl > 2.5e-5 and (step1 < 0.036 or step2 < 0.042):
            bonus = 1.45 if step1 < 0.016 else (1.20 if step2 < 0.028 else 1.0)
            ranked.append(((gl / math.sqrt(n)) * bonus, n))
    ranked.sort(reverse=True)
    return [x[1] for x in ranked[:num_targets]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=4950000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    targets = select_40_dualwall_targets(cur, front_L, num_targets=2 * args.num_shards)
    my_two = targets[args.shard :: args.num_shards]

    print(
        f"[W79A-LargeTrustFixedLP-DualWall s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"TARGETS={my_two} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    deadline = t0 + args.budget_sec
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

    def record_win_n(n_w, Q_cand, S_cand, src_tag, w_cnt):
        nonlocal tot_ds
        L_ref = wins[n_w][0] if n_w in wins else cur[n_w][0]
        if S_cand >= L_ref - 1e-9:
            return
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n_w, Q_cand)
        if ok and Lf < L_ref - 1e-9:
            ds = (Lf - L_ref) / math.sqrt(n_w)
            tot_ds += ds
            wins[n_w] = (float(Lf), xf, yf, tf)
            cur[n_w] = (float(Lf), np.column_stack([xf, yf, tf]))
            print(
                f"  [W79A DUALWALL-LP WIN!!] n={n_w:4d} [{src_tag}] -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    for pi, n in enumerate(my_two):
        rem_all = deadline - time.time()
        if rem_all < 140.0:
            break
        puz_b = (rem_all - 15.0) / max(1, len(my_two) - pi)
        puz_dl = time.time() + puz_b
        bseed = args.seed_off + args.shard * 8191 + pi * 997

        L0 = cur[n][0]
        P0 = cur[n][1]
        P1 = RE.normalize_to_origin(cur[n + 1][1])

        # Build candidate deletions: edge squares (`edge < 0.02`) + max-void + near-band on `cur[n+1]`
        # PLUS orthogonal X-wall x Y-wall delete-2 pairs on `cur[n+2]`!
        t_scr = time.time()
        h1 = 0.5 * (np.abs(np.cos(P1[:, 2])) + np.abs(np.sin(P1[:, 2])))
        sx1, sy1, _, _ = spans_xy(P1)
        edge_ids = np.flatnonzero(
            (P1[:, 0] - h1 < 0.02)
            | (sx1 - (P1[:, 0] + h1) < 0.02)
            | (P1[:, 1] - h1 < 0.02)
            | (sy1 - (P1[:, 1] + h1) < 0.02)
        ).tolist()
        d_nn = cKDTree(P1[:, :2]).query(P1[:, :2], k=min(5, len(P1)))[0]
        void_ids = np.argsort(-d_nn[:, 1:].mean(axis=1))[:12].tolist()
        nb_ids = w73a.select_nearband_indices(P1, max_cands=64)
        cand_1 = list(dict.fromkeys(void_ids + edge_ids[:36] + nb_ids))[:84]

        jobs = [
            (P1, (idx,), f"lt1_{idx}", bseed + 10 + k_i)
            for k_i, idx in enumerate(cand_1)
        ]
        if n + 2 <= 1000 and cur[n + 2][1] is not None and cur[n + 2][0] - L0 < 0.042:
            P2 = RE.normalize_to_origin(cur[n + 2][1])
            orth_p = gen_orthogonal_del2_pairs(P2, max_pairs=20, seed=bseed + 301)
            nb_p = w51d.gen_del2_pairs(P2, max_pairs=10)
            all_p2 = list(dict.fromkeys(orth_p + nb_p))[:26]
            for k_i, (u, v) in enumerate(all_p2):
                jobs.append((P2, (u, v), f"lt2_{u}_{v}", bseed + 500 + k_i))

        cavs = []
        ex_scr = ProcessPoolExecutor(max_workers=args.procs)
        try:
            futs = [ex_scr.submit(eval_large_trust_cavity_job, jb) for jb in jobs]
            try:
                for f in as_completed(futs, timeout=54.0):
                    tg, Sc, Qc = f.result()
                    if Qc is not None and math.isfinite(Sc):
                        if Sc < cur[n][0] - 1e-9:
                            Qp_c, Sp_c = w51d.deep_contact_polish(
                                n, Qc, puz_dl, slp_sec=36.0, fq_sec=20
                            )
                            record_win_n(n, Qp_c, Sp_c, f"scr_{tg}", 0)
                            Qc, Sc = Qp_c, Sp_c
                        cavs.append((Sc, tg, Qc))
            except TimeoutError:
                pass
        finally:
            w51d.terminate_pool(ex_scr)

        cavs.sort(key=lambda z: z[0])
        if not cavs:
            cavs = [(L0, "own_fallback", P0)]

        S_c0, tg_c0, Q_c0 = cavs[0]
        ad_c0 = w53h.topology_angle_dist(Q_c0, P0)
        cav_pair = [(S_c0, ad_c0, tg_c0, Q_c0)]
        for S_ck, tg_ck, Q_ck in cavs[1:]:
            if w53h.topology_angle_dist(Q_ck, Q_c0) >= 0.06:
                cav_pair.append((S_ck, w53h.topology_angle_dist(Q_ck, P0), tg_ck, Q_ck))
                break
        if len(cav_pair) < 2 and len(cavs) >= 2:
            S_ck, tg_ck, Q_ck = cavs[1]
            cav_pair.append((S_ck, w53h.topology_angle_dist(Q_ck, P0), tg_ck, Q_ck))

        print(
            f"  [W79A lt-screen] n={n:4d} jobs={len(cavs)}/{len(jobs)} "
            f"cav0={tg_c0}({S_c0 - L0:+.5f},dA={ad_c0:.2f}) ({time.time() - t_scr:.1f}s)",
            flush=True,
        )

        # Phase 2: WideHopper on top 2 large-trust cavities -> `santa_ga` v8 -> Kink Polish!
        nbrs = []
        if n > 4 and math.isqrt(n - 1) ** 2 != (n - 1) and cur[n - 1][1] is not None:
            nbrs.append(cur[n - 1][1])

        ex_hop = ProcessPoolExecutor(max_workers=args.procs)
        try:
            for ci, (S_ci, ad_ci, tg_ci, Q_ci) in enumerate(cav_pair):
                rem_puz = puz_dl - time.time()
                if rem_puz < 160.0:
                    break
                hop_b = max(90.0, min(320.0 if ci == 0 else 190.0, 0.38 * rem_puz))
                L0_cur = cur[n][0]
                Qb_c, Sb_c, (_, _, wt_c, nr_c) = w63k.continuous_momentum_widehopper(
                    ex_hop,
                    n,
                    Q_ci,
                    L0_cur,
                    bseed + 40000 + ci * 7919,
                    hop_b,
                    deadline,
                    lambda Qx, Sx, stg, wc, _n=n, _tg=tg_ci: record_win_n(
                        _n, Qx, Sx, f"{_tg}->{stg}", wc
                    ),
                )
                Sk_c, Qk_c = kps.kink_contact_lp_polish(
                    Qb_c, rounds=32, seed=bseed + 211 + ci, max_seconds=10.0
                )
                if Sk_c < Sb_c:
                    Qb_c, Sb_c = Qk_c, Sk_c
                record_win_n(n, Qb_c, Sb_c, f"{tg_ci}->wh->kink", wt_c)
                if Sb_c < cur[n][0] + 0.016:
                    nbrs.append(Qb_c)
        finally:
            w51d.terminate_pool(ex_hop)

        rem_ga = puz_dl - time.time()
        if rem_ga >= 120.0:
            ga_sec = max(105.0, min(340.0, rem_ga - 25.0))
            L0_ga = cur[n][0]
            Qw, _, js = RE.run_santa_ga(
                cur[n][1],
                nbr_packings=nbrs,
                seed=bseed + 777,
                sec=ga_sec,
                islands=6,
                pop=14,
                offspring=28,
                steps=125,
                threads=args.procs,
                timeout_s=ga_sec + 55.0,
            )
            if Qw is not None:
                rq_w = RE.run_fq_quench_args(
                    Qw,
                    [
                        "--loosen",
                        "1.0003",
                        "--outer",
                        "25",
                        "--pit",
                        "450",
                        "--finish",
                        "--stag-w",
                        "0",
                        "--rmin",
                        "1e-12",
                        "--deadline",
                        "10.0",
                    ],
                    timeout_s=13.0,
                )
                if rq_w is not None:
                    Qp_w, Sp_w = w51d.deep_contact_polish(
                        n, rq_w[1], puz_dl, slp_sec=38.0, fq_sec=22
                    )
                    Sk_w, Qk_w = kps.kink_contact_lp_polish(
                        Qp_w, rounds=38, seed=bseed + 991, max_seconds=11.0
                    )
                    if Sk_w < Sp_w:
                        Qp_w, Sp_w = Qk_w, Sk_w
                    record_win_n(n, Qp_w, Sp_w, "lt_gav8_win", js.get("wins", 1))
            print(
                f"  [W79A done] n={n:4d} s_best={cur[n][0]:.7f} (dL={cur[n][0] - L0:+.6f})",
                flush=True,
            )

    save_ckpt()
    print(
        f"[W79A s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
