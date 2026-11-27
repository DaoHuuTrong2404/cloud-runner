#!/usr/bin/env python3
"""Wave 53S: Connected-Component Bar Splicer + Pocket-Void Relocator + Rust `harden reharden`
+ Near-Miss Re-Escalation -> 3-Round Deep-Polish `eta<=0.28` Wider-Hopper!

Fixes ALL 4 root causes discovered in Wave 52S forensics (`n=702`, `n=758` wins + `756 +9e-6` near-miss!):
  1. DESCENDING donor priority (`999 -> 160`, plus explicit top-gap & W52S near-miss donors FIRST:
     `976, 965, 929, 914, 849, 815, 813, 804, 793, 757, 744, 705, 632, 598, 510, 499, 461, 455, 417, 333, 298`)
     so high-N donors are NEVER starved by Phase-A timeout!
  2. Local-frame Connected-Component Bar Extractor (`extract_bars_cc`, `dv <= 0.32`) that cleanly decomposes
     100% of `965, 976, 914, 793, 632, 499, 455, 298` without merging adjacent curved/staggered bars!
  3. Exact Wall-Preserving `(-1.0, -1.0)` Cut + `relocate_crushed_to_voids` (Pocket Conservation Law:
     moves any `-0.94` step-transition colliding upright square into the vacated pocket, turning
     `632 -> 584/583` (`-0.0106` below R32!), `455 -> 414`, `298 -> 265 -> 234`, `965 -> 904 -> 845 -> 680`
     into clean `-0.01..-0.24` starts!) + BOTH `harden reharden` (soft-to-rigid unjamming) AND `squeeze2`!
  4. Fixes `w51d.compact_deleted` `(Sc, Qc)` 2-tuple unpacking so LP Compactor always refines quenched cuts!
"""

import argparse
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

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
TOOLS_DIR = os.path.abspath(os.path.join(HERE, "..", "tools"))
for _d in (DS_DIR, TOOLS_DIR, HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import rust_engines as RE
import squeeze2 as sq2
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w52s_staggered_band_quench_transplanter as w52s

PRIORITY_DONORS = [
    965,
    976,
    914,
    804,
    793,
    632,
    499,
    455,
    333,
    298,
    813,
    811,
    872,
    421,
    987,
    510,
    598,
    744,
    705,
    992,
    417,
    461,
    815,
    757,
    849,
    905,
    846,
    789,
    734,
    681,
    858,
    807,
    750,
    999,
    990,
    928,
    927,
    861,
    818,
    816,
    745,
    703,
    701,
]


def extract_bars_cc(P, idx_mask):
    """Decompose tilted squares `idx_mask` into 1-row parallel bars via local-frame connected components."""
    if len(idx_mask) < 8:
        return None
    ang = ((P[idx_mask, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
    pts = P[idx_mask]
    best = None
    for mode in (0, 1):
        m = len(idx_mask)
        adj = [[] for _ in range(m)]
        for i in range(m):
            ti = float(ang[i])
            ux, uy = (
                (math.cos(ti), math.sin(ti))
                if mode == 0
                else (-math.sin(ti), math.cos(ti))
            )
            vx, vy = -uy, ux
            for j in range(i + 1, m):
                if abs(float(ang[i]) - float(ang[j])) > 0.16:
                    continue
                dx = float(pts[j, 0] - pts[i, 0])
                dy = float(pts[j, 1] - pts[i, 1])
                if (
                    0.82 <= abs(dx * ux + dy * uy) <= 1.25
                    and abs(dx * vx + dy * vy) <= 0.32
                ):
                    adj[i].append(j)
                    adj[j].append(i)
        vis = [False] * m
        comps = []
        for i in range(m):
            if vis[i]:
                continue
            st = [i]
            vis[i] = True
            comp = []
            while st:
                u = st.pop()
                comp.append(int(idx_mask[u]))
                for w in adj[u]:
                    if not vis[w]:
                        vis[w] = True
                        st.append(w)
            comps.append(np.array(comp, dtype=int))
        big_s = [c for c in comps if len(c) >= 3]
        if len(big_s) >= 5 and max(len(c) for c in comps) <= 20:
            th_med = float(np.median(ang))
            nx, ny = (
                (-math.sin(th_med), math.cos(th_med))
                if mode == 0
                else (math.cos(th_med), math.sin(th_med))
            )
            big_s.sort(key=lambda c: float(P[c, 0].mean() * nx + P[c, 1].mean() * ny))
            cxs = [float(P[c, 0].mean()) for c in big_s]
            cys = [float(P[c, 1].mean()) for c in big_s]
            ax = 0 if (max(cxs) - min(cxs)) >= (max(cys) - min(cys)) else 1
            sc = sum(len(c) for c in big_s)
            if best is None or sc > best[0]:
                best = (sc, ax, th_med, big_s)
    if best is not None:
        return best[1:]
    return w52s.cluster_tilted_bars(P, idx_mask)


def worst_close_pairs(P, thr=-0.24):
    dx = P[:, 0][:, None] - P[:, 0][None, :]
    dy = P[:, 1][:, None] - P[:, 1][None, :]
    ii, jj = np.where(np.triu((dx * dx + dy * dy) < 2.2, 1))
    bad = []
    for i, j in zip(ii, jj):
        ddx, ddy = float(P[j, 0] - P[i, 0]), float(P[j, 1] - P[i, 1])
        best = -1e9
        for t0 in (
            float(P[i, 2]),
            float(P[i, 2]) + math.pi / 2.0,
            float(P[j, 2]),
            float(P[j, 2]) + math.pi / 2.0,
        ):
            ct, st = math.cos(t0), math.sin(t0)
            ri = 0.5 * (abs(math.cos(P[i, 2] - t0)) + abs(math.sin(P[i, 2] - t0)))
            rj = 0.5 * (abs(math.cos(P[j, 2] - t0)) + abs(math.sin(P[j, 2] - t0)))
            best = max(best, abs(ddx * ct + ddy * st) - ri - rj)
        if best < thr:
            bad.append((best, int(i), int(j)))
    bad.sort()
    return bad


def relocate_crushed_to_voids(Q_in):
    """Pocket Conservation Law: relocate any severely overlapping tip square (`gap < -0.25`) into
    the newly vacated pocket left by a staggered step transition!
    """
    Q = RE.normalize_to_origin(Q_in)
    S = RE.side_exact(Q)
    for _ in range(4):
        bad = worst_close_pairs(Q, thr=-0.25)
        if not bad:
            break
        _, i, j = bad[0]
        ai = abs(((Q[i, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0)
        aj = abs(((Q[j, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0)
        victim = i if ai <= aj else j
        other = np.delete(Q, victim, axis=0)
        h0 = 0.5 * (np.abs(np.cos(other[:, 2])) + np.abs(np.sin(other[:, 2])))
        gx, gy = np.meshgrid(
            np.linspace(0.51, S - 0.51, 68), np.linspace(0.51, S - 0.51, 68)
        )
        cand_pts = []
        for px, py in zip(gx.ravel(), gy.ravel()):
            sep = np.maximum(
                np.abs(other[:, 0] - px) - 0.5 - h0,
                np.abs(other[:, 1] - py) - 0.5 - h0,
            )
            cand_pts.append((float(sep.min()), float(px), float(py)))
        cand_pts.sort(reverse=True)
        best_v = (-1e9, 0.5, 0.5, 0.0)
        for _, px, py in cand_pts[:28]:
            nn = int(np.argmin((other[:, 0] - px) ** 2 + (other[:, 1] - py) ** 2))
            for ang in (0.0, float(other[nn, 2])):
                mgt = 1e9
                for k in range(len(other)):
                    ddx, ddy = px - float(other[k, 0]), py - float(other[k, 1])
                    if ddx * ddx + ddy * ddy > 2.2:
                        continue
                    b = -1e9
                    for t0 in (
                        ang,
                        ang + math.pi / 2.0,
                        float(other[k, 2]),
                        float(other[k, 2]) + math.pi / 2.0,
                    ):
                        ct, st = math.cos(t0), math.sin(t0)
                        ri = 0.5 * (abs(math.cos(ang - t0)) + abs(math.sin(ang - t0)))
                        rj = 0.5 * (
                            abs(math.cos(other[k, 2] - t0))
                            + abs(math.sin(other[k, 2] - t0))
                        )
                        b = max(b, abs(ddx * ct + ddy * st) - ri - rj)
                    if b < mgt:
                        mgt = b
                if mgt > best_v[0]:
                    best_v = (mgt, px, py, ang)
        Q[victim] = [best_v[1], best_v[2], best_v[3]]
    return Q


def exact_wall_cut_twoband(P_in, ip, ineg, kp, kn):
    """Exact (-1.0, -1.0) outer-wall-preserving two-band cut + pocket relocation."""
    ax_p, _, bars_p = ip
    ax_n, _, bars_n = ineg
    if ax_p == ax_n:
        return None
    a = E.to_origin(P_in.copy())
    bp = bars_p[kp]
    bn = bars_n[kn]
    cp = float(a[bp, ax_p].mean())
    cn = float(a[bn, ax_n].mean())
    p_lo, p_hi = float(a[bp, ax_n].min()) - 0.45, float(a[bp, ax_n].max()) + 0.45
    n_lo, n_hi = float(a[bn, ax_p].min()) - 0.45, float(a[bn, ax_p].max()) + 0.45

    drop = np.zeros(len(a), bool)
    drop[bp] = True
    drop[bn] = True
    in_bars = np.zeros(len(a), bool)
    in_bars[np.concatenate(bars_p)] = True
    in_bars[np.concatenate(bars_n)] = True
    for i in range(len(a)):
        if in_bars[i]:
            continue
        if abs(a[i, ax_p] - cp) < 0.52 and (a[i, ax_n] < p_lo or a[i, ax_n] > p_hi):
            drop[i] = True
        elif abs(a[i, ax_n] - cn) < 0.52 and (a[i, ax_p] < n_lo or a[i, ax_p] > n_hi):
            drop[i] = True
    Q = a[~drop].copy()
    kept_idx = np.where(~drop)[0]
    p_right = set(np.concatenate(bars_p[kp + 1 :]).tolist())
    p_left = set(np.concatenate(bars_p[:kp]).tolist())
    n_right = set(np.concatenate(bars_n[kn + 1 :]).tolist())
    n_left = set(np.concatenate(bars_n[:kn]).tolist())
    for qi, orig_i in enumerate(kept_idx):
        if orig_i in p_right:
            Q[qi, ax_p] -= 1.0
        elif orig_i not in p_left and a[orig_i, ax_p] > cp:
            Q[qi, ax_p] -= 1.0
        if orig_i in n_right:
            Q[qi, ax_n] -= 1.0
        elif orig_i not in n_left and a[orig_i, ax_n] > cn:
            Q[qi, ax_n] -= 1.0
    return relocate_crushed_to_voids(Q)


def heal_cut_hybrid(P_raw, target_L_ref, seed=1):
    """Dual Seam Healer: tries fast Numba `squeeze2` AND Rust `harden reharden` -> `squeeze2` + `Compactor`!"""
    if P_raw is None or not (80 <= len(P_raw) <= 1000):
        return None
    best = None

    def _finish_squeeze(P_cand, r_max_iter=5500):
        x0, y0, t0 = sq2.normalize(P_cand[:, 0], P_cand[:, 1], P_cand[:, 2])
        S_init = max(float(sq2.side(x0, y0, t0)), float(target_L_ref)) * 1.0015
        ok, rx, ry, rt, _ = sq2.relax_at(x0, y0, t0, S_init, max_iter=r_max_iter)
        if not ok:
            ok, rx, ry, rt, _ = sq2.relax_at(
                x0, y0, t0, S_init * 1.012, max_iter=r_max_iter + 1500
            )
            if not ok:
                return None
        _, x2, y2, t2 = sq2.squeeze(
            rx, ry, rt, rel0=2.5e-3, rel_min=5e-7, max_iter=4500, max_tries=42
        )
        Lf, fx, fy, ft, good = sq2.finalize(x2, y2, t2, tol=2e-12)
        if not good:
            return None
        Q = E.to_origin(np.column_stack([fx, fy, ft]))
        try:
            Sc, Qc = w51d.compact_deleted(Q, max_iter=35, patience=2)
            if Qc is not None and Sc < Lf:
                return Qc, float(Sc)
        except Exception:
            pass
        return Q, float(E.side(Q))

    # Path 1: Direct Numba `squeeze2`
    r1 = _finish_squeeze(P_raw, r_max_iter=5500)
    if r1 is not None:
        best = r1

    # Path 2: If Path 1 failed or landed > target_L_ref + 0.003, run Nakajima/Ostashev `harden reharden`!
    if best is None or best[1] > float(target_L_ref) + 0.003:
        rh = RE.run_harden_reharden(
            P_raw,
            s_ref=float(target_L_ref),
            seed=seed,
            r0=0.17,
            stiff0=0.30,
            pre=1600,
            hold=6500,
            release=3200,
            settle=2400,
            hold_f=1.001,
            noise=0.0045,
            timeout_s=15.0,
        )
        if rh is not None:
            r2 = _finish_squeeze(rh[1], r_max_iter=4500)
            if r2 is not None and (best is None or r2[1] < best[1]):
                best = r2

    return best


def generate_donor_splices(args_tuple):
    src_n, P_src, cur_L_map, base_seed = args_tuple
    P = E.to_origin(P_src)
    ang = ((P[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
    pos_idx = np.where(ang > 0.03)[0]
    neg_idx = np.where(ang < -0.03)[0]

    ip = extract_bars_cc(P, pos_idx)
    ineg = extract_bars_cc(P, neg_idx)

    out = []
    seen = set()

    def _reg(Q, S, tag):
        nn = len(Q)
        if not (80 <= nn <= 1000):
            return
        L_cur = cur_L_map.get(nn, float(math.ceil(math.sqrt(nn))))
        if S < L_cur + 0.014:
            k = (nn, round(S, 4))
            if k not in seen:
                seen.add(k)
                out.append((nn, float(S), tag, Q))

    # Mode A: Single-band / Staircase cuts (`965 -> 904 -> 845 -> 788 -> 733 -> 680`, `333 -> 298`, ...)
    if (ip is not None and len(neg_idx) < 12) or (
        ineg is not None and len(pos_idx) < 12
    ):
        info = ip if (ip is not None and len(neg_idx) < 12) else ineg
        stack_ax, _, bars = info
        # Test BOTH `len(b)` values and transitions (`sz=11`, `sz=12`, `sz=13` in `965`!)
        by_trans = {}
        for k in range(2, len(bars) - 2):
            key = (len(bars[k]), len(bars[k + 1]))
            by_trans.setdefault(key, []).append(k)
        reps = [ks[len(ks) // 2] for _, ks in sorted(by_trans.items())[:5]]
        best_q1 = None
        for idx_k, kr in enumerate(reps):
            P1_raw = w52s.cut_one_band_axis(P, bars, stack_ax, kr, n_bars_del=1)
            if P1_raw is not None:
                P1_rel = relocate_crushed_to_voids(P1_raw)
                nn1 = len(P1_rel)
                rh1 = heal_cut_hybrid(
                    P1_rel,
                    cur_L_map.get(nn1, math.sqrt(nn1) + 0.56),
                    seed=base_seed + idx_k,
                )
                if rh1 is not None:
                    _reg(rh1[0], rh1[1], f"sb1_{src_n}_b{len(bars[kr])}")
                    if best_q1 is None or rh1[1] < best_q1[1]:
                        best_q1 = rh1
        # Chain 2nd step down (`965 -> 904 -> 845`)
        if best_q1 is not None:
            Q1, _ = best_q1
            a1 = ((Q1[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
            m1 = (
                np.where(a1 > 0.03)[0] if len(neg_idx) < 12 else np.where(a1 < -0.03)[0]
            )
            inf1 = extract_bars_cc(Q1, m1)
            if inf1 is not None and len(inf1[2]) >= 7:
                P2_raw = w52s.cut_one_band_axis(
                    Q1, inf1[2], inf1[0], len(inf1[2]) // 2, n_bars_del=1
                )
                if P2_raw is not None:
                    P2_rel = relocate_crushed_to_voids(P2_raw)
                    nn2 = len(P2_rel)
                    rh2 = heal_cut_hybrid(
                        P2_rel,
                        cur_L_map.get(nn2, math.sqrt(nn2) + 0.56),
                        seed=base_seed + 41,
                    )
                    if rh2 is not None:
                        _reg(rh2[0], rh2[1], f"sb2_{src_n}")
        # 5-bar periodic cut (`965 -> 679/680`)
        if len(bars) >= 22:
            P5_raw = w52s.cut_one_band_axis(
                P, bars, stack_ax, len(bars) // 2, n_bars_del=5
            )
            if P5_raw is not None:
                P5_rel = relocate_crushed_to_voids(P5_raw)
                nn5 = len(P5_rel)
                rh5 = heal_cut_hybrid(
                    P5_rel,
                    cur_L_map.get(nn5, math.sqrt(nn5) + 0.56),
                    seed=base_seed + 77,
                )
                if rh5 is not None:
                    _reg(rh5[0], rh5[1], f"sb5_{src_n}")

    # Mode B: Orthogonal Two-Band Cuts (`exact_wall_cut_twoband` + `w52s.cut_two_band_staggered`)
    if ip is not None and ineg is not None:
        _, _, bars_p = ip
        _, _, bars_n = ineg
        sz_p, sz_n = {}, {}
        for k in range(2, len(bars_p) - 2):
            sz_p.setdefault((len(bars_p[k]), len(bars_p[k + 1])), []).append(k)
        for k in range(2, len(bars_n) - 2):
            sz_n.setdefault((len(bars_n[k]), len(bars_n[k + 1])), []).append(k)
        reps_p = [
            ks[len(ks) // 2]
            for _, ks in sorted(sz_p.items(), key=lambda kv: -len(kv[1]))[:2]
        ]
        reps_n = [
            ks[len(ks) // 2]
            for _, ks in sorted(sz_n.items(), key=lambda kv: -len(kv[1]))[:2]
        ]

        best_tb1 = None
        si = 0
        for kp in reps_p:
            for kn in reps_n:
                si += 1
                tag_sz = f"s{len(bars_p[kp])}x{len(bars_n[kn])}"
                # Try both exact wall-preserving cut (+relocate) AND mean-step cut!
                for mname, P_cut in (
                    ("tw", exact_wall_cut_twoband(P, ip, ineg, kp, kn)),
                    ("tb", w52s.cut_two_band_staggered(P, ip, ineg, kp, kn)),
                ):
                    if P_cut is None:
                        continue
                    nt = len(P_cut)
                    rh = heal_cut_hybrid(
                        P_cut,
                        cur_L_map.get(nt, math.sqrt(nt) + 0.7),
                        seed=base_seed + 100 + si * 7 + (0 if mname == "tw" else 1),
                    )
                    if rh is not None:
                        _reg(rh[0], rh[1], f"{mname}1_{src_n}_{tag_sz}")
                        if best_tb1 is None or rh[1] < best_tb1[1]:
                            best_tb1 = rh

        # Chain Step 2 (`298 -> 265 -> 234`, `632 -> 583 -> 536`!)
        if best_tb1 is not None:
            Q1, _ = best_tb1
            a1 = ((Q1[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
            ip1 = extract_bars_cc(Q1, np.where(a1 > 0.03)[0])
            in1 = extract_bars_cc(Q1, np.where(a1 < -0.03)[0])
            if (
                ip1 is not None
                and in1 is not None
                and len(ip1[2]) >= 5
                and len(in1[2]) >= 5
            ):
                kp2, kn2 = len(ip1[2]) // 2, len(in1[2]) // 2
                for mname, P_cut2 in (
                    ("tw2", exact_wall_cut_twoband(Q1, ip1, in1, kp2, kn2)),
                    ("tb2", w52s.cut_two_band_staggered(Q1, ip1, in1, kp2, kn2)),
                ):
                    if P_cut2 is None:
                        continue
                    nt2 = len(P_cut2)
                    rh2 = heal_cut_hybrid(
                        P_cut2,
                        cur_L_map.get(nt2, math.sqrt(nt2) + 0.7),
                        seed=base_seed + 211 + (0 if mname == "tw2" else 1),
                    )
                    if rh2 is not None:
                        _reg(rh2[0], rh2[1], f"{mname}_{src_n}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2220)
    ap.add_argument("--r1-sec", type=int, default=200)
    ap.add_argument("--r2-sec", type=int, default=180)
    ap.add_argument("--r3-sec", type=int, default=190)
    ap.add_argument("--seed-off", type=int, default=550000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    _dm = np.array(
        [[0.5, 0.5, 0.0], [1.6, 0.5, 0.1], [0.5, 1.6, 0.0]], dtype=np.float64
    )
    sq2.squeeze(_dm[:, 0], _dm[:, 1], _dm[:, 2], max_iter=20, max_tries=3)

    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )
    cur_L_map = {n: cur[n][0] for n in range(1, 1001)}

    # Priority donors FIRST + remaining donors DESCENDING (`999 -> 180`)!
    seen_d = set()
    donors = []
    for n in PRIORITY_DONORS + list(range(999, 179, -1)):
        if n in seen_d or n < 160 or n > 1000:
            continue
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        seen_d.add(n)
        donors.append(n)

    my_donors = donors[args.shard :: args.num_shards][:18]
    tasks = [
        (n, cur[n][1], cur_L_map, args.seed_off + args.shard * 991 + di * 37)
        for di, n in enumerate(my_donors)
    ]

    cands = []
    ex_gen = ProcessPoolExecutor(max_workers=args.procs)
    try:
        t_gen = time.time()
        futs = [ex_gen.submit(generate_donor_splices, tk) for tk in tasks]
        try:
            for f in as_completed(futs, timeout=360.0):
                try:
                    for nn, S, tag, Q in f.result():
                        gap = S - cur[nn][0]
                        if gap < 0.012:
                            cands.append((gap, nn, S, tag, Q))
                except Exception:
                    continue
                if time.time() - t_gen > 345.0:
                    break
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_gen)

    cands.sort(key=lambda z: z[0])
    seen_nt = {}
    queued = []
    for gap, nn, S, tag, Q in cands:
        cnt = seen_nt.get(nn, 0)
        if cnt >= 2:
            continue
        seen_nt[nn] = cnt + 1
        queued.append((gap, nn, S, tag, Q))
        if len(queued) >= 9:
            break

    print(
        f"[W53S-CCBarReharden s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_donors[:6]={my_donors[:6]} -> cands(gap<0.012)={len(cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
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

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, S_init, tag, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 50.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            if S_init < L0_cur + 0.00035:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    n, Q_init, deadline, slp_sec=38.0, fq_sec=26
                )
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        tot_ds += (Lf0 - L0_cur) / math.sqrt(n)
                        wins[n] = (float(Lf0), xf0, yf0, tf0)
                        L0_cur = float(Lf0)
                        print(
                            f"  [W53S DIRECT SPLICE WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf0:.7f}!",
                            flush=True,
                        )
                        save_ckpt()

            r1 = min(args.r1_sec, max(30, int(0.35 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(25, int(0.30 * (rem - 20.0))))
            r3 = min(args.r3_sec, max(25, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + 20000 + args.shard * 10007 + i * 601
            Qb, Sb, (tr, ft, wn) = w51d.deep_3round_widehopper(
                ex_hop, n, Q_init, L0_cur, bseed, r1, r2, r3, deadline
            )
            if Sb < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, Qb)
                if ok and Lf < L0_cur - 1e-9:
                    dL = Lf - cur[n][0]
                    tot_ds += (Lf - L0_cur) / math.sqrt(n)
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W53S WIN!!] n={n:4d} [{tag}_wh28] gap0={gap0:+.5f} tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W53S done] n={n:4d} [{tag}] gap0={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W53S s={args.shard} EXC] {exc}", flush=True)
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W53S s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
