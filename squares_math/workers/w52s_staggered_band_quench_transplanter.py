#!/usr/bin/env python3
"""Wave 52S: Staggered-Band Normal-Cluster Splicer + Numba `squeeze2` Seam Quench
-> 3-Round Deep-Polish `eta<=0.28` Wider-Hopper!

Breaks the exact geometric wall proved by Wave 50-51 forensics:
  - Elite packings (`965, 793, 804, 914, 976, 632, 849, 499, 500, 455, 333, 298`) use STAGGERED
    tilted bands (`[5, 5, 6, 6]`, `[8, 8, 9, 9]`, `[11, 11, 12, 12, 12]`), so `E.band_summary`'s
    `np.ptp(size[bars]) == 0` and `min_gap >= -1e-8` gates rejected 100% of them!
  - Now:
    1. Clusters tilted squares by bar-normal projection (`u_cos` vs `u_sin`), immune to staggered
       bar lengths and Hopper angle jitter!
    2. Mode A (Single-Band / Staircase 1-Bar, 2-Bar & 5-Bar Cuts — `965 -> 903/904 -> 844/845 -> ...`,
       `679/680`, `730`, `530`, `401`): cuts interior bars of EVERY distinct bar size (`sz=11` vs `sz=12`!),
       peels outer upright row(s) + shifts upright columns, and quenches `-0.02..-0.15` seam overlaps
       to `E < 1e-26` in `<0.4s` via Numba `squeeze2.squeeze`, plus generates `+1` void-insert and `-1`
       delete-1 neighbors so EVERY target residue `d` (`904`, `680`, `738`, `456`, `234`, `491`, `367`) is hit!
    3. Mode B (Orthogonal Two-Band Staggered `(bar_pos, bar_neg)` Cuts — `793 -> 738`, `499 -> 456`,
       `455 -> 414`, `298 -> 265 -> 234`, `632 -> 583 -> 536 -> 491 -> 448 -> 407 -> 367`):
       Tests representative interior bars of EACH bar size in `pos` x EACH bar size in `neg` + upright
       row/col slice -> Numba `squeeze2.squeeze` seam heal -> 2-step chain + `+-1` square neighbors!
    4. Phase B (clean isolated pool): Hunts top `< +0.012` fresh topologies with 3-Round Deep-Polish
       `eta<=0.28` Wider-Hopper (`w51d.deep_3round_widehopper`)!
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

import squeeze2 as sq2
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r


def cluster_tilted_bars(P, idx_mask):
    """Cluster tilted square indices `idx_mask` into parallel bars via optimal normal projection."""
    if len(idx_mask) < 8:
        return None
    ang = ((P[idx_mask, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
    th = float(np.median(ang))
    pts = P[idx_mask]
    u_cos = pts[:, 0] * math.cos(th) + pts[:, 1] * math.sin(th)
    u_sin = -pts[:, 0] * math.sin(th) + pts[:, 1] * math.cos(th)
    for pname, proj in (("u_cos", u_cos), ("u_sin", u_sin)):
        ord_p = np.argsort(proj)
        ps = proj[ord_p]
        sp = np.where(np.diff(ps) > 0.30)[0]
        st = np.concatenate([[0], sp + 1])
        en = np.concatenate([sp + 1, [len(ps)]])
        bars = [idx_mask[ord_p[st[k] : en[k]]] for k in range(len(st))]
        if 6 <= len(bars) <= 36 and max(len(b) for b in bars) <= 20:
            # Determine primary stacking axis (0=x if bars step along x, 1=y if bars step along y)
            cxs = [float(P[b, 0].mean()) for b in bars]
            cys = [float(P[b, 1].mean()) for b in bars]
            stack_ax = 0 if (max(cxs) - min(cxs)) >= (max(cys) - min(cys)) else 1
            return stack_ax, th, bars
    return None


def fast_quench_candidate(P_raw, target_L_ref):
    """Heal up to ~0.25 seam overlap via Numba L-BFGS `squeeze2.squeeze` (`<0.4s`!)."""
    if len(P_raw) < 80 or len(P_raw) > 1000:
        return None
    x0, y0, t0 = sq2.normalize(P_raw[:, 0], P_raw[:, 1], P_raw[:, 2])
    S_init = max(float(sq2.side(x0, y0, t0)), float(target_L_ref)) * 1.002
    ok, rx, ry, rt, _ = sq2.relax_at(x0, y0, t0, S_init, max_iter=6000)
    if not ok:
        ok, rx, ry, rt, _ = sq2.relax_at(x0, y0, t0, S_init * 1.012, max_iter=8000)
        if not ok:
            return None
    S_sq, x2, y2, t2 = sq2.squeeze(
        rx, ry, rt, rel0=3e-3, rel_min=5e-7, max_iter=5000, max_tries=45
    )
    Lf, fx, fy, ft, good = sq2.finalize(x2, y2, t2, tol=2e-12)
    if not good:
        return None
    Q = E.to_origin(np.column_stack([fx, fy, ft]))
    # Quick LP Compactor pass to settle upright grid contacts
    try:
        Qc, Sc, ov = w51d.compact_deleted(Q, max_iter=35, patience=2)
        if Qc is not None and Sc < Lf:
            return Qc, float(Sc)
    except Exception:
        pass
    return Q, float(E.side(Q))


def quick_insert_one(P_base, L_ref):
    """Insert 1 square at best Voronoi void in `P_base` and quench via `squeeze2`!"""
    P = E.to_origin(P_base)
    S = float(E.side(P))
    gx, gy = np.meshgrid(np.linspace(0.6, S - 0.6, 28), np.linspace(0.6, S - 0.6, 28))
    c0, s0 = np.cos(P[:, 2]), np.sin(P[:, 2])
    h0 = 0.5 * (np.abs(c0) + np.abs(s0))
    best_v = (-1e9, S * 0.5, S * 0.5, 0.0)
    for px, py in zip(gx.ravel(), gy.ravel()):
        dx = np.abs(P[:, 0] - px) - 0.5 - h0
        dy = np.abs(P[:, 1] - py) - 0.5 - h0
        sep = np.maximum(dx, dy)
        msep = float(sep.min())
        if msep > best_v[0]:
            # Also try matching angle of nearest tilted neighbor!
            nn_i = int(np.argmin((P[:, 0] - px) ** 2 + (P[:, 1] - py) ** 2))
            best_v = (msep, float(px), float(py), float(P[nn_i, 2]))
    P_ins = np.vstack([P, [[best_v[1], best_v[2], best_v[3]]]])
    return fast_quench_candidate(P_ins, L_ref)


def cut_one_band_axis(P_in, bars, stack_ax, k_rem, n_bars_del=1):
    """Cut `n_bars_del` bars starting at `k_rem` along `stack_ax`, peel `n_bars_del` outer upright strips on `1-stack_ax`,
    and shift `stack_ax` upright columns!
    """
    other_ax = 1 - stack_ax
    a = E.to_origin(P_in.copy())
    h = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    L = float(max((a[:, 0] + h).max(), (a[:, 1] + h).max()))

    all_bar_idx = np.concatenate(bars)
    bar_mask = np.zeros(len(a), bool)
    bar_mask[all_bar_idx] = True

    b_first = bars[k_rem]
    b_after = bars[k_rem + n_bars_del]
    d_stack = float(a[b_after, stack_ax].mean() - a[b_first, stack_ax].mean())
    d_other = float(a[b_after, other_ax].mean() - a[b_first, other_ax].mean())

    T_L = a[np.concatenate(bars[:k_rem])].copy()
    T_R = a[np.concatenate(bars[k_rem + n_bars_del :])].copy()
    T_R[:, stack_ax] -= d_stack
    if abs(d_other) < 0.25 * n_bars_del:
        T_R[:, other_ax] -= d_other

    up_all = a[~bar_mask].copy()
    # Choose outer wall (high vs low on `other_ax`) with more pure upright squares to peel `n_bars_del` rows!
    hi_cnt = int((up_all[:, other_ax] > L - n_bars_del).sum())
    lo_cnt = int((up_all[:, other_ax] < float(n_bars_del)).sum())
    if hi_cnt >= lo_cnt:
        up_crop = up_all[up_all[:, other_ax] <= L - n_bars_del].copy()
    else:
        up_crop = up_all[up_all[:, other_ax] >= float(n_bars_del)].copy()
        up_crop[:, other_ax] -= float(n_bars_del)
        T_L[:, other_ax] -= float(n_bars_del)
        T_R[:, other_ax] -= float(n_bars_del)

    # In each 1-unit bin along `other_ax`, outside b_first's `other_ax` span delete `n_bars_del` upright squares & shift!
    o_lo = float(a[b_first, other_ax].min()) - 0.45
    o_hi = float(a[b_first, other_ax].max()) + 0.45
    s_cut = float(a[b_first, stack_ax].mean())
    kept_up = []
    max_bin = int(math.ceil(L)) + 1
    for ob in range(max_bin):
        row = up_crop[
            (up_crop[:, other_ax] >= ob) & (up_crop[:, other_ax] < ob + 1.0)
        ].copy()
        if len(row) == 0:
            continue
        row = row[np.argsort(row[:, stack_ax])]
        omid = float(row[:, other_ax].mean())
        if o_lo <= omid <= o_hi:
            rL = row[row[:, stack_ax] < s_cut].copy()
            rR = row[row[:, stack_ax] >= s_cut].copy()
            rR[:, stack_ax] -= float(n_bars_del)
            if len(rL):
                kept_up.append(rL)
            if len(rR):
                kept_up.append(rR)
        else:
            if len(row) <= n_bars_del:
                continue
            dists = np.abs(row[:, stack_ax] - s_cut)
            del_idxs = np.argsort(dists)[:n_bars_del]
            s_max_del = float(row[del_idxs, stack_ax].max())
            keep_m = np.ones(len(row), bool)
            keep_m[del_idxs] = False
            rem_r = row[keep_m].copy()
            rem_r[rem_r[:, stack_ax] > s_max_del, stack_ax] -= float(n_bars_del)
            kept_up.append(rem_r)
    if not kept_up:
        return None
    return np.vstack([T_L, T_R, np.vstack(kept_up)])


def cut_two_band_staggered(P_in, info_pos, info_neg, kp, kn):
    """Cut 1 interior bar `kp` from `pos` band AND 1 interior bar `kn` from `neg` band + 1 upright row & col!"""
    ax_p, _, bars_p = info_pos
    ax_n, _, bars_n = info_neg
    if ax_p == ax_n:
        return None
    a = E.to_origin(P_in.copy())
    bp_rem, bp_nxt = bars_p[kp], bars_p[kp + 1]
    bn_rem, bn_nxt = bars_n[kn], bars_n[kn + 1]

    dp_s = float(a[bp_nxt, ax_p].mean() - a[bp_rem, ax_p].mean())
    dn_s = float(a[bn_nxt, ax_n].mean() - a[bn_rem, ax_n].mean())
    cp_cut = float(a[bp_rem, ax_p].mean())
    cn_cut = float(a[bn_rem, ax_n].mean())

    Tp_L = a[np.concatenate(bars_p[:kp])].copy()
    Tp_R = a[np.concatenate(bars_p[kp + 1 :])].copy()
    Tp_R[:, ax_p] -= dp_s
    # Also shift Tp if it lies above cn_cut on ax_n!
    Tp_L[Tp_L[:, ax_n] > cn_cut + 0.5, ax_n] -= 1.0
    Tp_R[Tp_R[:, ax_n] > cn_cut + 0.5, ax_n] -= 1.0

    Tn_L = a[np.concatenate(bars_n[:kn])].copy()
    Tn_R = a[np.concatenate(bars_n[kn + 1 :])].copy()
    Tn_R[:, ax_n] -= dn_s
    Tn_L[Tn_L[:, ax_p] > cp_cut + 0.5, ax_p] -= 1.0
    Tn_R[Tn_R[:, ax_p] > cp_cut + 0.5, ax_p] -= 1.0

    t_mask = np.zeros(len(a), bool)
    t_mask[np.concatenate(bars_p)] = True
    t_mask[np.concatenate(bars_n)] = True
    up = a[~t_mask].copy()

    # Delete 1 strip along ax_p near cp_cut (outside bp_rem's span on ax_n) AND 1 strip along ax_n near cn_cut!
    # Cleanest lattice operation: remove upright squares in [cp_cut - 0.52, cp_cut + 0.52] (outside bp_rem)
    # and in [cn_cut - 0.52, cn_cut + 0.52] (outside bn_rem), then shift > cp_cut by -1 on ax_p and > cn_cut by -1 on ax_n!
    p_o_lo = float(a[bp_rem, ax_n].min()) - 0.45
    p_o_hi = float(a[bp_rem, ax_n].max()) + 0.45
    n_o_lo = float(a[bn_rem, ax_p].min()) - 0.45
    n_o_hi = float(a[bn_rem, ax_p].max()) + 0.45

    in_p_cut = (np.abs(up[:, ax_p] - cp_cut) < 0.52) & (
        (up[:, ax_n] < p_o_lo) | (up[:, ax_n] > p_o_hi)
    )
    up1 = up[~in_p_cut].copy()
    up1[up1[:, ax_p] > cp_cut, ax_p] -= 1.0

    cn_cut_adj = cn_cut - (1.0 if cn_cut > cp_cut else 0.0)
    n_o_lo_adj = n_o_lo - (1.0 if n_o_lo > cp_cut else 0.0)
    n_o_hi_adj = n_o_hi - (1.0 if n_o_hi > cp_cut else 0.0)
    in_n_cut = (np.abs(up1[:, ax_n] - cn_cut_adj) < 0.52) & (
        (up1[:, ax_p] < n_o_lo_adj) | (up1[:, ax_p] > n_o_hi_adj)
    )
    up2 = up1[~in_n_cut].copy()
    up2[up2[:, ax_n] > cn_cut_adj, ax_n] -= 1.0

    return np.vstack([Tp_L, Tp_R, Tn_L, Tn_R, up2])


def generate_staggered_transplants(args_tuple):
    src_n, P_src, cur_L_map = args_tuple
    P = E.to_origin(P_src)
    ang = ((P[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
    pos_idx = np.where(ang > 0.03)[0]
    neg_idx = np.where(ang < -0.03)[0]

    info_pos = cluster_tilted_bars(P, pos_idx)
    info_neg = cluster_tilted_bars(P, neg_idx)

    out = []
    seen = set()

    def _register(Q, S, tag):
        nn = len(Q)
        if not (80 <= nn <= 1000):
            return
        L_cur = cur_L_map.get(nn, float(math.ceil(math.sqrt(nn))))
        if S < L_cur + 0.015:
            k = (nn, round(S, 4))
            if k not in seen:
                seen.add(k)
                out.append((nn, float(S), tag, Q))

    def _quench_and_neighbors(P_raw, base_tag):
        if P_raw is None or not (80 <= len(P_raw) <= 1000):
            return None
        nn0 = len(P_raw)
        res = fast_quench_candidate(P_raw, cur_L_map.get(nn0, math.sqrt(nn0) + 0.6))
        if res is None:
            return None
        Qc, Sc = res
        _register(Qc, Sc, base_tag)
        # Also test +1 void insertion (e.g. 903 -> 904, 679 -> 680!) & -1 deletion if Sc is competitive!
        L0_n = cur_L_map.get(nn0, Sc)
        if Sc < L0_n + 0.020:
            if nn0 + 1 <= 1000:
                ri = quick_insert_one(Qc, cur_L_map.get(nn0 + 1, Sc))
                if ri is not None:
                    _register(ri[0], ri[1], base_tag + "_ins1")
            if nn0 - 1 >= 80:
                cands = w32a.pick_near_band(Qc, max_cand=6)
                for idx in cands[:2]:
                    Qd, Sd = w32a.fast_delete_and_compact(Qc, idx)
                    if Qd is not None:
                        _register(Qd.astype(np.float64), float(Sd), base_tag + "_del1")
        return Qc, Sc

    # Mode A: Single-band / Staircase cuts (e.g. 965 -> 903/904, 679/680; 333 -> 298 -> 265 -> 234!)
    if (info_pos is not None and len(neg_idx) < 12) or (
        info_neg is not None and len(pos_idx) < 12
    ):
        info = info_pos if (info_pos is not None and len(neg_idx) < 12) else info_neg
        stack_ax, _, bars = info
        # Pick up to 2 representative interior bars for EACH distinct bar size!
        by_sz = {}
        for k in range(3, len(bars) - 3):
            by_sz.setdefault(len(bars[k]), []).append(k)
        reps = []
        for sz, ks in sorted(by_sz.items(), key=lambda kv: -len(kv[1])):
            reps.append(ks[len(ks) // 2])
            if len(ks) >= 4:
                reps.append(ks[-2])
        for kr in reps[:4]:
            P1 = cut_one_band_axis(P, bars, stack_ax, kr, n_bars_del=1)
            r1 = _quench_and_neighbors(P1, f"sb1_{src_n}_k{kr}")
            # Chain 2nd step down (e.g. 965 -> 904 -> 845!)
            if r1 is not None:
                Q1, _ = r1
                ang1 = ((Q1[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
                m1 = (
                    np.where(ang1 > 0.03)[0]
                    if len(neg_idx) < 12
                    else np.where(ang1 < -0.03)[0]
                )
                inf1 = cluster_tilted_bars(Q1, m1)
                if inf1 is not None and len(inf1[2]) >= 8:
                    mid_k = len(inf1[2]) // 2
                    P2 = cut_one_band_axis(Q1, inf1[2], inf1[0], mid_k, n_bars_del=1)
                    _quench_and_neighbors(P2, f"sb2_{src_n}")
        # Also test 5-bar periodic cut if >= 22 bars (965 -> 679/680!)
        if len(bars) >= 22:
            mid5 = len(bars) // 2
            P5 = cut_one_band_axis(P, bars, stack_ax, mid5, n_bars_del=5)
            _quench_and_neighbors(P5, f"sb5_{src_n}")

    # Mode B: Orthogonal Two-Band Staggered Cuts (`793->738`, `499->456`, `455->414`, `298->265->234`, `632->583->491`!)
    if info_pos is not None and info_neg is not None:
        _, _, bars_p = info_pos
        _, _, bars_n = info_neg
        sz_p, sz_n = {}, {}
        for k in range(2, len(bars_p) - 2):
            sz_p.setdefault(len(bars_p[k]), []).append(k)
        for k in range(2, len(bars_n) - 2):
            sz_n.setdefault(len(bars_n[k]), []).append(k)
        reps_p = [
            ks[len(ks) // 2]
            for _, ks in sorted(sz_p.items(), key=lambda kv: -len(kv[1]))[:2]
        ]
        reps_n = [
            ks[len(ks) // 2]
            for _, ks in sorted(sz_n.items(), key=lambda kv: -len(kv[1]))[:2]
        ]

        best_step1 = None
        for kp in reps_p:
            for kn in reps_n:
                P_tb = cut_two_band_staggered(P, info_pos, info_neg, kp, kn)
                r_tb = _quench_and_neighbors(
                    P_tb, f"tb1_{src_n}_s{len(bars_p[kp])}x{len(bars_n[kn])}"
                )
                if r_tb is not None and (best_step1 is None or r_tb[1] < best_step1[1]):
                    best_step1 = r_tb

        # Chain 2nd step (`298 -> 265 -> 234`, `632 -> 583 -> 536`!)
        if best_step1 is not None:
            Q1, _ = best_step1
            ang1 = ((Q1[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
            ip1 = cluster_tilted_bars(Q1, np.where(ang1 > 0.03)[0])
            in1 = cluster_tilted_bars(Q1, np.where(ang1 < -0.03)[0])
            if (
                ip1 is not None
                and in1 is not None
                and len(ip1[2]) >= 6
                and len(in1[2]) >= 6
            ):
                P_tb2 = cut_two_band_staggered(
                    Q1, ip1, in1, len(ip1[2]) // 2, len(in1[2]) // 2
                )
                _quench_and_neighbors(P_tb2, f"tb2_{src_n}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--r1-sec", type=int, default=180)
    ap.add_argument("--r2-sec", type=int, default=165)
    ap.add_argument("--r3-sec", type=int, default=175)
    ap.add_argument("--seed-off", type=int, default=530000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    # Warm squeeze2 Numba kernels once before forking!
    _dm = np.array(
        [[0.5, 0.5, 0.0], [1.6, 0.5, 0.1], [0.5, 1.6, 0.0]], dtype=np.float64
    )
    sq2.squeeze(_dm[:, 0], _dm[:, 1], _dm[:, 2], max_iter=20, max_tries=3)

    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )
    cur_L_map = {n: cur[n][0] for n in range(1, 1001)}

    donors = []
    for n in range(160, 1001):
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        donors.append(n)
    my_donors = donors[args.shard :: args.num_shards]
    tasks = [(n, cur[n][1], cur_L_map) for n in my_donors]

    cands = []
    ex_gen = ProcessPoolExecutor(max_workers=args.procs)
    try:
        t_gen = time.time()
        futs = [ex_gen.submit(generate_staggered_transplants, tk) for tk in tasks]
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
        if len(queued) >= 10:
            break

    print(
        f"[W52S-StaggeredQuench s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"donors={len(my_donors)} -> cands(gap<0.012)={len(cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
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
            if rem < 45.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            if S_init < L0_cur + 0.00030:
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
                            f"  [W52S DIRECT QUENCH WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf0:.7f}!",
                            flush=True,
                        )
                        save_ckpt()

            r1 = min(args.r1_sec, max(30, int(0.35 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(25, int(0.30 * (rem - 20.0))))
            r3 = min(args.r3_sec, max(25, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + args.shard * 10007 + i * 601
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
                        f"  [W52S WIN!!] n={n:4d} [{tag}_wh28] gap0={gap0:+.5f} tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W52S done] n={n:4d} [{tag}] gap0={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W52S s={args.shard} EXC] {exc}", flush=True)
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W52S s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
