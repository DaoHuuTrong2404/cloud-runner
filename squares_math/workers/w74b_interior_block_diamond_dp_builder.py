#!/usr/bin/env python3
"""Wave 74B (`w74b_interior_block_diamond_dp_builder.py`):
80-Core (`20` Shards) Interior `a x b` Rotated Block + 45-deg Diamond DP Row-Split Constructor
(`block_scan` / `block_build` / `diamond_scan2` / `diamond_build` in `@njit` Numba!)
-> Rust `fq` Active-Set Quench -> Angle-Pinned `santa_ga` v6 -> Continuous WideHopper!

Why this is a genuinely orthogonal basin constructor:
  - Boundary L-gadget DP (`fdp8`) ONLY places tilted squares along the right/top margin, leaving
    wrong-basin puzzles (`367` tilted=90 vs `368` tilted=168, `491` tilted=101 vs `492` tilted=147,
    `738` tilted=120 vs `739` tilted=222, `456` tilted=60 vs `457` tilted=167, `234` tilted=36)
    unreachable!
  - Here, we scan interior `a x b` blocks at both 45 deg (`diamond_scan2`) and general angles
    `theta in [14..42 deg]` (`block_scan` with `E=12` DP row-offset optimization), build the exact
    `N`-square geometry (preferentially pruning surplus upright squares right at the block's corners
    so `fq` / `santa_ga` v6 can squeeze the cavity!), quench via Rust `fq` + `kkt_sqp_polish_fq`,
    and run `santa_ga` v6 + `continuous_momentum_widehopper`!
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
from numba import njit

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
TOOLS_DIR = os.path.abspath(os.path.join(HERE, "..", "tools"))
for _d in (DS_DIR, TOOLS_DIR, HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import dp8_cache_w68 as pdc
import fault_dp_8sym as fdp8
import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w56g_unified_global_faultdp as w56g
import w63k_1puzzle_per_shard_killshot as w63k


@njit(cache=False)
def ib_verts(px, py, a, b, c, s):
    vx = np.empty(4, np.float64)
    vy = np.empty(4, np.float64)
    vx[0] = px
    vy[0] = py
    vx[1] = px + a * c
    vy[1] = py + a * s
    vx[2] = px + a * c - b * s
    vy[2] = py + a * s + b * c
    vx[3] = px - b * s
    vy[3] = py + b * c
    return vx, vy


@njit(cache=False)
def ib_band_extent(vx, vy, y0, y1):
    lo = 1e18
    hi = -1e18
    hit = False
    for i in range(4):
        if vy[i] >= y0 and vy[i] <= y1:
            if vx[i] < lo:
                lo = vx[i]
            if vx[i] > hi:
                hi = vx[i]
            hit = True
        j = (i + 1) & 3
        for k_c in range(2):
            yc = y0 if k_c == 0 else y1
            if (vy[i] - yc) * (vy[j] - yc) < 0.0:
                t = (yc - vy[i]) / (vy[j] - vy[i])
                xc = vx[i] + t * (vx[j] - vx[i])
                if xc < lo:
                    lo = xc
                if xc > hi:
                    hi = xc
                hit = True
    return hit, lo, hi


@njit(cache=False)
def ib_rows_dp(S, K, W, vx, vy, E):
    f = S - K
    NEG = -1000000000
    cnt = np.empty((K, E + 1), np.int64)
    for j in range(K):
        for q in range(E + 1):
            y0 = j + f * q / E
            hit, lo, hi = ib_band_extent(vx, vy, y0, y0 + 1.0)
            if not hit:
                cnt[j, q] = int(math.floor(W + 1e-12))
            elif lo < -1e-12 or hi > W + 1e-12:
                cnt[j, q] = NEG
            else:
                cnt[j, q] = int(math.floor(lo + 1e-12)) + int(
                    math.floor(W - hi + 1e-12)
                )
    best = cnt[0].copy()
    for j in range(1, K):
        run = NEG
        nb = np.empty(E + 1, np.int64)
        for q in range(E + 1):
            if best[q] > run:
                run = best[q]
            nb[q] = run + cnt[j, q]
        best = nb
    mx = NEG
    for q in range(E + 1):
        if best[q] > mx:
            mx = best[q]
    return mx


@njit(cache=False)
def ib_best_count_dp(S, K, a, b, th, cols, steps, E):
    c = math.cos(th)
    s = math.sin(th)
    W = S - cols
    bw = a * c + b * s
    bh = a * s + b * c
    wx = W - bw
    wy = S - bh
    if wx < 0.0 or wy < 0.0:
        return -1, 0.0, 0.0
    best = -1
    bpx = 0.0
    bpy = 0.0
    for i in range(steps + 1):
        px = b * s + wx * i / steps
        for k in range(steps + 1):
            py = wy * k / steps
            vx, vy = ib_verts(px, py, a, b, c, s)
            v = ib_rows_dp(S, K, W, vx, vy, E)
            if v > best:
                best = v
                bpx = px
                bpy = py
    return best + a * b + cols * K, bpx, bpy


def build_interior_block_packing(S, K, a, b, th, cols, px, py, n, E=12, prune_mode=0):
    """Build exact N-square interior `a x b` rotated block + DP-offset rows + `cols` right columns."""
    c, s = math.cos(th), math.sin(th)
    vx, vy = ib_verts(px, py, a, b, c, s)
    W = S - cols
    f = S - K
    cnt = np.zeros((K, E + 1), dtype=int)
    seg = {}
    for j in range(K):
        for q in range(E + 1):
            y0 = j + f * q / E
            hit, lo, hi = ib_band_extent(vx, vy, y0, y0 + 1.0)
            if not hit:
                cnt[j, q] = int(math.floor(W + 1e-12))
                seg[j, q] = (None, None)
            elif lo < -1e-12 or hi > W + 1e-12:
                cnt[j, q] = -(10**9)
                seg[j, q] = (None, None)
            else:
                cnt[j, q] = int(math.floor(lo + 1e-12)) + int(
                    math.floor(W - hi + 1e-12)
                )
                seg[j, q] = (lo, hi)

    best = cnt[0].astype(float)
    arg = []
    for j in range(1, K):
        run = -1e18
        ri = 0
        nb = np.zeros(E + 1)
        ar = np.zeros(E + 1, dtype=int)
        for q in range(E + 1):
            if best[q] > run:
                run = best[q]
                ri = q
            nb[q] = run + cnt[j, q]
            ar[q] = ri
        arg.append(ar)
        best = nb
    q = int(np.argmax(best))
    qs = [q]
    for j in range(K - 1, 0, -1):
        q = int(arg[j - 1][q])
        qs.append(q)
    qs = qs[::-1]

    ux, uy = c, s
    vx_d, vy_d = -s, c
    blk_rows = []
    for i in range(a):
        for j in range(b):
            bx = px + (i + 0.5) * ux + (j + 0.5) * vx_d
            by = py + (i + 0.5) * uy + (j + 0.5) * vy_d
            blk_rows.append((bx, by, th % (math.pi / 2)))

    up_rows = []
    for cc in range(cols):
        for j in range(K):
            yy = 0.5 + j * (S - 1.0) / max(1, K - 1) if K > 1 else 0.5 * S
            up_rows.append((W + cc + 0.5, yy, 0.0))
    for j in range(K):
        y0 = j + f * qs[j] / E
        lo, hi = seg[j, qs[j]]
        if lo is None:
            for i in range(int(math.floor(W + 1e-12))):
                up_rows.append((i + 0.5, y0 + 0.5, 0.0))
        else:
            for i in range(int(math.floor(lo + 1e-12))):
                up_rows.append((i + 0.5, y0 + 0.5, 0.0))
            for i in range(int(math.floor(W - hi + 1e-12))):
                up_rows.append((W - i - 0.5, y0 + 0.5, 0.0))

    tot_m = len(blk_rows) + len(up_rows)
    if tot_m < n:
        return None
    drop = tot_m - n
    if drop > 0 and len(up_rows) >= drop:
        if prune_mode == 1 and blk_rows:
            # Drop surplus upright squares closest to the rotated block boundary so L-BFGS/fq squeezes inward!
            bcx = sum(r[0] for r in blk_rows) / len(blk_rows)
            bcy = sum(r[1] for r in blk_rows) / len(blk_rows)
            up_scored = sorted(
                up_rows,
                key=lambda r: (r[0] - bcx) ** 2 + (r[1] - bcy) ** 2,
                reverse=True,
            )
            up_rows = up_scored[: len(up_rows) - drop]
        else:
            up_rows = up_rows[: len(up_rows) - drop]

    all_sq = (blk_rows + up_rows)[:n]
    if len(all_sq) != n:
        return None
    return np.array(all_sq, dtype=np.float64)


def scan_interior_block_designs(n, L0, max_designs=8):
    """Scan both 45-deg diamonds and general-angle `a x b` interior blocks whose coarse DP side
    is within `min(K + 0.999, L0 + 0.26)` (pre-squeeze — before `fq`/L-BFGS slides upright rows into
    the rotated block's diagonal wedges!).
    """
    K = math.isqrt(n)
    d = n - K * K
    if K < 7 or d <= 0:
        return []
    f_cur = L0 - K
    hi_cap = min(0.999, max(0.76, f_cur + 0.26))
    lo_cap = max(0.38, f_cur - 0.030)
    if hi_cap <= lo_cap + 1e-4:
        return []

    ang_deg = [45.0, 26.565, 21.801, 30.964, 18.435, 33.690, 24.0, 36.870]
    cands = []
    a_step = 1 if K <= 15 else 2
    area_lo = max(6, n // 8)
    area_hi = max(area_lo + 4, int(0.58 * n))

    for deg in ang_deg:
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        steps_s = 12
        E_dp = 8
        for cols in (0, 1, 2):
            W_hi = K + hi_cap - cols
            S_hi = K + hi_cap
            a_min = max(3, K // 4)
            a_max = min(int(1.40 * K), 2 * K)
            for a in range(a_min, a_max + 1, a_step):
                b_lo = max(2, area_lo // a)
                b_hi = min(a, area_hi // a)
                b_step = 1 if (b_hi - b_lo) <= 6 else 2
                for b in range(b_lo, b_hi + 1, b_step):
                    ab_list = (
                        [(a, b)]
                        if (a == b or abs(deg - 45.0) < 0.1)
                        else [(a, b), (b, a)]
                    )
                    for aa, bb in ab_list:
                        if (
                            aa * c + bb * s > W_hi - 0.05
                            or aa * s + bb * c > S_hi - 0.05
                        ):
                            continue
                        cnt_hi, _, _ = ib_best_count_dp(
                            S_hi, K, aa, bb, th, cols, steps_s, E_dp
                        )
                        if cnt_hi < n:
                            continue
                        lo_s, hi_s = K + lo_cap, S_hi
                        for _ in range(9):
                            mid_s = 0.5 * (lo_s + hi_s)
                            if (
                                ib_best_count_dp(
                                    mid_s, K, aa, bb, th, cols, steps_s, E_dp
                                )[0]
                                >= n
                            ):
                                hi_s = mid_s
                            else:
                                lo_s = mid_s
                        cands.append((hi_s, aa, bb, th, deg, cols))

    cands.sort(key=lambda z: z[0])
    # Deduplicate by (aa, bb, round(deg,1), cols) and build fine coordinates (cap to top 24 coarse candidates)!
    built = []
    seen_cfg = set()
    for s_coarse, aa, bb, th, deg, cols in cands[:32]:
        sig = (aa, bb, round(deg, 1), cols)
        if sig in seen_cfg:
            continue
        seen_cfg.add(sig)
        cnt_f, bpx, bpy = ib_best_count_dp(s_coarse + 2e-3, K, aa, bb, th, cols, 36, 12)
        if cnt_f < n:
            continue
        for pm in (0, 1):
            P_blk = build_interior_block_packing(
                s_coarse + 2e-3, K, aa, bb, th, cols, bpx, bpy, n, E=12, prune_mode=pm
            )
            if P_blk is not None:
                tag = f"ib_{aa}x{bb}_t{deg:.1f}_c{cols}_p{pm}"
                built.append((s_coarse, tag, P_blk))
                break
        if len(built) >= max_designs:
            break
    return built


def eval_quench_ib_job(job):
    tag, n, L0, P_ib, seed = job
    try:
        rq = RE.run_fq_quench_args(
            P_ib,
            [
                "--seed",
                str(seed),
                "--loosen",
                "1.008",
                "--outer",
                "55",
                "--pit",
                "340",
                "--flip-top",
                "22",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "13.0",
            ],
            s_ref=L0,
            timeout_s=16.0,
        )
        if rq is None:
            return tag, 1e9, None
        Sp, Qp = RE.kkt_sqp_polish_fq(rq[1], timeout_each=5.5)
        return tag, float(Sp), Qp
    except Exception:
        return tag, 1e9, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--dp-cache-npz", default="")
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=3850000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    # Warm @njit functions once
    _ = ib_best_count_dp(8.6, 8, 3, 3, 0.45, 0, 4, 4)

    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    dp_ctx = None
    if args.dp_cache_npz and os.path.exists(args.dp_cache_npz):
        try:
            dp_ctx = pdc.load_dp8_cache(args.dp_cache_npz)
        except Exception:
            dp_ctx = None

    ranked = []
    for n in range(95, 999):
        K = math.isqrt(n)
        if K * K == n or cur[n][1] is None:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gl > 2.5e-5:
            ranked.append((gl / math.sqrt(n), n))
    ranked.sort(reverse=True)
    my_two = [x[1] for x in ranked[: 2 * args.num_shards]][
        args.shard :: args.num_shards
    ]

    print(
        f"[W74B-InteriorBlockDP s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"dp_cache={dp_ctx is not None} TARGETS={my_two} ({time.time() - t0:.1f}s)",
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
            print(
                f"  [W74B INTERIOR-BLOCK WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_two):
            rem_all = deadline - time.time()
            if rem_all < 130.0:
                break
            puz_b = (rem_all - 15.0) / max(1, len(my_two) - pi)
            puz_dl = time.time() + puz_b
            bseed = args.seed_off + args.shard * 8191 + pi * 997

            L0 = wins[n][0] if n in wins else cur[n][0]
            P0 = cur[n][1]

            # Stage 1 (~35s): Scan & Quench Interior Rotated Block / Diamond Designs!
            t_ib = time.time()
            ib_designs = scan_interior_block_designs(n, L0, max_designs=8)
            quenched_ib = []
            if ib_designs:
                q_jobs = [
                    (tg, n, L0, P_ib, bseed + 100 + ji)
                    for ji, (_, tg, P_ib) in enumerate(ib_designs)
                ]
                futs_q = [ex_hop.submit(eval_quench_ib_job, jb) for jb in q_jobs]
                try:
                    for f in as_completed(futs_q, timeout=38.0):
                        tg, Sq, Qq = f.result()
                        if Qq is not None and math.isfinite(Sq):
                            if Sq < L0 + 0.0008:
                                Qp_i, Sp_i = w51d.deep_contact_polish(
                                    n, Qq, puz_dl, slp_sec=36.0, fq_sec=20
                                )
                                record_win_n(n, Qp_i, Sp_i, tg, 0)
                                Qq, Sq = Qp_i, Sp_i
                            quenched_ib.append((Sq, tg, Qq))
                except TimeoutError:
                    w51d.terminate_pool(ex_hop)
                    ex_hop = ProcessPoolExecutor(max_workers=args.procs)

            quenched_ib.sort(key=lambda z: z[0])
            best_ib_str = (
                f"{quenched_ib[0][1]}({quenched_ib[0][0] - L0:+.5f})"
                if quenched_ib
                else "none"
            )
            print(
                f"  [W74B ib-scan] n={n:4d} designs={len(ib_designs)} quenched={len(quenched_ib)} best={best_ib_str} ({time.time() - t_ib:.1f}s)",
                flush=True,
            )

            # Choose primary start: if an interior block quenched within L0 + 0.014, invade that pure interior-block basin!
            nbrs = [z[2] for z in quenched_ib[:4]]
            K = math.isqrt(n)
            D = n - K * K
            if dp_ctx is not None and K <= fdp8.WMAX and 0 < D <= fdp8.DMAX:
                donors_c, donor_n_c, G_c, dp_c, PG_c, PM_c, PQ_c = dp_ctx
                for L_est, dtag, P_comp in fdp8.trace_and_compose_v2(
                    dp_c, PG_c, PM_c, PQ_c, G_c, donors_c, donor_n_c, K, D, top_m=2
                ):
                    if L_est < L0 + 0.016:
                        rh = w56g.quench_composed_chain(P_comp, L0)
                        if rh is not None and rh[1] < L0 + 0.014:
                            nbrs.append(rh[0])

            nbrs.append(P0)
            if n + 1 <= 1000 and cur[n + 1][1] is not None:
                nbrs.append(cur[n + 1][1])

            if quenched_ib and quenched_ib[0][0] < L0 + 0.014:
                S_prim, tg_prim, P_prim = quenched_ib[0]
            else:
                P_prim = (
                    np.column_stack([wins[n][1], wins[n][2], wins[n][3]])
                    if n in wins
                    else P0
                )
                S_prim = wins[n][0] if n in wins else L0
                tg_prim = "cur_own_ibnbr"

            rem_puz = max(140.0, puz_dl - time.time())
            ga_sec = min(460.0 if pi == 0 else 360.0, max(120.0, 0.56 * rem_puz))
            Qw, Qc, js = RE.run_santa_ga(
                P_prim,
                nbr_packings=nbrs,
                seed=bseed,
                sec=ga_sec,
                islands=6,
                pop=14,
                offspring=28,
                steps=125,
                threads=args.procs,
                timeout_s=ga_sec + 35.0,
            )
            print(
                f"  [W74B GAv6 done] n={n:4d} [{tg_prim}] gens={js.get('gens', 0)} ga_wins={js.get('wins', 0)} s_in={S_prim:.7f} -> s_best={js.get('s_best', S_prim):.7f} (L0={L0:.7f})",
                flush=True,
            )

            hop_P = P_prim
            hop_tag = tg_prim
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
                    timeout_s=12.5,
                )
                if rq_w is not None:
                    Qp_w, Sp_w = w51d.deep_contact_polish(
                        n, rq_w[1], puz_dl, slp_sec=42.0, fq_sec=24
                    )
                    record_win_n(n, Qp_w, Sp_w, f"{tg_prim}->gav6", js.get("wins", 1))
                    hop_P = Qp_w
                    hop_tag = f"{tg_prim}->gav6"

            L0_cur = wins[n][0] if n in wins else cur[n][0]
            hop_b = max(90.0, min(deadline - time.time() - 14.0, puz_dl - time.time()))
            Qb, Sb, (_, _, wt, nr) = w63k.continuous_momentum_widehopper(
                ex_hop,
                n,
                hop_P,
                L0_cur,
                bseed + 80000,
                hop_b,
                deadline,
                lambda Qx, Sx, stg, wc, _n=n, _tg=hop_tag: record_win_n(
                    _n, Qx, Sx, f"{_tg}->{stg}", wc
                ),
            )
            record_win_n(n, Qb, Sb, f"{hop_tag}->whFin", wt)
            ad_end = w53h.topology_angle_dist(Qb, P0)
            print(
                f"  [W74B done] n={n:4d} [{hop_tag}] dA={ad_end:.2f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W74B s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
