#!/usr/bin/env python3
"""Wave 75B (`w75b_k2p1_and_gobel_staircase_forge.py`):
80-Core (`20` Shards, 1-Puzzle-Per-Shard Laser Focus = `7,200` Core-Seconds / Puzzle!)
`K^2 + 1` Sub-`0.500` Sheared 4-Staircase `w x h` Block (`Gobel/Ellsworth`) & Sub-Rectangle Forge
+ Diagonal-Mirror `santa_ga` v7 + Multi-Step L-Extension Cascade!

Targets the `~0.0028` LB goldmine uncovered across `K = 19..31` (`n = K^2 + 1`):
  - `K=31 (962)`: `f_cur = 0.503133` vs `f_front = 0.489514` (`gap = +0.01362`!)
  - `K=27 (730)`: `f_cur = 0.511625` vs `f_front = 0.498476` (`gap = +0.01315`!)
  - `K=28 (785)`: `f_cur = 0.509228` vs `f_front = 0.498476` (`gap = +0.01075`!)
  - `K=30 (901)`: `f_cur = 0.503133` vs `f_front = 0.493394` (`gap = +0.00974`!)
  - `K=23 (530)`: `f_cur = 0.524612` vs `f_front = 0.516088` (`gap = +0.00852`!)
  - `K=20 (401)`: `f_cur = 0.532985` vs `f_front = 0.526135` (`gap = +0.00685`!)
  - `K=29 (842)`: `f_cur = 0.503133` vs `f_front = 0.496609` (`gap = +0.00652`!)
  - `K=24 (577)`: `f_cur = 0.521812` vs `f_front = 0.516074` (`gap = +0.00574`!)
  - `K=19 (362)`: `f_cur = 0.535534` vs `f_front = 0.532894` (`gap = +0.00264`!)
  - Plus `677 (26^2+1)`, `485 (22^2+1)`, `442 (21^2+1)`, `626 (25^2+1)`!

Combines 3 orthogonal `K^2 + 1` constructors per shard:
  1. Sheared 4-Corner Upright Staircase `(TL, TR, BR, BL)` + `a x b` Tilted Block (`Gobel/Ellsworth`
     generalized across shallow angles `17..45 deg`) quenched in sub-rectangles `W x H + 1` via
     `harden` + `fq --rect-dx/--rect-dy` (`exact mechanism of n=842's 20x28+1 = 0.5031331!`),
  2. Center-Preserving & Corner Frame Peels from `(K+1)^2 + 1` (`401->362`, `626->577`, `842->785->730`)
     + Sub-Block Core Stretch (`842`'s `20x28+1` core stretched to `21x30+1`, `22x31+1` for `901, 962`!),
  3. `800s` of 6-Island `santa_ga` v7 (with 2 Diagonal-Mirror Symmetry Islands!) + `800s` of
     6-Round `continuous_momentum_widehopper` + automatic `add_L` cascade to all `(K + c)^2 + 1 <= 1000`!
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

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
TOOLS_DIR = os.path.abspath(os.path.join(HERE, "..", "tools"))
for _d in (DS_DIR, TOOLS_DIR, HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w63k_1puzzle_per_shard_killshot as w63k
import w75a_upward_nm1_basin_invasion_gav7 as w75a

# 20-Shard Schedule: 2 independent seeds/angles on top 7 `K^2+1` gaps + 1 shard on next 6!
K2P1_20 = [
    962,  # s=0:  31^2+1 (variant A)
    730,  # s=1:  27^2+1 (variant A)
    785,  # s=2:  28^2+1 (variant A)
    901,  # s=3:  30^2+1 (variant A)
    530,  # s=4:  23^2+1 (variant A)
    401,  # s=5:  20^2+1 (variant A)
    842,  # s=6:  29^2+1 (variant A)
    577,  # s=7:  24^2+1
    362,  # s=8:  19^2+1
    677,  # s=9:  26^2+1
    485,  # s=10: 22^2+1
    442,  # s=11: 21^2+1
    626,  # s=12: 25^2+1
    962,  # s=13: 31^2+1 (variant B)
    730,  # s=14: 27^2+1 (variant B)
    785,  # s=15: 28^2+1 (variant B)
    901,  # s=16: 30^2+1 (variant B)
    530,  # s=17: 23^2+1 (variant B)
    401,  # s=18: 20^2+1 (variant B)
    842,  # s=19: 29^2+1 (variant B)
]


def add_L_upright(P_in):
    """Exact Gobel/Ellsworth +1 L-border extension: `K^2 + d -> (K+1)^2 + d` with `S -> S + 1.0`!"""
    P = RE.normalize_to_origin(P_in)
    S = RE.side_exact(P)
    S1 = S + 1.0
    m = int(math.floor(S1 - 1e-9))
    extra = []
    for j in range(m):
        extra.append((S1 - 0.5, j + 0.5, 0.0))
    for i in range(m - 1):
        extra.append((i + 0.5, S1 - 0.5, 0.0))
    return np.vstack([P, np.array(extra, dtype=np.float64)])


def build_sheared_gobel_rect_candidates(K, variant=0, max_jobs=14):
    """Construct sheared 4-corner staircase `(TL, TR, BR, BL)` + `a x b` tilted block inside
    sub-rectangles `W x H + 1` (`W <= K, H <= K`) so `harden` + `fq --rect-dx/--rect-dy` relaxes
    into `20x28+1`-class sub-`0.500` gadgets!
    """
    tri = [t * (t + 1) // 2 for t in range(K + 3)]
    angles_deg = (
        [20.9, 24.5, 31.0, 45.0, 18.4]
        if variant == 0
        else [22.6, 26.6, 35.5, 45.0, 19.6]
    )

    # Candidate sub-rectangle sizes `(W, H)`: both aspect-ratio ~ 0.70 (like 20x28 in n=842!) and square KxK!
    wh_pairs = []
    for dw in (0, int(round(0.28 * K)), int(round(0.22 * K)), int(round(0.33 * K)), 2):
        W = max(14, K - dw)
        for dh in (0, 1):
            H = max(W, K - dh)
            if (W, H) not in wh_pairs and W <= K and H <= K:
                wh_pairs.append((W, H))

    shapes = []
    for W, H in wh_pairs:
        m_tgt = W * H + 1
        # Search 4 corner staircases + a x b tilted block == m_tgt
        for a in range(max(6, W // 2), min(W + 3, 24)):
            for b in range(a, min(H + 4, 28)):
                rem_u = m_tgt - a * b
                if rem_u < 12 or rem_u > 4 * tri[min(W, H) - 1]:
                    continue
                # Find symmetric or near-symmetric staircase 4-tuples `(t1, t2, t3, t4)` summing to `rem_u`!
                t_est = int((math.sqrt(1.0 + 2.0 * rem_u) - 1.0) * 0.5)
                found_tup = None
                for t1 in range(max(1, t_est - 3), min(W, t_est + 4)):
                    for t2 in range(max(1, t_est - 3), min(W, t_est + 4)):
                        r2 = rem_u - tri[t1] - tri[t2]
                        if r2 < 2:
                            continue
                        for t3 in range(max(1, t1 - 2), min(W, t1 + 3)):
                            r4 = r2 - tri[t3]
                            if r4 <= 0:
                                continue
                            t4 = int(round((math.sqrt(1.0 + 8.0 * r4) - 1.0) * 0.5))
                            if (
                                t4 >= 1
                                and tri[t4] == r4
                                and t1 + t2 <= W + 1
                                and t2 + t3 <= H + 1
                            ):
                                found_tup = (t1, t2, t3, t4)
                                break
                        if found_tup is not None:
                            break
                    if found_tup is not None:
                        break
                if found_tup is None:
                    continue
                TL, TR, BR, BL = found_tup
                deg = angles_deg[len(shapes) % len(angles_deg)]
                th = math.radians(deg)
                c, s = math.cos(th), math.sin(th)
                sx_r = W + 0.505
                sy_r = H + 0.505
                rows = []
                # 4 upright corner staircases
                for corner, tc in (("TL", TL), ("TR", TR), ("BR", BR), ("BL", BL)):
                    for col_c in range(tc):
                        for row_c in range(tc - col_c):
                            ux = (
                                col_c + 0.5
                                if corner in ("TL", "BL")
                                else sx_r - (col_c + 0.5)
                            )
                            uy = (
                                sy_r - (row_c + 0.5)
                                if corner in ("TL", "TR")
                                else row_c + 0.5
                            )
                            rows.append((ux, uy, 0.0))
                # Centered `a x b` tilted block inside `[0, sx_r] x [0, sy_r]`
                cx_b, cy_b = 0.5 * sx_r, 0.5 * sy_r
                for i in range(a):
                    for j in range(b):
                        du = i - 0.5 * (a - 1)
                        dv = j - 0.5 * (b - 1)
                        bx = np.clip(cx_b + du * c - dv * s, 0.52, sx_r - 0.52)
                        by = np.clip(cy_b + du * s + dv * c, 0.52, sy_r - 0.52)
                        rows.append((float(bx), float(by), th))
                if len(rows) == m_tgt:
                    # Prefer shapes where block dimensions fit comfortably inside `[sx_r, sy_r]`
                    span_pen = max(0.0, (a * c + b * s) - sx_r) + max(
                        0.0, (a * s + b * c) - sy_r
                    )
                    shapes.append(
                        (
                            span_pen,
                            f"gob_{W}x{H}_{a}x{b}_t{deg:.1f}",
                            W,
                            H,
                            np.array(rows, dtype=np.float64),
                        )
                    )
    shapes.sort(key=lambda z: z[0])
    return shapes[:max_jobs]


def embed_rect_into_k2p1(P_rect, W, H, K):
    """Pad a `W x H + 1` packing (`W <= K, H <= K`) with `(K - W)` right upright columns and
    `(K - H)` top upright rows into an exact `K^2 + 1` packing!
    """
    P = RE.normalize_to_origin(P_rect)
    he = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    w_box = float((P[:, 0] + he).max())
    h_box = float((P[:, 1] + he).max())
    f = max(w_box - W, h_box - H)
    S_full = K + f
    extra = []
    # Add `K - W` columns of `H` squares on the right of `[0, W + f] x [0, H + f]`
    for c_i in range(K - W):
        xc = W + f + c_i + 0.5
        for r_j in range(H):
            yc = 0.5 + r_j * (H + f - 1.0) / max(1, H - 1) if H > 1 else 0.5 * (H + f)
            extra.append((xc, yc, 0.0))
    # Add `K - H` rows of `K` squares across the top `[0, K + f] x [H + f, K + f]`
    for r_i in range(K - H):
        yc = H + f + r_i + 0.5
        for c_j in range(K):
            xc = 0.5 + c_j * (S_full - 1.0) / max(1, K - 1) if K > 1 else 0.5 * S_full
            extra.append((xc, yc, 0.0))
    if not extra:
        return P
    return np.vstack([P, np.array(extra, dtype=np.float64)])


def eval_rect_gobel_job(job):
    tag, W, H, K, L0_full, P_rect, seed = job
    try:
        f_ref = L0_full - K
        dx = float(W - H)
        s_rect_ref = max(W, H) + f_ref + 0.015
        rh = RE.run_harden_reharden(
            P_rect,
            s_ref=s_rect_ref,
            seed=int(seed),
            r0=0.22,
            stiff0=0.28,
            pre=2600,
            hold=8500,
            release=4000,
            settle=3000,
            hold_f=1.003,
            noise=0.0055,
            timeout_s=13.5,
        )
        P_mid = rh[1] if rh is not None else P_rect
        rect_args = (
            ["--rect-dx", f"{abs(dx):.6f}"]
            if dx < -1e-6
            else (["--rect-dy", f"{abs(dx):.6f}"] if dx > 1e-6 else [])
        )
        rq = RE.run_fq_quench_args(
            P_mid,
            [
                "--seed",
                str(int(seed) + 17),
                "--round-r0",
                "0.09",
                "--round-steps",
                "14",
                "--round-press",
                "--round-press-outer",
                "6",
                "--loosen",
                "1.007",
                "--outer",
                "52",
                "--pit",
                "320",
                "--flip-top",
                "22",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "12.5",
            ]
            + rect_args,
            s_ref=s_rect_ref,
            timeout_s=16.0,
        )
        if rq is None:
            return tag, math.inf, None
        P_full = embed_rect_into_k2p1(rq[1], W, H, K)
        if len(P_full) != K * K + 1:
            return tag, math.inf, None
        Sp, Qp = RE.kkt_sqp_polish_fq(P_full, timeout_each=5.5)
        return tag, float(Sp), Qp
    except Exception:
        return tag, math.inf, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=4050000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )

    n = K2P1_20[args.shard % len(K2P1_20)]
    variant = 1 if args.shard >= 13 else 0
    K = math.isqrt(n)
    L0 = cur[n][0]
    P0 = cur[n][1]
    bseed = args.seed_off + args.shard * 9973

    print(
        f"[W75B-K2P1-GobelForge s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"TARGET n={n} ({K}^2+1, var={variant}, f_cur={L0 - K:.7f}) ({time.time() - t0:.1f}s)",
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

    def record_win_with_lextend(n_w, Q_cand, S_cand, src_tag, w_cnt):
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
                f"  [W75B K2P1 WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            # Cascade automatic +1 L-border extensions to all `(K' + c)^2 + 1 <= 1000`!
            Q_ext = np.column_stack([xf, yf, tf])
            Kw = math.isqrt(n_w)
            for Kp in range(Kw + 1, 32):
                n_ext = Kp * Kp + 1
                if n_ext > 1000:
                    break
                Q_ext = add_L_upright(Q_ext)
                S_ext = RE.side_exact(Q_ext)
                L_ext_ref = wins[n_ext][0] if n_ext in wins else cur[n_ext][0]
                if S_ext < L_ext_ref - 1e-9:
                    ok_e, Lf_e, xf_e, yf_e, tf_e = f2d.stabilize_decimal(n_ext, Q_ext)
                    if ok_e and Lf_e < L_ext_ref - 1e-9:
                        ds_e = (Lf_e - L_ext_ref) / math.sqrt(n_ext)
                        tot_ds += ds_e
                        wins[n_ext] = (float(Lf_e), xf_e, yf_e, tf_e)
                        print(
                            f"    [W75B L-EXT CASCADE!!] n={n_ext:4d} ({Kp}^2+1) {cur[n_ext][0]:.7f} -> {Lf_e:.7f} (dS={ds_e:+.8f})!",
                            flush=True,
                        )
            save_ckpt()

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        # Phase 1 (~110s): Sheared 4-Staircase Sub-Rectangle Forge + `(K+1)^2+1 -> K^2+1` Frame Peels!
        t_p1 = time.time()
        gob_shapes = build_sheared_gobel_rect_candidates(
            K, variant=variant, max_jobs=14
        )
        gob_jobs = [
            (tg, W, H, K, L0, P_r, bseed + 100 + ji)
            for ji, (_, tg, W, H, P_r) in enumerate(gob_shapes)
        ]

        peel_jobs = []
        for c_up in (1, 2):
            n_up = (K + c_up) ** 2 + 1
            if n_up <= 1000 and cur[n_up][1] is not None:
                for ji, (ptag, P_cp) in enumerate(
                    w75a.center_preserving_frame_peels(cur[n_up][1], n)[:4]
                ):
                    peel_jobs.append(
                        (
                            f"{ptag}_from{n_up}",
                            n,
                            L0,
                            P_cp,
                            bseed + 300 + 20 * c_up + ji,
                        )
                    )

        donors = []
        if gob_jobs:
            futs_g = [ex_hop.submit(eval_rect_gobel_job, jb) for jb in gob_jobs]
            try:
                for f in as_completed(futs_g, timeout=75.0):
                    tg, Sg, Qg = f.result()
                    if Qg is not None and math.isfinite(Sg):
                        if Sg < L0 + 0.0008:
                            Qp_g, Sp_g = w51d.deep_contact_polish(
                                n, Qg, deadline, slp_sec=40.0, fq_sec=24
                            )
                            record_win_with_lextend(n, Qp_g, Sp_g, tg, 0)
                            Qg, Sg = Qp_g, Sp_g
                        donors.append((Sg, tg, Qg))
            except TimeoutError:
                w51d.terminate_pool(ex_hop)
                ex_hop = ProcessPoolExecutor(max_workers=args.procs)

        if peel_jobs:
            futs_p = [
                ex_hop.submit(w75a.eval_soft_nm1_heal_job, jb) for jb in peel_jobs
            ]
            try:
                for f in as_completed(futs_p, timeout=48.0):
                    tg, Sp, Qp = f.result()
                    if Qp is not None and math.isfinite(Sp):
                        if Sp < L0 + 0.0008:
                            Qp_2, Sp_2 = w51d.deep_contact_polish(
                                n, Qp, deadline, slp_sec=40.0, fq_sec=24
                            )
                            record_win_with_lextend(n, Qp_2, Sp_2, tg, 0)
                            Qp, Sp = Qp_2, Sp_2
                        donors.append((Sp, tg, Qp))
            except TimeoutError:
                w51d.terminate_pool(ex_hop)
                ex_hop = ProcessPoolExecutor(max_workers=args.procs)

        donors.sort(key=lambda z: z[0])
        best_d_str = f"{donors[0][1]}({donors[0][0] - L0:+.5f})" if donors else "none"
        print(
            f"  [W75B phase1] n={n:4d} gob_shapes={len(gob_shapes)} peels={len(peel_jobs)} "
            f"quenched={len(donors)} best={best_d_str} ({time.time() - t_p1:.1f}s)",
            flush=True,
        )

        # Phase 2 (~820s): 6-Island Angle-Pinned + Diagonal-Mirror `santa_ga` v7!
        L0_ga = wins[n][0] if n in wins else cur[n][0]
        P0_ga = (
            np.column_stack([wins[n][1], wins[n][2], wins[n][3]]) if n in wins else P0
        )
        # Variant 1 starts from best distinct donor if within +0.025; Variant 0 starts from P0_ga with donors in --nbr!
        if variant == 1 and donors and donors[0][0] < L0_ga + 0.025:
            S_in, tg_in, P_in = donors[0]
            nbrs = [z[2] for z in donors[1:5]] + [P0_ga]
        else:
            S_in, tg_in, P_in = L0_ga, "cur_k2p1", P0_ga
            nbrs = [z[2] for z in donors[:5]]
        if n + 1 <= 1000 and cur[n + 1][1] is not None:
            nbrs.append(cur[n + 1][1])

        rem = deadline - time.time()
        ga_sec = min(840.0, max(240.0, 0.50 * (rem - 20.0)))
        Qw, Qc, js = RE.run_santa_ga(
            P_in,
            nbr_packings=nbrs,
            seed=bseed,
            sec=ga_sec,
            islands=6,
            pop=14,
            offspring=28,
            steps=130,
            threads=args.procs,
            timeout_s=ga_sec + 40.0,
        )
        print(
            f"  [W75B GAv7 done] n={n:4d} [{tg_in}] gens={js.get('gens', 0)} swaps={js.get('swaps', 0)} "
            f"ga_wins={js.get('wins', 0)} s_in={S_in:.7f} -> s_best={js.get('s_best', S_in):.7f} (L0={L0_ga:.7f})",
            flush=True,
        )

        hop_P = P_in
        hop_tag = tg_in
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
                    "12.0",
                ],
                timeout_s=15.0,
            )
            if rq_w is not None:
                Qp_w, Sp_w = w51d.deep_contact_polish(
                    n, rq_w[1], deadline, slp_sec=48.0, fq_sec=28
                )
                record_win_with_lextend(
                    n, Qp_w, Sp_w, f"{tg_in}->gav7", js.get("wins", 1)
                )
                hop_P = Qp_w
                hop_tag = f"{tg_in}->gav7"

        # Phase 3 (~800s = 6 full rounds!): Continuous Momentum WideHopper!
        L0_cur = wins[n][0] if n in wins else cur[n][0]
        hop_b = max(120.0, deadline - time.time() - 15.0)
        Qb, Sb, (_, _, wt, nr) = w63k.continuous_momentum_widehopper(
            ex_hop,
            n,
            hop_P,
            L0_cur,
            bseed + 77000,
            hop_b,
            deadline,
            lambda Qx, Sx, stg, wc, _n=n, _tg=hop_tag: record_win_with_lextend(
                _n, Qx, Sx, f"{_tg}->{stg}", wc
            ),
        )
        record_win_with_lextend(n, Qb, Sb, f"{hop_tag}->whFin", wt)
        ad_end = w53h.topology_angle_dist(Qb, P0)
        print(
            f"  [W75B done] n={n:4d} [{hop_tag}] dA={ad_end:.2f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
            flush=True,
        )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W75B s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
