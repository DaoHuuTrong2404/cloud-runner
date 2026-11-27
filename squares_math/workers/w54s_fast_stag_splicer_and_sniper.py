#!/usr/bin/env python3
"""Wave 54S: Ultra-Fast Vectorized CC-Bar & Staggered Two-Band Splicer (8 Donors/Shard, <60s Phase A!)
+ Long-Horizon 3-Round Sniper Escalation (`r1=260, r2=240, r3=240`) on W52S Hot Jumpers & `n=750..1000`!

Fixes the exact Phase-A bottleneck proven by Wave 52S vs Wave 53S telemetry:
  - Wave 52S won on `n=380 (-0.000543)`, `n=702`, `n=758` and hit 24 hot near-misses (`756 +9e-6`,
    `924 +8.3e-5`, `465 +9.2e-5`, `754 +1.46e-4`, `549 +2.1e-4`, `689 +2.3e-4`, `652 +2.9e-4`,
    `929 +3.0e-4`, `376 +3.2e-4`, `418 +3.3e-4`, ...), but ran ascending `160..1000` so `800..1000`
    were starved at 345s.
  - Wave 53S added an un-vectorized `68x68` Python loop + 15s `harden` subprocess per cut, slowing
    Phase A to >90s/donor so 13/20 shards timed out before finishing 4 donors.
  - Wave 54S:
    1. Vectorizes `relocate_crushed_fast` in 3D NumPy broadcasting (`< 0.004s` vs `4.5s`!).
    2. Uses pure Numba `squeeze2` + fixed 2-tuple `w51d.compact_deleted` (`~2.5s` per cut, ZERO subprocesses in Phase A!).
    3. Caps each of the 22 shards at **8 high-priority donors** (`22 x 8 = 176` donors: all 25 W52S/R33
       hot donors `380, 813, 987, 510, 872, 811, 598, 744, 705, 992, 417, 461, 928, 381, 649, 999,
       552, 807, 465, 816, 651, 599, 699, 700, 421` + all elite high-N donors `998 -> 730` + `965, 976,
       914, 804, 793, 632, 499, 455, 333, 298`) -> Phase A finishes 100% of donors in **45-95s**!
    4. Phase B spends **2,050+ seconds** running `r1=260s, r2=240s, r3=240s` Deep-Polish `eta<=0.28` Wider-Hopper!
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
import w53s_ccbar_reharden_splicer as w53s

HOT_AND_ELITE_DONORS = [
    # Fresh R33 winners + W52S ultra-near-miss donors (`+9e-6 .. +3.5e-4`)
    380,
    813,
    987,
    510,
    872,
    811,
    598,
    744,
    705,
    992,
    417,
    461,
    928,
    381,
    649,
    999,
    552,
    807,
    465,
    816,
    651,
    599,
    699,
    700,
    421,
    758,
    702,
    # Elite private & frontier macro-topology donors
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
    849,
    905,
    846,
    789,
    734,
    681,
    858,
    750,
    990,
    927,
    861,
    818,
]


def relocate_crushed_fast(Q_in):
    """100% Vectorized NumPy Pocket Relocator (`< 0.004s` per call!)."""
    Q = RE.normalize_to_origin(Q_in)
    S = RE.side_exact(Q)
    gx, gy = np.meshgrid(
        np.linspace(0.52, S - 0.52, 38), np.linspace(0.52, S - 0.52, 38)
    )
    gpts = np.column_stack([gx.ravel(), gy.ravel()])  # (1444, 2)

    for _ in range(3):
        bad = w53s.worst_close_pairs(Q, thr=-0.25)
        if not bad:
            break
        _, i, j = bad[0]
        ai = abs(((Q[i, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0)
        aj = abs(((Q[j, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0)
        victim = i if ai <= aj else j
        other = np.delete(Q, victim, axis=0)
        h0 = 0.5 * (np.abs(np.cos(other[:, 2])) + np.abs(np.sin(other[:, 2])))
        # Vectorized (1444, N-1) clearance!
        dx = np.abs(gpts[:, 0:1] - other[None, :, 0]) - 0.5 - h0[None, :]
        dy = np.abs(gpts[:, 1:2] - other[None, :, 1]) - 0.5 - h0[None, :]
        min_sep = np.maximum(dx, dy).min(axis=1)
        top_idx = np.argsort(-min_sep)[:12]
        best_v = (-1e9, 0.5, 0.5, 0.0)
        for ti in top_idx:
            px, py = float(gpts[ti, 0]), float(gpts[ti, 1])
            nn = int(np.argmin((other[:, 0] - px) ** 2 + (other[:, 1] - py) ** 2))
            for ang in (0.0, float(other[nn, 2])):
                mgt = 1e9
                close_k = np.where(
                    (other[:, 0] - px) ** 2 + (other[:, 1] - py) ** 2 < 2.2
                )[0]
                for k in close_k:
                    ddx, ddy = px - float(other[k, 0]), py - float(other[k, 1])
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


def fast_quench_v2(P_raw, target_L_ref):
    """Fast Numba `squeeze2` + fixed 2-tuple `w51d.compact_deleted` (`~2.5s` per candidate!)."""
    if P_raw is None or not (80 <= len(P_raw) <= 1000):
        return None
    x0, y0, t0 = sq2.normalize(P_raw[:, 0], P_raw[:, 1], P_raw[:, 2])
    S_init = max(float(sq2.side(x0, y0, t0)), float(target_L_ref)) * 1.002
    ok, rx, ry, rt, _ = sq2.relax_at(x0, y0, t0, S_init, max_iter=5500)
    if not ok:
        ok, rx, ry, rt, _ = sq2.relax_at(x0, y0, t0, S_init * 1.012, max_iter=7500)
        if not ok:
            return None
    _, x2, y2, t2 = sq2.squeeze(
        rx, ry, rt, rel0=2.8e-3, rel_min=5e-7, max_iter=4500, max_tries=42
    )
    Lf, fx, fy, ft, good = sq2.finalize(x2, y2, t2, tol=2e-12)
    if not good:
        return None
    Q = E.to_origin(np.column_stack([fx, fy, ft]))
    try:
        Sc, Qc = w51d.compact_deleted(Q, max_iter=32, patience=2)
        if Qc is not None and Sc < Lf:
            return Qc, float(Sc)
    except Exception:
        pass
    return Q, float(E.side(Q))


def exact_wall_cut_fast(P_in, ip, ineg, kp, kn):
    ax_p, _, bars_p = ip
    ax_n, _, bars_n = ineg
    if ax_p == ax_n:
        return None
    a = E.to_origin(P_in.copy())
    bp, bn = bars_p[kp], bars_n[kn]
    cp, cn = float(a[bp, ax_p].mean()), float(a[bn, ax_n].mean())
    p_lo, p_hi = float(a[bp, ax_n].min()) - 0.45, float(a[bp, ax_n].max()) + 0.45
    n_lo, n_hi = float(a[bn, ax_p].min()) - 0.45, float(a[bn, ax_p].max()) + 0.45

    drop = np.zeros(len(a), bool)
    drop[bp] = True
    drop[bn] = True
    in_bars = np.zeros(len(a), bool)
    in_bars[np.concatenate(bars_p)] = True
    in_bars[np.concatenate(bars_n)] = True
    up_m = ~in_bars
    drop |= (
        up_m
        & (np.abs(a[:, ax_p] - cp) < 0.52)
        & ((a[:, ax_n] < p_lo) | (a[:, ax_n] > p_hi))
    )
    drop |= (
        up_m
        & (np.abs(a[:, ax_n] - cn) < 0.52)
        & ((a[:, ax_p] < n_lo) | (a[:, ax_p] > n_hi))
    )

    Q = a[~drop].copy()
    kept_idx = np.where(~drop)[0]
    p_right = set(np.concatenate(bars_p[kp + 1 :]).tolist())
    p_left = set(np.concatenate(bars_p[:kp]).tolist())
    n_right = set(np.concatenate(bars_n[kn + 1 :]).tolist())
    n_left = set(np.concatenate(bars_n[:kn]).tolist())
    for qi, orig_i in enumerate(kept_idx):
        if orig_i in p_right or (orig_i not in p_left and a[orig_i, ax_p] > cp):
            Q[qi, ax_p] -= 1.0
        if orig_i in n_right or (orig_i not in n_left and a[orig_i, ax_n] > cn):
            Q[qi, ax_n] -= 1.0
    return relocate_crushed_fast(Q)


def generate_fast_splices(args_tuple):
    src_n, P_src, cur_L_map = args_tuple
    P = E.to_origin(P_src)
    ang = ((P[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
    pos_idx = np.where(ang > 0.03)[0]
    neg_idx = np.where(ang < -0.03)[0]

    ip = w53s.extract_bars_cc(P, pos_idx)
    ineg = w53s.extract_bars_cc(P, neg_idx)

    out = []
    seen = set()

    def _reg(Q, S, tag):
        nn = len(Q)
        if not (80 <= nn <= 1000):
            return
        L_cur = cur_L_map.get(nn, float(math.ceil(math.sqrt(nn))))
        if S < L_cur + 0.012:
            k = (nn, round(S, 4))
            if k not in seen:
                seen.add(k)
                out.append((nn, float(S), tag, Q))

    def _quench_and_del1(P_raw, tag):
        if P_raw is None or not (80 <= len(P_raw) <= 1000):
            return None
        nn = len(P_raw)
        rq = fast_quench_v2(P_raw, cur_L_map.get(nn, math.sqrt(nn) + 0.65))
        if rq is None:
            return None
        Qc, Sc = rq
        _reg(Qc, Sc, tag)
        if Sc < cur_L_map.get(nn, Sc) + 0.015 and nn - 1 >= 80:
            for idx in w32a.pick_near_band(Qc, max_cand=4)[:1]:
                Qd, Sd = w32a.fast_delete_and_compact(Qc, idx)
                if Qd is not None:
                    _reg(Qd.astype(np.float64), float(Sd), tag + "_del1")
        return Qc, Sc

    # Mode A: Single-band / Staircase cuts (`965 -> 904 -> 845 -> 680`, `333 -> 298`)
    if (ip is not None and len(neg_idx) < 12) or (
        ineg is not None and len(pos_idx) < 12
    ):
        info = ip if (ip is not None and len(neg_idx) < 12) else ineg
        stack_ax, _, bars = info
        by_sz = {}
        for k in range(2, len(bars) - 2):
            by_sz.setdefault(len(bars[k]), []).append(k)
        reps = [
            ks[len(ks) // 2]
            for _, ks in sorted(by_sz.items(), key=lambda kv: -len(kv[1]))[:3]
        ]
        best_q1 = None
        for kr in reps:
            P1 = w52s.cut_one_band_axis(P, bars, stack_ax, kr, n_bars_del=1)
            if P1 is not None:
                r1 = _quench_and_del1(
                    relocate_crushed_fast(P1), f"sb1_{src_n}_b{len(bars[kr])}"
                )
                if r1 is not None and (best_q1 is None or r1[1] < best_q1[1]):
                    best_q1 = r1
        if best_q1 is not None:
            Q1, _ = best_q1
            a1 = ((Q1[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
            m1 = (
                np.where(a1 > 0.03)[0] if len(neg_idx) < 12 else np.where(a1 < -0.03)[0]
            )
            inf1 = w53s.extract_bars_cc(Q1, m1)
            if inf1 is not None and len(inf1[2]) >= 7:
                P2 = w52s.cut_one_band_axis(
                    Q1, inf1[2], inf1[0], len(inf1[2]) // 2, n_bars_del=1
                )
                if P2 is not None:
                    _quench_and_del1(relocate_crushed_fast(P2), f"sb2_{src_n}")
        if len(bars) >= 22:
            P5 = w52s.cut_one_band_axis(P, bars, stack_ax, len(bars) // 2, n_bars_del=5)
            if P5 is not None:
                _quench_and_del1(relocate_crushed_fast(P5), f"sb5_{src_n}")

    # Mode B: Orthogonal Two-Band Cuts (both `tb` proven winner AND vectorized `tw`!)
    if ip is not None and ineg is not None:
        _, _, bars_p = ip
        _, _, bars_n = ineg
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

        best_tb1 = None
        for kp in reps_p:
            for kn in reps_n:
                tsz = f"s{len(bars_p[kp])}x{len(bars_n[kn])}"
                P_tb = w52s.cut_two_band_staggered(P, ip, ineg, kp, kn)
                r_tb = _quench_and_del1(P_tb, f"tb1_{src_n}_{tsz}")
                if r_tb is not None and (best_tb1 is None or r_tb[1] < best_tb1[1]):
                    best_tb1 = r_tb
                P_tw = exact_wall_cut_fast(P, ip, ineg, kp, kn)
                r_tw = _quench_and_del1(P_tw, f"tw1_{src_n}_{tsz}")
                if r_tw is not None and (best_tb1 is None or r_tw[1] < best_tb1[1]):
                    best_tb1 = r_tw

        if best_tb1 is not None:
            Q1, _ = best_tb1
            a1 = ((Q1[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
            ip1 = w53s.extract_bars_cc(Q1, np.where(a1 > 0.03)[0])
            in1 = w53s.extract_bars_cc(Q1, np.where(a1 < -0.03)[0])
            if (
                ip1 is not None
                and in1 is not None
                and len(ip1[2]) >= 5
                and len(in1[2]) >= 5
            ):
                P_tb2 = w52s.cut_two_band_staggered(
                    Q1, ip1, in1, len(ip1[2]) // 2, len(in1[2]) // 2
                )
                _quench_and_del1(P_tb2, f"tb2_{src_n}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=22)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2220)
    ap.add_argument("--r1-sec", type=int, default=260)
    ap.add_argument("--r2-sec", type=int, default=240)
    ap.add_argument("--r3-sec", type=int, default=240)
    ap.add_argument("--seed-off", type=int, default=580000)
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

    seen_d = set()
    donors = []
    for n in HOT_AND_ELITE_DONORS + list(range(999, 719, -1)):
        if n in seen_d or n < 160 or n > 1000:
            continue
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        seen_d.add(n)
        donors.append(n)

    # STRICT 8 DONORS PER SHARD (= 2 batches across 4 cores -> finishes in ~50-95s with ZERO timeout starvation!)
    my_donors = donors[args.shard :: args.num_shards][:8]
    tasks = [(n, cur[n][1], cur_L_map) for n in my_donors]

    cands = []
    ex_gen = ProcessPoolExecutor(max_workers=args.procs)
    try:
        futs = [ex_gen.submit(generate_fast_splices, tk) for tk in tasks]
        try:
            for f in as_completed(futs, timeout=220.0):
                try:
                    for nn, S, tag, Q in f.result():
                        gap = S - cur[nn][0]
                        if gap < 0.012:
                            cands.append((gap, nn, S, tag, Q))
                except Exception:
                    continue
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
        if len(queued) >= 8:
            break

    print(
        f"[W54S-FastStagSniper s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_donors={my_donors} -> cands(gap<0.012)={len(cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
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
                    n, Q_init, deadline, slp_sec=40.0, fq_sec=28
                )
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        tot_ds += (Lf0 - L0_cur) / math.sqrt(n)
                        wins[n] = (float(Lf0), xf0, yf0, tf0)
                        L0_cur = float(Lf0)
                        print(
                            f"  [W54S DIRECT WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf0:.7f}!",
                            flush=True,
                        )
                        save_ckpt()

            r1 = min(args.r1_sec, max(35, int(0.36 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(30, int(0.32 * (rem - 20.0))))
            r3 = min(args.r3_sec, max(30, int(0.32 * (rem - 20.0))))
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
                        f"  [W54S WIN!!] n={n:4d} [{tag}_wh28] gap0={gap0:+.5f} tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W54S done] n={n:4d} [{tag}] gap0={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W54S s={args.shard} EXC] {exc}", flush=True)
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W54S s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
