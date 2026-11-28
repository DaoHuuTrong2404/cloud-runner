#!/usr/bin/env python3
"""Wave 64M (`w64m_tolerant_multilattice_window_milp.py`): Tolerant Multi-Tilt Empirical-Lattice
Window 0-1 MILP (`ov_tol in {-0.012, -0.026, -0.042}`) + R35 Diagonal/Grid Inflation + Continuous WideHopper!

Unlocks Ross Wade's `sq_research.py` empirical multi-tilt lattice grower (`lattice_vectors` + `candidates` + `max_set`)
by removing its rigid `-1e-12` zero-overlap wall:
  - In `w28b` (last run 15 submissions ago at R20), `R.candidates` and `R.max_set` rejected any candidate square
    with `pair_gap < -1e-12`, so tightly packed R35 configurations `P[n - 1]` could NEVER accept `+1` extra square!
  - Allowing gentle seam tolerance `ov_tol in {-0.012, -0.026, -0.042}` inside a surgical `[x0, x0 + w] x [y0, y0 + h]`
    pocket lets HiGHS 0-1 MILP extend neighboring tilted & upright lattices to pack `n_in + 1` squares inside
    `P[n - 1]` at `S_try < L[n]`, which plain `fq quench --loosen 1.02` immediately relaxes into a legal exact-`n`
    basin!
  - Also sweeps `R.best_diag(k=1,2)` + `R.best_inflation2(k=1,2)` + `E.transplant` across our upgraded R35 packings,
    escalating all novel `< L0 + 0.012` basins via Continuous-Momentum `eta <= 0.28` LP WideHopper!
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial import cKDTree

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import highspy
import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
TOOLS_DIR = os.path.abspath(os.path.join(HERE, "..", "tools"))
for _d in (DS_DIR, TOOLS_DIR, HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import rust_engines as RE
import sq_engine as E
import sq_research as R
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w63k_1puzzle_per_shard_killshot as w63k


def candidates_tolerant(P, S, x0, y0, w, h, ov_tol=-0.025, gens=5, margin=3.0):
    """Port of `sq_research.candidates` allowing gentle seam overlap `ov_tol < 0` for `fq` relaxation!"""
    cx, cy = P[:, 0], P[:, 1]
    inwin = (cx > x0) & (cx < x0 + w) & (cy > y0) & (cy < y0 + h)
    near = (
        (cx > x0 - margin)
        & (cx < x0 + w + margin)
        & (cy > y0 - margin)
        & (cy < y0 + h + margin)
    )
    t = E.fold(P[:, 2])
    cands = [P[inwin]]
    groups = {}
    for i in np.where(near)[0]:
        groups.setdefault(round(float(t[i]), 3), []).append(i)
    for key, idx in groups.items():
        idx = np.array(idx)
        if abs(key) < 1e-6:
            V = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], float)
        else:
            V = R.lattice_vectors(P, np.where(np.abs(t - key) < 2e-3)[0])
            if len(V) == 0:
                continue
        front = P[idx].copy()
        seen = set()
        for _ in range(gens):
            new = (front[:, None, :2] + V[None, :, :]).reshape(-1, 2)
            new = np.concatenate([new, np.full((len(new), 1), P[idx[0], 2])], axis=1)
            keep = (
                (new[:, 0] > x0 - 0.6)
                & (new[:, 0] < x0 + w + 0.6)
                & (new[:, 1] > y0 - 0.6)
                & (new[:, 1] < y0 + h + 0.6)
            )
            new = new[keep]
            if len(new) == 0:
                break
            keys = [tuple(np.round(r / np.array([0.01, 0.01, 1e-3]))) for r in new]
            sel = [i for i, k in enumerate(keys) if k not in seen]
            for i in sel:
                seen.add(keys[i])
            front = new[sel]
            if len(front) == 0:
                break
            cands.append(front)
    C = np.vstack(cands)
    key = np.round(C / np.array([0.01, 0.01, 1e-3]))
    C = C[np.sort(np.unique(key, axis=0, return_index=True)[1])]
    C = C[R.wall_ok(C, S, tol=abs(ov_tol))]
    fixed = P[~inwin]
    if len(fixed) and len(C):
        T = cKDTree(fixed[:, :2])
        pr = []
        for i, js in enumerate(T.query_ball_point(C[:, :2], 1.4143)):
            for j in js:
                pr.append((i, j))
        bad = np.zeros(len(C), bool)
        if pr:
            pr = np.array(pr, dtype=np.int64)
            g = R.gaps_to(np.ascontiguousarray(C), np.ascontiguousarray(fixed), pr)
            bad[pr[g < ov_tol, 0]] = True
        C = C[~bad]
    return C, inwin


def max_set_tolerant(C, ov_tol=-0.025, time_limit=3.5):
    """Exact 0-1 HiGHS Maximum Independent Set on `C` with conflict threshold `pair_gap < ov_tol`."""
    n = len(C)
    if n == 0:
        return np.zeros(0, bool), 0
    T = cKDTree(C[:, :2])
    pr = np.array(sorted(T.query_pairs(1.4143)), dtype=np.int64).reshape(-1, 2)
    if len(pr):
        g = R.gaps_to(np.ascontiguousarray(C), np.ascontiguousarray(C), pr)
        pr = pr[g < ov_tol]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 0.0)
    h.addCols(
        n,
        -np.ones(n),
        np.zeros(n),
        np.ones(n),
        0,
        np.array([], np.int32),
        np.array([], np.int32),
        np.array([]),
    )
    h.changeColsIntegrality(
        n,
        np.arange(n, dtype=np.int32),
        np.array([highspy.HighsVarType.kInteger] * n),
    )
    if len(pr):
        m = len(pr)
        starts = np.arange(0, 2 * m, 2, dtype=np.int32)
        index = pr.reshape(-1).astype(np.int32)
        h.addRows(
            m, -np.inf * np.ones(m), np.ones(m), 2 * m, starts, index, np.ones(2 * m)
        )
    h.run()
    x = np.array(h.getSolution().col_value)
    if len(x) != n or not np.all(np.isfinite(x)):
        return np.zeros(n, bool), 0
    sel = x > 0.5
    return sel, int(sel.sum())


def eval_window_milp_puzzle(job_tup):
    n, L_ref, P_prev_in, L_prev, seed0 = job_tup
    rng = np.random.default_rng(int(seed0))
    P0 = E.to_origin(P_prev_in)
    S_try = min(float(L_ref) - 0.0005, float(L_prev) + 0.016)

    try:
        V = E.empty_area(P0, float(L_prev), 8)
        flat = np.argsort(-V.ravel())[:10]
        ny = V.shape[1]
        pockets = [(int(idx // ny) + 0.5, int(idx % ny) + 0.5) for idx in flat]
    except Exception:
        pockets = [(L_prev * 0.5, L_prev * 0.5)]

    tilts = np.abs(E.fold(P0[:, 2]))
    til = np.where(tilts > 0.02)[0]
    if len(til) > 0:
        for q in rng.choice(til, size=min(6, len(til)), replace=False):
            pockets.append((float(P0[q, 0]), float(P0[q, 1])))

    raw_hits = []
    for pi, (cx, cy) in enumerate(pockets[:12]):
        if len(raw_hits) >= 5:
            break
        w = float(rng.choice([6.0, 7.5, 9.0]))
        h = float(rng.choice([6.0, 7.5, 9.0]))
        x0 = max(0.0, min(S_try - w, cx - 0.5 * w))
        y0 = max(0.0, min(S_try - h, cy - 0.5 * h))
        for ov_tol in (-0.012, -0.028, -0.042):
            try:
                C, inwin = candidates_tolerant(
                    P0, S_try, x0, y0, w, h, ov_tol=ov_tol, gens=5, margin=3.0
                )
                n_in = int(inwin.sum())
                if len(C) <= n_in or len(C) > 420:
                    continue
                sel, val = max_set_tolerant(C, ov_tol=ov_tol, time_limit=3.0)
                if val >= n_in + 1:
                    Q = np.vstack([P0[~inwin], C[sel]])[:n].copy()
                    if len(Q) == n:
                        raw_hits.append((f"wmilp_p{pi}_t{int(abs(ov_tol) * 1000)}", Q))
                        break
            except Exception:
                continue

    quenched = []
    for qi, (tag, Q_raw) in enumerate(raw_hits[:5]):
        rq = RE.run_fq_quench_args(
            Q_raw,
            [
                "--seed",
                str(int(seed0) + qi * 19),
                "--loosen",
                "1.022",
                "--outer",
                "58",
                "--pit",
                "300",
                "--flip-top",
                "22",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "21.0",
            ],
            s_ref=S_try,
            timeout_s=24.0,
        )
        if rq is not None:
            Sq, Qq = float(rq[0]), rq[1]
            if Sq < L_ref + 0.0145:
                quenched.append((n, Sq, tag, Qq))
    quenched.sort(key=lambda z: z[1])
    return quenched[:2]


def eval_r35_inflate_diag_donor(job_tup):
    m, arr_m, cur_L_map, seed0 = job_tup
    rng = np.random.default_rng(int(seed0))
    P_m = E.to_origin(arr_m)
    raw = []
    for up in (True, False):
        try:
            out = E.transplant(P_m, up)
            if out is not None:
                raw.append((f"tr{'U' if up else 'D'}_{m}", out[0]))
        except Exception:
            pass
    for k in (1, 2):
        try:
            Q1, _, _ = R.best_inflation2(P_m, k, tol=-0.015, rng=rng, n_random=80)
            if Q1 is not None and 120 <= len(Q1) <= 1002:
                raw.append((f"inf{k}_{m}", Q1))
        except Exception:
            pass
        try:
            Q2, _, _ = R.best_diag(P_m, k, tol=-0.015, steps=32)
            if Q2 is not None and 120 <= len(Q2) <= 1002:
                raw.append((f"diag{k}_{m}", Q2))
        except Exception:
            pass

    hits = []
    for ri, (tag, Q) in enumerate(raw):
        for drop in (0, 1):
            nn = len(Q) - drop
            if not (120 <= nn <= 1000) or math.isqrt(nn) ** 2 == nn:
                continue
            L_ref = cur_L_map.get(nn, math.sqrt(nn) + 0.65)
            Qsub = RE.normalize_to_origin(Q if drop == 0 else Q[:nn].copy())
            if RE.side_exact(Qsub) > L_ref + 0.05:
                continue
            rq = RE.run_fq_quench_args(
                Qsub,
                [
                    "--seed",
                    str(int(seed0) + ri * 7 + drop),
                    "--loosen",
                    "1.018",
                    "--outer",
                    "52",
                    "--pit",
                    "260",
                    "--flip-top",
                    "20",
                    "--finish",
                    "--stag-w",
                    "0",
                    "--deadline",
                    "18.0",
                ],
                s_ref=L_ref,
                timeout_s=21.0,
            )
            if rq is not None and float(rq[0]) < L_ref + 0.0135:
                hits.append((nn, float(rq[0]), f"{tag}_d{drop}", rq[1]))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=18)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1820)
    ap.add_argument("--seed-off", type=int, default=1680000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    P_warm = E.to_origin(
        np.array([[0.5, 0.5, 0.0], [1.6, 0.5, 0.1], [0.5, 1.6, 0.0]], dtype=np.float64)
    )
    E.empty_area(P_warm, 3.0, 4)
    R.clash_pairs(
        P_warm, E.neighbour_pairs(P_warm, 1.45), np.full(3, -1, np.int64), 0.0
    )
    R.gaps_to(P_warm, P_warm, np.array([[0, 1]], dtype=np.int64))

    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )
    cur_L_map = {n: cur[n][0] for n in range(1, 1001)}
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    # Select n where P[n-1] is strictly tighter than P[n] AND n has frontier gap!
    milp_targets = []
    for n in range(110, 1001):
        K = math.isqrt(n)
        if K * K == n or math.isqrt(n - 1) ** 2 == (n - 1):
            continue
        if cur[n - 1][0] < cur[n][0] - 2e-4:
            gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
            if gl > 6.0e-5:
                milp_targets.append((gl / math.sqrt(n), n))
    milp_targets.sort(reverse=True)
    my_milp = [x[1] for x in milp_targets[: args.num_shards * 4]][
        args.shard :: args.num_shards
    ]

    donors = [
        m
        for m in range(160, 999)
        if math.isqrt(m) ** 2 != m and cur[m][0] < math.isqrt(m) + 1 - 1e-6
    ]
    my_donors = donors[args.shard :: args.num_shards][:8]

    print(
        f"[W64M-TolerantMILP s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_milp={my_milp} my_donors={my_donors} ({time.time() - t0:.1f}s)",
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
                f"  [W64M MILP WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    screened = []
    ex_sc = ProcessPoolExecutor(max_workers=args.procs)
    try:
        futs = []
        for i, n in enumerate(my_milp):
            futs.append(
                ex_sc.submit(
                    eval_window_milp_puzzle,
                    (
                        n,
                        cur[n][0],
                        cur[n - 1][1],
                        cur[n - 1][0],
                        args.seed_off + args.shard * 3109 + i * 71,
                    ),
                )
            )
        for j, m in enumerate(my_donors):
            futs.append(
                ex_sc.submit(
                    eval_r35_inflate_diag_donor,
                    (
                        m,
                        cur[m][1],
                        cur_L_map,
                        args.seed_off + 15000 + args.shard * 2003 + j * 43,
                    ),
                )
            )
        try:
            for f in as_completed(futs, timeout=540.0):
                for nn, Sq, tg, Qq in f.result():
                    L0 = wins[nn][0] if nn in wins else cur[nn][0]
                    if Sq < L0 - 1e-9:
                        Qp0, Sp0 = w51d.deep_contact_polish(
                            nn, Qq, deadline, slp_sec=40.0, fq_sec=24
                        )
                        record_win_n(nn, Qp0, Sp0, tg + "_dir", 0)
                    ad = w53h.topology_angle_dist(Qq, cur[nn][1])
                    if Sq < L0 + 0.0135 and ad > 0.030:
                        screened.append((Sq - L0, nn, tg, Sq, ad, Qq))
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_sc)

    screened.sort(key=lambda z: z[0])
    queued = []
    seen_n = {}
    for item in screened:
        nn = item[1]
        if seen_n.get(nn, 0) >= 2:
            continue
        seen_n[nn] = seen_n.get(nn, 0) + 1
        queued.append(item)
        if len(queued) >= 4:
            break

    print(
        f"[W64M s={args.shard}] screened={len(screened)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, tag, S_init, adist, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 80.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            sub_b = (rem - 20.0) / max(1, len(queued) - i)
            Qb, Sb, (_, _, wt, nr) = w63k.continuous_momentum_widehopper(
                ex_hop,
                n,
                Q_init,
                L0_cur,
                args.seed_off + 80000 + args.shard * 8009 + i * 401,
                sub_b,
                deadline,
                lambda Qc, Sc, stg, wc, _n=n, _tg=tag: record_win_n(
                    _n, Qc, Sc, f"{_tg}->{stg}", wc
                ),
            )
            record_win_n(n, Qb, Sb, f"{tag}->whFin", wt)
            print(
                f"  [W64M done] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W64M s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
