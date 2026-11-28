#!/usr/bin/env python3
"""Wave 60A (`w60a_faithful_ladd_and_rust_hop.py`): Faithful Approximate L-Addition (`ladd.py`)
(`K - 1 -> K` Upward Tilted/Staggered Cross & Outer-Border Refill) + Compiled Rust `packer hhop` / `hop`!

Orthogonal upward shell constructor (`n0 -> n = n0 + d` where `d in [2*(K-1)+1 .. 2*(K-1)+6]`):
  - Reaches `K = 31` (`n = 962..999`, e.g. `976 <- 911/912/913`) where NO `K + 1` donor exists!
  - Exploits ultra-tight `K - 1` records (`234 <- 201` diff `-0.116`, `367 <- 326` diff `-0.146`,
    `491 <- 444` diff `-0.112`, `333 <- 294` diff `-0.114`, `261 <- 226` diff `-0.143`,
    `904 <- 842` diff `-0.060` using our new R34 `n=842` record!).
  - Vectorized 49-point `tear` cut scorer (`<0.05s` at `n=950`) + edge L-border cuts (`cx, cy in {0, s}`)
    + plain `fq quench --loosen 1.02 / 1.05` (`s_ref = S = s_don + w`, NO `relocate_crushed`, NO `--round-r0`)
    + compiled Rust `packer hhop` & `packer hop` escalation + KKT `deep_contact_polish` + `Decimal64`!
"""

import argparse
import json
import math
import os
import random
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

import ladd as LA
import lcut as LC
import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w59l_faithful_lcut_and_rust_hop as w59l


def best_ladd_cuts_fast(sq_deg, s, step=0.08, m=10, sep=0.25):
    """Vectorized exact port of `ladd.best_cuts` (`tear` across 7x7 sample grid per square)."""
    P = LC.samples(sq_deg)
    X, Y = P[..., 0], P[..., 1]
    ns = float(P.shape[1])
    Cx = X.mean(axis=1, keepdims=True)
    Cy = Y.mean(axis=1, keepdims=True)
    g = np.unique(np.append(np.arange(0.0, float(s) + 1e-9, step), [0.0, float(s)]))
    TX = (X[:, :, None] >= g) != (Cx[:, :, None] >= g)
    TY = (Y[:, :, None] >= g) != (Cy[:, :, None] >= g)
    CX = TX.sum(axis=1)
    CY = TY.sum(axis=1)
    G = len(g)
    S_mat = np.empty((G, G), dtype=np.int64)
    for i in range(G):
        C = CX[:, i, None] + CY
        near = np.nonzero(CX[:, i])[0]
        if len(near):
            C[near] -= (TX[near, :, i][:, :, None] & TY[near]).sum(axis=1)
        S_mat[i] = C.sum(axis=0)
    order = np.lexsort((np.tile(g, G), np.repeat(g, G), S_mat.ravel()))
    out = []
    for q in order:
        i, j = divmod(int(q), G)
        r = (float(S_mat[i, j]) / ns, float(g[i]), float(g[j]))
        if all(abs(r[1] - u[1]) + abs(r[2] - u[2]) > sep for u in out):
            out.append(r)
            if len(out) >= m:
                break
    return out


def eval_faithful_ladd_job(job_tup):
    tag, sq_don_list, s_don, d_add, n_want, cx, cy, wmax, tmax, loosen, seed = job_tup
    try:
        rng = random.Random(int(seed))
        w = rng.uniform(1.0, float(wmax))
        S_box, C_deg = LA.add_l(
            sq_don_list,
            float(s_don),
            int(d_add),
            float(cx),
            float(cy),
            w,
            rng,
            float(tmax),
        )
        if len(C_deg) != n_want:
            return tag, 1e9, None, 99
        Q_raw = np.asarray(C_deg, dtype=np.float64)
        Q_raw[:, 0] = np.clip(Q_raw[:, 0], 0.5, S_box - 0.5)
        Q_raw[:, 1] = np.clip(Q_raw[:, 1], 0.5, S_box - 0.5)
        Q_raw[:, 2] = np.radians(Q_raw[:, 2] % 90.0)

        rq = RE.run_fq_quench_args(
            Q_raw,
            [
                "--seed",
                str(int(seed)),
                "--loosen",
                repr(float(loosen)),
                "--outer",
                "60",
                "--pit",
                "320",
                "--flip-top",
                "24",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "24.5",
            ],
            s_ref=S_box,
            timeout_s=27.5,
        )
        if rq is None:
            return tag, 1e9, None, 99
        Sq, Qq = float(rq[0]), rq[1]
        k_ceil = math.ceil(math.sqrt(n_want) - 1e-12)
        n_jam = w59l.count_jam_lines_k(Qq, k_ceil)
        return tag, Sq, Qq, n_jam
    except Exception:
        return tag, 1e9, None, 99


def pick_kminus1_ladd_jobs(n, cur, bseed, max_jobs=28):
    K = math.isqrt(n)
    Km = K - 1
    if Km < 9:
        return [], []
    L0 = cur[n][0]
    donors = []
    # Check d from 2*Km + 1 (exact L count) up to 2*Km + 6 (tilted high-density refill!)
    for d in range(2 * Km + 1, 2 * Km + 7):
        n0 = n - d
        if Km * Km < n0 < K * K and cur[n0][1] is not None:
            s0 = cur[n0][0]
            # Prioritize small s0 + 1 plus gentle extra density (d - (2*Km+1))
            extra_d = max(0, d - (2 * Km + 1))
            score_don = (s0 + 1.0 - L0) + 0.012 * extra_d
            if s0 + 1.0 < L0 + 0.012:
                donors.append((score_don, n0, d, s0))
    donors.sort()
    top_donors = donors[:4]
    if not top_donors:
        return [], []

    jobs = []
    don_ids = []
    j_idx = 0
    for _, n0, d_add, _ in top_donors:
        don_ids.append(n0)
        P_don = RE.normalize_to_origin(cur[n0][1])
        S_don = RE.side_exact(P_don)
        P_deg = P_don.copy()
        P_deg[:, 2] = np.degrees(P_deg[:, 2]) % 90.0
        sq_list = [tuple(map(float, r)) for r in P_deg]
        cuts = best_ladd_cuts_fast(P_deg, S_don, step=0.08, m=5, sep=0.25)
        for ci, (tear_sc, cx, cy) in enumerate(cuts):
            for rep_i, lo in enumerate((1.02, 1.05)):
                if len(jobs) >= max_jobs:
                    break
                tmax = 22.0 if d_add > 2 * Km + 1 else 18.0
                wmax = 1.07 if d_add > 2 * Km + 2 else 1.05
                jobs.append(
                    (
                        f"ladd{n0}_d{d_add}_c{ci}_r{rep_i}",
                        sq_list,
                        S_don,
                        d_add,
                        n,
                        cx,
                        cy,
                        wmax,
                        tmax,
                        lo,
                        bseed + j_idx,
                    )
                )
                j_idx += 1
    return jobs[:max_jobs], don_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1820)
    ap.add_argument("--seed-off", type=int, default=960000)
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

    # Includes ALL shells up to n=1000 (especially K=31 `962, 976, ...` + top gap puzzles!)
    targets = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        if K * K == n:
            continue
        gap_l = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gap_l > 8.0e-5:
            bonus = 0.00015 if K == 31 else 0.0
            targets.append((gap_l / math.sqrt(n) + bonus, n))
    targets.sort(reverse=True)
    my_slice = [x[1] for x in targets[: args.num_shards * 4]][
        args.shard :: args.num_shards
    ]

    print(
        f"[W60A-FaithfulLAdd s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_slice={my_slice} ({time.time() - t0:.1f}s)",
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

    screened = []
    ex_sc = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_slice):
            if time.time() - t0 > 560.0:
                break
            L0 = cur[n][0]
            P0 = cur[n][1]
            bseed = args.seed_off + args.shard * 4507 + pi * 211
            jobs, don_ids = pick_kminus1_ladd_jobs(n, cur, bseed, max_jobs=28)
            if not jobs:
                continue

            ts0 = time.time()
            futs = [ex_sc.submit(eval_faithful_ladd_job, jb) for jb in jobs]
            hits = []
            direct_best = None
            try:
                for f in as_completed(futs, timeout=225.0):
                    tg, Sq, Qq, njam = f.result()
                    if Qq is None:
                        continue
                    gap0 = Sq - L0
                    ad = w53h.topology_angle_dist(Qq, P0) if P0 is not None else 1.0
                    if gap0 < -1e-9:
                        if direct_best is None or Sq < direct_best[0]:
                            direct_best = (Sq, tg, Qq, ad, njam)
                    if gap0 < -1e-9 or (gap0 < 0.0165 and ad > 0.035):
                        hits.append(
                            (
                                gap0 + 0.0015 * min(njam, 2),
                                gap0,
                                n,
                                tg,
                                Sq,
                                ad,
                                njam,
                                Qq,
                            )
                        )
            except TimeoutError:
                w51d.terminate_pool(ex_sc)
                ex_sc = ProcessPoolExecutor(max_workers=args.procs)

            if direct_best is not None:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    n, direct_best[2], deadline, slp_sec=40.0, fq_sec=24
                )
                L0_cur = wins[n][0] if n in wins else cur[n][0]
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        ds0 = (Lf0 - L0_cur) / math.sqrt(n)
                        tot_ds += ds0
                        wins[n] = (float(Lf0), xf0, yf0, tf0)
                        print(
                            f"  [W60A DIRECT LADD WIN!!] n={n:4d} [{direct_best[1]}] {cur[n][0]:.7f} -> {Lf0:.7f} (dS={ds0:+.8f})!",
                            flush=True,
                        )
                        save_ckpt()

            hits.sort(key=lambda z: z[0])
            uniq_n = []
            for h in hits:
                if all(abs(h[4] - u[4]) > 4e-6 for u in uniq_n):
                    uniq_n.append(h)
                if len(uniq_n) >= 3:
                    break
            b_str = (
                f"{uniq_n[0][1]:+.6f}(dA={uniq_n[0][5]:.2f},jam={uniq_n[0][6]})@{uniq_n[0][3]}"
                if uniq_n
                else "none"
            )
            print(
                f"  [W60A screen] n={n:4d}<-{don_ids} jobs={len(jobs)} hits={len(hits)} best={b_str} ({time.time() - ts0:.1f}s)",
                flush=True,
            )
            for u in uniq_n[:2]:
                screened.append(u)
    finally:
        w51d.terminate_pool(ex_sc)

    screened.sort(key=lambda z: z[0])
    queued = screened[:6]
    print(
        f"[W60A s={args.shard}] Phase B (Rust packer hhop + hop): hunting {len(queued)}/{len(screened)} ladd basins ({time.time() - t0:.1f}s)!",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (_, gap0, n, tag, S_init, adist, njam, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 65.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            best_Q, best_S = Q_init.copy(), float(S_init)

            if best_S < L0_cur + 0.0005:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    n, best_Q, deadline, slp_sec=36.0, fq_sec=22
                )
                if Sp0 < best_S:
                    best_Q, best_S = Qp0, Sp0

            rem = deadline - time.time()
            per_slot = min(
                175.0, max(45.0, (rem - 35.0) / max(1, min(3, len(queued) - i)))
            )
            bseed = args.seed_off + 50000 + args.shard * 9001 + i * 313
            k_ceil = math.ceil(math.sqrt(n) - 1e-12)
            forbid_k = k_ceil if (L0_cur < k_ceil + 1e-6 and njam == 0) else 0
            ts_goal = min(best_S, L0_cur) - 4.5e-5

            rust_tasks = [
                ("hhop", best_Q, ts_goal, bseed + 1, forbid_k, False, per_slot),
                ("hhop", best_Q, ts_goal, bseed + 2, forbid_k, True, per_slot),
                ("hop", best_Q, ts_goal, bseed + 3, forbid_k, False, per_slot),
                ("hop", best_Q, ts_goal, bseed + 4, forbid_k, True, per_slot),
            ]
            futs_r = [ex_hop.submit(w59l.run_rust_worker_task, rt) for rt in rust_tasks]
            best_mode = "init"
            try:
                for fr in as_completed(futs_r, timeout=per_slot + 35.0):
                    rr = fr.result()
                    if rr is not None and rr[0] < best_S - 1e-10:
                        best_S, best_Q, best_mode = rr[0], rr[1], rr[2]
            except TimeoutError:
                w51d.terminate_pool(ex_hop)
                ex_hop = ProcessPoolExecutor(max_workers=args.procs)

            if best_S < L0_cur + 0.0018 or best_S < S_init - 2e-4:
                best_Q, best_S = w51d.deep_contact_polish(
                    n, best_Q, deadline, slp_sec=45.0, fq_sec=26
                )

            if best_S < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, best_Q)
                if ok and Lf < L0_cur - 1e-9:
                    ds = (Lf - L0_cur) / math.sqrt(n)
                    tot_ds += ds
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W60A RUST WIN!!] n={n:4d} [{tag}->{best_mode}] dA={adist:.2f} {cur[n][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f})!",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W60A done] n={n:4d} [{tag}->{best_mode}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={best_S - L0_cur:+.6f}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W60A s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
