#!/usr/bin/env python3
"""Wave 58P (`w58p_fresh_pub_escalator.py`): 18-Minute Lightning Escalation on Brand-New
Late-Evening Public Drop Basins (`n=491, 556, 707` + any `pub_v15` wins)!

24 shards x 4 cores = 96 Free Public Cores (`budget-sec 1020`, finishes cleanly before `23:42 UTC`!):
  - Shards 0..11 (48 Cores): 4x independent multi-seed `1000s` KKT Wider-Hopper escalations (`r1=340, r2=320, r3=320`)
    directly on the brand-new `n=491`, `n=556`, and `n=707` packings (which have never seen our 3-Round
    `eta<=0.28` Wider-Hopper + second-order KKT `newton_finish` + SQP Hessian!).
  - Shards 12..17 (24 Cores): Exhaustive `delete-1` (`490, 555, 706`), CC-bar splices (`w54s`), and exact-`N`
    `lcut` transplants downward from `491 -> 446..449`, `556 -> 508..511`, `707 -> 653..656`!
  - Shards 18..23 (24 Cores): 8-Sym 4D Fault-DP re-composition incorporating the brand-new `(22,22,7)` (`n=491`),
    `(23,23,27)` (`n=556`), and `(26,26,31)` (`n=707`) gadgets!
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

import fault_dp_8sym as fdp8
import lcut as LC
import rust_engines as RE
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w54s_fast_stag_splicer_and_sniper as w54s
import w56g_unified_global_faultdp as w56g
import w57l_inexact_lcut_exact_n as w57l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--pub-v15-npz", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=24)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1020)
    ap.add_argument("--seed-off", type=int, default=810000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )
    cur_L_map = {n: cur[n][0] for n in range(1, 1001)}

    pub_n_list = []
    if os.path.exists(args.pub_v15_npz):
        with np.load(args.pub_v15_npz) as zp:
            for k in sorted(zp.files):
                if k.startswith("x_"):
                    pub_n_list.append(int(k[2:]))
    if not pub_n_list:
        pub_n_list = [491, 556, 707]

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

    queued = []
    if args.shard < 12:
        # Direct multi-seed KKT + Wider-Hopper on fresh pub_v15 packings!
        tgt_n = pub_n_list[args.shard % len(pub_n_list)]
        P0 = cur[tgt_n][1]
        L0 = cur[tgt_n][0]
        queued.append((0.0, tgt_n, f"pubhop_s{args.shard}", L0, P0))
        r1_c, r2_c, r3_c = 330, 310, 310
    elif args.shard < 18:
        # Delete-1 + CC-bar splices + exact-N lcut downward from fresh pub_v15 donors!
        sub_s = args.shard - 12
        tgt_don = pub_n_list[sub_s % len(pub_n_list)]
        P_don = cur[tgt_don][1]
        cands = []
        # 1) Exhaustive near-band delete-1 -> tgt_don - 1
        nn1 = tgt_don - 1
        for didx in w32a.pick_near_band(P_don, max_cand=24)[:16]:
            Qd, Sd = w32a.fast_delete_and_compact(P_don, didx)
            if Qd is not None and float(Sd) < cur[nn1][0] + 0.014:
                cands.append(
                    (
                        float(Sd) - cur[nn1][0],
                        nn1,
                        f"del1_{tgt_don}_{didx}",
                        float(Sd),
                        Qd.astype(np.float64),
                    )
                )
        # 2) CC-bar splices
        for nn, Sq, tg, Qq in w54s.generate_fast_splices((tgt_don, P_don, cur_L_map)):
            if Sq < cur[nn][0] + 0.014:
                cands.append((Sq - cur[nn][0], nn, tg, Sq, Qq))
        # 3) Exact-N lcut to shell K - 1
        Kd = math.isqrt(tgt_don)
        Dd = tgt_don - Kd * Kd
        Km = Kd - 1
        P_norm = RE.normalize_to_origin(P_don)
        S_norm = RE.side_exact(P_norm)
        P_deg = P_norm.copy()
        P_deg[:, 2] = np.degrees(P_deg[:, 2]) % 90.0
        ex_lc = ProcessPoolExecutor(max_workers=args.procs)
        try:
            lc_jobs = []
            for nw in range(
                max(80, Km * Km + max(1, Dd - 3)),
                min(Km * Km + Dd + 2, Kd * Kd - 1) + 1,
            ):
                cuts = LC.best_cuts(
                    P_deg, S_norm, tgt_don - nw, step=0.07, m=2, sep=0.20
                )
                for ci, (_, cx, cy, _) in enumerate(cuts):
                    lc_jobs.append(
                        (
                            f"lcut_{tgt_don}_to_{nw}_c{ci}",
                            P_deg,
                            S_norm,
                            cx,
                            cy,
                            tgt_don - nw,
                            nw,
                            cur[nw][0],
                            args.seed_off + args.shard * 401 + nw + ci,
                        )
                    )
            futs = [ex_lc.submit(w57l.eval_lcut_job, jb) for jb in lc_jobs[:12]]
            try:
                for f in as_completed(futs, timeout=80.0):
                    tg, Sq, Qq = f.result()
                    if Qq is not None and Sq < cur[len(Qq)][0] + 0.014:
                        cands.append((Sq - cur[len(Qq)][0], len(Qq), tg, Sq, Qq))
            except TimeoutError:
                pass
        finally:
            w51d.terminate_pool(ex_lc)
        cands.sort(key=lambda z: z[0])
        queued = cands[sub_s // len(pub_n_list) :: 2][:3]
        r1_c, r2_c, r3_c = 250, 220, 220
    else:
        # 8-Sym 4D Fault-DP re-solve with fresh pub_v15 donors!
        dp_s = args.shard - 18
        donors = [
            cur[n][1]
            for n in range(5, 1001)
            if math.isqrt(n) ** 2 != n
            and cur[n][0] < math.isqrt(n) + 1 - 1e-6
            and cur[n][1] is not None
        ]
        G = fdp8.extract_gadgets(donors)
        dp0, PG0, PM0, PQ0 = fdp8.solve_dp(G)
        cands = []
        check_ns = [
            n
            for n in range(100, 1001)
            if math.isqrt(n) ** 2 != n and 1 <= n - math.isqrt(n) ** 2 <= fdp8.DMAX
        ][dp_s::6]
        for n in check_ns:
            if time.time() - t0 > 180.0:
                break
            K = math.isqrt(n)
            D = n - K * K
            L0 = cur[n][0]
            for L_est, dtag, P_comp in fdp8.trace_and_compose(
                dp0, PG0, PM0, PQ0, G, donors, K, D, top_m=2
            ):
                if L_est < L0 + 0.008:
                    rh = w56g.quench_composed_chain(P_comp, L0)
                    if rh is not None and rh[1] < L0 + 0.012:
                        cands.append((rh[1] - L0, n, dtag, rh[1], rh[0]))
        cands.sort(key=lambda z: z[0])
        queued = cands[:3]
        r1_c, r2_c, r3_c = 240, 210, 210

    print(
        f"[W58P s={args.shard}/{args.num_shards}] score={tot_score:.10f} pub_n={pub_n_list} queued={len(queued)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, nn, tag, S_init, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 45.0:
                break
            L0_cur = wins[nn][0] if nn in wins else cur[nn][0]
            Qp0, Sp0 = w51d.deep_contact_polish(
                nn, Q_init, deadline, slp_sec=32.0, fq_sec=20
            )
            if Sp0 < L0_cur - 1e-9:
                ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(nn, Qp0)
                if ok0 and Lf0 < L0_cur - 1e-9:
                    tot_ds += (Lf0 - L0_cur) / math.sqrt(nn)
                    wins[nn] = (float(Lf0), xf0, yf0, tf0)
                    L0_cur = float(Lf0)
                    print(
                        f"  [W58P DIRECT WIN!!] n={nn:4d} [{tag}] {cur[nn][0]:.7f} -> {Lf0:.7f}!",
                        flush=True,
                    )
                    save_ckpt()
            r1 = min(r1_c, max(35, int(0.35 * (rem - 18.0))))
            r2 = min(r2_c, max(30, int(0.30 * (rem - 18.0))))
            r3 = min(r3_c, max(30, int(0.30 * (rem - 18.0))))
            bseed = args.seed_off + 20000 + args.shard * 14009 + i * 877
            Qb, Sb, (tr, ft, wn) = w51d.deep_3round_widehopper(
                ex_hop, nn, Qp0, L0_cur, bseed, r1, r2, r3, deadline
            )
            if Sb < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(nn, Qb)
                if ok and Lf < L0_cur - 1e-9:
                    ds = (Lf - L0_cur) / math.sqrt(nn)
                    tot_ds += ds
                    wins[nn] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W58P HOP WIN!!] n={nn:4d} [{tag}] {cur[nn][0]:.7f} -> {Lf:.7f} (dL={Lf - cur[nn][0]:+.7f}, dS={ds:+.8f}, w={wn})",
                        flush=True,
                    )
                    save_ckpt()
            else:
                print(
                    f"  [W58P done] n={nn:4d} [{tag}] dEnd={Sb - L0_cur:+.6f} w={wn}",
                    flush=True,
                )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W58P s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
