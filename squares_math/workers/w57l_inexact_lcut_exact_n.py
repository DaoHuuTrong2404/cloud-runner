#!/usr/bin/env python3
"""Wave 57L (`w57l_inexact_lcut_exact_n.py`): Vectorized 7x7 Area-Scored Inexact L-Cut (`lcut.py`)
for Exact `n0 -> n` Shell-Down Transplantation across ALL Top 110 High-Gap Puzzles!

Solves the fundamental bar-cut discrete-jump limitation:
  - Previous bar-cut waves (`W52S..W55B`) deleted whole tilted bars (`b_p + b_n`), skipping 85% of our
    highest-gap puzzles (`234, 367, 491, 333, 738, 456, 730, 261, 904, 414, 332, 804, 914, 748, ...`).
  - `lcut.best_cuts` + `lcut.apply_cut` takes ANY superior donor `n0` in shell `K + 1` (where `L[n0] - 1.0`
    is smaller than `L[n]`!), finds the `[cx, cx+1) u [cy, cy+1)` strip union minimizing area damage across
    49 sample points/square, and removes EXACTLY `d = n0 - n` squares so `len(C) == n` 100% of the time!
  - Relaxes each exact-`n` cut via compiled Rust `fq --round-r0 0.08 --finish --stag-w 0`, then escalates
    the sharpest new basins through KKT-polished 3-Round Wider-Hopper!
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

import lcut as LC
import rust_engines as RE
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w54s_fast_stag_splicer_and_sniper as w54s


def eval_lcut_job(job_tup):
    tag, P_don_deg, S_don, cx, cy, d_rem, n_want, L0_want, seed = job_tup
    try:
        C_deg = LC.apply_cut(P_don_deg, float(cx), float(cy), int(d_rem))
        if len(C_deg) != n_want:
            return tag, 1e9, None
        s1 = max(float(S_don) - 1.0, float(L0_want))
        Q_raw = np.asarray(C_deg, dtype=np.float64)
        Q_raw[:, 0] = np.clip(Q_raw[:, 0], 0.5, s1 - 0.5)
        Q_raw[:, 1] = np.clip(Q_raw[:, 1], 0.5, s1 - 0.5)
        Q_raw[:, 2] = np.radians(Q_raw[:, 2])
        Q_raw = w54s.relocate_crushed_fast(Q_raw)

        rq = RE.run_fq_quench_args(
            Q_raw,
            [
                "--seed",
                str(int(seed)),
                "--round-r0",
                "0.080",
                "--round-steps",
                "8",
                "--round-maxit",
                "950",
                "--round-gtol",
                "1e-8",
                "--hold-f",
                "1.003",
                "--loosen",
                "1.022",
                "--outer",
                "45",
                "--pit",
                "240",
                "--flip-top",
                "16",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "13.5",
            ],
            s_ref=s1,
            timeout_s=15.5,
        )
        if rq is None:
            return tag, 1e9, None
        return tag, float(rq[0]), rq[1]
    except Exception:
        return tag, 1e9, None


def pick_kplus1_donors(n, cur):
    K = math.isqrt(n)
    if K >= 31:
        return []
    D = n - K * K
    Kp = K + 1
    lo = max(Kp * Kp + 1, Kp * Kp + D - 2)
    hi = min(1000, Kp * Kp + D + 9, (Kp + 1) * (Kp + 1) - 1)
    cands = []
    for n0 in range(lo, hi + 1):
        if cur[n0][1] is None or math.isqrt(n0) ** 2 == n0:
            continue
        f0 = cur[n0][0] - Kp
        if f0 < 0.999:
            cands.append((f0, n0))
    cands.sort()
    return [n0 for _, n0 in cands[:3]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=22)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1780)
    ap.add_argument("--r1-sec", type=int, default=225)
    ap.add_argument("--r2-sec", type=int, default=205)
    ap.add_argument("--r3-sec", type=int, default=205)
    ap.add_argument("--seed-off", type=int, default=760000)
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

    targets = []
    for n in range(100, 961):
        K = math.isqrt(n)
        if K * K == n or K >= 31:
            continue
        gap_l = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gap_l > 1.0e-4:
            targets.append((gap_l / math.sqrt(n), n))
    targets.sort(reverse=True)
    my_slice = [x[1] for x in targets[: args.num_shards * 5]][
        args.shard :: args.num_shards
    ]

    print(
        f"[W57L-InexactLCut s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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
            if time.time() - t0 > 320.0:
                break
            L0 = cur[n][0]
            P0 = cur[n][1]
            donors_n0 = pick_kplus1_donors(n, cur)
            if not donors_n0:
                continue
            bseed = args.seed_off + args.shard * 3109 + pi * 127
            jobs = []
            for di, n0 in enumerate(donors_n0):
                P_don = RE.normalize_to_origin(cur[n0][1])
                S_don = RE.side_exact(P_don)
                P_deg = P_don.copy()
                P_deg[:, 2] = np.degrees(P_deg[:, 2]) % 90.0
                cuts = LC.best_cuts(P_deg, S_don, n0 - n, step=0.06, m=4, sep=0.18)
                for ci, (sc, cx, cy, _) in enumerate(cuts):
                    jobs.append(
                        (
                            f"lcut_{n0}_c{ci}",
                            P_deg,
                            S_don,
                            cx,
                            cy,
                            n0 - n,
                            n,
                            L0,
                            bseed + di * 50 + ci,
                        )
                    )
            if not jobs:
                continue
            ts0 = time.time()
            futs = [ex_sc.submit(eval_lcut_job, jb) for jb in jobs]
            hits = []
            try:
                for f in as_completed(futs, timeout=95.0):
                    tg, Sq, Qq = f.result()
                    if Qq is None:
                        continue
                    gap0 = Sq - L0
                    ad = w53h.topology_angle_dist(Qq, P0) if P0 is not None else 1.0
                    if gap0 < -1e-9 or (gap0 < 0.0135 and ad > 0.05):
                        hits.append((gap0, n, tg, Sq, ad, Qq))
            except TimeoutError:
                w51d.terminate_pool(ex_sc)
                ex_sc = ProcessPoolExecutor(max_workers=args.procs)

            hits.sort(key=lambda z: z[0])
            b_str = (
                f"{hits[0][0]:+.6f}(dA={hits[0][4]:.2f})@{hits[0][2]}"
                if hits
                else "none"
            )
            print(
                f"  [W57L screen] n={n:4d}<-{donors_n0} hits={len(hits)} best={b_str} ({time.time() - ts0:.1f}s)",
                flush=True,
            )
            for h in hits[:2]:
                screened.append(h)
    finally:
        w51d.terminate_pool(ex_sc)

    screened.sort(key=lambda z: z[0])
    queued = screened[:6]
    print(
        f"[W57L s={args.shard}] Phase B: hunting {len(queued)}/{len(screened)} exact-n lcut basins ({time.time() - t0:.1f}s)!",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, tag, S_init, adist, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 50.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            if S_init < L0_cur + 0.00035:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    n, Q_init, deadline, slp_sec=36.0, fq_sec=22
                )
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        tot_ds += (Lf0 - L0_cur) / math.sqrt(n)
                        wins[n] = (float(Lf0), xf0, yf0, tf0)
                        L0_cur = float(Lf0)
                        print(
                            f"  [W57L DIRECT LCUT WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf0:.7f}!",
                            flush=True,
                        )
                        save_ckpt()

            r1 = min(args.r1_sec, max(35, int(0.35 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(30, int(0.30 * (rem - 20.0))))
            r3 = min(args.r3_sec, max(30, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + 30000 + args.shard * 12011 + i * 787
            Qb, Sb, (tr, ft, wn) = w51d.deep_3round_widehopper(
                ex_hop, n, Q_init, L0_cur, bseed, r1, r2, r3, deadline
            )
            if Sb < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, Qb)
                if ok and Lf < L0_cur - 1e-9:
                    ds = (Lf - L0_cur) / math.sqrt(n)
                    tot_ds += ds
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W57L HOP WIN!!] n={n:4d} [{tag}] dA={adist:.2f} {cur[n][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={wn})",
                        flush=True,
                    )
                    save_ckpt()
                else:
                    print(
                        f"  [W57L miss] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f}",
                        flush=True,
                    )
            else:
                print(
                    f"  [W57L done] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} w={wn}",
                    flush=True,
                )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W57L s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
