#!/usr/bin/env python3
"""Wave 57T (`w57t_pitch_stretch_and_lband2_milp.py`): Bidirectional Whole-Arm Pitch Stretch (`stretch.py`)
+ First-Principles Split-MILP L-Band Constructor (`lband2.assemble_split` with `rich=0.25`)!

72 Free Public Cores (`18` shards x `4` cores):
  - Shards 0..9 (40 Pitch-Stretch Cores): Stretches our sharpest tilted-arm record packings BOTH upward
    (`K -> K + 1`, propagating e.g. `380 -> 421+`, `702 -> 757+`, `758 -> 815+`!) AND downward (`K -> K - 1`)
    using measured arm pitch `d`, plus single-axis cuts + `delete-1`, quenched in compiled Rust `fq`!
  - Shards 10..17 (32 Split-MILP `lband2` Cores): Builds brand-new L-band geometries from first principles
    across `(N_arm in {3,4,5,6}, mode in {'right', 'right_opp'})` with `rich=0.25` fine upright junction grids,
    quenched in Rust `fq` -> KKT-polished 3-Round Wider-Hopper!
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

import lband2 as LB2
import rust_engines as RE
import stretch as ST
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w54s_fast_stag_splicer_and_sniper as w54s


def eval_stretch_donor(args_tuple):
    src_n, P_src_rad, cur_L_map, seed0 = args_tuple
    P = RE.normalize_to_origin(P_src_rad)
    raw_cands = []

    for stp, lbl in ((1, "st_up1"), (-1, "st_dn1")):
        try:
            b, _ = ST.stretch(P, stp)
            if b is not None and 80 <= len(b) <= 1000:
                raw_cands.append((f"{lbl}_{src_n}", b))
        except Exception:
            pass

    for ax in (0, 1):
        try:
            c = ST.pick_cuts(P, ax)
            if c is not None:
                for op, ol in ((1, f"st_ax{ax}p_{src_n}"), (-1, f"st_ax{ax}m_{src_n}")):
                    b1, _ = ST.apply(P, ax, c, op)
                    if b1 is not None and 80 <= len(b1) <= 1000:
                        raw_cands.append((ol, b1))
        except Exception:
            pass

    out = []
    for idx, (tag, b_raw) in enumerate(raw_cands):
        nn = len(b_raw)
        L_ref = cur_L_map.get(nn, math.sqrt(nn) + 0.65)
        Q_pre = w54s.relocate_crushed_fast(b_raw)
        rq = RE.run_fq_quench_args(
            Q_pre,
            [
                "--seed",
                str(seed0 + idx * 17),
                "--round-r0",
                "0.075",
                "--round-steps",
                "8",
                "--round-maxit",
                "950",
                "--round-gtol",
                "1e-8",
                "--hold-f",
                "1.003",
                "--loosen",
                "1.020",
                "--outer",
                "44",
                "--pit",
                "230",
                "--flip-top",
                "16",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "12.5",
            ],
            s_ref=max(RE.side_exact(Q_pre) * 0.99, L_ref),
            timeout_s=14.5,
        )
        if rq is None:
            continue
        Sq, Qq, _ = rq
        if Sq < L_ref + 0.0135:
            out.append((nn, float(Sq), tag, Qq))
            if nn - 1 >= 80:
                L_m1 = cur_L_map.get(nn - 1, Sq)
                for didx in w32a.pick_near_band(Qq, max_cand=4)[:1]:
                    Qd, Sd = w32a.fast_delete_and_compact(Qq, didx)
                    if Qd is not None and float(Sd) < L_m1 + 0.0135:
                        out.append(
                            (nn - 1, float(Sd), tag + "_d1", Qd.astype(np.float64))
                        )
    return out


def eval_lband2_job(args_tuple):
    n, K, f_try, N_arm, mode, ph_t, ph_r, L0, seed = args_tuple
    tag = f"lb2_N{N_arm}_{mode[:4]}_{ph_t:.1f}_{ph_r:.1f}"
    try:
        cnt, sol = LB2.assemble_split(
            K,
            f_try,
            N_arm,
            ph_t,
            ph_r,
            mode,
            ec=3,
            phases=4,
            jm=2.8,
            rich=0.25,
            time_limit=4.5,
            tol=0.012,
        )
        if sol is None or cnt < n:
            return tag, 1e9, None
        Q = np.asarray(sol, dtype=np.float64)
        while len(Q) > n:
            # Drop excess squares nearest to the tilted junction so count == n!
            cands_rm = w32a.pick_near_band(Q, max_cand=6)
            rm_i = int(cands_rm[0]) if len(cands_rm) else len(Q) - 1
            Q = np.delete(Q, rm_i, axis=0)
        Q = w54s.relocate_crushed_fast(Q)
        rq = RE.run_fq_quench_args(
            Q,
            [
                "--seed",
                str(int(seed)),
                "--round-r0",
                "0.065",
                "--round-steps",
                "6",
                "--round-maxit",
                "900",
                "--round-gtol",
                "1e-8",
                "--hold-f",
                "1.002",
                "--loosen",
                "1.018",
                "--outer",
                "44",
                "--pit",
                "240",
                "--flip-top",
                "16",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "13.0",
            ],
            s_ref=L0,
            timeout_s=15.0,
        )
        if rq is None:
            return tag, 1e9, None
        return tag, float(rq[0]), rq[1]
    except Exception:
        return tag, 1e9, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=18)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1780)
    ap.add_argument("--r1-sec", type=int, default=225)
    ap.add_argument("--r2-sec", type=int, default=205)
    ap.add_argument("--r3-sec", type=int, default=205)
    ap.add_argument("--seed-off", type=int, default=780000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
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
    if args.shard < 10:
        # Pitch-Stretch Shards (0..9)
        donors = []
        seen_d = set()
        for n in w54s.HOT_AND_ELITE_DONORS + list(range(980, 220, -7)):
            if n in seen_d or not (180 <= n <= 999):
                continue
            K = math.isqrt(n)
            if K * K == n or cur[n][0] >= K + 1 - 1e-6 or cur[n][1] is None:
                continue
            seen_d.add(n)
            donors.append(n)
        my_donors = donors[args.shard :: 10][:8]
        print(
            f"[W57T-PitchStretch s={args.shard}/{args.num_shards}] score={tot_score:.10f} donors={my_donors}",
            flush=True,
        )
        ex_st = ProcessPoolExecutor(max_workers=args.procs)
        cands = []
        try:
            futs = [
                ex_st.submit(
                    eval_stretch_donor,
                    (
                        dn,
                        cur[dn][1],
                        cur_L_map,
                        args.seed_off + args.shard * 997 + i * 41,
                    ),
                )
                for i, dn in enumerate(my_donors)
            ]
            try:
                for f in as_completed(futs, timeout=280.0):
                    for nn, Sq, tg, Qq in f.result():
                        ad = (
                            w53h.topology_angle_dist(Qq, cur[nn][1])
                            if cur[nn][1] is not None
                            else 1.0
                        )
                        gap0 = Sq - cur[nn][0]
                        if gap0 < -1e-9 or (gap0 < 0.0130 and ad > 0.05):
                            cands.append((gap0, nn, tg, Sq, ad, Qq))
            except TimeoutError:
                pass
        finally:
            w51d.terminate_pool(ex_st)
        cands.sort(key=lambda z: z[0])
        seen_n = {}
        for item in cands:
            if seen_n.get(item[1], 0) < 2:
                seen_n[item[1]] = seen_n.get(item[1], 0) + 1
                queued.append(item)
            if len(queued) >= 6:
                break
    else:
        # First-Principles `lband2` Split-MILP Shards (10..17)
        lb_shard = args.shard - 10
        lb_num = max(1, args.num_shards - 10)
        targets = []
        for n in range(150, 960):
            K = math.isqrt(n)
            if K * K == n:
                continue
            f0 = cur[n][0] - K
            gap_l = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
            if 0.62 < f0 < 0.995 and gap_l > 1.2e-4:
                targets.append((gap_l / math.sqrt(n), n))
        targets.sort(reverse=True)
        my_slice = [x[1] for x in targets[: lb_num * 6]][lb_shard::lb_num]
        print(
            f"[W57T-LBand2MILP s={args.shard}/{args.num_shards}] score={tot_score:.10f} slice={my_slice}",
            flush=True,
        )
        ex_lb = ProcessPoolExecutor(max_workers=args.procs)
        cands = []
        try:
            for pi, n in enumerate(my_slice):
                if time.time() - t0 > 320.0:
                    break
                K = math.isqrt(n)
                L0 = cur[n][0]
                f_try = max(0.55, (L0 - K) - 0.001)
                bseed = args.seed_off + args.shard * 1801 + pi * 73
                jobs = []
                for na in (3, 4, 5, 6):
                    if na >= K - 3:
                        continue
                    for md in ("right", "right_opp"):
                        jobs.append(
                            (
                                n,
                                K,
                                f_try,
                                na,
                                md,
                                (pi * 0.25) % 1.0,
                                ((pi + 1) * 0.35) % 1.0,
                                L0,
                                bseed + na * 11 + (0 if md == "right" else 5),
                            )
                        )
                futs = [ex_lb.submit(eval_lband2_job, jb) for jb in jobs]
                hits = []
                try:
                    for f in as_completed(futs, timeout=90.0):
                        tg, Sq, Qq = f.result()
                        if Qq is None:
                            continue
                        gap0 = Sq - L0
                        ad = (
                            w53h.topology_angle_dist(Qq, cur[n][1])
                            if cur[n][1] is not None
                            else 1.0
                        )
                        if gap0 < -1e-9 or (gap0 < 0.0135 and ad > 0.05):
                            hits.append((gap0, n, tg, Sq, ad, Qq))
                except TimeoutError:
                    w51d.terminate_pool(ex_lb)
                    ex_lb = ProcessPoolExecutor(max_workers=args.procs)
                hits.sort(key=lambda z: z[0])
                print(
                    f"  [W57T lband2] n={n:4d} hits={len(hits)} best={hits[0][0]:+.6f}@{hits[0][2]}"
                    if hits
                    else f"  [W57T lband2] n={n:4d} hits=0",
                    flush=True,
                )
                cands.extend(hits[:2])
        finally:
            w51d.terminate_pool(ex_lb)
        cands.sort(key=lambda z: z[0])
        queued = cands[:6]

    print(
        f"[W57T s={args.shard}] Phase B: hunting {len(queued)} targets ({time.time() - t0:.1f}s)!",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, nn, tag, S_init, adist, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 50.0:
                break
            L0_cur = wins[nn][0] if nn in wins else cur[nn][0]
            if S_init < L0_cur + 0.00035:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    nn, Q_init, deadline, slp_sec=36.0, fq_sec=22
                )
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(nn, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        tot_ds += (Lf0 - L0_cur) / math.sqrt(nn)
                        wins[nn] = (float(Lf0), xf0, yf0, tf0)
                        L0_cur = float(Lf0)
                        print(
                            f"  [W57T DIRECT WIN!!] n={nn:4d} [{tag}] {cur[nn][0]:.7f} -> {Lf0:.7f}!",
                            flush=True,
                        )
                        save_ckpt()

            r1 = min(args.r1_sec, max(35, int(0.35 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(30, int(0.30 * (rem - 20.0))))
            r3 = min(args.r3_sec, max(30, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + 30000 + args.shard * 13003 + i * 829
            Qb, Sb, (tr, ft, wn) = w51d.deep_3round_widehopper(
                ex_hop, nn, Q_init, L0_cur, bseed, r1, r2, r3, deadline
            )
            if Sb < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(nn, Qb)
                if ok and Lf < L0_cur - 1e-9:
                    ds = (Lf - L0_cur) / math.sqrt(nn)
                    tot_ds += ds
                    wins[nn] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W57T HOP WIN!!] n={nn:4d} [{tag}] dA={adist:.2f} {cur[nn][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={wn})",
                        flush=True,
                    )
                    save_ckpt()
                else:
                    print(
                        f"  [W57T miss] n={nn:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f}",
                        flush=True,
                    )
            else:
                print(
                    f"  [W57T done] n={nn:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} w={wn}",
                    flush=True,
                )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W57T s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
