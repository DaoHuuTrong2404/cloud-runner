#!/usr/bin/env python3
"""Wave 62T (`w62t_faithful_stretch_and_fillfast_transplant.py`): Faithful Whole-Arm Pitch Stretch
(`stretch.py` WITHOUT `relocate_crushed_fast` / `--round-r0`) + `fillfast` Exact Staircase Tilted-Channel
Transplant (`58ms` simplicial + MILP axis refill) + Hybrid Rust `packer` -> LP WideHopper!

Two clean, high-precision constructive mechanisms:
  1. Faithful `stretch.py` (`K -> K +- 1` + single-axis `apply` across ALL elite tilted-arm donors):
     Removes `w57t`'s destructive `relocate_crushed_fast` + `--round-r0` time-starvation and runs plain
     `fq quench --loosen 1.02 / 1.04 --deadline 22.0` on exact pitch-shifted arm seams!
  2. `fillfast.fill(s_try, T_deg, 'exact', tau)` Tilted-Channel Transplant (`fillfast.py` + `gen.py`):
     Extracts the tilted channel `T` from superior neighbor donors (`n+1, n+2, n-1, n + 2K + 1`), rescales
     into `[0, s_try]^2` at `s_try = L0 - 0.003`, and solves the exact Maximum Independent Set of staircase
     + 4-corner upright squares around `T` (`tau in {0.012, 0.028}`)! Whenever `len(X) >= n`, quenches `X[:n]`
     with plain `fq --loosen 1.02` and escalates through Hybrid Rust `packer` -> `w51d.deep_3round_widehopper`!
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
import scipy.optimize._milp as _milp

if hasattr(_milp, "_highs_wrapper"):
    _hw = _milp._highs_wrapper
    _milp._highs_wrapper = lambda *a: _hw(*a[:-1], dict(a[-1], threads=1))

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
TOOLS_DIR = os.path.abspath(os.path.join(HERE, "..", "tools"))
for _d in (DS_DIR, TOOLS_DIR, HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import fillfast as FF
import rust_engines as RE
import stretch as ST
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w59l_faithful_lcut_and_rust_hop as w59l


def eval_faithful_stretch_donor(job_tup):
    src_n, P_src, cur_L_map, seed0 = job_tup
    P = RE.normalize_to_origin(P_src)
    raw = []
    for stp, lbl in ((1, "st_up1"), (-1, "st_dn1")):
        try:
            b, _ = ST.stretch(P, stp)
            if b is not None and 90 <= len(b) <= 1000:
                raw.append((f"{lbl}_{src_n}", b))
        except Exception:
            pass
    for ax in (0, 1):
        try:
            c = ST.pick_cuts(P, ax)
            if c is not None:
                for op, ol in ((1, f"st_ax{ax}p_{src_n}"), (-1, f"st_ax{ax}m_{src_n}")):
                    b1, _ = ST.apply(P, ax, c, op)
                    if b1 is not None and 90 <= len(b1) <= 1000:
                        raw.append((ol, b1))
        except Exception:
            pass

    out = []
    for idx, (tag, b_raw) in enumerate(raw):
        nn = len(b_raw)
        L_ref = cur_L_map.get(nn, math.sqrt(nn) + 0.65)
        Q_in = RE.normalize_to_origin(b_raw)
        s_in = RE.side_exact(Q_in)
        if s_in > L_ref + 0.25:
            continue
        for rep_i, lo in enumerate((1.02, 1.04)):
            rq = RE.run_fq_quench_args(
                Q_in,
                [
                    "--seed",
                    str(int(seed0) + idx * 5 + rep_i),
                    "--loosen",
                    repr(float(lo)),
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
                s_ref=min(s_in, L_ref + 0.02),
                timeout_s=24.0,
            )
            if rq is not None:
                Sq, Qq = float(rq[0]), rq[1]
                if Sq < L_ref + 0.0155:
                    out.append((nn, Sq, f"{tag}_l{int(round(lo * 100))}", Qq))
                    break
    return out


def eval_fillfast_transplant(job_tup):
    n_want, L0_want, don_list, seed0 = job_tup
    hits = []
    for di, (m_don, P_don) in enumerate(don_list):
        P = RE.normalize_to_origin(P_don)
        s_don = RE.side_exact(P)
        half_pi = 0.5 * math.pi
        a_mod = np.mod(P[:, 2], half_pi)
        tilts = np.minimum(a_mod, half_pi - a_mod)
        t_idx = np.where(tilts > 0.02)[0]
        if len(t_idx) < 8 or len(t_idx) >= n_want:
            continue

        for ti, (ds_off, tau) in enumerate(
            ((-0.002, 0.015), (-0.005, 0.028), (0.001, 0.008))
        ):
            s_try = L0_want + ds_off
            scale = s_try / max(s_don, 1e-9)
            T_deg = P[t_idx].copy()
            T_deg[:, 0] = np.clip(T_deg[:, 0] * scale, 0.52, s_try - 0.52)
            T_deg[:, 1] = np.clip(T_deg[:, 1] * scale, 0.52, s_try - 0.52)
            T_deg[:, 2] = np.degrees(T_deg[:, 2]) % 90.0
            try:
                X_deg = FF.fill(s_try, T_deg, mode="exact", tau=tau, time_limit=3.0)
            except Exception:
                continue
            if len(X_deg) < n_want:
                continue
            # Keep all tilted squares + upright squares sorted from corners inward
            X_use = X_deg[:n_want].copy()
            X_use[:, 2] = np.radians(X_use[:, 2])
            rq = RE.run_fq_quench_args(
                X_use,
                [
                    "--seed",
                    str(int(seed0) + di * 11 + ti),
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
                s_ref=s_try,
                timeout_s=24.0,
            )
            if rq is not None:
                Sq, Qq = float(rq[0]), rq[1]
                if Sq < L0_want + 0.0155:
                    hits.append((n_want, Sq, f"ff_{m_don}_t{int(tau * 1000)}", Qq))
                    break
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=18)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=1340000)
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

    # 1) Select 6 elite tilted-arm donors per shard for faithful `stretch.py` + 4 high-gap targets for `fillfast`!
    donors_all = []
    for n in range(160, 1000):
        K = math.isqrt(n)
        if K * K == n or cur[n][1] is None or cur[n][0] >= K + 1 - 1e-6:
            continue
        # Best donors have tiny gap to public frontier (already near world-record!)
        gap_f = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        donors_all.append((gap_f, -(n % 17), n))
    donors_all.sort()
    my_stretch_donors = [x[2] for x in donors_all[args.shard :: args.num_shards][:6]]

    gap_targets = []
    for n in range(120, 1001):
        K = math.isqrt(n)
        if K * K == n:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gl > 8.0e-5:
            gap_targets.append((gl / math.sqrt(n), n))
    gap_targets.sort(reverse=True)
    my_ff_targets = [x[1] for x in gap_targets[: args.num_shards * 4]][
        args.shard :: args.num_shards
    ]

    print(
        f"[W62T-Stretch+FillFast s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"st_donors={my_stretch_donors} ff_targets={my_ff_targets} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    screened = []
    ex_sc = ProcessPoolExecutor(max_workers=args.procs)
    try:
        futs = []
        for i, dn in enumerate(my_stretch_donors):
            futs.append(
                ex_sc.submit(
                    eval_faithful_stretch_donor,
                    (
                        dn,
                        cur[dn][1],
                        cur_L_map,
                        args.seed_off + args.shard * 2111 + i * 37,
                    ),
                )
            )
        for j, tn in enumerate(my_ff_targets):
            nb_dons = []
            for m in (tn + 1, tn + 2, tn - 1):
                if (
                    50 <= m <= 1000
                    and cur[m][1] is not None
                    and math.isqrt(m) ** 2 != m
                ):
                    nb_dons.append((m, cur[m][1]))
            if nb_dons:
                futs.append(
                    ex_sc.submit(
                        eval_fillfast_transplant,
                        (
                            tn,
                            cur[tn][0],
                            nb_dons,
                            args.seed_off + 9000 + args.shard * 1709 + j * 53,
                        ),
                    )
                )

        try:
            for f in as_completed(futs, timeout=540.0):
                for nn, Sq, tg, Qq in f.result():
                    L0 = wins[nn][0] if nn in wins else cur[nn][0]
                    P0 = cur[nn][1]
                    if Sq < L0 - 1e-9:
                        Qp0, Sp0 = w51d.deep_contact_polish(
                            nn, Qq, deadline, slp_sec=40.0, fq_sec=24
                        )
                        ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(nn, Qp0)
                        if ok0 and Lf0 < L0 - 1e-9:
                            ds0 = (Lf0 - L0) / math.sqrt(nn)
                            tot_ds += ds0
                            wins[nn] = (float(Lf0), xf0, yf0, tf0)
                            L0 = float(Lf0)
                            print(
                                f"  [W62T DIRECT WIN!!] n={nn:4d} [{tg}] {cur[nn][0]:.7f} -> {Lf0:.7f} (dS={ds0:+.8f})!",
                                flush=True,
                            )
                            save_ckpt()
                    ad = w53h.topology_angle_dist(Qq, P0) if P0 is not None else 1.0
                    if Sq < L0 - 1e-9 or (Sq < L0 + 0.0155 and ad > 0.030):
                        screened.append((Sq - L0, nn, tg, Sq, ad, Qq))
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_sc)

    screened.sort(key=lambda z: z[0])
    uniq = []
    seen_n = {}
    for item in screened:
        nn = item[1]
        if seen_n.get(nn, 0) >= 2:
            continue
        seen_n[nn] = seen_n.get(nn, 0) + 1
        uniq.append(item)
        if len(uniq) >= 5:
            break

    print(
        f"[W62T s={args.shard}] Phase B: hunting {len(uniq)}/{len(screened)} stretch/fillfast basins ({time.time() - t0:.1f}s)!",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, tag, S_init, adist, Q_init) in enumerate(uniq):
            rem = deadline - time.time()
            if rem < 75.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            best_Q, best_S = RE.normalize_to_origin(Q_init), float(S_init)

            per_slot = min(125.0, max(40.0, (rem - 60.0) * 0.25))
            bseed = args.seed_off + 70000 + args.shard * 9001 + i * 313
            k_ceil = math.ceil(math.sqrt(n) - 1e-12)
            forbid_k = k_ceil if L0_cur < k_ceil + 1e-6 else 0
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

            rem2 = deadline - time.time()
            wn_cnt = 0
            if best_S < L0_cur + 0.0048 and rem2 > 85.0:
                r1 = min(155, max(35, int(0.35 * (rem2 - 25.0))))
                r2 = min(140, max(30, int(0.30 * (rem2 - 25.0))))
                r3 = min(140, max(30, int(0.30 * (rem2 - 25.0))))
                Qw, Sw, (_, _, wn_cnt) = w51d.deep_3round_widehopper(
                    ex_hop, n, best_Q, L0_cur, bseed + 700, r1, r2, r3, deadline
                )
                if Sw < best_S - 1e-10:
                    best_S, best_Q, best_mode = Sw, Qw, best_mode + "->wh3r"
            elif best_S < L0_cur + 0.0020 or best_S < S_init - 2e-4:
                best_Q, best_S = w51d.deep_contact_polish(
                    n, best_Q, deadline, slp_sec=44.0, fq_sec=26
                )

            if best_S < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, best_Q)
                if ok and Lf < L0_cur - 1e-9:
                    ds = (Lf - L0_cur) / math.sqrt(n)
                    tot_ds += ds
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W62T HYBRID WIN!!] n={n:4d} [{tag}->{best_mode}] dA={adist:.2f} {cur[n][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={wn_cnt})!",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W62T done] n={n:4d} [{tag}->{best_mode}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={best_S - L0_cur:+.6f} w={wn_cnt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W62T s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
