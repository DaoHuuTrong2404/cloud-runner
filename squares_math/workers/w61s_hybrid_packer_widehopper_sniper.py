#!/usr/bin/env python3
"""Wave 61S (`w61s_hybrid_packer_widehopper_sniper.py`): Multi-Seed `bandflip` + Plain `del1`
-> Native Rust `packer hhop`/`hop` Fresh-Topology Harvester -> `w51d.deep_3round_widehopper` Sniper!

Closes the exact handoff gap discovered in Wave 59P logs (`task-7277`):
  - In `W59P`, compiled Rust `packer hhop` & `packer hop` discovered genuinely distinct topologies (`dA > 0.025`)
    within micro-margins of our R35 records (`n=638 +1e-6`, `n=738 +2e-6`, `n=418 +1.8e-5`, `n=414 +2.3e-5`,
    `n=966 +6.03e-4`), then exited without running `w51d.deep_3round_widehopper` (exact linearized LP contact
    steps + Residual Contact Continuation + `fq --sqp --finish`) on those new basins!
  - Here, for each of the Top 64 all-shell gap puzzles (incl. `K=31` `962, 976, 966` & `K^2+1` `401, 530, 730, 785`):
    1. Screens `8` faithful `bandflip` junction swaps + `4` plain-`fq` near-band `del1` seeds (`~28s`).
    2. Runs 4-core native Rust `packer hhop` & `packer hop` (`~135s`) to harvest the tightest fresh topology (`dA > 0.020`).
    3. Chains that fresh Rust basin directly into `w51d.deep_3round_widehopper` (`r1=155, r2=140, r3=140`)!
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

import bandflip as BF
import rust_engines as RE
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w59p_rust_packer_hop_assault as w59p


def eval_bflip_or_del1_job(job_tup):
    kind, P_src, s_ref, aux_idx, loosen, seed = job_tup
    try:
        if kind == "bflip":
            sq_deg = P_src.copy()
            sq_deg[:, 2] = np.degrees(sq_deg[:, 2]) % 90.0
            B = BF.bands(sq_deg, min_size=11)
            if not B:
                return None
            bi = int(aux_idx) % len(B)
            rng = random.Random(int(seed))
            out_deg, _ = BF.flip(float(s_ref), sq_deg, B[bi], rng, B)
            Q_raw = np.asarray(out_deg, dtype=np.float64)
            Q_raw[:, 0] = np.clip(Q_raw[:, 0], 0.5, float(s_ref) - 0.5)
            Q_raw[:, 1] = np.clip(Q_raw[:, 1], 0.5, float(s_ref) - 0.5)
            Q_raw[:, 2] = np.radians(Q_raw[:, 2])
            tag = f"bf{bi}_l{int(round(loosen * 100))}"
        else:
            Q_raw = np.delete(P_src, int(aux_idx), axis=0)
            tag = f"del1_{int(aux_idx)}"

        rq = RE.run_fq_quench_args(
            Q_raw,
            [
                "--seed",
                str(int(seed)),
                "--loosen",
                repr(float(loosen)),
                "--outer",
                "55",
                "--pit",
                "280",
                "--flip-top",
                "20",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "15.5",
            ],
            s_ref=float(s_ref),
            timeout_s=18.0,
        )
        if rq is None:
            return None
        return float(rq[0]), rq[1], tag
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=1180000)
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

    # Prioritize W59P micro-gap puzzles + K=31 + K^2+1 + top all-shell gaps!
    hot_w59 = {
        401,
        414,
        418,
        638,
        738,
        966,
        962,
        976,
        730,
        785,
        530,
        367,
        759,
        495,
        376,
        746,
        591,
    }
    targets = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        if K * K == n:
            continue
        gap_l = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gap_l > 6.0e-5 or n in hot_w59:
            bonus = (0.00035 if n in hot_w59 else 0.0) + (
                0.00018 if (K == 31 or n == K * K + 1) else 0.0
            )
            targets.append((gap_l / math.sqrt(n) + bonus, n))
    targets.sort(reverse=True)
    my_slice = [x[1] for x in targets[: args.num_shards * 4]][
        args.shard :: args.num_shards
    ]

    print(
        f"[W61S-HybridSniper s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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

    ex = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_slice):
            rem = deadline - time.time()
            if rem < 80.0:
                break
            puzzles_left = max(1, len(my_slice) - pi)
            puz_budget = min(465.0, max(95.0, (rem - 25.0) / puzzles_left))
            tp0 = time.time()

            L0 = wins[n][0] if n in wins else cur[n][0]
            P0 = RE.normalize_to_origin(cur[n][1])
            bseed = args.seed_off + args.shard * 7001 + pi * 503

            # 1) Screen 8 bandflip + 4 del1 seeds (~32s across 4 cores)
            sc_jobs = []
            for j in range(6):
                sc_jobs.append(
                    ("bflip", P0, L0, j % 2, 1.02 if j % 2 == 0 else 1.05, bseed + j)
                )
            if (
                n < 1000
                and cur[n + 1][1] is not None
                and math.isqrt(n + 1) ** 2 != (n + 1)
                and (cur[n + 1][0] - L0) < 0.025
            ):
                P1 = RE.normalize_to_origin(cur[n + 1][1])
                try:
                    _, _, _, _, load = E.skeleton(P1)
                except Exception:
                    load = np.zeros(len(P1))
                half_pi = 0.5 * math.pi
                a_mod = np.mod(P1[:, 2], half_pi)
                tilts = np.minimum(a_mod, half_pi - a_mod)
                near = np.where(tilts > 0.015)[0]
                if len(near) > 0:
                    order = near[np.argsort(load[near])]
                    for k_i, r_idx in enumerate(order[:4]):
                        sc_jobs.append(
                            ("del1", P1, L0, int(r_idx), 1.010, bseed + 20 + k_i)
                        )

            alt_cands = []
            futs_sc = [ex.submit(eval_bflip_or_del1_job, sj) for sj in sc_jobs]
            try:
                for fs in as_completed(futs_sc, timeout=55.0):
                    rs = fs.result()
                    if rs is not None:
                        Ss, Qs, tgs = rs
                        ad = w53h.topology_angle_dist(Qs, P0)
                        if Ss < L0 - 1e-9 or (Ss < L0 + 0.0145 and ad > 0.025):
                            alt_cands.append((Ss, Qs, tgs, ad))
            except TimeoutError:
                w51d.terminate_pool(ex)
                ex = ProcessPoolExecutor(max_workers=args.procs)

            alt_cands.sort(key=lambda z: z[0])
            if alt_cands and alt_cands[0][0] < L0 - 1e-9:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    n, alt_cands[0][1], deadline, slp_sec=40.0, fq_sec=24
                )
                ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                if ok0 and Lf0 < L0 - 1e-9:
                    ds0 = (Lf0 - L0) / math.sqrt(n)
                    tot_ds += ds0
                    wins[n] = (float(Lf0), xf0, yf0, tf0)
                    L0 = float(Lf0)
                    P0 = RE.normalize_to_origin(np.column_stack([xf0, yf0, tf0]))
                    print(
                        f"  [W61S DIRECT SEED WIN!!] n={n:4d} [{alt_cands[0][2]}] {cur[n][0]:.7f} -> {Lf0:.7f}!",
                        flush=True,
                    )
                    save_ckpt()

            # 2) Stage 1: Native Rust `packer hhop` & `packer hop` (~125s across 4 cores)
            s1_slot = min(130.0, max(45.0, (puz_budget - (time.time() - tp0)) * 0.34))
            k_ceil = math.ceil(math.sqrt(n) - 1e-12)
            forbid_k = k_ceil if L0 < k_ceil + 1e-6 else 0
            ts_goal = L0 - 4.5e-5

            slots = [
                ("hhop", P0, ts_goal, bseed + 101, forbid_k, False, s1_slot, "cur"),
                ("hhop", P0, ts_goal, bseed + 102, forbid_k, True, s1_slot, "cur"),
                ("hop", P0, ts_goal, bseed + 103, forbid_k, True, s1_slot, "cur"),
            ]
            if alt_cands:
                slots.append(
                    (
                        "combo",
                        alt_cands[0][1],
                        ts_goal,
                        bseed + 104,
                        forbid_k,
                        True,
                        s1_slot,
                        alt_cands[0][2],
                    )
                )
            else:
                slots.append(
                    ("hop", P0, ts_goal, bseed + 104, forbid_k, False, s1_slot, "cur")
                )

            futs_p = [ex.submit(w59p.eval_packer_slot, sl) for sl in slots]
            best_S, best_Q, best_tag = L0, P0, "cur_init"
            fresh_S, fresh_Q, fresh_tag = 1e9, None, ""
            try:
                for fp in as_completed(futs_p, timeout=s1_slot + 38.0):
                    rr = fp.result()
                    if rr is None:
                        continue
                    Sr, Qr, tr_tag = rr
                    if Sr < best_S - 1e-10:
                        best_S, best_Q, best_tag = Sr, Qr, tr_tag
                    ad = w53h.topology_angle_dist(Qr, P0)
                    if ad > 0.020 and Sr < fresh_S:
                        fresh_S, fresh_Q, fresh_tag = Sr, Qr, tr_tag
            except TimeoutError:
                w51d.terminate_pool(ex)
                ex = ProcessPoolExecutor(max_workers=args.procs)

            if fresh_Q is None and alt_cands:
                fresh_S, fresh_Q, fresh_tag = (
                    alt_cands[0][0],
                    alt_cands[0][1],
                    alt_cands[0][2],
                )

            # 3) Stage 2: Exact LP Contact WideHopper (`w51d.deep_3round_widehopper`) on the fresh basin!
            hunt_Q, hunt_S, hunt_tag = best_Q, best_S, best_tag
            if hunt_S >= L0 - 1e-10 and fresh_Q is not None and fresh_S < L0 + 0.0055:
                hunt_Q, hunt_S, hunt_tag = fresh_Q, fresh_S, fresh_tag

            wn_cnt = 0
            rem_puz = puz_budget - (time.time() - tp0)
            if (
                (hunt_tag != "cur_init" or n in hot_w59)
                and rem_puz > 65.0
                and (deadline - time.time()) > 80.0
            ):
                r1 = min(135, max(35, int(0.40 * (rem_puz - 15.0))))
                r2 = min(115, max(30, int(0.30 * (rem_puz - 15.0))))
                r3 = min(115, max(25, int(0.30 * (rem_puz - 15.0))))
                Qw, Sw, (_, _, wn_cnt) = w51d.deep_3round_widehopper(
                    ex, n, hunt_Q, L0, bseed + 500, r1, r2, r3, deadline
                )
                if Sw < hunt_S - 1e-10:
                    hunt_S, hunt_Q, hunt_tag = Sw, Qw, hunt_tag + "->wh3r"

            if hunt_S < L0 - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, hunt_Q)
                if ok and Lf < L0 - 1e-9:
                    ds = (Lf - L0) / math.sqrt(n)
                    tot_ds += ds
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W61S HYBRID WIN!!] n={n:4d} [{hunt_tag}] {cur[n][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={wn_cnt}) ({time.time() - tp0:.1f}s)!",
                        flush=True,
                    )
                    save_ckpt()
                    continue

            print(
                f"  [W61S done] n={n:4d} dEnd={hunt_S - L0:+.6f} [{hunt_tag}] w={wn_cnt} ({time.time() - tp0:.1f}s)",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex)

    save_ckpt()
    print(
        f"[W61S s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
