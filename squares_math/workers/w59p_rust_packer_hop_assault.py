#!/usr/bin/env python3
"""Wave 59P (`w59p_rust_packer_hop_assault.py`): Native Compiled Rust `packer hop` + `packer hhop`
All-Shell Assault (`n = 100..1000`, INCLUDING `K = 31` `962, 976` & `K^2 + 1` `401, 530, 730, 785, 901`)!

First-ever deployment of compiled Rust `packer` (`fq_crate/src/main.rs`) across our R35 frontier:
  - `packer hop`: descending-target (`--s (L0 - 4.5e-5) --descend 5e-5`) region-move + `best_hole(300)`
    + `strip_slide` + `cluster_rotate` + `jam_lines` row-break basin hopping (`~60,000` native L-BFGS hops!).
  - `packer hhop`: hard-square collision event-chains (`max_slide` + `chain`) interleaved with `--soft 0.20`
    penalty-squeeze moves, both unfrozen AND orientation-agnostic far-grid frozen (`build_far_grid_freeze_mask`)!
  - Also injects a faithful `bandflip` junction-swap seed (`BF.bands` + `BF.flip` + plain `fq --loosen 1.025`)
    so Worker 3 explores opposite/same-tilt junction swaps on `K^2 + 1` and mixed-band puzzles!
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


def make_bandflip_seed(P0, seed):
    """Generate a faithful `bandflip` junction-swapped seed (`[0, s]^2`, plain `fq --loosen 1.025`)."""
    try:
        P_norm = RE.normalize_to_origin(P0)
        s0 = RE.side_exact(P_norm)
        sq_deg = P_norm.copy()
        sq_deg[:, 2] = np.degrees(sq_deg[:, 2]) % 90.0
        B = BF.bands(sq_deg, min_size=12)
        if not B:
            return None
        best = None
        for bi in range(min(2, len(B))):
            rng = random.Random(int(seed) + bi * 37)
            out_deg, _ = BF.flip(s0, sq_deg, B[bi], rng, B)
            Q_raw = np.asarray(out_deg, dtype=np.float64)
            Q_raw[:, 0] = np.clip(Q_raw[:, 0], 0.5, s0 - 0.5)
            Q_raw[:, 1] = np.clip(Q_raw[:, 1], 0.5, s0 - 0.5)
            Q_raw[:, 2] = np.radians(Q_raw[:, 2])
            rq = RE.run_fq_quench_args(
                Q_raw,
                [
                    "--seed",
                    str(int(seed) + bi),
                    "--loosen",
                    "1.025",
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
                    "15.0",
                ],
                s_ref=s0,
                timeout_s=17.5,
            )
            if rq is not None:
                Sq, Qq = float(rq[0]), rq[1]
                ad = w53h.topology_angle_dist(Qq, P_norm)
                if ad > 0.03 and (best is None or Sq < best[0]):
                    best = (Sq, Qq, f"bflip_b{bi}")
        return best
    except Exception:
        return None


def make_del1_seed(P_up, L0_want, seed):
    """Quick near-band delete-1 seed from `P[n+1]` quenched by `fq --loosen 1.008`."""
    try:
        P1 = RE.normalize_to_origin(P_up)
        s1 = RE.side_exact(P1)
        if s1 - L0_want > 0.024:
            return None
        try:
            _, _, _, _, load = E.skeleton(P1)
        except Exception:
            load = np.zeros(len(P1))
        half_pi = 0.5 * math.pi
        a_mod = np.mod(P1[:, 2], half_pi)
        tilts = np.minimum(a_mod, half_pi - a_mod)
        near = np.where(tilts > 0.015)[0]
        if len(near) == 0:
            return None
        order = near[np.argsort(load[near])]
        best = None
        for k_i, rem_idx in enumerate(order[:2]):
            Q_del = np.delete(P1, int(rem_idx), axis=0)
            rq = RE.run_fq_quench_args(
                Q_del,
                [
                    "--seed",
                    str(int(seed) + k_i),
                    "--loosen",
                    "1.008",
                    "--outer",
                    "45",
                    "--pit",
                    "260",
                    "--flip-top",
                    "20",
                    "--finish",
                    "--stag-w",
                    "0",
                    "--deadline",
                    "12.0",
                ],
                s_ref=L0_want,
                timeout_s=14.5,
            )
            if rq is not None:
                Sq, Qq = float(rq[0]), rq[1]
                if best is None or Sq < best[0]:
                    best = (Sq, Qq, f"del1_{int(rem_idx)}")
        return best
    except Exception:
        return None


def eval_packer_slot(slot_tup):
    kind, Q_in, target_s, seed, forbid_k, use_freeze, budget_s, tag_in = slot_tup
    try:
        fz = (
            RE.build_far_grid_freeze_mask(Q_in, buffer_dist=3.5) if use_freeze else None
        )
        sfx = "fz" if use_freeze else "all"
        if kind == "hhop":
            rh = RE.run_packer_hhop(
                Q_in,
                seed=seed,
                iters=110000,
                temp=1.1e-4 if not use_freeze else 1.7e-4,
                chains=8 if not use_freeze else 10,
                chainlen=1.05,
                gmax=0.016 if not use_freeze else 0.022,
                soft=0.16 if not use_freeze else 0.24,
                forbid_lines=forbid_k,
                frozen_mask=fz,
                timeout_s=budget_s,
            )
            if rh is None:
                return None
            return (rh[0], rh[1], f"{tag_in}->hhop_{sfx}")
        elif kind == "hop":
            rp = RE.run_packer_hop(
                Q_in,
                target_s=target_s,
                seed=seed,
                iters=110000,
                temp=0.24 if not use_freeze else 0.30,
                amp=0.92 if not use_freeze else 1.08,
                maxit=2600,
                descend=4.5e-5,
                linepen=0.01,
                frozen_mask=fz,
                timeout_s=budget_s,
            )
            if rp is None:
                return None
            return (rp[0], rp[1], f"{tag_in}->{rp[2]}_{sfx}")
        else:
            # Combo: half budget `hhop` -> half budget `hop` on fresh topology seed!
            b1 = max(25.0, 0.50 * budget_s)
            cur_Q = Q_in
            cur_S = RE.side_exact(cur_Q)
            cur_tag = tag_in
            rh = RE.run_packer_hhop(
                cur_Q,
                seed=seed,
                iters=80000,
                temp=1.5e-4,
                chains=9,
                chainlen=1.05,
                gmax=0.020,
                soft=0.22,
                forbid_lines=forbid_k,
                frozen_mask=fz,
                timeout_s=b1,
            )
            if rh is not None and rh[0] < cur_S:
                cur_S, cur_Q, cur_tag = rh[0], rh[1], f"{tag_in}->hhop_{sfx}"
            b2 = max(25.0, budget_s - b1)
            ts2 = min(cur_S, target_s)
            rp = RE.run_packer_hop(
                cur_Q,
                target_s=ts2,
                seed=seed + 77,
                iters=80000,
                temp=0.26,
                amp=1.0,
                maxit=2600,
                descend=4.5e-5,
                linepen=0.01,
                frozen_mask=fz,
                timeout_s=b2,
            )
            if rp is not None and rp[0] < cur_S:
                cur_S, cur_Q, cur_tag = rp[0], rp[1], f"{cur_tag}->{rp[2]}_{sfx}"
            return (cur_S, cur_Q, cur_tag)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1820)
    ap.add_argument("--seed-off", type=int, default=910000)
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

    # Include ALL shells up to n=1000 (including K=31 `962, 976` and `K^2+1` puzzles!)
    targets = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        if K * K == n:
            continue
        gap_l = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gap_l > 8.0e-5:
            # Slight priority boost for K=31 and K^2+1 puzzles that lcut cannot reach!
            bonus = 0.00018 if (K == 31 or n == K * K + 1) else 0.0
            targets.append((gap_l / math.sqrt(n) + bonus, n))
    targets.sort(reverse=True)
    my_slice = [x[1] for x in targets[: args.num_shards * 4]][
        args.shard :: args.num_shards
    ]

    print(
        f"[W59P-RustPackerAssault s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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
            if rem < 75.0:
                break
            puzzles_left = max(1, len(my_slice) - pi)
            puz_budget = min(455.0, max(90.0, (rem - 25.0) / puzzles_left))
            tp0 = time.time()

            L0 = wins[n][0] if n in wins else cur[n][0]
            P0 = RE.normalize_to_origin(cur[n][1])
            bseed = args.seed_off + args.shard * 6007 + pi * 409

            # Generate fast alternative seeds (bandflip + del1) in parallel (~16s)!
            alt_seed = None
            f_bf = ex.submit(make_bandflip_seed, P0, bseed + 11)
            f_d1 = (
                ex.submit(make_del1_seed, cur[n + 1][1], L0, bseed + 29)
                if (
                    n < 1000
                    and cur[n + 1][1] is not None
                    and math.isqrt(n + 1) ** 2 != (n + 1)
                )
                else None
            )
            alt_cands = []
            try:
                rbf = f_bf.result(timeout=22.0)
                if rbf is not None and rbf[0] < L0 + 0.015:
                    alt_cands.append(rbf)
                if f_d1 is not None:
                    rd1 = f_d1.result(timeout=18.0)
                    if rd1 is not None and rd1[0] < L0 + 0.015:
                        alt_cands.append(rd1)
            except Exception:
                w51d.terminate_pool(ex)
                ex = ProcessPoolExecutor(max_workers=args.procs)

            if alt_cands:
                alt_cands.sort(key=lambda z: z[0])
                alt_seed = alt_cands[0]
                if alt_seed[0] < L0 - 1e-9:
                    Qp0, Sp0 = w51d.deep_contact_polish(
                        n, alt_seed[1], deadline, slp_sec=36.0, fq_sec=22
                    )
                    if Sp0 < L0 - 1e-9:
                        ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                        if ok0 and Lf0 < L0 - 1e-9:
                            ds0 = (Lf0 - L0) / math.sqrt(n)
                            tot_ds += ds0
                            wins[n] = (float(Lf0), xf0, yf0, tf0)
                            L0 = float(Lf0)
                            P0 = RE.normalize_to_origin(
                                np.column_stack([xf0, yf0, tf0])
                            )
                            print(
                                f"  [W59P DIRECT SEED WIN!!] n={n:4d} [{alt_seed[2]}] {cur[n][0]:.7f} -> {Lf0:.7f}!",
                                flush=True,
                            )
                            save_ckpt()

            s1_slot = min(215.0, max(55.0, (puz_budget - (time.time() - tp0)) * 0.62))
            k_ceil = math.ceil(math.sqrt(n) - 1e-12)
            forbid_k = k_ceil if L0 < k_ceil + 1e-6 else 0
            ts_goal = L0 - 4.5e-5

            slots = [
                ("hhop", P0, ts_goal, bseed + 101, forbid_k, False, s1_slot, "cur"),
                ("hhop", P0, ts_goal, bseed + 102, forbid_k, True, s1_slot, "cur"),
                ("hop", P0, ts_goal, bseed + 103, forbid_k, False, s1_slot, "cur"),
            ]
            if alt_seed is not None:
                slots.append(
                    (
                        "combo",
                        alt_seed[1],
                        ts_goal,
                        bseed + 104,
                        forbid_k,
                        True,
                        s1_slot,
                        alt_seed[2],
                    )
                )
            else:
                slots.append(
                    ("hop", P0, ts_goal, bseed + 104, forbid_k, True, s1_slot, "cur")
                )

            futs = [ex.submit(eval_packer_slot, sl) for sl in slots]
            best_S, best_Q, best_tag = L0, P0, "cur_init"
            best_alt_S, best_alt_Q, best_alt_tag = 1e9, None, ""
            try:
                for f in as_completed(futs, timeout=s1_slot + 40.0):
                    rr = f.result()
                    if rr is None:
                        continue
                    Sr, Qr, tr_tag = rr
                    if Sr < best_S - 1e-10:
                        best_S, best_Q, best_tag = Sr, Qr, tr_tag
                    ad = w53h.topology_angle_dist(Qr, P0)
                    if ad > 0.025 and Sr < best_alt_S:
                        best_alt_S, best_alt_Q, best_alt_tag = Sr, Qr, tr_tag
            except TimeoutError:
                w51d.terminate_pool(ex)
                ex = ProcessPoolExecutor(max_workers=args.procs)

            cand_Q, cand_S, cand_tag = best_Q, best_S, best_tag
            if (
                cand_S >= L0 - 1e-10
                and best_alt_Q is not None
                and best_alt_S < L0 + 0.0022
            ):
                cand_Q, cand_S, cand_tag = best_alt_Q, best_alt_S, best_alt_tag

            if cand_S < L0 - 1e-10 or (cand_tag != "cur_init" and cand_S < L0 + 0.0022):
                cand_Q, cand_S = w51d.deep_contact_polish(
                    n, cand_Q, deadline, slp_sec=44.0, fq_sec=26
                )
                # Stage 2 Sniper if strictly improved or within +0.0005!
                rem_puz = puz_budget - (time.time() - tp0)
                if (
                    cand_S < L0 + 0.0005
                    and rem_puz > 65.0
                    and (deadline - time.time()) > 85.0
                ):
                    s2_slot = min(135.0, max(45.0, rem_puz - 20.0))
                    ts2 = min(cand_S, L0) - 3.0e-5
                    slots2 = [
                        (
                            "hhop",
                            cand_Q,
                            ts2,
                            bseed + 201,
                            forbid_k,
                            False,
                            s2_slot,
                            cand_tag,
                        ),
                        (
                            "hhop",
                            cand_Q,
                            ts2,
                            bseed + 202,
                            forbid_k,
                            True,
                            s2_slot,
                            cand_tag,
                        ),
                        (
                            "hop",
                            cand_Q,
                            ts2,
                            bseed + 203,
                            forbid_k,
                            False,
                            s2_slot,
                            cand_tag,
                        ),
                        (
                            "hop",
                            cand_Q,
                            ts2,
                            bseed + 204,
                            forbid_k,
                            True,
                            s2_slot,
                            cand_tag,
                        ),
                    ]
                    futs2 = [ex.submit(eval_packer_slot, sl) for sl in slots2]
                    try:
                        for f2 in as_completed(futs2, timeout=s2_slot + 40.0):
                            rr2 = f2.result()
                            if rr2 is not None and rr2[0] < cand_S - 1e-10:
                                cand_S, cand_Q, cand_tag = (
                                    rr2[0],
                                    rr2[1],
                                    rr2[2] + "_s2",
                                )
                    except TimeoutError:
                        w51d.terminate_pool(ex)
                        ex = ProcessPoolExecutor(max_workers=args.procs)
                    cand_Q, cand_S = w51d.deep_contact_polish(
                        n, cand_Q, deadline, slp_sec=48.0, fq_sec=28
                    )

            if cand_S < L0 - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cand_Q)
                if ok and Lf < L0 - 1e-9:
                    ds = (Lf - L0) / math.sqrt(n)
                    tot_ds += ds
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W59P RUST WIN!!] n={n:4d} [{cand_tag}] {cur[n][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}) ({time.time() - tp0:.1f}s)!",
                        flush=True,
                    )
                    save_ckpt()
                    continue

            alt_msg = (
                f"alt={best_alt_S - L0:+.6f}@{best_alt_tag}"
                if best_alt_Q is not None
                else "alt=none"
            )
            print(
                f"  [W59P done] n={n:4d} best={cand_S - L0:+.6f} [{cand_tag}] {alt_msg} ({time.time() - tp0:.1f}s)",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex)

    save_ckpt()
    print(
        f"[W59P s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
