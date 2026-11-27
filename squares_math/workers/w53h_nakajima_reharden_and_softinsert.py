#!/usr/bin/env python3
"""Wave 53H: Nakajima / Ostashev Soft-to-Rigid `harden reharden` Unjammer + Soft `n-1 -> n` Inserter
-> 3-Round Deep-Polish `eta<=0.28` Wider-Hopper!

Directly solves the hard-square angular jamming barrier proven in Wave 52I (`0/72` when inserting
a hard unit square into `n-1`) AND Wave 51 (`dEnd = 0.000000` local basin trap):
  - Arm A (Soft-to-Rigid `n-1 + 1` Insertion): Places 1 square into Voronoi / tilted-band pockets of
    `P[n-1]`, softens ALL squares to rounded cores (`r0 = 0.19`, initial stiffness `0.22 -> 1.0`) inside
    the held box via compiled Rust `harden reharden` (`~0.5s`), letting squares slide past corners and
    absorb the new square into the tilted band, then finishes with `squeeze2` + LP `Compactor`!
  - Arm B (Direct Ostashev `reharden` Basin Unjamming on `P[n]`): Runs diverse `(r0, hold_f, noise, sym_md)`
    soft-to-rigid momentum trajectories on our highest-gap R32 puzzles (`234, 367, 491, 333, 738, 456,
    730, 261, 904, 962, 414, 332, 804, 914, ...`), keeping ONLY immediate wins OR genuinely distinct
    tilted topologies (`sorted_angle_diff > 0.05` vs R32) for Phase-B 3-Round Wider-Hopper!
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
import squeeze2 as sq2
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w52i_step_anomaly_void_inserter as w52i


def topology_angle_dist(P_a, P_b):
    """Sorted folded-angle L1 distance to distinguish genuinely new basins from R32 clones."""
    aa = np.sort(
        np.abs(((P_a[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0)
    )
    ab = np.sort(
        np.abs(((P_b[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0)
    )
    return float(np.sum(np.abs(aa - ab)))


def eval_soft_trial(args_tuple):
    mode_tag, P_raw, s_ref, seed, r0, stiff0, hold_f, noise, sym_md = args_tuple
    try:
        rh = RE.run_harden_reharden(
            P_raw,
            s_ref=s_ref,
            seed=seed,
            r0=r0,
            stiff0=stiff0,
            pre=2200,
            hold=8500,
            release=3800,
            settle=2800,
            hold_f=hold_f,
            noise=noise,
            sym_md=sym_md,
            timeout_s=18.0,
        )
        if rh is None:
            return mode_tag, 1e9, None
        _, Q_h = rh
        x0, y0, t0 = sq2.normalize(Q_h[:, 0], Q_h[:, 1], Q_h[:, 2])
        S_sq, x2, y2, t2 = sq2.squeeze(
            x0, y0, t0, rel0=2e-3, rel_min=5e-7, max_iter=4500, max_tries=42
        )
        Lf, fx, fy, ft, good = sq2.finalize(x2, y2, t2, tol=2e-12)
        if not good:
            return mode_tag, 1e9, None
        Q = E.to_origin(np.column_stack([fx, fy, ft]))
        try:
            Sc, Qc = w51d.compact_deleted(Q, max_iter=35, patience=2)
            if Qc is not None and Sc < Lf:
                return mode_tag, float(Sc), Qc
        except Exception:
            pass
        return mode_tag, float(E.side(Q)), Q
    except Exception:
        return mode_tag, 1e9, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2220)
    ap.add_argument("--r1-sec", type=int, default=200)
    ap.add_argument("--r2-sec", type=int, default=180)
    ap.add_argument("--r3-sec", type=int, default=190)
    ap.add_argument("--max-puzzles", type=int, default=84)
    ap.add_argument("--seed-off", type=int, default=560000)
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
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    targets = []
    for n in range(120, 1001):
        K = math.isqrt(n)
        if K * K == n:
            continue
        gap_ldr = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gap_ldr > 1.2e-4:
            targets.append((gap_ldr / math.sqrt(n), n))
    targets.sort(reverse=True)
    my_slice = [x[1] for x in targets[: args.max_puzzles]][
        args.shard :: args.num_shards
    ]

    print(
        f"[W53H-NakajimaReharden s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_slice({len(my_slice)})={my_slice} ({time.time() - t0:.1f}s)",
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

    # PHASE A: Isolated Soft-to-Rigid Screening Pool
    screened = []
    ex_rh = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_slice):
            if time.time() - t0 > 400.0:
                break
            L0 = cur[n][0]
            P0 = cur[n][1]
            jobs = []
            bseed = args.seed_off + args.shard * 1301 + pi * 53

            # Arm A: 8 Soft-to-Rigid `n-1 + 1` Insertion trials when step > 0.008
            step1 = L0 - cur[n - 1][0]
            if step1 > 0.008 and cur[n - 1][1] is not None:
                ins_seeds = w52i.build_insertion_seeds(
                    cur[n - 1][1], n, seed0=bseed + 7
                )
                # Pick 4 Voronoi + 4 band-split starts
                vor = [s for s in ins_seeds if s[0].startswith("vor")][:4]
                bsp = [s for s in ins_seeds if s[0].startswith("bsplit")][:4]
                for ji, (stag, Pr) in enumerate(vor + bsp):
                    jobs.append(
                        (
                            f"sins_{stag}",
                            Pr,
                            cur[n - 1][0] + 0.004,
                            bseed + 100 + ji,
                            0.19 if ji % 2 == 0 else 0.23,
                            0.22,
                            1.0015,
                            0.0055,
                            0,
                        )
                    )

            # Arm B: 8 Direct Ostashev `reharden` Basin-Unjamming trials on `P[n]`
            cfgs = [
                (0.10, 0.45, 0.9996, 0.0035, 0),
                (0.14, 0.35, 0.9998, 0.0045, 0),
                (0.18, 0.28, 1.0005, 0.0055, 0),
                (0.22, 0.22, 1.0012, 0.0065, 0),
                (0.12, 0.40, 0.9997, 0.0040, 1),
                (0.17, 0.30, 1.0004, 0.0050, 1),
                (0.15, 0.32, 1.0000, 0.0075, 0),
                (0.20, 0.25, 1.0008, 0.0055, 0),
            ]
            for ci, (r0, st0, hf, ns, smd) in enumerate(cfgs):
                jobs.append(
                    (f"rehard_{ci}", P0, L0, bseed + 200 + ci, r0, st0, hf, ns, smd)
                )

            ts0 = time.time()
            futs = [ex_rh.submit(eval_soft_trial, jb) for jb in jobs]
            hits = []
            try:
                for f in as_completed(futs, timeout=52.0):
                    tg, S_q, Q_q = f.result()
                    if Q_q is None:
                        continue
                    gap0 = S_q - L0
                    adist = topology_angle_dist(Q_q, P0)
                    # Keep immediate improvements OR genuinely distinct topologies (`adist > 0.06`) within +0.012!
                    if gap0 < -1e-8 or (gap0 < 0.012 and adist > 0.06):
                        hits.append((gap0, S_q, adist, tg, Q_q))
            except TimeoutError:
                w51d.terminate_pool(ex_rh)
                ex_rh = ProcessPoolExecutor(max_workers=args.procs)

            hits.sort(key=lambda z: z[0])
            b_str = (
                f"{hits[0][0]:+.6f}(dA={hits[0][2]:.2f})@{hits[0][3]}"
                if hits
                else "none_distinct"
            )
            print(
                f"  [W53H screen] n={n:4d} (step=+{step1:.4f}) distinct_hits={len(hits)} best={b_str} ({time.time() - ts0:.1f}s)",
                flush=True,
            )
            picked = []
            for gap0, S_q, adist, tg, Q_q in hits:
                if all(abs(S_q - ps) > 1e-5 for _, ps, _, _, _ in picked):
                    picked.append((gap0, S_q, adist, tg, Q_q))
                if len(picked) >= 2:
                    break
            for gap0, S_q, adist, tg, Q_q in picked:
                screened.append((gap0, n, tg, S_q, adist, Q_q))
    finally:
        w51d.terminate_pool(ex_rh)

    screened.sort(key=lambda z: z[0])
    queued = screened[:9]
    print(
        f"[W53H s={args.shard}] Phase B (CLEAN POOL): hunting {len(queued)}/{len(screened)} distinct soft-unjammed starts ({time.time() - t0:.1f}s)!",
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
                    n, Q_init, deadline, slp_sec=38.0, fq_sec=26
                )
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        tot_ds += (Lf0 - L0_cur) / math.sqrt(n)
                        wins[n] = (float(Lf0), xf0, yf0, tf0)
                        L0_cur = float(Lf0)
                        print(
                            f"  [W53H DIRECT SOFT WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf0:.7f}!",
                            flush=True,
                        )
                        save_ckpt()

            r1 = min(args.r1_sec, max(30, int(0.35 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(25, int(0.30 * (rem - 20.0))))
            r3 = min(args.r3_sec, max(25, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + 30000 + args.shard * 10007 + i * 601
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
                        f"  [W53H WIN!!] n={n:4d} [{tag}_wh28] gap0={gap0:+.5f} dA={adist:.2f} tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W53H done] n={n:4d} [{tag}] gap0={gap0:+.5f} dA={adist:.2f} -> dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W53H s={args.shard} EXC] {exc}", flush=True)
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W53H s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
