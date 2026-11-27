#!/usr/bin/env python3
"""Wave 52I: Step-Jump Anomaly (`n-1 -> n` & `n-2 -> n`) 32-Mode Void/Bar Inserter
+ Numba `squeeze2` Quench -> 3-Round Deep-Polish `eta<=0.28` Wider-Hopper!

Directly targets the exact 148 step-jump anomalies uncovered in R32 where `P[n-1]` has a modern
two-band / staircase topology while `P[n]` is trapped in an older relic (`step = L[n] - L[n-1] > 0.012`):
  - `n=234<-233` (`15.7782 -> 15.8229`, step `+0.0447`, `dS_gap = 0.001198`!)
  - `n=367<-366` (`19.6458 -> 19.6815`, step `+0.0357`, `dS_gap = 0.000729`!)
  - `n=491<-490` (`22.6370 -> 22.6769`, step `+0.0399`, `dS_gap = 0.000690`!)
  - `n=333<-332` (`18.7395 -> 18.7710`, step `+0.0315`, `dS_gap = 0.000601`!)
  - `n=738<-737` (`27.6388 -> 27.6730`, step `+0.0342`, `dS_gap = 0.000588`!)
  - `n=456<-455` (`21.8182 -> 21.8467`, step `+0.0285`, `dS_gap = 0.000583` — 136-tilt 2-band -> 58-tilt relic!)
  - `n=261<-260` (`16.6335 -> 16.6788`, step `+0.0453`, `dS_gap = 0.000480`!)
  - `n=904<-903` (`30.5332 -> 30.5629`, step `+0.0297`, `dS_gap = 0.000475`!)
  - `n=414<-413` (`20.8170 -> 20.8407`, step `+0.0237`, `dS_gap = 0.000432`!)
  - `n=804<-803`, `914<-913`, `748<-747`, `371<-370`, `500<-499`, ...

Phase A (isolated pool): Screens 32 diverse 1-square (and 4 2-square `n-2 -> n`) insertion geometries
per puzzle via Numba `squeeze2.squeeze` + LP `Compactor` (`~8s` per puzzle across 4 cores!).
Phase B (clean pool): Attacks top `< +0.012` quenched starts with 3-Round Deep-Polish `eta<=0.28` Wider-Hopper!
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

import lns_band
import squeeze2 as sq2
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r


def build_insertion_seeds(P_prev, n_target, seed0=1000):
    """Generate ~28 diverse 1-square insertion states `P_raw` of length `n_target` from `P_prev` (`n_target - 1`)."""
    rng = np.random.default_rng(int(seed0))
    P = E.to_origin(P_prev)
    S = float(E.side(P))
    ang = ((P[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
    tilted = np.where(np.abs(ang) > 0.025)[0]
    c0, s0 = np.cos(P[:, 2]), np.sin(P[:, 2])
    h0 = 0.5 * (np.abs(c0) + np.abs(s0))

    seeds = []

    # Family 1 (12 seeds): Top-6 spatially separated Voronoi clearance voids x (upright vs neighbor angle)
    gx, gy = np.meshgrid(
        np.linspace(0.55, S - 0.55, 34), np.linspace(0.55, S - 0.55, 34)
    )
    voids = []
    for px, py in zip(gx.ravel(), gy.ravel()):
        sep = np.maximum(
            np.abs(P[:, 0] - px) - 0.5 - h0, np.abs(P[:, 1] - py) - 0.5 - h0
        )
        voids.append((float(sep.min()), float(px), float(py)))
    voids.sort(reverse=True)
    picked_v = []
    for v_gap, px, py in voids:
        if all(math.hypot(px - qx, py - qy) > 1.35 for _, qx, qy in picked_v):
            picked_v.append((v_gap, px, py))
        if len(picked_v) >= 6:
            break
    for vi, (_, px, py) in enumerate(picked_v):
        nn_i = int(np.argmin((P[:, 0] - px) ** 2 + (P[:, 1] - py) ** 2))
        seeds.append((f"vor0_{vi}", np.vstack([P, [[px, py, 0.0]]])))
        th_n = (
            float(P[nn_i, 2])
            if abs(ang[nn_i]) > 0.025
            else float(rng.uniform(-0.48, 0.48))
        )
        seeds.append((f"vort_{vi}", np.vstack([P, [[px, py, th_n]]])))

    # Family 2 (10 seeds): Tilted-bar split / band-tip insertion (pushes half of a tilted bar along its axis by +0.5!)
    if len(tilted) >= 4:
        for bi in range(10):
            idx_t = int(tilted[rng.integers(len(tilted))])
            th_b = float(P[idx_t, 2])
            ux, uy = math.cos(th_b), math.sin(th_b)
            # Nudge neighbors along bar direction (ux, uy) or (-uy, ux) to make a clean 1-square pocket!
            Q = P.copy()
            ax_dir = (ux, uy) if (bi % 2 == 0) else (-uy, ux)
            rel = (Q[:, 0] - P[idx_t, 0]) * ax_dir[0] + (
                Q[:, 1] - P[idx_t, 1]
            ) * ax_dir[1]
            perp = abs(
                -(Q[:, 0] - P[idx_t, 0]) * ax_dir[1]
                + (Q[:, 1] - P[idx_t, 1]) * ax_dir[0]
            )
            push_m = (rel > 0.0) & (perp < 1.2)
            Q[push_m, 0] += 0.45 * ax_dir[0]
            Q[push_m, 1] += 0.45 * ax_dir[1]
            ins_pt = [
                np.clip(P[idx_t, 0] + 0.5 * ax_dir[0], 0.55, S),
                np.clip(P[idx_t, 1] + 0.5 * ax_dir[1], 0.55, S),
                th_b,
            ]
            seeds.append((f"bsplit_{bi}", np.vstack([Q, [ins_pt]])))

    # Family 3 (6 seeds): Band-reseed (`lns_band.reseed`) AFTER inserting 1 square into the tilted strip!
    for ri in range(6):
        c_idx = (
            int(tilted[rng.integers(len(tilted))])
            if len(tilted)
            else int(rng.integers(len(P)))
        )
        Q_aug = np.vstack(
            [
                P,
                [
                    [
                        np.clip(P[c_idx, 0] + rng.normal(0.0, 0.4), 0.55, S),
                        np.clip(P[c_idx, 1] + rng.normal(0.0, 0.4), 0.55, S),
                        float(P[c_idx, 2]),
                    ]
                ],
            ]
        )
        rx, ry, rt, _, _ = lns_band.reseed(
            Q_aug[:, 0], Q_aug[:, 1], Q_aug[:, 2], rng, S + 0.04, band=True
        )
        seeds.append((f"reseed_{ri}", np.column_stack([rx, ry, rt])))

    return seeds


def eval_insert_seed(args_tuple):
    tag, P_raw, L_target_cur = args_tuple
    try:
        x0, y0, t0 = sq2.normalize(P_raw[:, 0], P_raw[:, 1], P_raw[:, 2])
        S_try = max(float(L_target_cur) + 0.008, float(sq2.side(x0, y0, t0)))
        ok, rx, ry, rt, _ = sq2.relax_at(x0, y0, t0, S_try, max_iter=5500)
        if not ok:
            ok, rx, ry, rt, _ = sq2.relax_at(x0, y0, t0, S_try * 1.012, max_iter=7000)
            if not ok:
                return tag, 1e9, None
        _, x2, y2, t2 = sq2.squeeze(
            rx, ry, rt, rel0=2.5e-3, rel_min=5e-7, max_iter=4500, max_tries=42
        )
        Lf, fx, fy, ft, good = sq2.finalize(x2, y2, t2, tol=2e-12)
        if not good:
            return tag, 1e9, None
        Q = E.to_origin(np.column_stack([fx, fy, ft]))
        Sc, Qc = w51d.compact_deleted(Q, max_iter=30, patience=2)
        if Qc is not None and Sc < Lf:
            return tag, float(Sc), Qc
        return tag, float(E.side(Q)), Q
    except Exception:
        return tag, 1e9, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--r1-sec", type=int, default=180)
    ap.add_argument("--r2-sec", type=int, default=165)
    ap.add_argument("--r3-sec", type=int, default=175)
    ap.add_argument("--max-puzzles", type=int, default=108)
    ap.add_argument("--seed-off", type=int, default=535000)
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

    anomalies = []
    for n in range(120, 1001):
        K = math.isqrt(n)
        if K * K == n or n == K * K + 1 or cur[n][0] >= K + 1 - 1e-6:
            continue
        step1 = cur[n][0] - cur[n - 1][0]
        gap_ldr = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if step1 > 0.011 and gap_ldr > 1e-4:
            prio = (gap_ldr / math.sqrt(n)) + max(0.0, step1 - 0.012) * 0.012
            anomalies.append((prio, n, step1))

    anomalies.sort(reverse=True)
    selected = [x[1] for x in anomalies[: args.max_puzzles]]
    my_slice = selected[args.shard :: args.num_shards]
    deadline = t0 + args.budget_sec

    print(
        f"[W52I-StepAnomalyInsert s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_slice({len(my_slice)})={my_slice} ({time.time() - t0:.1f}s)",
        flush=True,
    )

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

    # PHASE A: Isolated Void-Insertion Screening Pool
    screened = []
    ex_ins = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_slice):
            if time.time() - t0 > 380.0:
                break
            L0 = cur[n][0]
            raw_seeds = build_insertion_seeds(
                cur[n - 1][1], n, seed0=args.seed_off + args.shard * 997 + pi * 31
            )
            ts0 = time.time()
            futs = [
                ex_ins.submit(eval_insert_seed, (tg, Pr, L0)) for tg, Pr in raw_seeds
            ]
            hits = []
            try:
                for f in as_completed(futs, timeout=45.0):
                    tg, S_q, Q_q = f.result()
                    if Q_q is not None and S_q < L0 + 0.014:
                        hits.append((S_q - L0, S_q, tg, Q_q))
            except TimeoutError:
                w51d.terminate_pool(ex_ins)
                ex_ins = ProcessPoolExecutor(max_workers=args.procs)
            hits.sort(key=lambda z: z[0])
            b_str = f"{hits[0][0]:+.6f}@{hits[0][2]}" if hits else "none(<+0.014)"
            print(
                f"  [W52I screen] n={n:4d}<-{n - 1:4d} (step=+{L0 - cur[n - 1][0]:.4f}) hits={len(hits)} best={b_str} ({time.time() - ts0:.1f}s)",
                flush=True,
            )
            picked = []
            for gap0, S_q, tg, Q_q in hits:
                if all(abs(S_q - ps) > 5e-6 for _, ps, _, _ in picked):
                    picked.append((gap0, S_q, tg, Q_q))
                if len(picked) >= 2:
                    break
            for gap0, S_q, tg, Q_q in picked:
                screened.append((gap0, n, tg, S_q, Q_q))
    finally:
        w51d.terminate_pool(ex_ins)

    screened.sort(key=lambda z: z[0])
    print(
        f"[W52I s={args.shard}] Phase B (CLEAN POOL): hunting {len(screened)} best inserted starts ({time.time() - t0:.1f}s)!",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, tag, S_init, Q_init) in enumerate(screened):
            rem = deadline - time.time()
            if rem < 45.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            if S_init < L0_cur + 0.00030:
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
                            f"  [W52I DIRECT INSERT WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf0:.7f}!",
                            flush=True,
                        )
                        save_ckpt()

            r1 = min(args.r1_sec, max(30, int(0.35 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(25, int(0.30 * (rem - 20.0))))
            r3 = min(args.r3_sec, max(25, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + 5000 + args.shard * 10007 + i * 601
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
                        f"  [W52I WIN!!] n={n:4d} [{tag}_wh28] gap0={gap0:+.5f} tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{i + 1}/{len(screened)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W52I done] n={n:4d} [{tag}] gap0={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{i + 1}/{len(screened)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W52I s={args.shard} EXC] {exc}", flush=True)
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W52I s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
