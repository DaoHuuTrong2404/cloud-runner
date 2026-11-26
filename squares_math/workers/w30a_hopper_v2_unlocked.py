#!/usr/bin/env python3
"""Wave 30A/C: HopperV2 — Unlocked-Band & Seam-Cluster Force-Guided LP Hopper!

Eliminates the #1 bottleneck of stock sq_engine.Hopper (which welded all tilted bars of size>=3
and capped cluster size at 9 / rho<=2.3, causing applicable()==False or freezing clean crystals!):
  1. Unlocks ALL tilted squares (body size <= 28) + immediate seam-adjacent upright squares!
  2. Multi-scale cluster radius rho in [1.1, 3.4] up to 16 squares + larger axis expansion eta up to 0.22!
  3. Three coordinated force-guided moves:
       - Rigid cluster rotation & translation around centroid (breaks interlocking arch jams)
       - Coherent bar re-tilt (rotates tilted squares together by +-0.5..4 deg while relaxing seam)
       - Independent force-guided jitter + palette angle switching
  4. Runs BOTH HopperV2 (unlocked seam/band) AND stock Hopper + exhaustive n+1 removal screening
     + fq micro-kick -> HopperV2 ping-pong + strict Decimal64 >= 1.05e-9 certification!
"""

import argparse
import json
import math
import os
import sys
import tempfile
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

import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12


def safe_open(P):
    try:
        return E.to_origin(E.open_contacts(P))
    except Exception:
        return E.to_origin(P)


class HopperV2:
    """Enhanced Force-Guided LP Cluster Hopper with Unlocked Tilted Bands & Seam Junctions."""

    def __init__(self, P, seed=0, tol=1e-3, unlock_bands=True):
        self.rng = np.random.default_rng(seed)
        self.tol = tol
        self.unlock_bands = unlock_bands
        self.stats = dict(trials=0, fit=0, promising=0, wins=0)
        self.history = []
        self.t0 = time.time()
        self.reset(E.to_origin(P))

    def reset(self, P):
        P = E.to_origin(P)
        n = len(P)
        body, kind, start, size = E.find_bodies(P, self.tol)
        tilts = np.abs(E.fold(P[:, 2]))
        is_tilted = tilts > 0.015

        if self.unlock_bands:
            # Unlock ALL loose pieces + ALL tilted squares (unless huge solid > 28) + seam neighbors!
            core = (size[body] <= 4) | (is_tilted & (size[body] <= 28))
            free_ = core.copy()
            for i, j in E.neighbour_pairs(np.ascontiguousarray(P), 1.50):
                if is_tilted[i] and not is_tilted[j]:
                    free_[j] = True
                if is_tilted[j] and not is_tilted[i]:
                    free_[i] = True
                if core[i] and size[body[j]] <= 16:
                    free_[j] = True
                if core[j] and size[body[i]] <= 16:
                    free_[i] = True
            # Cap unlocked singles at 320 so HiGHS LP stays blazingly fast (<50ms/eval)
            idx_free = np.where(free_)[0]
            if len(idx_free) > 320:
                # Prioritize tilted & seam squares over interior upright squares
                prio = is_tilted[idx_free].astype(np.float64) * 2.0 + (
                    size[body[idx_free]] <= 4
                ).astype(np.float64)
                # Keep top 320
                keep = idx_free[np.argsort(-prio)[:320]]
                free_ = np.zeros(n, dtype=bool)
                free_[keep] = True
            loose = np.where(free_)[0]
        else:
            core = size[body] <= 2
            free_ = core.copy()
            for i, j in E.neighbour_pairs(np.ascontiguousarray(P), 1.45):
                if core[i] and size[body[j]] <= 12:
                    free_[j] = True
                if core[j] and size[body[i]] <= 12:
                    free_[i] = True
            loose = np.where(free_)[0]

        self.C = E.Compactor(P, self.tol, singles=loose)
        self.P = self.C.P
        self.S = float(E.extent(self.P)[1])
        self.loose = np.zeros(n, dtype=bool)
        self.loose[loose] = True
        self.is_tilted = np.abs(E.fold(self.P[:, 2])) > 0.015
        self.L = E.LocalLP(self.C)
        loads = np.maximum(
            np.nan_to_num(self.L.loads(), nan=0.0, posinf=0.0, neginf=0.0), 0.0
        )
        self.w = loads * self.loose
        # Fallback weight on tilted/seam squares if dual loads vanish on loose set
        if self.w.sum() <= 1e-12 and self.loose.any():
            self.w = self.loose.astype(np.float64)
        t_nz = E.fold(self.P[:, 2])
        t_nz = t_nz[np.abs(t_nz) > 0.02]
        self.palette = np.concatenate([[0.0], np.unique(np.round(t_nz, 3))[:10]])
        self.cen = E.body_centres(self.P, self.C.body, self.C.size)
        self.C.direction(self.P, self.S)
        self.C.repair(self.P)
        self.warm = dict(self.C.basis)
        self.history.append((time.time() - self.t0, self.S))

    def applicable(self):
        return bool(self.L.ok and np.isfinite(self.w).all() and self.w.sum() > 0)

    def trial(self):
        rng = self.rng
        P = self.P
        S = self.S
        C = self.C
        self.stats["trials"] += 1

        i0 = int(rng.choice(len(P), p=self.w / self.w.sum()))
        c0 = P[i0, :2] + rng.normal(0, 0.45, 2)
        rho = float(rng.uniform(1.1, 3.3 if self.unlock_bands else 2.3))
        d = np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1])
        cluster = np.where((d < rho) & self.loose)[0]
        max_cl = 16 if self.unlock_bands else 9
        if len(cluster) == 0:
            return False
        if len(cluster) > max_cl:
            # Take closest max_cl loose squares
            cluster = cluster[np.argsort(d[cluster])[:max_cl]]
            rho = float(d[cluster].max()) + 0.15

        near = np.where(d < rho + 1.5)[0]
        ring = np.where((d >= rho + 1.5) & (d < rho + 3.9))[0]
        sub = np.concatenate([near, ring])
        free = np.zeros(len(sub), np.bool_)
        free[: len(near)] = self.loose[near] | (d[near] < rho)

        Q = P.copy()
        room = rng.random() < 0.68
        eta = 0.0
        axis = 0
        up = None
        if room:
            eta = float(rng.uniform(0.03, 0.22 if self.unlock_bands else 0.12))
            axis = int(rng.integers(2))
            up = self.cen[C.body, axis] > c0[axis] + rng.uniform(-1.0, 1.0)
            Q[up, axis] += eta
            if E.min_gap_over(Q, C.near) < -1e-9:
                return False

        move_kind = rng.random()
        if self.unlock_bands and move_kind < 0.22 and len(cluster) >= 3:
            # Move A: Rigid cluster rotation + shift around centroid
            a = float(rng.normal(0, 0.08))
            cx, cy = Q[cluster, 0].mean(), Q[cluster, 1].mean()
            dx, dy = Q[cluster, 0] - cx, Q[cluster, 1] - cy
            ca, sa_ = math.cos(a), math.sin(a)
            Q[cluster, 0] = cx + ca * dx - sa_ * dy + rng.normal(0, 0.06)
            Q[cluster, 1] = cy + sa_ * dx + ca * dy + rng.normal(0, 0.06)
            Q[cluster, 2] += a + rng.normal(0, 0.02, len(cluster))
        elif self.unlock_bands and move_kind < 0.44 and np.any(self.is_tilted[cluster]):
            # Move B: Coordinated bar re-tilt on tilted squares + light seam jiggle
            dt = float(rng.normal(0, 0.055))
            for i in cluster:
                if self.is_tilted[i]:
                    Q[i, 2] += dt + rng.normal(0, 0.015)
                    Q[i, :2] += rng.normal(0, 0.08, 2)
                else:
                    Q[i, :2] += rng.normal(0, 0.10, 2)
                    if rng.random() < 0.20:
                        Q[i, 2] = float(rng.choice(self.palette))
        else:
            # Move C: Force-guided independent jitter + palette angle switch
            for i in cluster:
                Q[i, :2] += rng.normal(0, 0.12 if rng.random() < 0.5 else 0.30, 2)
                u = rng.random()
                if u < 0.35:
                    pass
                elif u < 0.62:
                    Q[i, 2] = float(rng.choice(self.palette))
                elif u < 0.82:
                    Q[i, 2] += rng.normal(0, 0.05)
                else:
                    Q[i, 2] = float(rng.uniform(-0.78, 0.78))

        R, E_val = E.relax(Q[sub], free, S + (eta if room else 0.0) + 1e-9, 260, 1e-16)
        if E_val <= 1e-14 and room:
            Q[near] = R[: len(near)]
            Q[up, axis] -= eta
            R, E_val = E.relax(Q[sub], free, S + 1e-9, 260, 1e-16)
        if E_val > 1e-14:
            return False

        Q[near] = R[: len(near)]
        shift = np.abs(Q[near] - P[near]).max(axis=1)
        if shift.max() < 1e-3:
            return False
        Q[near[shift <= 1e-9]] = P[near[shift <= 1e-9]]
        self.stats["fit"] += 1

        pred = self.L.evaluate(Q, near[shift > 1e-9])
        if pred is None or pred > S - 2e-8:
            return False
        self.stats["promising"] += 1

        C.basis = dict(self.warm)
        Q2, S2, ov = C.compact(Q, max_iter=65, patience=1)
        if ov > 2e-9 or S2 > S - 1e-8:
            if not np.array_equal(C.anchor, self.P[:, :2]):
                C.build(self.P, np.where(self.loose)[0])
            C.P = self.P
            return False

        self.stats["wins"] += 1
        self.reset(Q2)
        return True

    def run(self, seconds):
        t = time.time()
        while time.time() - t < seconds and self.applicable():
            self.trial()
        return self.P, self.S


def warm_numba():
    P = E.to_origin(
        np.array(
            [
                [0.5, 0.5, 0.0],
                [1.5, 0.5, 0.0],
                [0.5, 1.5, 0.0],
                [1.5, 1.5, 0.0],
                [2.5, 0.5, 0.15],
            ],
            dtype=np.float64,
        )
    )
    E.min_gap(P)
    E.side(P)
    safe_open(P)
    H = HopperV2(P, seed=0, unlock_bands=True)
    if H.applicable():
        H.run(1)


def delete_and_compact_v2(P1, idx):
    P = E.to_origin(np.delete(P1, idx, axis=0))
    body, kind, start, size = E.find_bodies(P, 1e-3)
    tilts = np.abs(E.fold(P[:, 2])) > 0.015
    core = (size[body] <= 3) | (tilts & (size[body] <= 24))
    free_ = core.copy()
    for u, v in E.neighbour_pairs(np.ascontiguousarray(P), 1.48):
        if core[u] and size[body[v]] <= 14:
            free_[v] = True
        if core[v] and size[body[u]] <= 14:
            free_[u] = True
    idx_f = np.where(free_)[0]
    if len(idx_f) > 300:
        idx_f = idx_f[:300]
    Q, S, ov = E.Compactor(P, 1e-3, singles=idx_f).compact(P, max_iter=60, patience=2)
    if ov > 1e-7:
        return None, 1e9
    return Q, float(S)


def pick_removal_candidates(P1, max_cand=64):
    P1 = E.to_origin(P1)
    n1 = len(P1)
    tilts = np.abs(E.fold(P1[:, 2]))
    tilted = np.where(tilts > 0.015)[0]
    near_band = set(int(i) for i in tilted)
    if len(tilted) > 0:
        for u, v in E.neighbour_pairs(np.ascontiguousarray(P1), 1.75):
            if u in near_band:
                near_band.add(int(v))
            if v in near_band:
                near_band.add(int(u))
    scores = np.zeros(n1, dtype=np.float64)
    try:
        H = HopperV2(P1, seed=0, unlock_bands=True)
        if H.L.ok:
            loads = np.maximum(
                np.nan_to_num(H.L.loads(), nan=0.0, posinf=0.0, neginf=0.0), 0.0
            )
            scores += loads * 10.0
    except Exception:
        pass
    for i in near_band:
        scores[i] += 0.25
    for i in tilted:
        scores[i] += 0.40
    return [int(i) for i in np.argsort(-scores)[:max_cand]]


def run_v2_chain(
    P_start, seed_base, first_sec, cascade_sec, unlock_bands=True, max_cascades=3
):
    best_P = E.to_origin(P_start)
    best_S = float(E.side(best_P))
    start_S = best_S
    try:
        H = HopperV2(best_P, seed=seed_base, unlock_bands=unlock_bands)
        if not H.applicable():
            return best_P, best_S, 0
        Q, S = H.run(first_sec)
        Q = safe_open(Q)
        S = float(E.side(Q))
        if S < best_S - 1e-10 and E.min_gap(Q) >= -1e-12:
            best_P, best_S = Q, S
    except Exception:
        return best_P, best_S, 0
    wins = 1 if best_S < start_S - 1e-9 else 0
    if wins:
        for c in range(max_cascades):
            prev_S = best_S
            try:
                # Alternate unlock_bands=True and False on cascades to hit both global & fine contacts!
                H2 = HopperV2(
                    best_P, seed=seed_base + (c + 1) * 1009, unlock_bands=(c % 2 == 0)
                )
                if not H2.applicable():
                    break
                Q2, S2 = H2.run(cascade_sec)
                Q2 = safe_open(Q2)
                S2 = float(E.side(Q2))
                if S2 < best_S - 1e-10 and E.min_gap(Q2) >= -1e-12:
                    best_P, best_S = Q2, S2
                    wins += 1
                if prev_S - best_S < 5e-7:
                    break
            except Exception:
                break
    return best_P, best_S, wins


def worker_v2(job):
    n, L0, arr0, arr_p1, L_p1, seed, v2_sec, del_sec, marathon = job
    best_S = L0
    best_P = None
    best_tag = ""

    # Phase 1: HopperV2 Unlocked-Band (3 seeds in normal mode, 5 seeds in marathon sniper mode!)
    n_seeds = 5 if marathon else 3
    for si in range(n_seeds):
        P_src = best_P if best_P is not None else arr0
        Q, S, w = run_v2_chain(
            P_src,
            seed + si * 7919,
            v2_sec,
            min(180, v2_sec),
            unlock_bands=(si != n_seeds - 1),
            max_cascades=3,
        )
        if S < best_S - 1e-8:
            best_S, best_P, best_tag = S, Q, f"hv2_s{si}"

    # Phase 2: Exhaustive n+1 Removal Screening (with Unlocked-Band Compaction!) -> HopperV2!
    if arr_p1 is not None and (L_p1 - best_S) < 0.042:
        max_c = 90 if marathon else 56
        cands = pick_removal_candidates(arr_p1, max_cand=max_c)
        holes = []
        t_scr = time.time()
        for idx in cands:
            if time.time() - t_scr > (75.0 if marathon else 48.0):
                break
            try:
                Qh, Sh = delete_and_compact_v2(arr_p1, idx)
                if Qh is not None and Sh < best_S + 0.016:
                    holes.append((Sh, idx, Qh))
            except Exception:
                continue
        holes.sort(key=lambda x: x[0])
        for hi, (Sh, idx, Qh) in enumerate(holes[: (4 if marathon else 2)]):
            if Sh < best_S - 1e-8:
                best_S, best_P, best_tag = Sh, Qh, f"del2v_{idx}"
            if Sh < best_S + 0.011:
                Q2, S2, _ = run_v2_chain(
                    Qh,
                    seed + 60000 + hi * 3571,
                    del_sec,
                    140,
                    unlock_bands=True,
                    max_cascades=2,
                )
                if S2 < best_S - 1e-8:
                    best_S, best_P, best_tag = S2, Q2, f"delhv2_{idx}"

    # Phase 3: Round-Press Escape Kick -> HopperV2 Ping-Pong!
    if os.path.exists(f2d.FQ_BIN):
        with tempfile.TemporaryDirectory(prefix=f"w30_k_{n}_") as tmp:
            P_src = best_P if best_P is not None else arr0
            kicks = (
                [("0.06", 0.010), ("0.14", 0.018)] if marathon else [("0.08", 0.012)]
            )
            for ki, (r0, pad) in enumerate(kicks):
                r_k = f2d.run_fq(
                    best_S + pad,
                    P_src,
                    tmp,
                    [
                        "--loosen",
                        "1.0",
                        "--round-r0",
                        r0,
                        "--round-compress",
                        "--round-press",
                        "--round-steps",
                        "7",
                        "--round-press-outer",
                        "12",
                        "--seed",
                        str(seed + 8888 + ki * 97),
                        "--flip-top",
                        "36",
                        "--pit",
                        "200",
                    ],
                    timeout=50,
                )
                if r_k is not None and r_k[0] < best_S + 0.008:
                    if r_k[0] < best_S - 1e-8:
                        best_S, best_P, best_tag = r_k[0], r_k[1], f"rpkick_{r0}"
                    Q3, S3, _ = run_v2_chain(
                        r_k[1],
                        seed + 99991 + ki * 503,
                        160,
                        130,
                        unlock_bands=True,
                        max_cascades=2,
                    )
                    if S3 < best_S - 1e-8:
                        best_S, best_P, best_tag = S3, Q3, f"rphv2_{r0}"

    # Phase 4: Final Polish + Strict Decimal64 >= 1.05e-9 Certification
    if best_P is not None and best_S < L0 - 1e-9:
        with tempfile.TemporaryDirectory(prefix=f"w30_fin_{n}_") as tmp:
            cur_a = safe_open(best_P)
            cur_s = float(E.side(cur_a))
            if os.path.exists(f2d.FQ_BIN):
                r_fin = f2d.run_fq(
                    cur_s,
                    cur_a,
                    tmp,
                    ["--loosen", "1.0", "--no-alm", "--flip-top", "56", "--pit", "550"],
                    timeout=65,
                )
                if r_fin is not None and r_fin[0] < cur_s:
                    cur_s, cur_a = r_fin[0], r_fin[1]
            try:
                cur_a = w12.lp_rot_polish(cur_a)
            except Exception:
                pass
            ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_a)
            if ok and Lf < L0 - 1e-9:
                return n, best_tag, (Lf, xf, yf, tf)
    return n, best_tag, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2800)
    ap.add_argument("--v2-sec", type=int, default=190)
    ap.add_argument("--del-sec", type=int, default=160)
    ap.add_argument("--min-rank", type=int, default=0)
    ap.add_argument("--max-targets", type=int, default=160)
    ap.add_argument("--marathon", action="store_true")
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    warm_numba()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    front_rows = json.load(open(args.frontier_json))
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    scored = []
    for n in range(150, 1000):
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        gap_ds = max(0.0, (cur[n][0] - front_L.get(n, cur[n][0])) / math.sqrt(n))
        if gap_ds > 1e-7:
            scored.append((gap_ds, n))
    scored.sort(reverse=True)
    targets = [n for _, n in scored[args.min_rank : args.min_rank + args.max_targets]]
    my_slice = targets[args.shard :: args.num_shards]
    tag_mode = "Marathon" if args.marathon else "UnlockedV2"
    print(
        f"[W30-{tag_mode} s={args.shard}/{args.num_shards}] Base={tot_score:.10f} "
        f"(loaded {n_files} win files, total_front_gap={sum(s for s, _ in scored):.6f}), "
        f"my_targets({len(my_slice)})={my_slice} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    jobs = []
    for i, n in enumerate(my_slice):
        arr_p1 = None
        L_p1 = 1e9
        if n + 1 <= 1000:
            kp1 = math.isqrt(n + 1)
            if kp1 * kp1 != (n + 1) and cur[n + 1][0] < kp1 + 1 - 1e-6:
                arr_p1 = cur[n + 1][1]
                L_p1 = cur[n + 1][0]
        jobs.append(
            (
                n,
                cur[n][0],
                cur[n][1],
                arr_p1,
                L_p1,
                3010001
                + (500000 if args.marathon else 0)
                + args.shard * 10169
                + i * 331,
                args.v2_sec,
                args.del_sec,
                args.marathon,
            )
        )

    wins = {}
    tot_ds = 0.0
    done = 0

    def save_ckpt():
        payload = {}
        for k_n, (Lf, xf, yf, tf) in sorted(wins.items()):
            payload[f"L_{k_n}"] = np.array([Lf], dtype=np.float64)
            payload[f"x_{k_n}"] = np.asarray(xf, dtype=np.float64)
            payload[f"y_{k_n}"] = np.asarray(yf, dtype=np.float64)
            payload[f"t_{k_n}"] = np.asarray(tf, dtype=np.float64)
        np.savez_compressed(args.out_npz, **payload)

    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_v2, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W30 ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                dL = Lf - cur[n][0]
                if dL < -1e-9:
                    dS = dL / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W30-{tag_mode} WIN!!] n={n:4d} {tag:14s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W30-{tag_mode} s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W30-{tag_mode} s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
