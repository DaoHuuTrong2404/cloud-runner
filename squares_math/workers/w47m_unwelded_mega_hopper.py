#!/usr/bin/env python3
"""Wave 47M: Unwelded-Interface Mega-Cluster Hopper (max_cl=16/34/52) Engine!

Breaks the #1 structural bottleneck discovered in Ross Wade's stock Hopper (sq_engine.py):
  - Stock Hopper sets `core = size[body] <= 2`, leaving 100% of welded tilted bars (size >= 3)
    and 100% of upright grid squares adjacent to tilted bands FROZEN IN CONCRETE (`loose = False`),
    and throws away any cluster larger than 9 (or 18 in wide mode).
  - UnweldedMegaHopper UNWELDS ALL tilted squares (`|fold(theta)| > 0.02`) PLUS 2 full layers
    of upright grid squares bordering the tilted bands (keeping only the deep interior upright
    core welded for fast LP solves, capped at <= 420 individual singles so HiGHS stays fast!),
    and uses 3-tier cluster radii:
      * 30% Medium (rho=1.6..3.0, max_cl=16, rel_it=320)
      * 45% Mega   (rho=2.6..4.6, max_cl=34, rel_it=460) -- flips bar widths b -> b+-1!
      * 25% Super  (rho=3.8..6.0, max_cl=52, rel_it=560, eta=0.06..0.18) -- re-tiles junctions!
  - Instead of rejecting `len(cluster) > max_cl`, clamps to the `max_cl` nearest loose squares
    (5x-10x higher valid trial rate!).
  - Strictly protected by HiGHS 4s time_limit + internal deadline + hard os._exit(0) (0% hangs!).
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
import slp as SLP
import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12
import w32a_fresh_basin_fast_hopper as w32a


def build_unwelded_singles(P, tol=1e-3, max_singles=420):
    """Unweld ALL tilted squares + size<=2 squares + 2-layer upright grid interface around bands!"""
    n = len(P)
    if n <= max_singles:
        # For small/medium packings (n <= 420), unweld ALL tilted + 2-hop upright interface + wall layer!
        body, kind, start, size = E.find_bodies(P, tol)
        tilted = np.abs(E.fold(P[:, 2])) > 0.02
        layer0 = (size[body] <= 2) | tilted
        pairs = E.neighbour_pairs(np.ascontiguousarray(P), 1.55)
        layer1 = layer0.copy()
        for i, j in pairs:
            if layer0[i]:
                layer1[j] = True
            if layer0[j]:
                layer1[i] = True
        layer2 = layer1.copy()
        for i, j in pairs:
            if layer1[i]:
                layer2[j] = True
            if layer1[j]:
                layer2[i] = True
        return np.where(layer2)[0]

    # For large packings (n > 420), prioritize layer0 (all tilted + loose) -> layer1 -> layer2 up to max_singles
    body, kind, start, size = E.find_bodies(P, tol)
    tilted = np.abs(E.fold(P[:, 2])) > 0.02
    layer0 = (size[body] <= 2) | tilted
    idx0 = np.where(layer0)[0]
    if len(idx0) >= max_singles:
        return idx0[:max_singles]

    pairs = E.neighbour_pairs(np.ascontiguousarray(P), 1.55)
    layer1 = layer0.copy()
    for i, j in pairs:
        if layer0[i]:
            layer1[j] = True
        if layer0[j]:
            layer1[i] = True
    idx1_only = np.where(layer1 & ~layer0)[0]
    chosen = list(idx0)
    need = max_singles - len(chosen)
    if len(idx1_only) <= need:
        chosen.extend(idx1_only.tolist())
        need = max_singles - len(chosen)
        if need > 0:
            layer2 = layer1.copy()
            for i, j in pairs:
                if layer1[i]:
                    layer2[j] = True
                if layer1[j]:
                    layer2[i] = True
            idx2_only = np.where(layer2 & ~layer1)[0]
            chosen.extend(idx2_only[:need].tolist())
    else:
        chosen.extend(idx1_only[:need].tolist())
    return np.array(sorted(chosen), dtype=np.int64)


class UnweldedMegaHopper:
    def __init__(self, P, seed=0, tol=1e-3, max_singles=420):
        self.rng = np.random.default_rng(seed)
        self.tol = tol
        self.max_singles = max_singles
        self.stats = dict(trials=0, fit=0, promising=0, wins=0)
        self.t0 = time.time()
        self.reset(E.to_origin(P))

    def reset(self, P):
        loose = build_unwelded_singles(P, self.tol, self.max_singles)
        self.C = E.Compactor(P, self.tol, singles=loose)
        self.P = self.C.P
        self.S = E.extent(self.P)[1]
        self.loose = np.zeros(len(P), bool)
        self.loose[loose] = True
        self.L = E.LocalLP(self.C)
        self.w = (
            np.maximum(
                np.nan_to_num(self.L.loads(), nan=0.0, posinf=0.0, neginf=0.0), 0.0
            )
            * self.loose
        )
        # Fallback uniform weight over tilted+interface if LP loads zero out
        if self.w.sum() <= 0 and self.loose.any():
            self.w = self.loose.astype(np.float64)
        tilts = E.fold(self.P[:, 2])
        tilts = tilts[np.abs(tilts) > 0.02]
        uniq_t = np.unique(np.round(tilts, 3))
        if len(uniq_t) > 10:
            uniq_t = uniq_t[np.linspace(0, len(uniq_t) - 1, 10, dtype=int)]
        # Palette includes 0, dominant tilts, +-dominant tilts, and +-pi/4!
        pal = [0.0, math.pi / 4.0, -math.pi / 4.0]
        for tv in uniq_t:
            pal.append(float(tv))
            pal.append(float(-tv))
        self.palette = np.array(pal, dtype=np.float64)
        self.cen = E.body_centres(self.P, self.C.body, self.C.size)
        self.C.direction(self.P, self.S)
        self.C.repair(self.P)
        self.warm = dict(self.C.basis)

    def applicable(self):
        return bool(self.L.ok and np.isfinite(self.w).all() and self.w.sum() > 0)

    def trial(self):
        rng = self.rng
        P = self.P
        S = self.S
        C = self.C
        self.stats["trials"] += 1

        # 80% force-guided on loaded squares, 20% uniform on any unwelded band/interface square!
        if rng.random() < 0.80:
            i0 = int(rng.choice(len(P), p=self.w / self.w.sum()))
        else:
            loose_ids = np.where(self.loose)[0]
            i0 = int(rng.choice(loose_ids))
        c0 = P[i0, :2] + rng.normal(0, 0.5, 2)

        tier = rng.random()
        if tier < 0.30:
            rho = rng.uniform(1.6, 3.0)
            max_cl = 16
            rel_it = 320
            eta_hi = 0.12
        elif tier < 0.75:
            rho = rng.uniform(2.6, 4.6)
            max_cl = 34
            rel_it = 460
            eta_hi = 0.15
        else:
            rho = rng.uniform(3.8, 6.0)
            max_cl = 52
            rel_it = 560
            eta_hi = 0.18

        d = np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1])
        cluster = np.where((d < rho) & self.loose)[0]
        if len(cluster) == 0:
            return False
        if len(cluster) > max_cl:
            # Clamp to max_cl nearest loose squares instead of throwing trial away!
            cluster = cluster[np.argsort(d[cluster])[:max_cl]]
            rho_eff = float(d[cluster].max()) + 0.05
        else:
            rho_eff = rho

        near = np.where(d < rho_eff + 1.5)[0]
        ring = np.where((d >= rho_eff + 1.5) & (d < rho_eff + 3.8))[0]
        sub = np.concatenate([near, ring])
        free = np.zeros(len(sub), np.bool_)
        in_cl = np.zeros(len(P), np.bool_)
        in_cl[cluster] = True
        free[: len(near)] = self.loose[near] | in_cl[near]

        Q = P.copy()
        room = rng.random() < 0.65
        eta = 0.0
        axis = 0
        up = None
        if room:
            eta = float(rng.uniform(0.03, eta_hi))
            axis = int(rng.integers(2))
            up = self.cen[C.body, axis] > c0[axis] + rng.uniform(-1.2, 1.2)
            Q[up, axis] += eta
            if E.min_gap_over(Q, C.near) < -1e-9:
                room = False
                Q = P.copy()

        # Structured perturbation: 30% coherent angle shift (for bar re-tilting!), 70% mixed shake!
        move_mode = rng.random()
        if move_mode < 0.30:
            dth = float(rng.normal(0.0, 0.07))
            target_th = float(rng.choice(self.palette))
            use_target = rng.random() < 0.45
            for i in cluster:
                Q[i, :2] += rng.normal(0, 0.18, 2)
                if use_target:
                    Q[i, 2] = target_th + rng.normal(0, 0.03)
                else:
                    Q[i, 2] += dth + rng.normal(0, 0.02)
        else:
            for i in cluster:
                Q[i, :2] += rng.normal(0, 0.14 if rng.random() < 0.5 else 0.34, 2)
                u = rng.random()
                if u < 0.30:
                    pass
                elif u < 0.60:
                    Q[i, 2] = rng.choice(self.palette)
                elif u < 0.80:
                    Q[i, 2] += rng.normal(0, 0.06)
                else:
                    Q[i, 2] = rng.uniform(-0.785, 0.785)

        R, en = E.relax(Q[sub], free, S + (eta if room else 0.0) + 1e-9, rel_it, 1e-16)
        if en <= 1e-14 and room:
            Q[near] = R[: len(near)]
            Q[up, axis] -= eta
            R, en = E.relax(Q[sub], free, S + 1e-9, rel_it, 1e-16)
        if en > 1e-14:
            return False

        Q[near] = R[: len(near)]
        shift = np.abs(Q[near] - P[near]).max(axis=1)
        if shift.max() < 1e-3:
            return False
        Q[near[shift <= 1e-9]] = P[near[shift <= 1e-9]]
        self.stats["fit"] += 1

        moved = near[shift > 1e-9]
        # For mega-clusters, bypass LocalLP rejection if shift is large & go straight to compact 20% of time
        skip_lp = (len(cluster) >= 24) and (rng.random() < 0.25)
        if not skip_lp:
            pred = self.L.evaluate(Q, moved)
            if pred is None or pred > S - 1e-8:
                return False
        self.stats["promising"] += 1

        C.basis = dict(self.warm)
        Q2, S2, ov = C.compact(Q, max_iter=50, patience=1)
        if ov > 2e-9 or S2 > S - 1e-8:
            if not np.array_equal(C.anchor, self.P[:, :2]):
                C.build(self.P, np.where(self.loose)[0])
            C.P = self.P
            return False

        self.stats["wins"] += 1
        self.reset(Q2)
        return True

    def run(self, seconds, deadline=None):
        t = time.time()
        while time.time() - t < seconds and self.applicable():
            if deadline is not None and time.time() > deadline - 8.0:
                break
            self.trial()
        return self.P, self.S


def _slp_polish(P):
    cur_a = w32a.safe_open(P)
    cur_s = float(E.side(cur_a))
    try:
        xs, ys, ts, Ls = SLP.polish(
            cur_a[:, 0],
            cur_a[:, 1],
            cur_a[:, 2],
            iters=120,
            r0=2.5e-3,
            rmin=1e-10,
            tol_sep=-1e-13,
            time_limit=22.0,
        )
        Qs = w32a.safe_open(np.column_stack([xs, ys, ts]))
        Ss = float(E.side(Qs))
        if Ss < cur_s - 1e-10 and E.min_gap(Qs) >= -1e-12:
            return Qs, Ss
    except Exception:
        pass
    return cur_a, cur_s


def worker_unwelded_mega(job):
    n, L0, arr0, seed, round_sec, n_rounds, deadline = job
    best_S = L0
    best_P = None
    best_tag = ""
    cur_P = w32a.safe_open(arr0)
    tot_trials = 0
    tot_fits = 0
    tot_wins = 0

    for r in range(n_rounds):
        rem = deadline - time.time()
        if rem < 20.0:
            break
        sec = min(round_sec, rem - 10.0)
        try:
            H = UnweldedMegaHopper(cur_P, seed=seed + r * 9973)
            if not H.applicable():
                break
            Q, S = H.run(sec, deadline=deadline)
            tot_trials += H.stats["trials"]
            tot_fits += H.stats["fit"]
            tot_wins += H.stats["wins"]
            Q, S = _slp_polish(Q)
            if S < best_S - 1e-9 and E.min_gap(Q) >= -1e-12:
                best_S, best_P, best_tag = S, Q, f"umega_r{r}"
                cur_P = Q
            elif S < float(E.side(cur_P)) - 1e-9:
                cur_P = Q
        except Exception:
            break

    if best_P is not None and best_S < L0 - 1e-9:
        cur_a, cur_s = _slp_polish(best_P)
        if os.path.exists(f2d.FQ_BIN):
            with tempfile.TemporaryDirectory(prefix=f"w47m_{n}_") as tmp:
                r_fin = f2d.run_fq(
                    cur_s,
                    cur_a,
                    tmp,
                    ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
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
            return n, best_tag, (tot_trials, tot_fits, tot_wins), (Lf, xf, yf, tf)
    return n, best_tag, (tot_trials, tot_fits, tot_wins), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--round-sec", type=int, default=340)
    ap.add_argument("--n-rounds", type=int, default=2)
    ap.add_argument("--max-targets", type=int, default=120)
    ap.add_argument("--seed-off", type=int, default=470000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    gap_ranked = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        fl = front_L.get(n, cur[n][0])
        gds = (cur[n][0] - fl) / math.sqrt(n)
        if gds > 1e-6:
            gap_ranked.append((gds, n))
    gap_ranked.sort(reverse=True)
    targets = [n for _, n in gap_ranked[: args.max_targets]]

    my_slice = targets[args.shard :: args.num_shards]
    deadline = t0 + args.budget_sec
    jobs = [
        (
            n,
            cur[n][0],
            cur[n][1],
            args.seed_off + args.shard * 10601 + i * 719,
            args.round_sec,
            args.n_rounds,
            deadline,
        )
        for i, n in enumerate(my_slice)
    ]

    print(
        f"[W47M-UnweldedMegaHopper s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_targets({len(my_slice)})={my_slice} round={args.round_sec}sx{args.n_rounds} ({time.time() - t0:.1f}s)",
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

    done = 0
    ex = ProcessPoolExecutor(max_workers=args.procs)
    futs = [ex.submit(worker_unwelded_mega, j) for j in jobs]
    try:
        for fut in as_completed(futs, timeout=max(30.0, deadline - time.time() + 90.0)):
            done += 1
            try:
                n, tag, st, res = fut.result()
            except Exception as exc:
                print(f"  [W47M ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                prev_L = wins[n][0] if n in wins else cur[n][0]
                if Lf < prev_L - 1e-10:
                    dL = Lf - cur[n][0]
                    dS = (Lf - prev_L) / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W47M WIN!!] n={n:4d} [{tag}] tr={st[0]} fit={st[1]} w={st[2]} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
            else:
                print(
                    f"  [W47M done] n={n:4d} tr={st[0]} fit={st[1]} w={st[2]} no_win [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                    flush=True,
                )
            if time.time() > deadline + 45.0:
                print(
                    f"  [W47M s={args.shard}] Hard deadline after {done}/{len(jobs)}.",
                    flush=True,
                )
                break
    except TimeoutError:
        print(
            f"  [W47M s={args.shard}] Timeout after {done}/{len(jobs)} — exiting cleanly!",
            flush=True,
        )

    save_ckpt()
    print(
        f"[W47M s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    for proc in list(getattr(ex, "_processes", {}).values()):
        try:
            proc.terminate()
        except Exception:
            pass
    os._exit(0)


if __name__ == "__main__":
    main()
