#!/usr/bin/env python3
"""Wave 48A: Simulated-Annealing Unwelded Mega-Hopper (SA-UnweldedHopper) Engine!

Solves BOTH root causes identified by Wave 47 telemetry (where UnweldedMegaHopper generated
600-1,989 zero-overlap re-packings per puzzle, yet greedy `S2 < S - 1e-8` rejected 99.95% of them!):
  1. Cyclic Metropolis Simulated Annealing (`T: 0.0022 -> 2e-5`, ceiling `S2 <= best_S + 0.005`):
     Allows multi-hop topological barrier crossing across unwelded tilted bars + 2-layer upright
     grid interface (`max_cl = 16/34/52`), with automatic restart-to-best if stalled!
  2. 3-Stage Hybrid Finisher on `best_P`:
     Stage 1: SA-Unwelded-MegaHopper (`sa_sec`, T > 0 barrier crossing)
     Stage 2: T=0 Unwelded-MegaHopper Quench (`quench_sec`, unwelded bar/interface settle)
     Stage 3: Welded Rigid-Bar Wide-Hopper (`wide_sec`, fast rigid-bar slide) + SLP + fq + Decimal64!
  3. Pillar 2 Driver: Attacks Top-96 live-gap puzzles + goodpjw2008 `targets.csv` open near-miss
     `n+1 -> n` holes (which previously failed because welded greedy Hopper froze their bars!).
"""

import argparse
import csv
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
import w32a_fresh_basin_fast_hopper as w32a
import w47m_unwelded_mega_hopper as w47m


class SAUnweldedHopper(w47m.UnweldedMegaHopper):
    def __init__(self, P, seed=0, tol=1e-3, max_singles=420, max_drift=0.005):
        super().__init__(P, seed=seed, tol=tol, max_singles=max_singles)
        self.best_P = self.P.copy()
        self.best_S = float(self.S)
        self.max_drift = float(max_drift)
        self.stats["sa_accepts"] = 0
        self.stats["restarts"] = 0
        self._since_best = 0

    def sa_trial(self, temp):
        rng = self.rng
        P = self.P
        S = self.S
        C = self.C
        self.stats["trials"] += 1

        if rng.random() < 0.78:
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
            rel_it = 450
            eta_hi = 0.15
        else:
            rho = rng.uniform(3.8, 6.0)
            max_cl = 52
            rel_it = 540
            eta_hi = 0.18

        d = np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1])
        cluster = np.where((d < rho) & self.loose)[0]
        if len(cluster) == 0:
            return False
        if len(cluster) > max_cl:
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

        if rng.random() < 0.32:
            dth = float(rng.normal(0.0, 0.075))
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
                if u < 0.28:
                    pass
                elif u < 0.60:
                    Q[i, 2] = rng.choice(self.palette)
                elif u < 0.80:
                    Q[i, 2] += rng.normal(0, 0.06)
                else:
                    Q[i, 2] = rng.uniform(-0.785, 0.785)

        # Allow slight box breathing (+2.5*temp) during relaxation when temp > 0!
        box_allow = S + (2.5 * temp if temp > 0.0 else 0.0)
        R, en = E.relax(
            Q[sub], free, box_allow + (eta if room else 0.0) + 1e-9, rel_it, 1e-16
        )
        if en <= 1e-14 and room:
            Q[near] = R[: len(near)]
            Q[up, axis] -= eta
            R, en = E.relax(Q[sub], free, box_allow + 1e-9, rel_it, 1e-16)
        if en > 1e-14:
            return False

        Q[near] = R[: len(near)]
        shift = np.abs(Q[near] - P[near]).max(axis=1)
        if shift.max() < 1e-3:
            return False
        Q[near[shift <= 1e-9]] = P[near[shift <= 1e-9]]
        self.stats["fit"] += 1

        moved = near[shift > 1e-9]
        allow_up = 2.2 * temp if temp > 0.0 else -1e-8
        pred = self.L.evaluate(Q, moved)
        if pred is not None and pred > S + allow_up + 5e-4:
            return False
        if temp <= 0.0 and (pred is None or pred > S - 1e-8):
            if not ((len(cluster) >= 24) and (rng.random() < 0.20)):
                return False
        self.stats["promising"] += 1

        C.basis = dict(self.warm)
        Q2, S2, ov = C.compact(Q, max_iter=50, patience=1)
        if ov > 2e-9 or S2 > self.best_S + self.max_drift:
            if not np.array_equal(C.anchor, self.P[:, :2]):
                C.build(self.P, np.where(self.loose)[0])
            C.P = self.P
            return False

        dS = float(S2 - S)
        accept = False
        if dS < -1e-8:
            accept = True
        elif temp > 0.0 and dS < 3.5 * temp:
            if rng.random() < math.exp(-max(0.0, dS) / max(temp, 1e-9)):
                accept = True

        if not accept:
            if not np.array_equal(C.anchor, self.P[:, :2]):
                C.build(self.P, np.where(self.loose)[0])
            C.P = self.P
            return False

        if S2 < self.best_S - 1e-9:
            self.best_S = float(S2)
            self.best_P = Q2.copy()
            self.stats["wins"] += 1
            self._since_best = 0
        else:
            self.stats["sa_accepts"] += 1
            self._since_best += 1

        # Restart to best_P if wandered for 18 accepted uphill/sideways moves without new best
        if temp > 0.0 and self._since_best >= 18 and self.S > self.best_S + 2e-4:
            self._since_best = 0
            self.stats["restarts"] += 1
            self.reset(self.best_P.copy())
        else:
            self.reset(Q2)
        return True

    def run_anneal_and_quench(
        self, sa_sec, quench_sec, t_init=0.0022, t_min=2e-5, deadline=None
    ):
        t0 = time.time()
        # Stage 1: Geometric cooling SA (barrier crossing!)
        if sa_sec > 5.0:
            while self.applicable():
                now = time.time()
                if deadline is not None and now > deadline - 12.0:
                    break
                el = now - t0
                if el >= sa_sec:
                    break
                frac = min(1.0, max(0.0, el / sa_sec))
                temp = t_init * ((t_min / t_init) ** frac)
                self.sa_trial(temp)

        # Snap back to best_P before T=0 quench!
        if self.S > self.best_S + 1e-10:
            self.reset(self.best_P.copy())

        # Stage 2: Pure T=0 Unwelded Mega-Hopper Quench!
        t1 = time.time()
        while (time.time() - t1 < quench_sec) and self.applicable():
            if deadline is not None and time.time() > deadline - 10.0:
                break
            self.sa_trial(0.0)

        return self.best_P, self.best_S


def run_sa_unwelded_pipeline(
    P_init, L0, seed, sa_sec, quench_sec, wide_sec, deadline, tag_prefix="sa"
):
    """Full 3-stage SA-Unwelded -> Unwelded-Quench -> Welded-Wide-Hopper -> SLP -> fq -> Decimal64!"""
    cur_P = w32a.safe_open(P_init)
    best_S = float(E.side(cur_P))
    best_P = cur_P if (best_S < L0 - 1e-9 and E.min_gap(cur_P) >= -1e-12) else None
    best_tag = f"{tag_prefix}_raw" if best_P is not None else ""
    stats_out = (0, 0, 0, 0)

    rem = deadline - time.time()
    if rem > 25.0:
        s1 = min(sa_sec, max(10.0, 0.45 * (rem - 15.0)))
        s2 = min(quench_sec, max(10.0, 0.35 * (rem - 15.0)))
        try:
            H = SAUnweldedHopper(cur_P, seed=seed, max_drift=0.006)
            if H.applicable():
                Qa, Sa = H.run_anneal_and_quench(
                    s1, s2, t_init=0.0022, t_min=2e-5, deadline=deadline
                )
                stats_out = (
                    H.stats["trials"],
                    H.stats["fit"],
                    H.stats["sa_accepts"],
                    H.stats["wins"],
                )
                Qa, Sa = w47m._slp_polish(Qa)
                if Sa < min(L0, best_S) - 1e-9 and E.min_gap(Qa) >= -1e-12:
                    best_S, best_P, best_tag = Sa, Qa, f"{tag_prefix}_saum"
                if Sa < float(E.side(cur_P)) + 0.003:
                    cur_P = Qa
        except Exception:
            pass

    # Stage 3: Fast Welded Rigid-Bar Wide-Hopper finisher (slides re-tilted bars rapidly!)
    rem = deadline - time.time()
    if wide_sec > 5.0 and rem > 20.0:
        try:
            Hw = E.Hopper(
                w32a.safe_open(best_P if best_P is not None else cur_P),
                seed=seed + 31337,
                wide=True,
            )
            if Hw.applicable():
                Qw, Sw = Hw.run(min(wide_sec, rem - 12.0))
                Qw, Sw = w47m._slp_polish(Qw)
                if Sw < min(L0, best_S) - 1e-9 and E.min_gap(Qw) >= -1e-12:
                    best_S, best_P, best_tag = Sw, Qw, f"{tag_prefix}_sawh"
        except Exception:
            pass

    if best_P is not None and best_S < L0 - 1e-9:
        n = len(best_P)
        cur_a, cur_s = w47m._slp_polish(best_P)
        if os.path.exists(f2d.FQ_BIN) and (deadline - time.time()) > 15.0:
            with tempfile.TemporaryDirectory(prefix=f"w48_{n}_") as tmp:
                r_fin = f2d.run_fq(
                    cur_s,
                    cur_a,
                    tmp,
                    ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
                    timeout=min(60, max(15, int(deadline - time.time() - 5))),
                )
                if r_fin is not None and r_fin[0] < cur_s:
                    cur_s, cur_a = r_fin[0], r_fin[1]
        try:
            cur_a = w12.lp_rot_polish(cur_a)
        except Exception:
            pass
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_a)
        if ok and Lf < L0 - 1e-9:
            return best_tag, stats_out, (Lf, xf, yf, tf)
    return best_tag, stats_out, None


def worker_gap_and_target(job):
    n, L0, arr0, arr_p1, hint_idxs, seed, sa_sec, quench_sec, wide_sec, deadline = job
    best_res = None
    best_L = L0
    best_tag = ""
    agg_st = [0, 0, 0, 0]

    # Candidate 1: Current best packing under SA-UnweldedHopper!
    t_tag, st, res = run_sa_unwelded_pipeline(
        arr0, best_L, seed, sa_sec, quench_sec, wide_sec, deadline, tag_prefix="cur"
    )
    for k in range(4):
        agg_st[k] += st[k]
    if res is not None and res[0] < best_L - 1e-9:
        best_L, best_res, best_tag = res[0], res, t_tag

    # Candidate 2: Best near-miss n+1 -> n hole (from goodpjw targets.csv + near-band) under SA-UnweldedHopper!
    if arr_p1 is not None and (deadline - time.time()) > 80.0:
        cands = list(
            dict.fromkeys(
                list(hint_idxs) + list(w32a.pick_near_band(arr_p1, max_cand=180))
            )
        )
        best_h = None
        t_scr = time.time()
        for idx in cands:
            if time.time() - t_scr > 45.0 or (deadline - time.time()) < 60.0:
                break
            if not (0 <= idx < len(arr_p1)):
                continue
            try:
                Qh, Sh = w32a.fast_delete_and_compact(arr_p1, idx)
                if (
                    Qh is not None
                    and Sh < best_L + 0.012
                    and (best_h is None or Sh < best_h[0])
                ):
                    best_h = (Sh, idx, Qh)
            except Exception:
                continue
        if best_h is not None:
            t_tag2, st2, res2 = run_sa_unwelded_pipeline(
                best_h[2],
                best_L,
                seed + 77777,
                sa_sec,
                quench_sec,
                wide_sec,
                deadline,
                tag_prefix=f"del_{best_h[1]}",
            )
            for k in range(4):
                agg_st[k] += st2[k]
            if res2 is not None and res2[0] < best_L - 1e-9:
                best_L, best_res, best_tag = res2[0], res2, t_tag2

    return n, best_tag, tuple(agg_st), best_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--targets-csv", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--sa-sec", type=int, default=120)
    ap.add_argument("--quench-sec", type=int, default=90)
    ap.add_argument("--wide-sec", type=int, default=60)
    ap.add_argument("--max-targets", type=int, default=96)
    ap.add_argument("--seed-off", type=int, default=480000)
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

    tg_hints = {}
    if os.path.exists(args.targets_csv):
        with open(args.targets_csv) as f:
            for row in csv.DictReader(f):
                tg_hints.setdefault(int(row["n"]), []).append(int(row["removed"]))

    gap_ranked = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        fl = front_L.get(n, cur[n][0])
        gds = (cur[n][0] - fl) / math.sqrt(n)
        # Boost puzzles that also appear in goodpjw open near-miss targets!
        if n in tg_hints:
            gds += 0.00015
        if gds > 1e-6:
            gap_ranked.append((gds, n))
    gap_ranked.sort(reverse=True)
    targets = [n for _, n in gap_ranked[: args.max_targets]]

    my_slice = targets[args.shard :: args.num_shards]
    deadline = t0 + args.budget_sec
    jobs = []
    for i, n in enumerate(my_slice):
        arr_p1 = None
        if n + 1 <= 1000:
            k1 = math.isqrt(n + 1)
            if k1 * k1 != (n + 1) and cur[n + 1][0] < k1 + 1 - 1e-6:
                arr_p1 = cur[n + 1][1]
        jobs.append(
            (
                n,
                cur[n][0],
                cur[n][1],
                arr_p1,
                tg_hints.get(n, []),
                args.seed_off + args.shard * 10903 + i * 881,
                args.sa_sec,
                args.quench_sec,
                args.wide_sec,
                deadline,
            )
        )

    print(
        f"[W48A-SAUnwelded s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_targets({len(my_slice)})={my_slice} sa={args.sa_sec}s q={args.quench_sec}s w={args.wide_sec}s ({time.time() - t0:.1f}s)",
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
    futs = [ex.submit(worker_gap_and_target, j) for j in jobs]
    try:
        for fut in as_completed(futs, timeout=max(30.0, deadline - time.time() + 90.0)):
            done += 1
            try:
                n, tag, st, res = fut.result()
            except Exception as exc:
                print(f"  [W48A ERR] {exc}", flush=True)
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
                        f"  [W48A WIN!!] n={n:4d} [{tag}] tr={st[0]} fit={st[1]} acc={st[2]} w={st[3]} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
            else:
                print(
                    f"  [W48A done] n={n:4d} tr={st[0]} fit={st[1]} acc={st[2]} w={st[3]} no_win [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                    flush=True,
                )
            if time.time() > deadline + 45.0:
                break
    except TimeoutError:
        print(
            f"  [W48A s={args.shard}] Timeout after {done}/{len(jobs)} — exiting cleanly!",
            flush=True,
        )

    save_ckpt()
    print(
        f"[W48A s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
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
