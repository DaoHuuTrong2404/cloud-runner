#!/usr/bin/env python3
"""Wave 49A: Fault-Open Global-Compaction SA-Unwelded Engine + 4-Basin Cross-Topology Donors!

Fixes BOTH root causes uncovered by Wave 48 forensic telemetry:
  1. Dual-Path Fault-Open Global Compaction (`fop`):
     In W47M/W48A, when `room=True` opened a fault line (`Q[up, axis] += eta`, zero overlap) and
     `E.relax` found a zero-overlap cluster re-packing inside `[0, S + eta]^2`, line 157 snapped
     `Q[up, axis] -= eta` locally against a FROZEN `ring` (`free=False`), killing >95% of all
     aspect-ratio / bar-count / angle-changing moves BEFORE `Compactor` ever saw them!
     Now: if local snap fails (`en2 > 1e-14`), `Q_open` IS ALREADY A VALID ZERO-OVERLAP GLOBAL
     PACKING OF SIDE `<= S + eta`! We feed `Q_open` directly into global `C.compact(Q_open)`
     (throttled so trial throughput stays fast), allowing ALL `n` squares across both halves and
     perpendicular bands to slide simultaneously to close `+eta` into a new basin!
  2. Full-Packing Unwelding for `n <= 460` (`np.arange(n)`):
     Unwelds 100% of upright + tilted squares for `n <= 460` (`234, 261, 332, 333, 367, 401, 414,
     456`), and up to `460` tilted + 2-layer interface singles for `n > 460`!
  3. 2-Cycle Reheating Metropolis SA (`T: 0.0045 -> 3e-5` x 2 cycles) + Mid-Cycle `SLP.polish`
     nonlinear contact unlock -> `T=0` Unwelded Quench -> Welded Wide-Hopper -> `SLP` + `fq` +
     `lp_rot` + exact `Decimal64 >= 1.05e-9`!
  4. 4-Basin Cross-Topology Donor Attack per Top-96 Gap Puzzle:
     (a) Current `P[n]`, (b) `n+1 -> n` (`E.skeleton` dual-force + `targets.csv` + `near_band`),
     (c) `n-1 -> n` soft void-insertion (`233->234`, `366->367`, `737->738`!),
     (d) `n+2 -> n` 2-square skeleton deletion!
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
import slp as SLP
import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12
import w32a_fresh_basin_fast_hopper as w32a
import w44h_neighbor_donor_wide_hopper as w44h
import w47m_unwelded_mega_hopper as w47m


def load_closed_frontier(union_npz, win_globs):
    """Load frontier + close both L-borders AND best-boundary-square delete-1 cascades!"""
    cur, fresh, _, n_files = w12.load_frontier_with_wins(union_npz, win_globs)
    for _ in range(3):
        moved = 0
        for n in range(999, 10, -1):
            Lp1, sq = cur[n + 1]
            if Lp1 < cur[n][0] + 0.015:
                h = 0.5 * (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2])))
                x0, x1 = (sq[:, 0] - h).min(), (sq[:, 0] + h).max()
                y0, y1 = (sq[:, 1] - h).min(), (sq[:, 1] + h).max()
                on_wall = np.where(
                    ((sq[:, 0] - h) < x0 + 1e-4)
                    | ((sq[:, 0] + h) > x1 - 1e-4)
                    | ((sq[:, 1] - h) < y0 + 1e-4)
                    | ((sq[:, 1] + h) > y1 - 1e-4)
                )[0]
                cands = list(on_wall[:16]) + [len(sq) - 1]
                for idx in cands:
                    sub = np.delete(sq, idx, axis=0)
                    hs = np.delete(h, idx)
                    Ls = float(
                        max(
                            (sub[:, 0] + hs).max() - (sub[:, 0] - hs).min(),
                            (sub[:, 1] + hs).max() - (sub[:, 1] - hs).min(),
                        )
                    )
                    if Ls < cur[n][0] - 1e-8:
                        cur[n] = (Ls, sub)
                        fresh.add(n)
                        moved += 1
        for n in range(10, 1000):
            Ln, sq = cur[n]
            k = int(math.floor(Ln))
            n2 = n + 2 * k + 1
            if n2 <= 1000 and Ln + 1.0 + 1e-7 < cur[n2][0] - 1e-7:
                he = 0.5 * (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2])))
                xs = sq[:, 0] - np.min(sq[:, 0] - he)
                ys = sq[:, 1] - np.min(sq[:, 1] - he)
                L_act = max(float(np.max(xs + he)), float(np.max(ys + he)))
                L_new = L_act + 1.0 + 1e-7
                extra = []
                for i in range(k):
                    extra.append((L_new - 0.5, i + 0.5, 0.0))
                    extra.append((i + 0.5, L_new - 0.5, 0.0))
                extra.append((L_new - 0.5, L_new - 0.5, 0.0))
                sq2 = np.vstack(
                    [
                        np.column_stack([xs, ys, sq[:, 2]]),
                        np.array(extra, dtype=np.float64),
                    ]
                )
                cur[n2] = (w12.span(sq2), sq2)
                fresh.add(n2)
                moved += 1
        if not moved:
            break
    tot = sum(cur[n][0] / math.sqrt(n) - 1.0 for n in range(1, 1001))
    return cur, fresh, tot, n_files


def build_singles_w49(P, tol=1e-3, max_singles=460):
    """Unweld ALL squares when n <= max_singles; otherwise unweld all tilted + 2-layer interface!"""
    n = len(P)
    if n <= max_singles:
        return np.arange(n, dtype=np.int64)
    return w47m.build_unwelded_singles(P, tol=tol, max_singles=max_singles)


class FaultOpenSAHopper:
    def __init__(
        self, P, L_ref=None, seed=0, tol=1e-3, max_singles=460, max_drift=0.008
    ):
        self.rng = np.random.default_rng(seed)
        self.tol = tol
        self.max_singles = max_singles
        self.max_drift = float(max_drift)
        self.stats = dict(
            trials=0,
            fit=0,
            fop_calls=0,
            fop_accepts=0,
            sa_accepts=0,
            wins=0,
            ref_wins=0,
            restarts=0,
        )
        self.t0 = time.time()
        self.reset(E.to_origin(P))
        self.best_P = self.P.copy()
        self.best_S = float(self.S)
        self.L_ref = float(L_ref) if L_ref is not None else float(self.S)
        self._since_best = 0
        self._last_fop_t = 0.0

    def reset(self, P):
        loose = build_singles_w49(P, self.tol, self.max_singles)
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
        if self.w.sum() <= 0 and self.loose.any():
            self.w = self.loose.astype(np.float64)
        tilts = E.fold(self.P[:, 2])
        tilts = tilts[np.abs(tilts) > 0.02]
        uniq_t = np.unique(np.round(tilts, 3))
        if len(uniq_t) > 10:
            uniq_t = uniq_t[np.linspace(0, len(uniq_t) - 1, 10, dtype=int)]
        # Rich rational shear + existing angle palette!
        pal = [
            0.0,
            math.pi / 4.0,
            -math.pi / 4.0,
            math.atan2(2.0, 5.0),
            -math.atan2(2.0, 5.0),
            math.atan2(1.0, 2.0),
            -math.atan2(1.0, 2.0),
        ]
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

    def _accept_candidate(self, Q2, S2, temp, is_fop=False):
        S = self.S
        dS = float(S2 - S)
        accept = False
        if dS < -1e-8:
            accept = True
        elif temp > 0.0 and dS < 3.8 * temp:
            if self.rng.random() < math.exp(-max(0.0, dS) / max(temp, 1e-9)):
                accept = True
        if not accept:
            if not np.array_equal(self.C.anchor, self.P[:, :2]):
                self.C.build(self.P, np.where(self.loose)[0])
            self.C.P = self.P
            return False

        if is_fop:
            self.stats["fop_accepts"] += 1
        if S2 < self.best_S - 1e-9:
            self.best_S = float(S2)
            self.best_P = Q2.copy()
            self.stats["wins"] += 1
            if S2 < self.L_ref - 1e-9:
                self.stats["ref_wins"] += 1
            self._since_best = 0
        else:
            self.stats["sa_accepts"] += 1
            self._since_best += 1

        if temp > 0.0 and self._since_best >= 20 and self.S > self.best_S + 2.5e-4:
            self._since_best = 0
            self.stats["restarts"] += 1
            self.reset(self.best_P.copy())
        else:
            self.reset(Q2)
        return True

    def sa_trial(self, temp):
        rng = self.rng
        P = self.P
        S = self.S
        C = self.C
        self.stats["trials"] += 1

        if rng.random() < 0.76:
            i0 = int(rng.choice(len(P), p=self.w / self.w.sum()))
        else:
            loose_ids = np.where(self.loose)[0]
            i0 = int(rng.choice(loose_ids))
        c0 = P[i0, :2] + rng.normal(0, 0.5, 2)

        tier = rng.random()
        if tier < 0.28:
            rho = rng.uniform(1.6, 3.0)
            max_cl = 16
            rel_it = 320
            eta_hi = 0.11
        elif tier < 0.72:
            rho = rng.uniform(2.6, 4.7)
            max_cl = 36
            rel_it = 450
            eta_hi = 0.14
        else:
            rho = rng.uniform(3.8, 6.2)
            max_cl = 56
            rel_it = 550
            eta_hi = 0.17

        d = np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1])
        cluster = np.where((d < rho) & self.loose)[0]
        if len(cluster) == 0:
            return False
        if len(cluster) > max_cl:
            cluster = cluster[np.argsort(d[cluster])[:max_cl]]
            rho_eff = float(d[cluster].max()) + 0.05
        else:
            rho_eff = rho

        near = np.where(d < rho_eff + 1.55)[0]
        ring = np.where((d >= rho_eff + 1.55) & (d < rho_eff + 3.85))[0]
        sub = np.concatenate([near, ring])
        free = np.zeros(len(sub), np.bool_)
        in_cl = np.zeros(len(P), np.bool_)
        in_cl[cluster] = True
        free[: len(near)] = self.loose[near] | in_cl[near]

        Q = P.copy()
        room = rng.random() < 0.72
        eta = 0.0
        axis = 0
        up = None
        if room:
            eta = float(rng.uniform(0.022, eta_hi))
            axis = int(rng.integers(2))
            up = self.cen[C.body, axis] > c0[axis] + rng.uniform(-1.1, 1.1)
            Q[up, axis] += eta
            if E.min_gap_over(Q, C.near) < -1e-9:
                room = False
                Q = P.copy()

        # Perturb cluster (36% coherent group retilt/palette jump, 64% mixed shake!)
        if rng.random() < 0.36:
            dth = float(rng.normal(0.0, 0.085))
            target_th = float(rng.choice(self.palette))
            use_target = rng.random() < 0.48
            for i in cluster:
                Q[i, :2] += rng.normal(0, 0.19, 2)
                if use_target:
                    Q[i, 2] = target_th + rng.normal(0, 0.028)
                else:
                    Q[i, 2] += dth + rng.normal(0, 0.022)
        else:
            for i in cluster:
                Q[i, :2] += rng.normal(0, 0.14 if rng.random() < 0.5 else 0.35, 2)
                u = rng.random()
                if u < 0.26:
                    pass
                elif u < 0.60:
                    Q[i, 2] = rng.choice(self.palette)
                elif u < 0.82:
                    Q[i, 2] += rng.normal(0, 0.065)
                else:
                    Q[i, 2] = rng.uniform(-0.785, 0.785)

        box_allow = S + (2.5 * temp if temp > 0.0 else 0.0)
        R_open, en_open = E.relax(
            Q[sub], free, box_allow + (eta if room else 0.0) + 1e-9, rel_it, 1e-16
        )
        if en_open > 1e-14:
            return False

        # Save zero-overlap fault-opened state BEFORE trying local -= eta snap!
        Q_open = Q.copy()
        Q_open[near] = R_open[: len(near)]

        if room:
            Q_snap = Q_open.copy()
            Q_snap[up, axis] -= eta
            R_snap, en_snap = E.relax(
                Q_snap[sub], free, box_allow + 1e-9, rel_it, 1e-16
            )
            if en_snap <= 1e-14:
                Q_use = Q_snap
                Q_use[near] = R_snap[: len(near)]
            else:
                # BREAKTHROUGH FAULT-OPEN GLOBAL PATH!
                # Local cavity snap collided with frozen `ring` because cluster reshaped/retilted!
                # `Q_open` is ALREADY zero-overlap globally with side <= box_allow + eta!
                now_t = time.time()
                shift_op = np.abs(Q_open[near] - P[near]).max()
                if (
                    shift_op >= 2e-3
                    and (now_t - self._last_fop_t) >= 0.32
                    and rng.random() < 0.55
                ):
                    self._last_fop_t = now_t
                    self.stats["fit"] += 1
                    self.stats["fop_calls"] += 1
                    C.basis = dict(self.warm)
                    Q2, S2, ov = C.compact(Q_open, max_iter=55, patience=2)
                    if ov <= 2e-9 and S2 <= self.best_S + self.max_drift:
                        return self._accept_candidate(Q2, S2, temp, is_fop=True)
                    if not np.array_equal(C.anchor, self.P[:, :2]):
                        C.build(self.P, np.where(self.loose)[0])
                    C.P = self.P
                return False
        else:
            Q_use = Q_open

        shift = np.abs(Q_use[near] - P[near]).max(axis=1)
        if shift.max() < 1e-3:
            return False
        Q_use[near[shift <= 1e-9]] = P[near[shift <= 1e-9]]
        self.stats["fit"] += 1

        moved = near[shift > 1e-9]
        allow_up = 2.6 * temp if temp > 0.0 else -1e-8
        pred = self.L.evaluate(Q_use, moved)
        if pred is not None and pred > S + allow_up + 6e-4:
            return False
        if temp <= 0.0 and (pred is None or pred > S - 1e-8):
            if not ((len(cluster) >= 24) and (rng.random() < 0.22)):
                return False

        C.basis = dict(self.warm)
        Q2, S2, ov = C.compact(Q_use, max_iter=50, patience=1)
        if ov > 2e-9 or S2 > self.best_S + self.max_drift:
            if not np.array_equal(C.anchor, self.P[:, :2]):
                C.build(self.P, np.where(self.loose)[0])
            C.P = self.P
            return False

        return self._accept_candidate(Q2, S2, temp, is_fop=False)

    def _quick_slp_unlock(self):
        try:
            xs, ys, ts, _ = SLP.polish(
                self.best_P[:, 0],
                self.best_P[:, 1],
                self.best_P[:, 2],
                iters=45,
                r0=3.0e-3,
                rmin=1e-10,
                tol_sep=-1e-13,
                time_limit=8.0,
            )
            Qs = w32a.safe_open(np.column_stack([xs, ys, ts]))
            Ss = float(E.side(Qs))
            if Ss < self.best_S - 1e-9 and E.min_gap(Qs) >= -1e-12:
                self.best_S = Ss
                self.best_P = Qs.copy()
                if Ss < self.L_ref - 1e-9:
                    self.stats["ref_wins"] += 1
        except Exception:
            pass
        self.reset(self.best_P.copy())

    def run_reheated_anneal_and_quench(
        self, sa_sec, quench_sec, t_init=0.0045, t_min=3e-5, deadline=None
    ):
        # 2 Reheating SA cycles + mid-cycle SLP nonlinear unlock!
        if sa_sec > 8.0:
            n_cyc = 2 if sa_sec >= 40.0 else 1
            cyc_sec = sa_sec / float(n_cyc)
            for ci in range(n_cyc):
                tc = time.time()
                t_hi = t_init if ci == 0 else (0.55 * t_init)
                while self.applicable():
                    now = time.time()
                    if deadline is not None and now > deadline - 12.0:
                        break
                    el = now - tc
                    if el >= cyc_sec:
                        break
                    frac = min(1.0, max(0.0, el / cyc_sec))
                    temp = t_hi * ((t_min / t_hi) ** frac)
                    self.sa_trial(temp)
                if deadline is None or time.time() < deadline - 18.0:
                    self._quick_slp_unlock()

        if self.S > self.best_S + 1e-10:
            self.reset(self.best_P.copy())

        t1 = time.time()
        while (time.time() - t1 < quench_sec) and self.applicable():
            if deadline is not None and time.time() > deadline - 10.0:
                break
            self.sa_trial(0.0)

        return self.best_P, self.best_S


def run_fault_open_pipeline(
    P_init, L0, seed, sa_sec, quench_sec, wide_sec, deadline, tag_prefix="fop"
):
    """Full Fault-Open SA-Unwelded -> SLP -> Unwelded-Quench -> Welded-Wide-Hopper -> fq -> Decimal64!"""
    cur_P = w32a.safe_open(P_init)
    best_S = float(E.side(cur_P))
    best_P = cur_P if (best_S < L0 - 1e-9 and E.min_gap(cur_P) >= -1e-12) else None
    best_tag = f"{tag_prefix}_raw" if best_P is not None else ""
    stats_out = (0, 0, 0, 0, 0)

    rem = deadline - time.time()
    if rem > 22.0:
        s1 = min(sa_sec, max(10.0, 0.46 * (rem - 14.0)))
        s2 = min(quench_sec, max(8.0, 0.32 * (rem - 14.0)))
        try:
            H = FaultOpenSAHopper(cur_P, L_ref=L0, seed=seed, max_drift=0.008)
            if H.applicable():
                Qa, Sa = H.run_reheated_anneal_and_quench(
                    s1, s2, t_init=0.0045, t_min=3e-5, deadline=deadline
                )
                stats_out = (
                    H.stats["trials"],
                    H.stats["fit"],
                    H.stats["fop_accepts"],
                    H.stats["sa_accepts"],
                    H.stats["ref_wins"],
                )
                Qa, Sa = w47m._slp_polish(Qa)
                if Sa < best_S:
                    best_S = Sa
                if Sa < L0 - 1e-9 and E.min_gap(Qa) >= -1e-12:
                    if best_P is None or Sa < float(E.side(best_P)) - 1e-10:
                        best_P, best_tag = Qa, f"{tag_prefix}_fop"
                if Sa < float(E.side(cur_P)) + 0.004:
                    cur_P = Qa
        except Exception:
            pass

    rem = deadline - time.time()
    if wide_sec > 5.0 and rem > 18.0:
        try:
            Hw = E.Hopper(
                w32a.safe_open(best_P if best_P is not None else cur_P),
                seed=seed + 31337,
                wide=True,
            )
            if Hw.applicable():
                Qw, Sw = Hw.run(min(wide_sec, rem - 10.0))
                Qw, Sw = w47m._slp_polish(Qw)
                if Sw < best_S:
                    best_S = Sw
                if Sw < L0 - 1e-9 and E.min_gap(Qw) >= -1e-12:
                    if best_P is None or Sw < float(E.side(best_P)) - 1e-10:
                        best_P, best_tag = Qw, f"{tag_prefix}_fowh"
        except Exception:
            pass

    if best_P is not None and float(E.side(best_P)) < L0 - 1e-9:
        n = len(best_P)
        cur_a, cur_s = w47m._slp_polish(best_P)
        if os.path.exists(f2d.FQ_BIN) and (deadline - time.time()) > 12.0:
            with tempfile.TemporaryDirectory(prefix=f"w49_{n}_") as tmp:
                r_fin = f2d.run_fq(
                    cur_s,
                    cur_a,
                    tmp,
                    ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
                    timeout=min(55, max(12, int(deadline - time.time() - 4))),
                )
                if r_fin is not None and r_fin[0] < cur_s:
                    cur_s, cur_a = r_fin[0], r_fin[1]
        try:
            cur_a = w12.lp_rot_polish(cur_a)
        except Exception:
            pass
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_a)
        if ok and Lf < L0 - 1e-9:
            return best_tag, stats_out, float(Lf), (Lf, xf, yf, tf)
    return best_tag, stats_out, best_S, None


def worker_4basin_gap(job):
    (
        n,
        L0,
        arr0,
        arr_m1,
        L_m1,
        arr_p1,
        L_p1,
        arr_p2,
        L_p2,
        hint_idxs,
        seed,
        sa_sec,
        quench_sec,
        wide_sec,
        deadline,
    ) = job
    best_res = None
    best_L = L0
    closest_S = float("inf")
    best_tag = ""
    agg_st = [0, 0, 0, 0, 0]

    # Basin 1: Current best P[n] under Fault-Open SA-Unwelded!
    t_tag, st, cS, res = run_fault_open_pipeline(
        arr0, best_L, seed, sa_sec, quench_sec, wide_sec, deadline, tag_prefix="cur"
    )
    closest_S = min(closest_S, cS)
    for k in range(5):
        agg_st[k] += st[k]
    if res is not None and res[0] < best_L - 1e-9:
        best_L, best_res, best_tag = res[0], res, t_tag

    # Collect Cross-Topology Neighbor Donors (n+1 del1, n-1 void-insert, n+2 del2)
    donors = []
    if arr_p1 is not None and (deadline - time.time()) > 75.0:
        cands = list(
            dict.fromkeys(
                list(hint_idxs)
                + list(w44h._skeleton_and_band_indices(arr_p1, max_idx=40))
            )
        )
        best_d1 = None
        t_s = time.time()
        for idx in cands:
            if time.time() - t_s > 30.0:
                break
            if 0 <= idx < len(arr_p1):
                try:
                    Qd, Sd = w32a.fast_delete_and_compact(arr_p1, idx)
                    if (
                        Qd is not None
                        and Sd < best_L + 0.022
                        and (best_d1 is None or Sd < best_d1[0])
                    ):
                        best_d1 = (float(Sd), f"d1_{idx}", Qd)
                except Exception:
                    continue
        if best_d1 is not None:
            donors.append(best_d1)

    if arr_m1 is not None and (deadline - time.time()) > 75.0:
        try:
            ins_list = w44h.generate_insert_donors(
                arr_m1, L_m1, best_L, min(deadline, time.time() + 35.0), max_donors=1
            )
            donors.extend(ins_list)
        except Exception:
            pass

    if arr_p2 is not None and (deadline - time.time()) > 75.0:
        try:
            d2_list = w44h.generate_delete2_donors(
                arr_p2, L_p2, best_L, min(deadline, time.time() + 30.0), max_donors=1
            )
            donors.extend(d2_list)
        except Exception:
            pass

    donors.sort(key=lambda z: z[0])
    for di, (Sd, dtag, Qd) in enumerate(donors[:2]):
        if (deadline - time.time()) < 55.0:
            break
        t_tag2, st2, cS2, res2 = run_fault_open_pipeline(
            Qd,
            best_L,
            seed + 50021 * (di + 1),
            int(0.85 * sa_sec),
            int(0.85 * quench_sec),
            int(0.85 * wide_sec),
            deadline,
            tag_prefix=dtag,
        )
        closest_S = min(closest_S, cS2)
        for k in range(5):
            agg_st[k] += st2[k]
        if res2 is not None and res2[0] < best_L - 1e-9:
            best_L, best_res, best_tag = res2[0], res2, t_tag2

    return n, best_tag, tuple(agg_st), closest_S - L0, best_res


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
    ap.add_argument("--sa-sec", type=int, default=105)
    ap.add_argument("--quench-sec", type=int, default=70)
    ap.add_argument("--wide-sec", type=int, default=45)
    ap.add_argument("--max-targets", type=int, default=96)
    ap.add_argument("--seed-off", type=int, default=490000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, n_files = load_closed_frontier(args.union_npz, args.win_glob)
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
        if n in tg_hints:
            gds += 0.00012
        if gds > 1e-6:
            gap_ranked.append((gds, n))
    gap_ranked.sort(reverse=True)
    targets = [n for _, n in gap_ranked[: args.max_targets]]

    my_slice = targets[args.shard :: args.num_shards]
    deadline = t0 + args.budget_sec
    jobs = []
    for i, n in enumerate(my_slice):
        arr_m1, L_m1 = (None, 0.0)
        if n - 1 >= 50:
            km = math.isqrt(n - 1)
            if km * km != (n - 1) and cur[n - 1][0] < km + 1 - 1e-6:
                L_m1, arr_m1 = cur[n - 1]
        arr_p1, L_p1 = (None, 0.0)
        if n + 1 <= 1000:
            k1 = math.isqrt(n + 1)
            if k1 * k1 != (n + 1) and cur[n + 1][0] < k1 + 1 - 1e-6:
                L_p1, arr_p1 = cur[n + 1]
        arr_p2, L_p2 = (None, 0.0)
        if n + 2 <= 1000:
            k2 = math.isqrt(n + 2)
            if k2 * k2 != (n + 2) and cur[n + 2][0] < k2 + 1 - 1e-6:
                L_p2, arr_p2 = cur[n + 2]

        jobs.append(
            (
                n,
                cur[n][0],
                cur[n][1],
                arr_m1,
                L_m1,
                arr_p1,
                L_p1,
                arr_p2,
                L_p2,
                tg_hints.get(n, []),
                args.seed_off + args.shard * 10903 + i * 881,
                args.sa_sec,
                args.quench_sec,
                args.wide_sec,
                deadline,
            )
        )

    print(
        f"[W49A-FaultOpenSA s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
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
    futs = [ex.submit(worker_4basin_gap, j) for j in jobs]
    try:
        for fut in as_completed(futs, timeout=max(30.0, deadline - time.time() + 90.0)):
            done += 1
            try:
                n, tag, st, d_best, res = fut.result()
            except Exception as exc:
                print(f"  [W49A ERR] {exc}", flush=True)
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
                        f"  [W49A WIN!!] n={n:4d} [{tag}] tr={st[0]} fit={st[1]} fop={st[2]} acc={st[3]} rw={st[4]} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
            else:
                print(
                    f"  [W49A done] n={n:4d} tr={st[0]} fit={st[1]} fop={st[2]} acc={st[3]} rw={st[4]} dBest={d_best:+.6f} [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                    flush=True,
                )
            if time.time() > deadline + 45.0:
                break
    except TimeoutError:
        print(
            f"  [W49A s={args.shard}] Timeout after {done}/{len(jobs)} — exiting cleanly!",
            flush=True,
        )

    save_ckpt()
    print(
        f"[W49A s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
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
