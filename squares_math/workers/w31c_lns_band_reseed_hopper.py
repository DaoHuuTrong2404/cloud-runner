#!/usr/bin/env python3
"""Wave 31C: Numba L-BFGS Large-Neighborhood Search (Band-Reseed + Angle-Waves + Pin-Rotate) -> HopperV2 + SLP!

Bridges global basin-jumping moves (adapted from goodpjw_ds/tools/lns.py) with HopperV2 + SLP polish:
  - Kind 5/6 (Region & Band-Strip Lattice Reseed): wipes a disc or a strip along the tilted band
    and refills with a coherent rotated lattice at a new angle theta, then L-BFGS relaxes & squeezes!
  - Kind 7 (Band-Wide Angle Wave): coherent sinusoidal / linear ramp / alternating-rung angle wave
    across ALL tilted squares + L-BFGS squeeze!
  - Kind 8/9 (Pin-Rotate Escape): pins 1-2 squares rotated by +-8..32 deg, relaxes the whole box
    around them in L-BFGS, then unpins and squeezes!
  - Kind 0..4 (Cluster Rotation, Rigid Cluster Twist, Orientation Swap, Remove-Reinsert).
  - SECRET SAUCE: Not only keeps strict LNS wins, ALSO harvests distinct near-record basins
    (S < best_S + 0.0045 and angle/position shift > threshold) and feeds them straight into
    HopperV2 + Trust-Region SLP!
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

import sq_engine as E
import squeeze2 as SQ2
import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12
import w30a_hopper_v2_unlocked as w30
import w31a_slp_polish_and_exh_remove as w31a


def warm_all():
    w31a.warm_all()
    x = np.array([0.5, 1.5, 0.5, 1.5], dtype=np.float64)
    y = np.array([0.5, 0.5, 1.5, 1.5], dtype=np.float64)
    t = np.array([0.0, 0.0, 0.0, 0.05], dtype=np.float64)
    SQ2.squeeze(x, y, t, rel0=1e-3, rel_min=1e-4, max_iter=200, max_tries=4)


def knn(x, y, i, k):
    d = (x - x[i]) ** 2 + (y - y[i]) ** 2
    return np.argsort(d)[: k + 1]


def band_angles(x, y, t, rng, L):
    tm = np.abs(((t + np.pi / 4) % (np.pi / 2)) - np.pi / 4)
    T = np.where(tm > 0.02)[0]
    x = x.copy()
    y = y.copy()
    t = t.copy()
    if len(T) < 3:
        return x, y, t, 7, []
    P = np.stack([x[T], y[T]], 1)
    P0 = P - P.mean(0)
    _, V = np.linalg.eigh(np.cov(P0.T))
    u = P0 @ V[:, 1]
    amp = math.radians(rng.uniform(0.6, 6.5))
    mode = int(rng.integers(4))
    if mode == 0:
        per = rng.uniform(2.0, 20.0)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        d = amp * np.sin(2.0 * np.pi * u / per + ph)
    elif mode == 1:
        d = amp * (u - u.mean()) / max(float(np.ptp(u)), 1e-9) * 2.0
    elif mode == 2:
        v = P0 @ V[:, 0]
        d = amp * np.sign(v - np.median(v))
    else:
        d = rng.normal(0.0, amp, len(T))
    t[T] += d
    x[T] += rng.normal(0.0, 0.02, len(T))
    y[T] += rng.normal(0.0, 0.02, len(T))
    x = np.clip(x, 0.5, L - 0.5)
    y = np.clip(y, 0.5, L - 0.5)
    return x, y, t, 7, []


def reseed(x, y, t, rng, L, band=False):
    n = len(x)
    tm = np.abs(((t + np.pi / 4) % (np.pi / 2)) - np.pi / 4)
    tilted = np.where(tm > 0.02)[0]
    if band and len(tilted) >= 4:
        c = int(tilted[rng.integers(len(tilted))])
        P = np.stack([x[tilted], y[tilted]], 1)
        d0 = np.hypot(P[:, 0] - x[c], P[:, 1] - y[c])
        loc = P[d0 < 4.5]
        if len(loc) >= 3:
            C = np.cov((loc - loc.mean(0)).T)
            _, V = np.linalg.eigh(C)
            u = V[:, 1]
        else:
            u = np.array([1.0, 1.0]) / math.sqrt(2.0)
        vperp = np.array([-u[1], u[0]])
        half_len = rng.uniform(1.8, 5.5)
        half_w = rng.uniform(1.0, 2.5)
        rel = np.stack([x - x[c], y - y[c]], 1)
        a1 = rel @ u
        a2 = rel @ vperp
        sel = np.where((np.abs(a1) < half_len) & (np.abs(a2) < half_w))[0]
        med_t = (
            float(np.median(t[tilted[d0 < 4.5]])) if np.any(d0 < 4.5) else float(t[c])
        )
        theta = med_t + rng.normal(0.0, 0.16)
    else:
        c = (
            int(tilted[rng.integers(len(tilted))])
            if (len(tilted) > 0 and rng.random() < 0.75)
            else int(rng.integers(n))
        )
        R = rng.uniform(1.4, 3.3)
        sel = np.where(np.hypot(x - x[c], y - y[c]) < R)[0]
        theta = (
            rng.uniform(0.0, math.pi / 2.0)
            if rng.random() < 0.55
            else float(t[c]) + rng.normal(0.0, 0.30)
        )
    m = len(sel)
    if m < 2:
        return x.copy(), y.copy(), t.copy(), 5 + int(band), []
    e1 = np.array([math.cos(theta), math.sin(theta)])
    e2 = np.array([-math.sin(theta), math.cos(theta)])
    off = rng.uniform(0.0, 1.0, 2)
    gx, gy = np.meshgrid(np.arange(-8, 9), np.arange(-8, 9))
    pts = (gx.reshape(-1, 1) + off[0]) * e1 + (gy.reshape(-1, 1) + off[1]) * e2
    mx, my = float(x[sel].mean()), float(y[sel].mean())
    dist = np.hypot(pts[:, 0], pts[:, 1])
    pick = pts[np.argsort(dist)[:m]]
    x = x.copy()
    y = y.copy()
    t = t.copy()
    x[sel] = mx + pick[:, 0]
    y[sel] = my + pick[:, 1]
    t[sel] = theta + rng.normal(0.0, 0.01, m)
    x = np.clip(x, 0.5, L - 0.5)
    y = np.clip(y, 0.5, L - 0.5)
    return x, y, t, 5 + int(band), []


def perturb(x, y, t, rng, L):
    n = len(x)
    x = x.copy()
    y = y.copy()
    t = t.copy()
    # Bias toward the highest-impact structural moves: band_angles(7), band reseed(6), disc reseed(5), pin-rot(8,9), rigid twist(2)
    kind = int(rng.choice([2, 4, 5, 6, 6, 7, 7, 8, 9, 1]))
    tm = np.abs(((t + np.pi / 4) % (np.pi / 2)) - np.pi / 4)
    tilted = np.where(tm > 0.02)[0]
    i = (
        int(tilted[rng.integers(len(tilted))])
        if (len(tilted) > 0 and rng.random() < 0.8)
        else int(rng.integers(n))
    )

    if kind == 7:
        return band_angles(x, y, t, rng, L)
    if kind in (8, 9):
        js = [i] if kind == 8 else [int(q) for q in knn(x, y, i, 1)[:2]]
        for j in js:
            t[j] += float(rng.choice([-1, 1])) * math.radians(rng.uniform(8.0, 32.0))
        return x, y, t, kind, js
    if kind in (5, 6):
        return reseed(x, y, t, rng, L, band=(kind == 6))
    if kind == 1:
        cl = knn(x, y, i, int(rng.integers(4, 16)))
        t[cl] += rng.normal(0.0, 0.14, len(cl))
        x[cl] += rng.normal(0.0, 0.07, len(cl))
        y[cl] += rng.normal(0.0, 0.07, len(cl))
    elif kind == 2:
        cl = knn(x, y, i, int(rng.integers(4, 14)))
        a = float(rng.normal(0.0, 0.12))
        cx, cy = float(x[cl].mean()), float(y[cl].mean())
        dx, dy = x[cl] - cx, y[cl] - cy
        x[cl] = cx + dx * math.cos(a) - dy * math.sin(a)
        y[cl] = cy + dx * math.sin(a) + dy * math.cos(a)
        t[cl] += a
    else:
        cl = knn(x, y, i, int(rng.integers(2, 7)))
        j = int(cl[rng.integers(1, len(cl))])
        t[i] = t[j] + rng.normal(0.0, 0.05)
        x[cl] += rng.normal(0.0, 0.03, len(cl))
        y[cl] += rng.normal(0.0, 0.03, len(cl))
    x = np.clip(x, 0.5, L - 0.5)
    y = np.clip(y, 0.5, L - 0.5)
    return x, y, t, kind, []


def basin_distance(P1, P2):
    """Fast proxy for whether two packings are structurally distinct (angle distribution diff)."""
    t1 = np.sort(np.abs(E.fold(P1[:, 2])))
    t2 = np.sort(np.abs(E.fold(P2[:, 2])))
    return float(np.mean(np.abs(t1 - t2)))


def worker_lns(job):
    n, L0, arr0, seed, lns_sec, hop_sec, slp_sec = job
    rng = np.random.default_rng(seed)
    P0 = E.to_origin(np.asarray(arr0, dtype=np.float64))
    x, y, t = SQ2.normalize(P0[:, 0].copy(), P0[:, 1].copy(), P0[:, 2].copy())
    best_S = float(E.side(P0))
    best_P = None
    best_tag = ""

    # Phase 1: LNS basin-hopping with Numba L-BFGS squeeze2
    t_end = time.time() + lns_sec
    alt_basins = []  # (S, kind, Q_arr)
    while time.time() < t_end:
        px, py, pt, kind, pins = perturb(x, y, t, rng, best_S)
        big = kind in (5, 6, 7, 8, 9)
        gk = (
            6e-3
            if kind == 7
            else (rng.uniform(1e-2, 4e-2) if kind >= 8 else (2.2e-2 if big else 3e-3))
        )
        Sg = best_S * (1.0 + gk)
        if kind >= 8 and pins:
            fx_ = np.zeros(n, np.int64)
            fx_[pins] = 2
            ok, rx, ry, rt, _ = SQ2.relax_at(px, py, pt, Sg, 6000, fixed=fx_)
        else:
            ok, rx, ry, rt, _ = SQ2.relax_at(px, py, pt, Sg, 6000 if big else 3500)
        if not ok:
            continue
        S, x2, y2, t2 = SQ2.squeeze(
            rx, ry, rt, rel0=gk / 2.0, rel_min=3e-7, max_iter=3500, max_tries=42
        )
        if not math.isfinite(S):
            continue
        # Direct strict improvement!
        if S < best_S * (1.0 - 2e-9):
            S_hi, x3, y3, t3 = SQ2.squeeze(
                x2, y2, t2, rel0=1e-6, rel_min=5e-10, max_iter=15000, max_tries=50
            )
            Lf, fx, fy, ft, good = SQ2.finalize(x3, y3, t3, tol=2e-11)
            if good and Lf < best_S - 1e-9:
                Q = E.to_origin(np.column_stack([fx, fy, ft]))
                Lq = float(E.side(Q))
                if Lq < best_S - 1e-9:
                    best_S, best_P, best_tag = Lq, Q, f"lns_k{kind}"
                    x, y, t = SQ2.normalize(fx.copy(), fy.copy(), ft.copy())
        # Or distinct near-record basin -> harvest for HopperV2 bridge!
        elif S < best_S + 0.0045 and len(alt_basins) < 12:
            Lf, fx, fy, ft, good = SQ2.finalize(x2, y2, t2, tol=2e-11)
            if good and Lf < best_S + 0.0048:
                Q = E.to_origin(np.column_stack([fx, fy, ft]))
                if basin_distance(Q, P0) > 0.004:
                    if all(abs(Lf - s0) > 1e-5 for s0, _, _ in alt_basins):
                        alt_basins.append((float(E.side(Q)), kind, Q))

    # Phase 2: Bridge the best LNS winner + top-2 distinct near-record basins into HopperV2 + SLP!
    to_hop = []
    if best_P is not None:
        to_hop.append((best_S, best_tag, best_P))
    alt_basins.sort(key=lambda z: z[0])
    for Sa, kind, Qa in alt_basins[:2]:
        to_hop.append((Sa, f"lns_alt_k{kind}", Qa))

    for hi, (Sa, htag, Qa) in enumerate(to_hop):
        Qs, Ss = w31a.run_slp_polish(
            Qa, iters=80, r0=4e-3, time_limit=min(20.0, slp_sec)
        )
        if Ss < Sa:
            Qa, Sa = Qs, Ss
        if Sa < best_S - 1e-8:
            best_S, best_P, best_tag = Sa, Qa, f"{htag}_slp"
        Qh, Sh, _ = w30.run_v2_chain(
            Qa,
            seed + 51000 + hi * 4999,
            hop_sec,
            min(140, hop_sec),
            unlock_bands=True,
            max_cascades=3,
        )
        if Sh < best_S - 1e-8:
            best_S, best_P, best_tag = Sh, Qh, f"{htag}_hv2"
            Qs2, Ss2 = w31a.run_slp_polish(
                best_P, iters=120, r0=3e-3, time_limit=slp_sec
            )
            if Ss2 < best_S - 1e-9:
                best_S, best_P, best_tag = Ss2, Qs2, f"{htag}_hv2_slp"

    if best_P is not None and best_S < L0 - 1e-9:
        return w31a.finalize_candidate(n, L0, best_P, best_tag)
    return n, best_tag, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2800)
    ap.add_argument("--lns-sec", type=int, default=180)
    ap.add_argument("--hop-sec", type=int, default=130)
    ap.add_argument("--slp-sec", type=float, default=28.0)
    ap.add_argument("--min-rank", type=int, default=0)
    ap.add_argument("--max-targets", type=int, default=120)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    warm_all()
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
    jobs = [
        (
            n,
            cur[n][0],
            cur[n][1],
            3140001 + args.shard * 10169 + i * 331,
            args.lns_sec,
            args.hop_sec,
            args.slp_sec,
        )
        for i, n in enumerate(my_slice)
    ]

    print(
        f"[W31C-LNS s={args.shard}/{args.num_shards}] Base={tot_score:.10f} "
        f"(loaded {n_files} win files), my_targets({len(my_slice)})={my_slice} ({time.time() - t0:.1f}s)",
        flush=True,
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
        futs = [ex.submit(worker_lns, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W31C ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                dL = Lf - cur[n][0]
                if dL < -1e-9:
                    dS = dL / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W31C-LNS WIN!!] n={n:4d} {tag:20s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W31C s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W31C s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
