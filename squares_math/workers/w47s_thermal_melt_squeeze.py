#!/usr/bin/env python3
"""Wave 47S: Semi-Global / Full-Band Thermal Melt + Progressive E.relax Squeeze + UnweldedMegaHopper!

Escapes deep hardened basins (especially Top-60 gaps like n=234, 367, 491, 333, 738, 456, 261, 904,
and ALL K^2+1 tilted-core packings 401, 530, 577, 626, 677, 730, 785, 842, 901, 962!):
  - Move A (Coherent Band/Core Retilt + Melt): dilates S by +0.02..0.06, coherently shifts the
    entire tilted core angle by +-0.03..0.14 rad + shakes 2-layer upright interface, progressively
    squeezes via 3-stage Numba E.relax -> Unwelded Compactor!
  - Move B (Giant 6x6..9x9 Window Thermal Melt): melts 35..80 squares around high-load contacts,
    re-tiles with palette angles (0, +-med_t, +-45 deg), progressively squeezes -> Unwelded Compactor!
  - Move C (Fresh Global Starts for n <= 280): random + 45-deg/atan(1/2) tilted-core seeds relaxed
    at S_target = L0 - 0.002!
  - Every soft re-frozen basin landing within L0 + 0.014 gets UnweldedMegaHopper + SLP + fq + Decimal64!
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
import w32a_fresh_basin_fast_hopper as w32a
import w47m_unwelded_mega_hopper as w47m


def _progressive_squeeze(Q, free_mask, S_start, S_goal, iters=550):
    """3-stage progressive box squeeze using Numba E.relax."""
    cur = Q.copy()
    for S_step in (S_start, 0.5 * (S_start + S_goal), S_goal):
        cur, en = E.relax(cur, free_mask, float(S_step), iters, 1e-18)
        if en > 1e-14:
            if S_step == S_start:
                return None
            # Try one gentler step at midpoint
            cur_prev = cur
            cur, en = E.relax(cur_prev, free_mask, float(S_step + 0.015), iters, 1e-18)
            if en > 1e-14:
                return None
    return cur


def _unwelded_compact(Q):
    try:
        singles = w47m.build_unwelded_singles(Q, tol=1e-3, max_singles=420)
        Qc, Sc, ov = E.Compactor(Q, tol=1e-3, singles=singles).compact(
            max_iter=55, patience=2
        )
        if ov <= 2e-9 and E.min_gap(Qc) >= -1e-9:
            return Qc.astype(np.float64), float(Sc)
    except Exception:
        pass
    return None


def worker_thermal_melt(job):
    n, L0, arr0, seed, melt_sec, hop_sec, deadline = job
    rng = np.random.default_rng(seed)
    P0 = E.to_origin(w32a.safe_open(arr0))
    S0 = float(E.side(P0))

    ft = E.fold(P0[:, 2])
    til_angles = np.abs(ft[np.abs(ft) > 0.02])
    med_t = float(np.median(til_angles)) if len(til_angles) else 0.45
    palette = np.array(
        [0.0, med_t, -med_t, math.pi / 4.0, -math.pi / 4.0], dtype=np.float64
    )

    singles0 = w47m.build_unwelded_singles(P0, tol=1e-3, max_singles=420)
    free_interface = np.zeros(n, dtype=np.bool_)
    free_interface[singles0] = True
    all_free = np.ones(n, dtype=np.bool_)

    # Compute contact loads once for targeted window placement
    weights = np.ones(n, dtype=np.float64)
    try:
        _, forces, _ = E.skeleton(P0)
        ld = np.zeros(n, dtype=np.float64)
        for i, j, lam in forces:
            if 0 <= i < n:
                ld[i] += lam
            if 0 <= j < n:
                ld[j] += lam
        if ld.sum() > 0:
            weights = ld + 0.05 * ld.mean()
    except Exception:
        pass
    weights /= weights.sum()

    best_S = L0
    best_P = None
    best_tag = ""
    soft_basins = []  # (S, tag, Q)
    t_melt = time.time()
    n_tries = 0
    n_feasi = 0

    while (time.time() - t_melt < melt_sec) and (
        time.time() < deadline - hop_sec - 40.0
    ):
        n_tries += 1
        u = rng.random()
        Q = P0.copy()

        if (n <= 280) and (u < 0.20):
            # Move C: Fresh tilted-core / global relaxed start for medium top gaps (n=234, 261, 209, 240, 273)!
            S_try = S0 - 0.002 + rng.uniform(0.0, 0.025)
            Q = np.column_stack(
                [
                    rng.random(n) * (S_try - 1.0) + 0.5,
                    rng.random(n) * (S_try - 1.0) + 0.5,
                    rng.choice(palette, size=n) + rng.normal(0, 0.05, n),
                ]
            )
            Qr, en = E.relax(Q, all_free, float(S_try), 700, 1e-20)
            if en > 1e-16:
                continue
            res = _unwelded_compact(Qr)
            tag = "scratch"
        elif u < 0.55:
            # Move A: Coherent Tilted-Band / Core Retilt + Interface Melt!
            eps = float(rng.uniform(0.015, 0.065))
            S_melt = S0 + eps
            fac = S_melt / max(S0, 1e-9)
            Q[:, 0] *= fac
            Q[:, 1] *= fac
            dth = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.025, 0.14))
            til_mask = np.abs(ft) > 0.02
            Q[til_mask, 2] += dth + rng.normal(0, 0.025, int(til_mask.sum()))
            # Shake interface squares
            Q[free_interface, :2] += rng.normal(
                0, rng.uniform(0.08, 0.22), (int(free_interface.sum()), 2)
            )
            if rng.random() < 0.35:
                # Flip a few interface upright squares to tilted angle (encourages b -> b+1!)
                up_if = np.where(free_interface & ~til_mask)[0]
                if len(up_if) > 0:
                    k_flip = min(len(up_if), int(rng.integers(2, 10)))
                    pick = rng.choice(up_if, size=k_flip, replace=False)
                    Q[pick, 2] = med_t + dth + rng.normal(0, 0.04, k_flip)
            use_free = all_free if n <= 420 else free_interface
            Qr = _progressive_squeeze(Q, use_free, S_melt, S0 + 0.008, iters=520)
            if Qr is None:
                continue
            res = _unwelded_compact(Qr)
            tag = "retilt"
        else:
            # Move B: Giant 6x6..9x9 Window Thermal Re-Pack around loaded square!
            i0 = int(rng.choice(n, p=weights))
            cx, cy = P0[i0, 0], P0[i0, 1]
            rad = float(rng.uniform(3.2, 5.5))
            eps = float(rng.uniform(0.020, 0.075))
            S_melt = S0 + eps
            fac = S_melt / max(S0, 1e-9)
            Q[:, 0] *= fac
            Q[:, 1] *= fac
            d = np.hypot(P0[:, 0] - cx, P0[:, 1] - cy)
            win = d < rad
            n_win = int(win.sum())
            if n_win < 8 or n_win > 95:
                continue
            Q[win, :2] += rng.normal(0, rng.uniform(0.18, 0.42), (n_win, 2))
            Q[win, 2] = rng.choice(palette, size=n_win) + rng.normal(0, 0.06, n_win)
            fmask = (d < rad + 1.4) | free_interface
            if n <= 380:
                fmask = all_free
            Qr = _progressive_squeeze(Q, fmask, S_melt, S0 + 0.008, iters=520)
            if Qr is None:
                continue
            res = _unwelded_compact(Qr)
            tag = "winmelt"

        if res is None:
            continue
        Qc, Sc = res
        n_feasi += 1
        if Sc < best_S - 1e-9:
            best_S, best_P, best_tag = Sc, Qc, f"{tag}_raw"
        if Sc < L0 + 0.015:
            # Keep distinct soft basins (different side length or genuine win)
            if all(abs(Sc - bs) > 1e-5 for bs, _, _ in soft_basins):
                soft_basins.append((Sc, tag, Qc))
                soft_basins.sort(key=lambda z: z[0])
                soft_basins = soft_basins[:3]

    # Polish top-2 distinct soft re-frozen basins with UnweldedMegaHopper + SLP!
    for bi, (Sb, btag, Qb) in enumerate(soft_basins[:2]):
        rem = deadline - time.time()
        if rem < 25.0:
            break
        sec = min(hop_sec, rem - 12.0)
        try:
            H = w47m.UnweldedMegaHopper(
                w32a.safe_open(Qb), seed=seed + 50000 + bi * 7919
            )
            if H.applicable():
                Qh, Sh = H.run(sec, deadline=deadline)
                Qh, Sh = w47m._slp_polish(Qh)
                if Sh < best_S - 1e-9 and E.min_gap(Qh) >= -1e-12:
                    best_S, best_P, best_tag = Sh, Qh, f"{btag}_umega"
        except Exception:
            continue

    if best_P is not None and best_S < L0 - 1e-9:
        cur_a, cur_s = w47m._slp_polish(best_P)
        if os.path.exists(f2d.FQ_BIN):
            with tempfile.TemporaryDirectory(prefix=f"w47s_{n}_") as tmp:
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
            return n, best_tag, (n_tries, n_feasi, len(soft_basins)), (Lf, xf, yf, tf)
    return n, best_tag, (n_tries, n_feasi, len(soft_basins)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--melt-sec", type=int, default=360)
    ap.add_argument("--hop-sec", type=int, default=180)
    ap.add_argument("--max-targets", type=int, default=60)
    ap.add_argument("--seed-off", type=int, default=475000)
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

    # Priority 1: K^2+1 tilted-core ladder (401..962) + Top-50 live gaps!
    k2p1 = [730, 962, 785, 530, 401, 901, 842, 577, 677, 626]
    targets = []
    seen = set()
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
    for _, n in gap_ranked[:24]:
        targets.append(n)
        seen.add(n)
    for n in k2p1:
        if n not in seen:
            targets.append(n)
            seen.add(n)
    for _, n in gap_ranked[24:]:
        if len(targets) >= args.max_targets:
            break
        if n not in seen:
            targets.append(n)
            seen.add(n)

    my_slice = targets[args.shard :: args.num_shards]
    deadline = t0 + args.budget_sec
    jobs = [
        (
            n,
            cur[n][0],
            cur[n][1],
            args.seed_off + args.shard * 10837 + i * 829,
            args.melt_sec,
            args.hop_sec,
            deadline,
        )
        for i, n in enumerate(my_slice)
    ]

    print(
        f"[W47S-ThermalMeltSqueeze s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_targets({len(my_slice)})={my_slice} melt={args.melt_sec}s hop={args.hop_sec}s ({time.time() - t0:.1f}s)",
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
    futs = [ex.submit(worker_thermal_melt, j) for j in jobs]
    try:
        for fut in as_completed(futs, timeout=max(30.0, deadline - time.time() + 90.0)):
            done += 1
            try:
                n, tag, st, res = fut.result()
            except Exception as exc:
                print(f"  [W47S ERR] {exc}", flush=True)
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
                        f"  [W47S WIN!!] n={n:4d} [{tag}] tries={st[0]} feas={st[1]} bas={st[2]} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
            else:
                print(
                    f"  [W47S done] n={n:4d} tries={st[0]} feas={st[1]} bas={st[2]} no_win [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                    flush=True,
                )
            if time.time() > deadline + 45.0:
                print(
                    f"  [W47S s={args.shard}] Hard deadline after {done}/{len(jobs)}.",
                    flush=True,
                )
                break
    except TimeoutError:
        print(
            f"  [W47S s={args.shard}] Timeout after {done}/{len(jobs)} — exiting cleanly!",
            flush=True,
        )

    save_ckpt()
    print(
        f"[W47S s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
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
