#!/usr/bin/env python3
"""Wave 28C: De-Novo Two-Band Parametric Frames + Sub-0.50 Wide-Corridor Generator + Structured LNS.

Breaks topology traps (ancient n=18/n=54/45-deg packings like n=234, 456, 332, 367, 491, 738, 804, 914)
and attacks the Sub-0.50 (f < 0.50) K^2+1..4 frontier (n=730, 785, 842, 901, 962, 530, 401, 904):
  Part 1: De-Novo Two-Band Orthogonal Frames via sq_engine.scan_frames / build_frame (K=14..31)
          + Compactor + Hopper + fq Round-Press!
  Part 2: De-Novo Wide Single-Corridor Tilted-Band Generator (bV=3..22, theta=8..36 deg)
          for D=1..6, K=18..31 targeting f < 0.5000!
  Part 3: Structured LNS (band-wide angle waves, tilted-axis strip reseed, 8..32 deg pin-rotates,
          Numba L-BFGS relax -> Compactor -> Hopper -> fq) on Top-80 biggest frontier gaps!
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
    E.open_contacts(P)
    E.scan_frames(8, 20, 4)
    E.band_best(3, 0.3, 8.5, 8)
    E.relax(P, np.ones(5, np.bool_), 3.2, 20, 1e-16)
    H = E.Hopper(P, seed=0)
    if H.applicable():
        H.run(1)


def certify_candidate(n, arr, L_ref, hopper_sec=90, do_rp=True, seed=0):
    cur_P = E.to_origin(arr)
    try:
        Q, S, ov = E.Compactor(cur_P).compact(cur_P, max_iter=150, patience=3)
        if ov < 2e-9 and S < E.side(cur_P) + 1e-6:
            cur_P = E.to_origin(Q)
    except Exception:
        pass
    cur_S = float(E.side(cur_P))
    if cur_S > L_ref + 0.020:
        return None

    if hopper_sec > 0 and cur_S < L_ref + 0.015:
        try:
            H = E.Hopper(cur_P, seed=seed)
            if H.applicable():
                Q2, S2 = H.run(hopper_sec)
                Q2 = E.to_origin(E.open_contacts(Q2))
                if E.side(Q2) < cur_S and E.min_gap(Q2) >= -1e-12:
                    cur_P = Q2
                    cur_S = float(E.side(Q2))
        except Exception:
            pass

    if do_rp and os.path.exists(f2d.FQ_BIN) and cur_S < L_ref + 0.010:
        with tempfile.TemporaryDirectory(prefix=f"w28c_rp_{n}_") as tmp:
            for r0, pad in (("0.08", 0.012), ("0.18", 0.020)):
                r = f2d.run_fq(
                    cur_S + pad,
                    cur_P,
                    tmp,
                    [
                        "--loosen",
                        "1.0",
                        "--round-r0",
                        r0,
                        "--round-compress",
                        "--round-press",
                        "--round-steps",
                        "8",
                        "--round-press-outer",
                        "14",
                        "--seed",
                        str(seed + int(float(r0) * 1000)),
                        "--flip-top",
                        "36",
                        "--pit",
                        "220",
                    ],
                    timeout=55,
                )
                if r is not None and r[0] < cur_S - 1e-9:
                    cur_S, cur_P = r[0], r[1]

    if cur_S >= L_ref - 1e-9:
        return None

    with tempfile.TemporaryDirectory(prefix=f"w28c_fin_{n}_") as tmp:
        cur_a = E.to_origin(E.open_contacts(cur_P))
        cur_s = float(E.side(cur_a))
        if os.path.exists(f2d.FQ_BIN):
            r_fin = f2d.run_fq(
                cur_s,
                cur_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "48", "--pit", "450"],
                timeout=55,
            )
            if r_fin is not None and r_fin[0] < cur_s:
                cur_s, cur_a = r_fin[0], r_fin[1]
        try:
            cur_a = w12.lp_rot_polish(cur_a)
        except Exception:
            pass
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_a)
        if ok and Lf < L_ref - 1e-9:
            return (Lf, xf, yf, tf)
    return None


def frame_worker(job):
    """Part 1 + Part 2: scan_frames(K+1) + wide single-corridor generator for a single K."""
    K, cur_L_map, front_L_map, seed = job
    K_box = K + 1
    found = []

    # Part 1: De-novo two-band orthogonal frames for n in K^2+1 .. (K+1)^2-1
    bmax = min(18, max(6, K // 2 + 3))
    try:
        out, par = E.scan_frames(K_box, 320, bmax)
        # Check every n in this K band where analytic frame is promising or beats grid!
        cands = []
        for D in range(1, 2 * K + 1):
            n = K * K + D
            if n > 1000:
                continue
            S_an = float(out[n])
            if S_an >= K_box - 1e-6:
                continue
            L_ref = cur_L_map[n]
            L_fr = front_L_map.get(n, L_ref)
            # Prioritize huge gaps, grid-breaks, or close analytic hits
            margin = S_an - L_ref
            if margin < 0.022 or (L_ref >= K_box - 1e-5 and S_an < K_box - 1e-5):
                prio = (L_ref - L_fr) / math.sqrt(n) - max(0.0, margin) * 0.05
                cands.append((prio, margin, n, par[n].copy()))
        cands.sort(reverse=True)
        for prio, margin, n, p_row in cands[:8]:
            bV, tV, yV, bH, tH, yH = p_row
            try:
                sq_arr, S_f = E.build_frame(
                    K_box, int(bV), float(tV), float(yV), int(bH), float(tH), float(yH)
                )
                if len(sq_arr) >= n and E.min_gap(sq_arr[:n]) >= -1e-9:
                    res = certify_candidate(
                        n,
                        sq_arr[:n],
                        cur_L_map[n],
                        hopper_sec=100 if margin < 0.008 else 55,
                        do_rp=True,
                        seed=seed + n * 31,
                    )
                    if res is not None:
                        found.append((n, f"frame_{int(bV)}x{int(bH)}", res))
            except Exception:
                continue
    except Exception:
        pass

    # Part 2: Sub-0.50 Wide Single-Corridor Generator for D=1..5 (generalizing n=842 f=0.50317!)
    if K >= 18:
        best_corr = {}  # D -> (S, bV, th, yV, c_up)
        for c_up_delta in (0, 1):
            for bV in range(4, min(22, K)):
                c_up = K + c_up_delta - bV
                if c_up < 2:
                    continue
                for iv in range(12, 260):
                    th = 0.08 + 0.56 * iv / 260.0  # ~4.6 deg .. 36.7 deg
                    W = bV * math.cos(th) + math.sin(th)
                    S = c_up + W
                    if not (K + 0.44 <= S <= K + 0.62):
                        continue
                    cv, yv = E.band_best(bV, th, S, 16)
                    if cv < 0:
                        continue
                    tot_n = c_up * K + cv
                    D = tot_n - K * K
                    if 1 <= D <= 5 and K * K + D <= 1000:
                        if D not in best_corr or S < best_corr[D][0]:
                            best_corr[D] = (S, bV, th, yv, c_up)
        for D, (S, bV, th, yv, c_up) in sorted(best_corr.items()):
            n = K * K + D
            L_ref = cur_L_map[n]
            if S > L_ref + 0.025:
                continue
            try:
                cL = c_up // 2
                cR = c_up - cL
                W = bV * math.cos(th) + math.sin(th)
                sqs = []
                for i in range(cL):
                    for j in range(K):
                        sqs.append((i + 0.5, j + 0.5, 0.0))
                for X, Y, t in E.band_squares(bV, th, S, yv):
                    sqs.append((cL + X, Y, t))
                for i in range(cR):
                    for j in range(K):
                        sqs.append((cL + W + i + 0.5, j + 0.5, 0.0))
                arr = np.array(sqs[:n], dtype=np.float64)
                if len(arr) == n and E.min_gap(arr) >= -1e-9:
                    res = certify_candidate(
                        n,
                        arr,
                        L_ref,
                        hopper_sec=110,
                        do_rp=True,
                        seed=seed + n * 73 + bV,
                    )
                    if res is not None:
                        found.append((n, f"corr_b{bV}_d{D}", res))
            except Exception:
                continue

    return K, found


def lns_perturb(P, L, rng):
    """Structured LNS perturbations: band angle wave, tilted-axis reseed, pin-rotate, cluster rot."""
    n = len(P)
    Q = P.copy()
    x, y, t = Q[:, 0], Q[:, 1], Q[:, 2]
    tm = np.abs(E.fold(t))
    tilted = np.where(tm > 0.02)[0]
    kind = int(rng.choice([0, 1, 2, 3, 4]))

    if kind == 0 and len(tilted) >= 6:
        # Band-wide smooth / ramp / alternating angle wave (lns.py kind 7!)
        pts = np.stack([x[tilted], y[tilted]], axis=1)
        P0 = pts - pts.mean(0)
        _, V = np.linalg.eigh(np.cov(P0.T))
        u = P0 @ V[:, 1]
        amp = math.radians(rng.uniform(0.8, 6.5))
        mode = int(rng.integers(4))
        if mode == 0:
            per = rng.uniform(2.5, 18.0)
            ph = rng.uniform(0, 2 * math.pi)
            d = amp * np.sin(2 * math.pi * u / per + ph)
        elif mode == 1:
            d = amp * (u - u.mean()) / max(np.ptp(u), 1e-9) * 2.0
        elif mode == 2:
            v = P0 @ V[:, 0]
            d = amp * np.sign(v - np.median(v))
        else:
            d = rng.normal(0, amp, len(tilted))
        t[tilted] += d
        x[tilted] += rng.normal(0, 0.025, len(tilted))
        y[tilted] += rng.normal(0, 0.025, len(tilted))
        return Q, kind, 0.008, None

    if kind == 1 and len(tilted) >= 4:
        # Band strip reseed along tilted principal direction at new angle (lns.py kind 6!)
        c = int(rng.choice(tilted))
        pts = np.stack([x[tilted], y[tilted]], axis=1)
        d0 = np.hypot(pts[:, 0] - x[c], pts[:, 1] - y[c])
        loc = pts[d0 < 4.5]
        if len(loc) >= 3:
            _, V = np.linalg.eigh(np.cov((loc - loc.mean(0)).T))
            u = V[:, 1]
        else:
            u = np.array([1.0, 1.0]) / math.sqrt(2.0)
        vp = np.array([-u[1], u[0]])
        hl = rng.uniform(1.8, 5.2)
        hw = rng.uniform(1.0, 2.5)
        rel = np.stack([x - x[c], y - y[c]], axis=1)
        sel = np.where((np.abs(rel @ u) < hl) & (np.abs(rel @ vp) < hw))[0]
        if len(sel) >= 3:
            theta = (
                float(np.median(t[tilted[d0 < 4.5]]))
                if np.any(d0 < 4.5)
                else float(t[c])
            )
            theta += rng.normal(0, 0.18)
            e1 = np.array([math.cos(theta), math.sin(theta)])
            e2 = np.array([-math.sin(theta), math.cos(theta)])
            gx, gy = np.meshgrid(np.arange(-8, 9), np.arange(-8, 9))
            off = rng.uniform(0, 1, 2)
            lpts = (gx.reshape(-1, 1) + off[0]) * e1 + (gy.reshape(-1, 1) + off[1]) * e2
            mx, my = x[sel].mean(), y[sel].mean()
            pick = lpts[np.argsort(np.hypot(lpts[:, 0], lpts[:, 1]))[: len(sel)]]
            x[sel] = np.clip(mx + pick[:, 0], 0.5, L - 0.5)
            y[sel] = np.clip(my + pick[:, 1], 0.5, L - 0.5)
            t[sel] = theta + rng.normal(0, 0.015, len(sel))
            return Q, kind, 0.025, None

    if kind == 2:
        # Pin-rotate 1..2 squares near band by 8..32 deg!
        pool = tilted if len(tilted) > 0 and rng.random() < 0.8 else np.arange(n)
        i = int(rng.choice(pool))
        d2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
        cl = np.argsort(d2)[: int(rng.integers(2, 6))]
        for j in cl[:2]:
            t[j] += float(rng.choice([-1.0, 1.0])) * math.radians(
                rng.uniform(8.0, 32.0)
            )
        x[cl] += rng.normal(0, 0.04, len(cl))
        y[cl] += rng.normal(0, 0.04, len(cl))
        return Q, kind, 0.025, None

    if kind == 3:
        # Rigid rotation of local 4..14 square cluster around centroid
        pool = tilted if len(tilted) > 0 and rng.random() < 0.8 else np.arange(n)
        i = int(rng.choice(pool))
        d2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
        cl = np.argsort(d2)[: int(rng.integers(4, 15))]
        a = rng.normal(0, 0.14)
        cx, cy = x[cl].mean(), y[cl].mean()
        dx, dy = x[cl] - cx, y[cl] - cy
        x[cl] = np.clip(cx + dx * math.cos(a) - dy * math.sin(a), 0.5, L - 0.5)
        y[cl] = np.clip(cy + dx * math.sin(a) + dy * math.cos(a), 0.5, L - 0.5)
        t[cl] += a
        return Q, kind, 0.015, None

    # kind 4: local orientation flip (tilted <-> upright) + jiggle
    pool = tilted if len(tilted) > 0 and rng.random() < 0.8 else np.arange(n)
    i = int(rng.choice(pool))
    d2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
    cl = np.argsort(d2)[: int(rng.integers(4, 12))]
    j = int(cl[rng.integers(1, len(cl))])
    t[i] = t[j] + rng.normal(0, 0.08)
    x[cl] = np.clip(x[cl] + rng.normal(0, 0.05, len(cl)), 0.5, L - 0.5)
    y[cl] = np.clip(y[cl] + rng.normal(0, 0.05, len(cl)), 0.5, L - 0.5)
    return Q, kind, 0.015, None


def lns_worker(job):
    """Part 3: Structured LNS + Numba L-BFGS relax + Compactor + Hopper on Top-80 frontier gaps."""
    n, L0, arr0, seed, sec_budget = job
    rng = np.random.default_rng(seed)
    best_P = E.to_origin(arr0)
    best_S = float(E.side(best_P))
    best_tag = ""
    t_end = time.time() + sec_budget
    free_all = np.ones(n, dtype=np.bool_)

    while time.time() < t_end:
        Q, kind, grow, _ = lns_perturb(best_P, best_S, rng)
        Sg = best_S * (1.0 + grow)
        try:
            R_rel, E_val = E.relax(Q, free_all, Sg, 350, 1e-18)
            if E_val > 1e-14:
                continue
            Qc, Sc, ov = E.Compactor(R_rel).compact(R_rel, max_iter=120, patience=3)
            if ov < 2e-9 and Sc < best_S - 1e-8:
                best_P = E.to_origin(Qc)
                best_S = float(Sc)
                best_tag = f"lns_k{kind}"
                # Follow up immediately with short Hopper on new LNS basin!
                H = E.Hopper(best_P, seed=int(rng.integers(1 << 30)))
                if H.applicable():
                    Qh, Sh = H.run(45)
                    Qh = E.to_origin(E.open_contacts(Qh))
                    if E.side(Qh) < best_S and E.min_gap(Qh) >= -1e-12:
                        best_P = Qh
                        best_S = float(E.side(Qh))
        except Exception:
            continue

    if best_S < L0 - 1e-9:
        res = certify_candidate(
            n, best_P, L0, hopper_sec=60, do_rp=False, seed=seed + 999
        )
        if res is not None:
            return n, best_tag, res
    return n, best_tag, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2280)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    warm_numba()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    front_rows = json.load(open(args.frontier_json))
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}
    cur_L_map = {n: cur[n][0] for n in range(1, 1001)}

    # Part 1+2: K in 14..31 sharded across 8 shards
    ks = list(range(14, 32))
    my_ks = ks[args.shard :: args.num_shards]

    # Part 3: Top-80 biggest frontier gaps sharded across 8 shards (10 puzzles/shard)
    scored = []
    for n in range(150, 1000):
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        gap_ds = max(0.0, (cur[n][0] - front_L.get(n, cur[n][0])) / math.sqrt(n))
        if gap_ds > 1e-7:
            scored.append((gap_ds, n))
    scored.sort(reverse=True)
    my_lns = [n for _, n in scored[:80]][args.shard :: args.num_shards]

    print(
        f"[W28C-DeNovoFramesSub050LNS s={args.shard}/{args.num_shards}] Base={tot_score:.10f}, "
        f"my_Ks={my_ks}, my_LNS({len(my_lns)})={my_lns} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0

    def save_ckpt():
        payload = {}
        for k_n, (Lf, xf, yf, tf) in sorted(wins.items()):
            payload[f"L_{k_n}"] = np.array([Lf], dtype=np.float64)
            payload[f"x_{k_n}"] = np.asarray(xf, dtype=np.float64)
            payload[f"y_{k_n}"] = np.asarray(yf, dtype=np.float64)
            payload[f"t_{k_n}"] = np.asarray(tf, dtype=np.float64)
        np.savez_compressed(args.out_npz, **payload)

    def record_win(n, tag, res):
        nonlocal tot_ds
        Lf, xf, yf, tf = res
        ref = wins[n][0] if n in wins else cur[n][0]
        if Lf < ref - 1e-9:
            dS = (Lf - ref) / math.sqrt(n)
            tot_ds += dS
            wins[n] = res
            print(
                f"  [W28C WIN!!] n={n:4d} {tag:18s} {ref:.7f} -> {Lf:.7f} "
                f"dL={Lf - ref:.7f} dS={dS:.8f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
            save_ckpt()

    # Run Part 1+2 (De-Novo Frames + Sub-0.50 Wide Corridors)
    f_jobs = [
        (K, cur_L_map, front_L, 2830001 + args.shard * 10007 + K * 97) for K in my_ks
    ]
    with ProcessPoolExecutor(max_workers=min(args.procs, max(1, len(f_jobs)))) as ex:
        futs = [ex.submit(frame_worker, j) for j in f_jobs]
        for fut in as_completed(futs):
            try:
                K, found = fut.result()
                for n_new, tag, res in found:
                    record_win(n_new, tag, res)
            except Exception as exc:
                print(f"  [W28C FRAME ERR] {exc}", flush=True)
            if time.time() - t0 > args.budget_sec * 0.55:
                for f_rem in futs:
                    f_rem.cancel()
                break

    # Run Part 3 (Structured LNS on Top-80 frontier gaps)
    rem_sec = args.budget_sec - (time.time() - t0)
    if rem_sec > 90 and my_lns:
        per_sec = max(
            80.0, min(280.0, (rem_sec * args.procs) / max(1, len(my_lns)) * 0.85)
        )
        l_jobs = [
            (
                n,
                wins[n][0] if n in wins else cur[n][0],
                np.column_stack([wins[n][1], wins[n][2], wins[n][3]])
                if n in wins
                else cur[n][1],
                2835001 + args.shard * 10039 + i * 257,
                per_sec,
            )
            for i, n in enumerate(my_lns)
        ]
        with ProcessPoolExecutor(max_workers=args.procs) as ex:
            futs = [ex.submit(lns_worker, j) for j in l_jobs]
            for fut in as_completed(futs):
                try:
                    n, tag, res = fut.result()
                    if res is not None:
                        record_win(n, tag, res)
                except Exception as exc:
                    print(f"  [W28C LNS ERR] {exc}", flush=True)
                if time.time() - t0 > args.budget_sec:
                    print(
                        f"[W28C s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                        flush=True,
                    )
                    for f_rem in futs:
                        f_rem.cancel()
                    break

    save_ckpt()
    print(
        f"[W28C s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
