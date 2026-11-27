#!/usr/bin/env python3
"""Wave 44H: Bi-Directional Neighbor-Donor Topology Transporter + Wider-Move Hopper (max_cl=18)!

Directly solves the #1 failure mode uncovered by forensic anatomy of Top-45 remaining gaps
(n=234, 367, 491, 333, 738, 456, 804, 730, 261, 904, 748, 962, 414, 332, ...):
many top-gap puzzles are trapped in a suboptimal topology while their immediate neighbor
(n-1, n+1, or n+2) ALREADY holds the #1 world-record topology!
  - Operator 1 (n-1 -> n Soft Void-Expansion Insertion):
    Finds top-6 largest open voids in donor P_{n-1} (center-distance 0.80..0.89!), gently dilates
    the box to S_try = max(L_cur + delta, L_{n-1} + 0.05), inserts 1 square at optimal angles
    (0, +-med_tilt, pi/4), runs full-packing Numba L-BFGS E.relax(800) to smoothly open a zero-overlap
    pocket, then E.Compactor().compact() -> Wider-Move Hopper (max_cl=18) + SLP + fq + Decimal64!
  - Operator 2 (n+1 -> n & n+2 -> n Dual-Force Skeleton + Near-Band Multi-Deletion):
    Uses E.skeleton(P) dual load forces + near-band proximity to delete 1 square from n+1 or
    2 squares sequentially from n+2, compacts, and runs Wider-Move Hopper (max_cl=18) even when
    pre-Hopper S is up to +0.025 above L_cur(n) (because fresh cross-topology donors are SOFT!).
  - Operator 3 (from_scratch tilted-start relaxation for small/medium top gaps n <= 280):
    Generates fresh relaxed starts on hard medium puzzles like n=234, 261, 209, 240, 273.
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


def _slp_polish(P):
    cur_a = w32a.safe_open(P)
    cur_s = float(E.side(cur_a))
    try:
        xs, ys, ts, Ls = SLP.polish(
            cur_a[:, 0],
            cur_a[:, 1],
            cur_a[:, 2],
            iters=100,
            r0=2.5e-3,
            rmin=1e-10,
            tol_sep=-1e-13,
            time_limit=18.0,
        )
        Qs = w32a.safe_open(np.column_stack([xs, ys, ts]))
        Ss = float(E.side(Qs))
        if Ss < cur_s - 1e-10 and E.min_gap(Qs) >= -1e-12:
            return Qs, Ss
    except Exception:
        pass
    return cur_a, cur_s


def _find_top_voids(P, S_box, n_top=6):
    P0 = E.to_origin(P)
    ng = max(36, min(140, int(S_box * 4)))
    xs = np.linspace(0.55, max(0.60, S_box - 0.55), ng)
    ys = np.linspace(0.55, max(0.60, S_box - 0.55), ng)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    dx = pts[:, 0:1] - P0[:, 0:1].T
    dy = pts[:, 1:2] - P0[:, 1:2].T
    dmin = np.sqrt(np.min(dx * dx + dy * dy, axis=1))
    order = np.argsort(-dmin)
    picked = []
    for idx in order:
        cx, cy = float(pts[idx, 0]), float(pts[idx, 1])
        if all((cx - px) ** 2 + (cy - py) ** 2 > 1.4**2 for _, px, py in picked):
            picked.append((float(dmin[idx]), cx, cy))
            if len(picked) >= n_top:
                break
    return picked


def _skeleton_and_band_indices(P, max_idx=36):
    """Prioritize highest dual-force load squares from E.skeleton(P), followed by near-band squares."""
    n = len(P)
    ranked = []
    try:
        _, forces, _ = E.skeleton(P)
        loads = np.zeros(n, dtype=np.float64)
        for i, j, lam in forces:
            if 0 <= i < n:
                loads[i] += lam
            if 0 <= j < n:
                loads[j] += lam
        top_load = [int(i) for i in np.argsort(-loads) if loads[i] > 1e-9][:18]
        ranked.extend(top_load)
    except Exception:
        pass
    try:
        nb = w32a.pick_near_band(P, max_cand=max_idx)
        for idx in nb:
            if int(idx) not in ranked:
                ranked.append(int(idx))
    except Exception:
        pass
    return ranked[:max_idx]


def generate_insert_donors(P_m1, L_m1, L_cur, deadline, max_donors=2):
    """Operator 1: n-1 -> n soft void-expansion insertion + E.relax L-BFGS + Compactor."""
    donors = []
    if P_m1 is None or time.time() > deadline - 40.0:
        return donors
    P0 = E.to_origin(P_m1)
    ft = np.abs(E.fold(P0[:, 2]))
    tilted_angles = ft[ft >= 0.03]
    med_t = float(np.median(tilted_angles)) if len(tilted_angles) else 0.45
    cand_thetas = [0.0, med_t, -med_t, math.pi / 4.0]
    voids = _find_top_voids(P0, L_m1, n_top=6)
    all_free = np.ones(len(P0) + 1, dtype=np.bool_)

    # Try tightening/expanding box scales so E.relax smoothly opens a zero-overlap pocket
    scales = [
        max(L_cur + 0.015, L_m1 + 0.045),
        max(L_cur + 0.045, L_m1 + 0.085),
        max(L_cur + 0.095, L_m1 + 0.140),
    ]
    t_start = time.time()
    for vi, (vd, vx, vy) in enumerate(voids):
        if time.time() - t_start > 55.0 or time.time() > deadline - 35.0:
            break
        got_for_void = False
        for S_try in scales:
            if time.time() - t_start > 55.0 or time.time() > deadline - 35.0:
                break
            fac = S_try / max(L_m1, 1e-9)
            P_scaled = P0.copy()
            P_scaled[:, 0] *= fac
            P_scaled[:, 1] *= fac
            for ti, th in enumerate(cand_thetas):
                if time.time() - t_start > 55.0:
                    break
                P_aug = np.vstack(
                    [P_scaled, np.array([[vx * fac, vy * fac, th]], dtype=np.float64)]
                )
                try:
                    Qr, en = E.relax(P_aug, all_free, float(S_try), 700, 1e-22)
                    if en < 1e-18:
                        Qc, Sc, ov = E.Compactor(Qr).compact()
                        if ov <= 2e-9 and Sc < L_cur + 0.030:
                            donors.append(
                                (
                                    float(Sc),
                                    f"ins_m1_v{vi}_t{ti}",
                                    Qc.astype(np.float64),
                                )
                            )
                            got_for_void = True
                            break
                except Exception:
                    continue
            if got_for_void:
                break
    donors.sort(key=lambda z: z[0])
    picked = []
    for Sc, tag, Qc in donors:
        if all(abs(Sc - ps) > 1e-5 for ps, _, _ in picked):
            picked.append((Sc, tag, Qc))
        if len(picked) >= max_donors:
            break
    return picked


def generate_delete1_donors(P_p1, L_p1, L_cur, deadline, max_donors=2):
    """Operator 2A: n+1 -> n dual-force skeleton + near-band deletion -> Compactor."""
    donors = []
    if P_p1 is None or (L_p1 - L_cur) > 0.065 or time.time() > deadline - 40.0:
        return donors
    cands = _skeleton_and_band_indices(P_p1, max_idx=44)
    t_start = time.time()
    for idx in cands:
        if time.time() - t_start > 45.0 or time.time() > deadline - 35.0:
            break
        try:
            Qd, Sd = w32a.fast_delete_and_compact(P_p1, idx)
            if Qd is not None and Sd < L_cur + 0.025:
                donors.append((float(Sd), f"del1_p1_{idx}", Qd.astype(np.float64)))
        except Exception:
            continue
    donors.sort(key=lambda z: z[0])
    picked = []
    for Sd, tag, Qd in donors:
        if all(abs(Sd - ps) > 1e-5 for ps, _, _ in picked):
            picked.append((Sd, tag, Qd))
        if len(picked) >= max_donors:
            break
    return picked


def generate_delete2_donors(P_p2, L_p2, L_cur, deadline, max_donors=2):
    """Operator 2B: n+2 -> n sequential 2-square dual-force deletion -> Compactor."""
    donors = []
    if P_p2 is None or (L_p2 - L_cur) > 0.095 or time.time() > deadline - 40.0:
        return donors
    cands1 = _skeleton_and_band_indices(P_p2, max_idx=18)
    step1 = []
    t_start = time.time()
    for idx1 in cands1:
        if time.time() - t_start > 25.0 or time.time() > deadline - 35.0:
            break
        try:
            Q1, S1 = w32a.fast_delete_and_compact(P_p2, idx1)
            if Q1 is not None:
                step1.append((float(S1), idx1, Q1))
        except Exception:
            continue
    step1.sort(key=lambda z: z[0])
    for S1, idx1, Q1 in step1[:4]:
        if time.time() - t_start > 50.0 or time.time() > deadline - 35.0:
            break
        cands2 = _skeleton_and_band_indices(Q1, max_idx=12)
        for idx2 in cands2:
            if time.time() - t_start > 50.0 or time.time() > deadline - 35.0:
                break
            try:
                Q2, S2 = w32a.fast_delete_and_compact(Q1, idx2)
                if Q2 is not None and S2 < L_cur + 0.025:
                    donors.append(
                        (float(S2), f"del2_p2_{idx1}_{idx2}", Q2.astype(np.float64))
                    )
            except Exception:
                continue
    donors.sort(key=lambda z: z[0])
    picked = []
    for Sd, tag, Qd in donors:
        if all(abs(Sd - ps) > 1e-5 for ps, _, _ in picked):
            picked.append((Sd, tag, Qd))
        if len(picked) >= max_donors:
            break
    return picked


def worker_neighbor_donor(job):
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
        seed,
        hop_sec,
        n_rounds,
        deadline,
    ) = job

    best_S = L0
    best_P = None
    best_tag = ""

    # Gather diverse soft cross-topology donors from n-1, n+1, n+2!
    donors = []
    d_m1 = generate_insert_donors(arr_m1, L_m1, L0, deadline, max_donors=2)
    d_p1 = generate_delete1_donors(arr_p1, L_p1, L0, deadline, max_donors=2)
    d_p2 = generate_delete2_donors(arr_p2, L_p2, L0, deadline, max_donors=1)

    # Interleave so every available donor direction gets Hopper time!
    for lst in (d_m1, d_p1, d_p2):
        if lst:
            donors.append(lst[0])
    for lst in (d_m1, d_p1, d_p2):
        if len(lst) > 1:
            donors.append(lst[1])

    # Check if any raw donor already beats L0 directly!
    for Sd, tag, Qd in donors:
        Qs = w32a.safe_open(Qd)
        Ss = float(E.side(Qs))
        if Ss < best_S - 1e-9 and E.min_gap(Qs) >= -1e-12:
            best_S, best_P, best_tag = Ss, Qs, f"{tag}_raw"

    # Run Wider-Move Hopper (max_cl=18, wide=True) + SLP on every soft donor!
    for di, (Sd, tag, Qd) in enumerate(donors):
        rem = deadline - time.time()
        if rem < 25.0:
            break
        cur_d = w32a.safe_open(Qd)
        for r in range(n_rounds):
            rem = deadline - time.time()
            if rem < 18.0:
                break
            sec = min(hop_sec, rem - 10.0)
            try:
                H = E.Hopper(cur_d, seed=seed + di * 13007 + r * 3571, wide=True)
                if not H.applicable():
                    break
                Q2, S2 = H.run(sec)
                Q2, S2 = _slp_polish(Q2)
                if S2 < best_S - 1e-9 and E.min_gap(Q2) >= -1e-12:
                    best_S, best_P, best_tag = S2, Q2, f"{tag}_wh_r{r}"
                    cur_d = Q2
                elif S2 < float(E.side(cur_d)) - 1e-9:
                    cur_d = Q2
                # Early-stop hopeless donor if still far above best_S after round 0
                if r == 0 and float(E.side(cur_d)) > best_S + 0.015:
                    break
            except Exception:
                break

    # Final SLP + Rust fq + lp_rot + exact Decimal64 >= 1.05e-9 certification!
    if best_P is not None and best_S < L0 - 1e-9:
        cur_a, cur_s = _slp_polish(best_P)
        if os.path.exists(f2d.FQ_BIN):
            with tempfile.TemporaryDirectory(prefix=f"w44h_{n}_") as tmp:
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
            return n, best_tag, len(donors), (Lf, xf, yf, tf)
    return n, best_tag, len(donors), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2550)
    ap.add_argument("--hop-sec", type=int, default=160)
    ap.add_argument("--n-rounds", type=int, default=2)
    ap.add_argument("--seed-off", type=int, default=440000)
    ap.add_argument("--max-targets", type=int, default=96)
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

    # Rank ALL non-grid puzzles by live gap (L_cur - L_front)/sqrt(n)
    gap_ranked = []
    for n in range(100, 1000):
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

    def _valid_donor(k):
        if not (80 <= k <= 1000):
            return None, 1e9
        kk = math.isqrt(k)
        if kk * kk == k or cur[k][0] >= kk + 1 - 1e-6:
            return None, 1e9
        return cur[k][1], cur[k][0]

    jobs = []
    for i, n in enumerate(my_slice):
        am1, Lm1 = _valid_donor(n - 1)
        ap1, Lp1 = _valid_donor(n + 1)
        ap2, Lp2 = _valid_donor(n + 2)
        jobs.append(
            (
                n,
                cur[n][0],
                cur[n][1],
                am1,
                Lm1,
                ap1,
                Lp1,
                ap2,
                Lp2,
                args.seed_off + args.shard * 10301 + i * 613,
                args.hop_sec,
                args.n_rounds,
                deadline,
            )
        )

    print(
        f"[W44H-NeighborDonor+WideHop s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_targets({len(my_slice)})={my_slice} hop={args.hop_sec}sx{args.n_rounds} ({time.time() - t0:.1f}s)",
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
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_neighbor_donor, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, n_don, res = fut.result()
            except Exception as exc:
                print(f"  [W44H ERR] {exc}", flush=True)
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
                        f"  [W44H WIN!!] n={n:4d} [{tag}] donors={n_don} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"(dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
            else:
                print(
                    f"  [W44H done] n={n:4d} donors={n_don} no_win [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                    flush=True,
                )

    save_ckpt()
    print(
        f"[W44H s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
