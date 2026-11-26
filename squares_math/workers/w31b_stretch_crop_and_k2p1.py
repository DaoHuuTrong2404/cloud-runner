#!/usr/bin/env python3
"""Wave 31B: Cross-Scale Arm Stretcher + Sliding Window Crop + k^2+1 Sub-0.50 Grower + Grid-Break!

Completely orthogonal to same-n HopperV2: transports superior tilted-band & tilted-parallelogram
architectures ACROSS K -> K +- 1, +- 2 (n -> n +- (2K + 1)):
  1. Generalized Arm Stretcher (stretch.py + pure-upright-axis fallback for 1-arm layouts like
     n=842/626/442 -> directly grows n=842's 21.9-deg tilted parallelogram to full-height k=30 (901),
     k=31 (962) AND shrinks to k=28 (785), k=27 (730) to break f < 0.50!).
  2. Sliding Window Crop from donor m ~ n + 2K + 1 (e.g. 455->414, 499->456, 265->234, 405->367,
     818->762, 876->819) + exact count adjustment (+-1..3) + Numba L-BFGS squeeze2 -> HopperV2 -> SLP!
  3. A_GRID_BREAK Attack (605, 762, 819, 877, 1000): Voronoi near-band 1-square insertion into n-1
     + stretch donors -> squeeze2 -> HopperV2 -> SLP to crack pure integer grids L < K!
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
import stretch as ST
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


def upright_axis_cut(a, axis, op):
    """Fallback cut along an axis without a tilted arm: insert/delete 1 pure upright col/row."""
    a = E.to_origin(np.asarray(a, dtype=np.float64))
    L = float(E.side(a))
    deg = np.degrees(np.mod(a[:, 2], math.pi / 2))
    tilted = np.minimum(deg, 90.0 - deg) > 2.5
    tx = a[tilted, axis]
    # Search candidate integer-ish lines away from tilted squares
    best_cut = None
    best_cnt = -1
    for c_i in range(2, max(3, int(L) - 1)):
        for off in (0.0, 0.5):
            cut = float(c_i) + off
            if cut < 1.2 or cut > L - 1.2:
                continue
            if len(tx) > 0 and np.any((tx > cut - 1.15) & (tx < cut + 0.65)):
                continue
            in_strip = (~tilted) & (a[:, axis] > cut - 1.0) & (a[:, axis] <= cut)
            cnt = int(in_strip.sum())
            if cnt >= int(L) - 3 and cnt > best_cnt:
                best_cnt = cnt
                best_cut = cut
    if best_cut is None:
        return None
    cut = best_cut
    beyond = a[:, axis] > cut
    in_strip = (~tilted) & (a[:, axis] > cut - 1.0) & (a[:, axis] <= cut)
    b = a.copy()
    if op > 0:
        b[beyond, axis] += 1.0
        cp = a[in_strip].copy()
        cp[:, axis] += 1.0
        b = np.vstack([b, cp])
    else:
        b[beyond, axis] -= 1.0
        b = b[~in_strip]
    return E.to_origin(b)


def generalized_stretch(a, steps, cut_shift=0):
    """Stretch packing by K -> K + steps (+-1, +-2), supporting both 2-arm and 1-arm layouts!"""
    b = E.to_origin(np.asarray(a, dtype=np.float64))
    op = 1 if steps > 0 else -1
    for _ in range(abs(steps)):
        for axis in (0, 1):
            c = ST.pick_cuts(b, axis)
            if c is not None:
                c_try = c + float(cut_shift)
                b2, _ = ST.apply(b, axis, c_try, op)
                if b2 is None:
                    b2, _ = ST.apply(b, axis, c, op)
                if b2 is None:
                    b2 = upright_axis_cut(b, axis, op)
            else:
                b2 = upright_axis_cut(b, axis, op)
            if b2 is None:
                return None
            b = E.to_origin(b2)
    return b


def find_slack_pockets(P, S, k_pockets=24):
    """Find the k_pockets largest empty points inside [0.5, S-0.5]^2."""
    xs = np.linspace(0.55, max(0.6, S - 0.55), min(90, max(20, int(S * 3))))
    ys = np.linspace(0.55, max(0.6, S - 0.55), min(90, max(20, int(S * 3))))
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    # Distance to nearest square center (chunked to keep RAM tiny)
    centers = P[:, :2]
    dmin = np.full(len(pts), 1e9, dtype=np.float64)
    for i0 in range(0, len(pts), 512):
        sub = pts[i0 : i0 + 512]
        d2 = ((sub[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        dmin[i0 : i0 + 512] = np.sqrt(d2.min(axis=1))
    order = np.argsort(-dmin)
    picked = []
    for idx in order:
        px, py = float(pts[idx, 0]), float(pts[idx, 1])
        if all((px - qx) ** 2 + (py - qy) ** 2 > 0.64 for qx, qy, _ in picked):
            # Inherit angle of nearest square
            d2_c = (centers[:, 0] - px) ** 2 + (centers[:, 1] - py) ** 2
            ang = float(P[int(np.argmin(d2_c)), 2])
            picked.append((px, py, ang))
            if len(picked) >= k_pockets:
                break
    return picked


def adjust_count_to_n(P, target_n, S_hint):
    """Trim or insert up to +-4 squares to match target_n cleanly."""
    P = E.to_origin(np.asarray(P, dtype=np.float64))
    diff = len(P) - target_n
    if abs(diff) > 4:
        return None
    if diff == 0:
        return P
    if diff > 0:
        # Remove diff squares sticking out farthest / most crowded on boundary
        h = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
        protrude = (
            np.maximum(0.0, (P[:, 0] + h) - S_hint)
            + np.maximum(0.0, (P[:, 1] + h) - S_hint)
            + np.maximum(0.0, h - P[:, 0])
            + np.maximum(0.0, h - P[:, 1])
        )
        # Also penalize overlap with neighbors via center distance < 0.99
        d2 = ((P[:, None, :2] - P[None, :, :2]) ** 2).sum(axis=2)
        np.fill_diagonal(d2, 99.0)
        crowd = np.maximum(0.0, 1.0 - np.sqrt(d2.min(axis=1)))
        score = protrude * 4.0 + crowd * 2.0 + 1e-3 * (P[:, 0] + P[:, 1])
        drop = np.argsort(-score)[:diff]
        return E.to_origin(np.delete(P, drop, axis=0))
    else:
        need = -diff
        pockets = find_slack_pockets(P, S_hint, k_pockets=max(need + 4, 8))
        if len(pockets) < need:
            return None
        add = np.array(pockets[:need], dtype=np.float64)
        return E.to_origin(np.vstack([P, add]))


def relax_and_squeeze(P, S_target_hi, max_iter=12000):
    """Relax a stretched/cropped/inserted start in Numba L-BFGS squeeze2 and finalize."""
    P = E.to_origin(np.asarray(P, dtype=np.float64))
    x, y, t = P[:, 0].copy(), P[:, 1].copy(), P[:, 2].copy()
    L_init = SQ2.side(x, y, t)
    # Clip gently inside S_init = max(L_init, S_target_hi) * 1.005
    S0 = max(L_init, S_target_hi) * 1.006
    x = np.clip(x, 0.5, S0 - 0.5)
    y = np.clip(y, 0.5, S0 - 0.5)
    # Try relaxing at S0, then squeeze down!
    S_out, x2, y2, t2 = SQ2.squeeze(
        x, y, t, rel0=4e-3, rel_min=5e-9, max_iter=max_iter, max_tries=55
    )
    if not math.isfinite(S_out):
        return None, 1e9
    Lf, xf, yf, tf, good = SQ2.finalize(x2, y2, t2, tol=2e-11)
    if not good:
        return None, 1e9
    Q = E.to_origin(np.column_stack([xf, yf, tf]))
    return Q, float(E.side(Q))


def worker_cross_scale(job):
    n, L0, arr0, donors, is_grid_break, seed, hop_sec, slp_sec = job
    best_S = L0
    best_P = None
    best_tag = ""
    candidates = []  # (S_est, tag, arr_n)

    # 1. Generalized Stretched Donors (K -> K +- 1, +- 2)
    for d_tag, d_arr, step in donors:
        if d_tag.startswith("stretch"):
            for cshift in (0, 1, -1):
                try:
                    b = generalized_stretch(d_arr, step, cut_shift=cshift)
                    if b is None:
                        continue
                    b_adj = adjust_count_to_n(b, n, L0)
                    if b_adj is not None:
                        candidates.append(
                            (float(E.side(b_adj)), f"{d_tag}_s{cshift:+d}", b_adj)
                        )
                except Exception:
                    continue

        elif d_tag.startswith("crop"):
            # Sliding window crop of (L_big - 1.0) x (L_big - 1.0)
            P_big = E.to_origin(d_arr)
            L_big = float(E.side(P_big))
            Sw = L_big - 1.0
            if Sw < 2.0:
                continue
            offsets = [
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (0.5, 0.5),
                (0.5, 0.0),
                (0.0, 0.5),
                (1.0, 0.5),
                (0.5, 1.0),
            ]
            for oi, (ox, oy) in enumerate(offsets):
                keep = (
                    (P_big[:, 0] >= ox + 0.12)
                    & (P_big[:, 0] <= ox + Sw - 0.12)
                    & (P_big[:, 1] >= oy + 0.12)
                    & (P_big[:, 1] <= oy + Sw - 0.12)
                )
                sub = P_big[keep].copy()
                if abs(len(sub) - n) <= 3:
                    sub[:, 0] -= ox
                    sub[:, 1] -= oy
                    b_adj = adjust_count_to_n(sub, n, Sw)
                    if b_adj is not None:
                        candidates.append((Sw, f"{d_tag}_w{oi}", b_adj))

        elif d_tag.startswith("ins_prev"):
            # Insert 1 square into n-1 (especially for A_GRID_BREAK: 605, 762, 819, 877, 1000!)
            P_prev = E.to_origin(d_arr)
            L_prev = float(E.side(P_prev))
            pockets = find_slack_pockets(
                P_prev, L_prev, k_pockets=20 if is_grid_break else 10
            )
            for pi, (px, py, ang) in enumerate(pockets):
                for ai, a_use in enumerate(
                    (ang, 0.0) if abs(ang) > 0.05 else (ang, 0.28)
                ):
                    b = np.vstack(
                        [P_prev, np.array([[px, py, a_use]], dtype=np.float64)]
                    )
                    candidates.append((L_prev + 0.005, f"ins_p{pi}_a{ai}", b))

    # Interleave candidates across families (stretch, crop, ins_prev) so every mechanism gets fair shots!
    by_fam = {}
    for item in candidates:
        fam = item[1].split("_")[0]
        by_fam.setdefault(fam, []).append(item)
    for fam in by_fam:
        by_fam[fam].sort(key=lambda z: z[0])
    interleaved = []
    fams = sorted(by_fam.keys())
    idx_f = {f: 0 for f in fams}
    while len(interleaved) < len(candidates):
        prog = False
        for f in fams:
            if idx_f[f] < len(by_fam[f]):
                interleaved.append(by_fam[f][idx_f[f]])
                idx_f[f] += 1
                prog = True
        if not prog:
            break

    t_rel = time.time()
    squeezed = []
    max_try = 24 if is_grid_break else 15
    for ci, (s_est, ctag, c_arr) in enumerate(interleaved[:max_try]):
        if time.time() - t_rel > (180.0 if is_grid_break else 130.0):
            break
        Q, Sq = relax_and_squeeze(c_arr, max(L0, s_est), max_iter=10000)
        if Q is not None:
            # Quick SLP tighten
            Qs, Ss = w31a.run_slp_polish(Q, iters=50, r0=5e-3, time_limit=15.0)
            if Ss < Sq:
                Q, Sq = Qs, Ss
            if Sq < best_S - 1e-8:
                best_S, best_P, best_tag = Sq, Q, f"{ctag}_sq"
            if Sq < best_S + (0.015 if is_grid_break else 0.010):
                squeezed.append((Sq, ctag, Q))

    # Deduplicate squeezed basins and launch HopperV2 + SLP!
    squeezed.sort(key=lambda z: z[0])
    picked = []
    for Sq, ctag, Q in squeezed:
        if all(abs(Sq - ps) > 5e-6 for ps, _, _ in picked):
            picked.append((Sq, ctag, Q))
        if len(picked) >= (4 if is_grid_break else 3):
            break

    for pi, (Sq, ctag, Q) in enumerate(picked):
        Qh, Sh, _ = w30.run_v2_chain(
            Q,
            seed + 41000 + pi * 5113,
            hop_sec,
            min(150, hop_sec),
            unlock_bands=True,
            max_cascades=3,
        )
        if Sh < best_S - 1e-8:
            best_S, best_P, best_tag = Sh, Qh, f"{ctag}_hv2"
            Qs2, Ss2 = w31a.run_slp_polish(
                best_P, iters=120, r0=3e-3, time_limit=slp_sec
            )
            if Ss2 < best_S - 1e-9:
                best_S, best_P, best_tag = Ss2, Qs2, f"{ctag}_hv2_slp"
            # Extra HopperV2 cascade on breakthrough!
            Qh2, Sh2, _ = w30.run_v2_chain(
                best_P,
                seed + 83000 + pi * 997,
                hop_sec,
                120,
                unlock_bands=False,
                max_cascades=2,
            )
            if Sh2 < best_S - 1e-8:
                best_S, best_P, best_tag = Sh2, Qh2, f"{ctag}_hv2x2"

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
    ap.add_argument("--hop-sec", type=int, default=140)
    ap.add_argument("--slp-sec", type=float, default=30.0)
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

    # Priority target list:
    #   1) ALL 5 A_GRID_BREAK puzzles (605, 762, 819, 877, 1000)
    #   2) ALL k^2 + 1..3 puzzles with gap > 0 (730, 785, 842, 901, 962, 530, 577, 401, ...)
    #   3) Top remaining frontier gaps (234, 367, 491, 333, 738, 456, 804, 261, 904, 748, 414, ...)
    grid_breaks = {605, 762, 819, 877, 1000}
    k2p1_vip = {401, 530, 577, 677, 730, 785, 842, 901, 962, 731, 786, 843, 902, 963}
    scored = []
    for n in range(150, 1001):
        K = math.isqrt(n)
        if K * K == n:
            continue
        is_gb = n in grid_breaks and cur[n][0] >= K + 1 - 1e-6
        if not is_gb and cur[n][0] >= K + 1 - 1e-6:
            continue
        gap_ds = max(0.0, (cur[n][0] - front_L.get(n, cur[n][0])) / math.sqrt(n))
        if gap_ds > 1e-7 or is_gb or n in k2p1_vip:
            boost = 0.01 if is_gb else (0.005 if n in k2p1_vip else 0.0)
            scored.append((gap_ds + boost, n, is_gb))
    scored.sort(reverse=True)
    picked = scored[: args.max_targets]
    my_slice = picked[args.shard :: args.num_shards]

    jobs = []
    for i, (_, n, is_gb) in enumerate(my_slice):
        k_cur = int(math.floor(cur[n][0]))
        donors = []
        # 1) Stretch DOWN from m ~ n + 2*k + 1 (K+1 -> K)
        for dm in (0, -1, 1):
            m = n + 2 * k_cur + 1 + dm
            if m <= 1000:
                km = math.isqrt(m)
                if km * km != m and cur[m][0] < km + 1 - 1e-6:
                    if cur[m][0] - 1.0 < cur[n][0] + 0.015:
                        donors.append((f"stretch_dn_{m}", cur[m][1], -1))
                        donors.append((f"crop_{m}", cur[m][1], -1))
        # 2) Stretch UP from m ~ n - (2*k - 1) (K-1 -> K)
        for dm in (0, -1, 1):
            m = n - (2 * k_cur - 1) + dm
            if m >= 150:
                km = math.isqrt(m)
                if km * km != m and cur[m][0] < km + 1 - 1e-6:
                    if cur[m][0] + 1.0 < cur[n][0] + 0.025:
                        donors.append((f"stretch_up_{m}", cur[m][1], +1))
        # 3) Special 2-step stretch for k^2+1 VIPs (e.g. 842 -> 962 (+2) and 842 -> 730 (-2)!)
        if n in k2p1_vip:
            for m_ref in (842, 626, 785, 442):
                k_ref = int(math.isqrt(m_ref))
                step = int(math.isqrt(n)) - k_ref
                if step in (-2, -1, 1, 2):
                    donors.append(
                        (f"stretch_k2p1_{m_ref}_st{step:+d}", cur[m_ref][1], step)
                    )
        # 4) Insert 1 square from n-1 (critical for A_GRID_BREAK & small-jump gaps like 234<-233, 367<-366!)
        if n - 1 >= 150:
            kp = math.isqrt(n - 1)
            if kp * kp != (n - 1) and cur[n - 1][0] < kp + 1 - 1e-6:
                if is_gb or (cur[n][0] - cur[n - 1][0] > 0.020):
                    donors.append((f"ins_prev_{n - 1}", cur[n - 1][1], 0))

        jobs.append(
            (
                n,
                cur[n][0],
                cur[n][1],
                donors,
                is_gb,
                3130001 + args.shard * 10169 + i * 331,
                args.hop_sec,
                args.slp_sec,
            )
        )

    print(
        f"[W31B-CrossScale s={args.shard}/{args.num_shards}] Base={tot_score:.10f} "
        f"(loaded {n_files} win files), my_targets({len(my_slice)})={[x[1] for x in my_slice]} ({time.time() - t0:.1f}s)",
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
        futs = [ex.submit(worker_cross_scale, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W31B ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                dL = Lf - cur[n][0]
                if dL < -1e-9:
                    dS = dL / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W31B-CrossScale WIN!!] n={n:4d} {tag:22s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W31B s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W31B s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
