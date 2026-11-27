#!/usr/bin/env python3
"""Wave 44F: Relaxed Multi-Cut Full + Half Bar-Transplant + Wider-Move Hopper (max_cl=18)!

Upgrades Wave 42F/43F Bar-Transplant in THREE proven ways:
  1. Multi-Cut Relaxed find_cuts_relaxed (tol_body=0.015, tol_tilt=0.03, min_bar=2, top-3 cut levels):
     recovers Hopper-polished tilted bands & b=2 bands that strict E.find_cut rejected!
  2. Half-Transplant (1 tilted bar + 1 upright edge strip): works on 1-band-only packings AND
     generates brand-new residue classes (n +- (b + row + floor(S))) on 2-band packings!
  3. 1-step & 2-step Full Transplant + Transplant-Delete-1, strictly filtered by min_gap >= -1e-8
     and E.Compactor().compact() (ov <= 2e-9), then hammered with goodpjw2008's Wider-Move Hopper
     (wide=True, max_cl=18) + SLP + Rust fq + lp_rot + exact Decimal64 >= 1.05e-9!
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
import w42f_transplant_wide_hopper as w42f


def find_cuts_relaxed(P, axis, tol_body=0.015, tol_tilt=0.03, min_bar=2, top_k=3):
    P = E.to_origin(P)
    S = E.side(P)
    body, size, cen, tilt = E.band_summary(P, tol_body)
    other = 1 - axis
    upright = np.abs(E.fold(P[:, 2])) < 0.025
    grid_body = np.argmax(np.where(np.abs(tilt) < 0.01, size, 0))
    is_bar = (np.abs(tilt) > 0.025) & (size >= min_bar)
    loose = (size[body] == 1) & ~upright
    cands = []
    for y in np.unique(np.round(P[body == grid_body, axis] + 0.5, 4)):
        if y < 2.2 or y > S - 2.2:
            continue
        strip = np.where(np.abs(P[:, axis] - y) < 1.75)[0]
        tl = strip[~upright[strip]]
        if len(tl) == 0 or not np.all(is_bar[body[tl]]):
            continue
        bars = np.unique(body[tl])
        if len(bars) < 2 or np.ptp(tilt[bars]) > tol_tilt or np.ptp(size[bars]) != 0:
            continue
        one = np.where(body == bars[0])[0]
        if np.ptp(P[one, other]) < np.ptp(P[one, axis]):
            continue
        above = bars[cen[bars, axis] > y]
        if len(above) == 0:
            continue
        dist = float(np.min(np.abs(P[loose, axis] - y))) if loose.any() else 9.0
        th = float(np.mean(tilt[bars]))
        bar_idx = np.where(body == above[np.argmin(cen[above, axis])])[0]
        cands.append((dist, float(y), th, bar_idx, body, size, cen, tilt))
    cands.sort(key=lambda z: -z[0])
    return cands[:top_k]


def _change_len_cut(P, axis, add, cut, tol_tilt=0.03, min_bar=2, gap=2e-4):
    P = E.to_origin(P)
    _, y, th, bar, body, size, cen, tilt = cut
    in_band = (
        (np.abs(tilt) > 0.025) & (size >= min_bar) & (np.abs(tilt - th) < tol_tilt)
    )[body]
    above = np.where(in_band, cen[body, axis] > y, P[:, axis] > y)
    row = np.where((~in_band) & (P[:, axis] > y - 1.0) & (P[:, axis] <= y))[0]
    Q = P.copy()
    if add:
        Q[above, axis] += 1.0 / math.cos(th) + 2 * gap
        new_bar = P[bar].copy()
        new_bar[:, axis] += gap
        new_row = P[row].copy()
        new_row[:, axis] += 1.0 + gap
        return np.vstack([Q, new_bar, new_row])
    Q[above, axis] -= 1.0
    keep = np.ones(len(P), bool)
    keep[bar] = False
    keep[row] = False
    return Q[keep]


def _compact_if_valid(Q):
    if len(Q) < 80 or len(Q) > 1000 or E.min_gap(Q) < -1e-8:
        return None
    Qc, Sc, ov = E.Compactor(Q).compact()
    if ov > 2e-9 or E.min_gap(Qc) < -1e-8:
        return None
    return Qc.astype(np.float64), float(Sc)


def _try_all_transplants_for_src(args_tuple):
    src_n, P = args_tuple
    out_list = []
    seen_local = set()

    def _add(Q, S, tag):
        nn = int(len(Q))
        if not (80 <= nn <= 1000):
            return
        key = (nn, round(S, 5), tag[:4])
        if key in seen_local:
            return
        seen_local.add(key)
        out_list.append((nn, float(S), tag, Q))

    # 1. Stock 1-step, 2-step & del-1 transplants (proven in W42-43)
    for up in (True, False):
        for nn, S, tag, Q in w42f._try_transplant_one((src_n, P, up)):
            _add(Q, S, tag)

    # 2. Relaxed multi-cut Full Transplant (handles Hopper-polished angles + b=2 bands!)
    try:
        c1s = find_cuts_relaxed(P, 1, top_k=3)
        c0s = find_cuts_relaxed(P, 0, top_k=3)
        for up in (True, False):
            got = False
            for c1 in c1s[:2]:
                Qa = _change_len_cut(P, 1, up, c1)
                if E.min_gap(Qa) < -1e-8:
                    continue
                for c0 in find_cuts_relaxed(Qa, 0, top_k=2):
                    Qb = _change_len_cut(Qa, 0, up, c0)
                    res = _compact_if_valid(Qb)
                    if res is not None:
                        Qc, Sc = res
                        _add(Qc, Sc, f"rtp1_{src_n}_{'u' if up else 'd'}")
                        got = True
                        # Quick del-1 on relaxed transplant
                        if len(Qc) - 1 >= 80:
                            try:
                                cands = w32a.pick_near_band(Qc, max_cand=10)
                                best_d = None
                                for idx in cands[:5]:
                                    Qd, Sd = w32a.fast_delete_and_compact(Qc, idx)
                                    if Qd is not None and (
                                        best_d is None or Sd < best_d[0]
                                    ):
                                        best_d = (Sd, Qd)
                                if best_d is not None:
                                    _add(
                                        best_d[1].astype(np.float64),
                                        float(best_d[0]),
                                        f"rtpd1_{src_n}_{'u' if up else 'd'}",
                                    )
                            except Exception:
                                pass
                        break
                if got:
                    break

        # 3. Valid Half-Transplant (1 tilted bar + 1 upright edge strip) on BOTH axes!
        for ax, cuts in ((1, c1s), (0, c0s)):
            other = 1 - ax
            # Half-Up
            for c in cuts[:2]:
                Qa = _change_len_cut(P, ax, True, c)
                if E.min_gap(Qa) < -1e-8:
                    continue
                Qa = E.to_origin(Qa)
                h = 0.5 * (np.abs(np.cos(Qa[:, 2])) + np.abs(np.sin(Qa[:, 2])))
                W = float((Qa[:, 0] + h).max())
                H = float((Qa[:, 1] + h).max())
                L_s = H if ax == 1 else W
                L_o = W if ax == 1 else H
                k = int(math.floor(L_s + 1e-9))
                strip = (
                    np.array(
                        [(L_o + 0.5 + 2e-4, j + 0.5, 0.0) for j in range(k)],
                        dtype=np.float64,
                    )
                    if other == 0
                    else np.array(
                        [(i + 0.5, L_o + 0.5 + 2e-4, 0.0) for i in range(k)],
                        dtype=np.float64,
                    )
                )
                Qb = np.vstack([Qa, strip])
                res = _compact_if_valid(Qb)
                if res is not None:
                    _add(res[0], res[1], f"htpU_{src_n}_a{ax}")
                    break
            # Half-Down
            for c in cuts[:2]:
                Qa = _change_len_cut(P, ax, False, c)
                if E.min_gap(Qa) < -1e-8:
                    continue
                Qa = E.to_origin(Qa)
                h = 0.5 * (np.abs(np.cos(Qa[:, 2])) + np.abs(np.sin(Qa[:, 2])))
                upright = np.abs(E.fold(Qa[:, 2])) < 0.025
                got_dn = False
                for side_hi in (False, True):
                    L_o = float((Qa[:, other] + h).max())
                    mask = (
                        (upright & (Qa[:, other] > L_o - 1.05))
                        if side_hi
                        else (upright & (Qa[:, other] < 1.05))
                    )
                    viol = np.any(
                        (~upright)
                        & (
                            (Qa[:, other] + h > L_o - 0.95)
                            if side_hi
                            else (Qa[:, other] - h < 0.95)
                        )
                    )
                    if viol or mask.sum() < 5:
                        continue
                    Qb = E.to_origin(Qa[~mask])
                    res = _compact_if_valid(Qb)
                    if res is not None:
                        _add(res[0], res[1], f"htpD_{src_n}_a{ax}")
                        got_dn = True
                        break
                if got_dn:
                    break
    except Exception:
        pass

    return out_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2550)
    ap.add_argument("--hop-sec", type=int, default=200)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-off", type=int, default=440000)
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

    srcs = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        if cur[n][0] < K + 1 - 1e-6 and n != K * K:
            srcs.append(n)
    my_srcs = srcs[args.shard :: args.num_shards]
    tp_tasks = [(n, cur[n][1]) for n in my_srcs]

    tp_cands = []
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        for res_list in ex.map(_try_all_transplants_for_src, tp_tasks):
            for n_new, S, tag, Q in res_list:
                gap = S - cur[n_new][0]
                if gap < 0.022:
                    tp_cands.append((gap, n_new, S, tag, Q))
    tp_cands.sort(key=lambda z: z[0])
    print(
        f"[W44F-RelaxedFull+HalfTransplant s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"swept={len(tp_tasks)} srcs -> promising={len(tp_cands)} "
        f"(direct_wins={sum(1 for z in tp_cands if z[0] < -1e-7)}) ({time.time() - t0:.1f}s)",
        flush=True,
    )

    deadline = t0 + args.budget_sec
    jobs = []
    seen_tp_key = set()
    for i, (gap, n_new, S, tag, Q) in enumerate(tp_cands):
        # Allow up to 2 distinct operator families (e.g. full vs half) per target n_new!
        fam = "half" if tag.startswith("htp") else "full"
        if (n_new, fam) in seen_tp_key:
            continue
        seen_tp_key.add((n_new, fam))
        jobs.append(
            (
                n_new,
                cur[n_new][0],
                Q,
                tag,
                args.seed_off + args.shard * 10007 + i * 97,
                args.hop_sec,
                args.n_seeds,
                deadline,
            )
        )

    # Fill remainder with top live-gap Wider-Move Hopper targets
    gap_ranked = []
    for n in range(120, 1001):
        K = math.isqrt(n)
        if n == K * K or cur[n][0] >= K + 1 - 1e-6:
            continue
        fl = front_L.get(n, cur[n][0])
        gds = (cur[n][0] - fl) / math.sqrt(n)
        if gds > 1e-6:
            gap_ranked.append((gds, n))
    gap_ranked.sort(reverse=True)
    top_gap_ns = [n for _, n in gap_ranked[:160]][args.shard :: args.num_shards]
    for j_i, n in enumerate(top_gap_ns):
        jobs.append(
            (
                n,
                cur[n][0],
                cur[n][1],
                "wide_gap",
                args.seed_off + 500000 + args.shard * 10007 + j_i * 131,
                args.hop_sec,
                args.n_seeds,
                deadline,
            )
        )

    print(
        f"[W44F s={args.shard}] Queued {len(jobs)} jobs ({len(seen_tp_key)} fresh full/half transplants + {len(top_gap_ns)} gap targets)",
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
        futs = [ex.submit(w42f.worker_wide_hopper_polish, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W44F ERR] {exc}", flush=True)
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
                        f"  [W44F WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"(dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()

    save_ckpt()
    print(
        f"[W44F s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
