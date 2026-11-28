#!/usr/bin/env python3
"""Wave 68S (`w68s_global_seg2_dp_and_small_seam_forge.py`):
1) Loads precomputed 47-Gadget (`W64G + W65F + W66G + W67S`!) 8-Sym DP Cache (`pool/dp_cache.npz`) in 0.25s!
2) Parallel 4-process global `trace_and_compose_v2(..., top_m=8)` quench sweep across 100% of all non-grid puzzles
   (`all_targets[shard :: 16]`), including single-link forged super-gadgets (`19x20+13 df=-0.011240`, `20x18+14`, etc.)!
3) 2nd-Generation Compact & Medium Merged-Seam `"seg"` Forge (`M_sq <= 320`, reusable multi-gadget building blocks!)
4) 6-Round Continuous-Momentum WideHopper on top 2 novel DP basins!
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

import fault_dp_8sym as fdp8
import dp8_cache_w68 as pdc
import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w56g_unified_global_faultdp as w56g
import w63k_1puzzle_per_shard_killshot as w63k
import w67s_merged_seam_seg_forge_and_global_dp as w67s


def quench_dp_chain_job(job):
    n, L0, dtag, P_comp = job
    rh = w56g.quench_composed_chain(P_comp, L0)
    if rh is None:
        return n, L0, dtag, None, math.inf
    Qc, Sc = rh
    if Sc < L0 + 0.00045:
        Sp, Qp = RE.kkt_sqp_polish_fq(Qc, timeout_each=6.0)
        if Sp < Sc:
            Qc, Sc = Qp, Sp
    return n, L0, dtag, Qc, float(Sc)


def collect_compact_seg_jobs(dp, PG, PM, PQ, G, donors, targets, max_sq=320):
    seg_map = {}
    for _, n, K, D in targets:
        if K > fdp8.WMAX or D > fdp8.DMAX:
            continue
        sub = dp[: K + 1, : K + 1, D, :]
        if not np.isfinite(sub).any():
            continue
        flat_idx = np.argsort(sub.ravel())[:18]
        for fi in flat_idx:
            cw, ch, q = np.unravel_index(int(fi), sub.shape)
            if not math.isfinite(float(sub[cw, ch, q])):
                break
            t_cw, t_ch, t_d, t_q = int(cw), int(ch), int(D), int(q)
            links = []
            ok_tr = True
            while t_cw or t_ch or t_d:
                g = int(PG[t_cw, t_ch, t_d, t_q])
                if g < 0:
                    ok_tr = False
                    break
                mode = int(PM[t_cw, t_ch, t_d, t_q])
                qp = int(PQ[t_cw, t_ch, t_d, t_q])
                links.append((g, mode))
                w, h, dd = int(G[g, 0]), int(G[g, 1]), int(G[g, 2])
                t_cw -= w + (1 if mode == 1 else 0)
                t_ch -= h + (1 if mode == 2 else 0)
                t_d -= dd
                t_q = qp
            if not ok_tr or len(links) < 2:
                continue
            links = links[::-1]
            for span in (2, 3):
                for st in range(len(links) - span + 1):
                    res = w67s.compose_subchain_segment(
                        links[st : st + span], G, donors
                    )
                    if res is None:
                        continue
                    key, f_init, Q_seg = res
                    if key[0] * key[1] + key[2] > max_sq:
                        continue
                    rows = np.where(
                        (G[:, 0].astype(int) == key[0])
                        & (G[:, 1].astype(int) == key[1])
                        & (G[:, 2].astype(int) == key[2])
                    )[0]
                    f_lib = float(np.min(G[rows, 3])) if len(rows) else f_init
                    f_ref = min(f_lib, f_init)
                    if f_init <= f_ref + 0.015:
                        prev = seg_map.get(key)
                        if prev is None or f_init < prev[1]:
                            seg_map[key] = (
                                f_ref,
                                f_init,
                                Q_seg,
                                prev[3] + 1 if prev else 1,
                            )
                        else:
                            seg_map[key] = (prev[0], prev[1], prev[2], prev[3] + 1)
    ranked = sorted(seg_map.items(), key=lambda kv: (-kv[1][3], kv[1][1] - kv[1][0]))
    return [(k, v[0], v[2]) for k, v in ranked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--dp-cache-npz", required=True)
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=2650000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    donors, donor_n, G, dp0, PG0, PM0, PQ0 = pdc.load_dp8_cache(args.dp_cache_npz)

    all_targets = []
    for n in range(40, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D <= 0:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        all_targets.append((gl / math.sqrt(n), n, K, D))
    all_targets.sort(reverse=True)

    my_sweep = all_targets[args.shard :: args.num_shards]
    chain_jobs = []
    for _, n, K, D in my_sweep:
        if K > fdp8.WMAX or D > fdp8.DMAX:
            continue
        L0 = cur[n][0]
        dp_ch = fdp8.trace_and_compose_v2(
            dp0, PG0, PM0, PQ0, G, donors, donor_n, K, D, top_m=8
        )
        for L_est, dtag, P_comp in dp_ch:
            if L_est < L0 + 0.022:
                chain_jobs.append((n, L0, dtag, P_comp))

    seg_all = collect_compact_seg_jobs(
        dp0, PG0, PM0, PQ0, G, donors, all_targets[:320], max_sq=320
    )
    my_segs = seg_all[args.shard :: args.num_shards][:10]
    seg_jobs = [
        (k, f_r, q_s, args.seed_off + args.shard * 4391 + si * 113)
        for si, (k, f_r, q_s) in enumerate(my_segs)
    ]

    print(
        f"[W68S-GlobalSeg2DP s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"donors={len(donors)} G={len(G)} chain_jobs={len(chain_jobs)} compact_segs={len(seg_jobs)}/{len(seg_all)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    deadline = t0 + args.budget_sec
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

    def record_win_n(n_w, Q_cand, S_cand, src_tag, w_cnt):
        nonlocal tot_ds
        L_ref = wins[n_w][0] if n_w in wins else cur[n_w][0]
        if S_cand >= L_ref - 1e-9:
            return
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n_w, Q_cand)
        if ok and Lf < L_ref - 1e-9:
            ds = (Lf - L_ref) / math.sqrt(n_w)
            tot_ds += ds
            wins[n_w] = (float(Lf), xf, yf, tf)
            print(
                f"  [W68S GLOBAL-SEG2 WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    full_cands = []
    upgraded_segs = []
    ex_pool = ProcessPoolExecutor(max_workers=args.procs)
    try:
        # Stage 1: Parallel global 47-gadget 8-Sym DP composition quench across all shard targets!
        futs_ch = [ex_pool.submit(quench_dp_chain_job, jb) for jb in chain_jobs]
        try:
            for f in as_completed(futs_ch, timeout=480.0):
                n, L0, dtag, Qc, Sc = f.result()
                if Qc is None:
                    continue
                L_now = wins[n][0] if n in wins else L0
                if Sc < L_now + 0.00030:
                    Qp, Sp = w51d.deep_contact_polish(
                        n, Qc, deadline, slp_sec=38.0, fq_sec=24
                    )
                    record_win_n(n, Qp, Sp, dtag + "_glbdp", 0)
                    Qc, Sc = Qp, Sp
                ad = w53h.topology_angle_dist(Qc, cur[n][1])
                if Sc < L_now + 0.0130 and ad > 0.035:
                    full_cands.append((Sc - L_now, n, dtag, Sc, ad, Qc))
        except TimeoutError:
            pass

        # Stage 2: Compact & 2nd-Gen `"seg"` Merged-Seam Forge (`M_sq <= 320`)!
        futs_sg = [ex_pool.submit(w67s.quench_merged_seg_job, jb) for jb in seg_jobs]
        try:
            for f in as_completed(futs_sg, timeout=380.0):
                key, f_ref, best_f, best_Q = f.result()
                if best_Q is not None and best_f < f_ref - 1e-7:
                    upgraded_segs.append((key, f_ref, best_f, best_Q))
                    print(
                        f"  [W68S COMPACT SEG UPGRADE!!] {key[0]}x{key[1]}+{key[2]}: {f_ref:.6f} -> {best_f:.6f} (df={best_f - f_ref:+.6f})!",
                        flush=True,
                    )
                    # Direct puzzle check if square `w == h`!
                    if key[0] == key[1]:
                        n_eq = key[0] * key[1] + key[2]
                        if 1 <= n_eq <= 1000:
                            S_eq = RE.side_exact(best_Q)
                            L_eq0 = wins[n_eq][0] if n_eq in wins else cur[n_eq][0]
                            if S_eq < L_eq0 + 0.00030:
                                Qp_e, Sp_e = w51d.deep_contact_polish(
                                    n_eq, best_Q, deadline, slp_sec=36.0, fq_sec=22
                                )
                                record_win_n(n_eq, Qp_e, Sp_e, "seg_sq_direct", 0)
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_pool)

    if upgraded_segs:
        gad_pl = {"keys": np.array([k for k, _, _, _ in upgraded_segs], dtype=np.int32)}
        for gi, (_, _, _, sq1) in enumerate(upgraded_segs):
            gad_pl[f"sq_{gi}"] = np.asarray(sq1, dtype=np.float64)
        np.savez_compressed(
            os.path.join(
                os.path.dirname(args.out_npz), f"w68s_gadgets_s{args.shard}.npz"
            ),
            **gad_pl,
        )

    full_cands.sort(key=lambda z: z[0])
    queued = []
    seen_n = {}
    for item in full_cands:
        nn = item[1]
        if seen_n.get(nn, 0) >= 1:
            continue
        seen_n[nn] = 1
        queued.append(item)
        if len(queued) >= 2:
            break

    for _, nn, _, _ in my_sweep:
        if len(queued) >= 2:
            break
        if seen_n.get(nn, 0) >= 1:
            continue
        L0_fb = wins[nn][0] if nn in wins else cur[nn][0]
        P0_fb = cur[nn][1]
        queued.append((0.0, nn, "cur_direct", L0_fb, 0.0, P0_fb))
        seen_n[nn] = 1

    print(
        f"[W68S s={args.shard}] upgraded_compact_segs={len(upgraded_segs)} screened={len(full_cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, tag, S_init, adist, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 80.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            sub_b = (rem - 20.0) / max(1, len(queued) - i)
            Qb, Sb, (_, _, wt, nr) = w63k.continuous_momentum_widehopper(
                ex_hop,
                n,
                Q_init,
                L0_cur,
                args.seed_off + 95000 + args.shard * 8009 + i * 401,
                sub_b,
                deadline,
                lambda Qc, Sc, stg, wc, _n=n, _tg=tag: record_win_n(
                    _n, Qc, Sc, f"{_tg}->{stg}", wc
                ),
            )
            record_win_n(n, Qb, Sb, f"{tag}->whFin", wt)
            print(
                f"  [W68S done] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W68S s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
