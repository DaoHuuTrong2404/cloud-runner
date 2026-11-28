#!/usr/bin/env python3
"""Wave 67S (`w67s_merged_seam_seg_forge_and_global_dp.py`): `joegun` `"seg"` (`Gadget + neighbours`
Merged-Seam Sub-Rectangle Quench Forge — `81.8%` shrink rate!) + Unified 24-Gadget 8-Sym DP + 6-Round WideHopper!

Implements the highest-yield gadget discovery mechanic from `joegun research.json` (`9 / 11 = 81.8%` shrink rate!):
  - In `fault_dp_8sym`, multi-gadget chains `(g_1, g_2, ...)` place individual gadgets separated by rigid straight
    fault seams (`EPS = 1e-8`), because each `g_i` was quenched in isolation with flat container walls!
  - `extract_merged_segments()` traces all active multi-gadget chains on `all_targets[:240]` (after loading ALL `24`
    upgraded gadgets from `W64G` + `W66G`!), extracts contiguous 2-gadget and 3-gadget sub-chains `(W_seg, H_seg, DM_seg)`
    (including their exact staircase upright complement cells so total count is `W_seg * H_seg + DM_seg`), and
    quenches the merged segment AS ONE PIECE WITH ALL INTERNAL SEAMS COMPLETELY FREE via `fq quench --rect-dx ... --rect-dy ...`!
  - Injects all shrunk `"seg"` super-gadgets back into `fault_dp_8sym.solve_dp`, saves them to `w67s_gadgets_s*.npz`,
    sweeps `trace_and_compose(..., top_m=6)`, and runs full 6-Round Continuous-Momentum WideHopper on top `2` novel basins!
"""

import argparse
import glob
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
import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w56g_unified_global_faultdp as w56g
import w63k_1puzzle_per_shard_killshot as w63k


def compose_subchain_segment(sub_links, G, donors):
    """Compose a contiguous sub-chain `sub_links = [(g_idx, mode), ...]` (`len >= 2`) into one exact
    `(W_seg, H_seg, DM_seg)` sub-rectangle gadget of `W_seg * H_seg + DM_seg` disjoint unit squares!
    """
    M = len(sub_links)
    if M < 2:
        return None
    W_seg = int(G[sub_links[0][0], 0]) + sum(
        int(G[g, 0]) + (1 if mode == 1 else 0) for g, mode in sub_links[1:]
    )
    H_seg = int(G[sub_links[0][0], 1]) + sum(
        int(G[g, 1]) + (1 if mode == 2 else 0) for g, mode in sub_links[1:]
    )
    DM_seg = sum(int(G[g, 2]) for g, _ in sub_links)
    if W_seg > fdp8.WMAX or H_seg > fdp8.WMAX or DM_seg > fdp8.DMAX:
        return None
    M_sq = W_seg * H_seg + DM_seg
    if M_sq > 560:
        return None

    f_max = max(float(G[g, 3]) for g, _ in sub_links)
    F = f_max + (M + 2) * fdp8.EPS
    a = np.full(W_seg, H_seg, dtype=int)
    b = np.full(W_seg, H_seg, dtype=int)
    gad = []
    ox, oy = 0, H_seg
    for m_i, (g, mode) in enumerate(sub_links):
        w, h, dm, di, si = (int(v) for v in G[g, [0, 1, 2, 6, 7]])
        if m_i > 0:
            ox += 1 if mode == 1 else 0
            oy -= 1 if mode == 2 else 0
        oy -= h
        if mode == 1:
            a[ox - 1] = b[ox - 1] = oy + h
        a[ox : ox + w] = oy
        b[ox : ox + w] = oy + h
        sq_g, _ = fdp8.crop_gadget(donors[di], si, w, h, dm)
        sq_g = sq_g.copy()
        sq_g[:, :2] *= 1.0 + 1e-11
        sq_g[:, 0] += ox + fdp8.EPS * (1 + m_i)
        sq_g[:, 1] += oy + fdp8.EPS * (M - m_i)
        gad.append(sq_g)
        ox += w

    if oy != 0 or ox != W_seg or not (np.all(a <= b) and np.all(a[1:] <= b[:-1])):
        return None

    cells = [(i + 0.5, j + 0.5, 0.0) for i in range(W_seg) for j in range(a[i])]
    cells += [
        (i + 0.5 + F, j + 0.5 + F, 0.0)
        for i in range(W_seg)
        for j in range(b[i], H_seg)
    ]
    parts = ([np.array(cells, dtype=np.float64).reshape(-1, 3)] if cells else []) + gad
    Q_seg = RE.normalize_to_origin(np.concatenate(parts, axis=0))
    if len(Q_seg) != M_sq:
        return None
    f_init = RE.gadget_overhang(Q_seg, W_seg, H_seg)
    return (W_seg, H_seg, DM_seg), f_init, Q_seg


def collect_merged_segment_jobs(dp, PG, PM, PQ, G, donors, targets):
    """Find contiguous 2-gadget and 3-gadget sub-chains along active DP chains where the bottleneck
    gadget decides the overhang!
    """
    seg_map = {}
    for _, n, K, D in targets:
        if K > fdp8.WMAX or D > fdp8.DMAX:
            continue
        sub = dp[: K + 1, : K + 1, D, :]
        if not np.isfinite(sub).any():
            continue
        flat_idx = np.argsort(sub.ravel())[:14]
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
                    res = compose_subchain_segment(links[st : st + span], G, donors)
                    if res is None:
                        continue
                    key, f_init, Q_seg = res
                    # Compare against best existing `(w, h, dm)` in G (if any)!
                    rows = np.where(
                        (G[:, 0].astype(int) == key[0])
                        & (G[:, 1].astype(int) == key[1])
                        & (G[:, 2].astype(int) == key[2])
                    )[0]
                    f_lib = float(np.min(G[rows, 3])) if len(rows) else f_init
                    f_ref = min(f_lib, f_init)
                    if f_init <= f_ref + 0.012:
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


def quench_merged_seg_job(job_tup):
    key, f_ref, Q_seg, seed0 = job_tup
    w, h, dm = int(key[0]), int(key[1]), int(key[2])
    m_min = min(w, h)
    dx, dy = float(w - m_min), float(h - m_min)
    P0 = RE.normalize_to_origin(Q_seg)
    s_in = float(m_min) + RE.gadget_overhang(P0, w, h) + 1e-8

    best_f = float(f_ref)
    best_Q = None
    schedules = [
        # 1) Pure free-seam smooth ALM + KKT finish (exact `joegun` `"seg"` recipe!)
        [
            "--rect-dx",
            repr(dx),
            "--rect-dy",
            repr(dy),
            "--seed",
            str(int(seed0)),
            "--loosen",
            "1.003",
            "--kick",
            "0.004",
            "--outer",
            "60",
            "--pit",
            "320",
            "--flip-top",
            "24",
            "--finish",
            "--stag-w",
            "0",
            "--deadline",
            "18.0",
        ],
        # 2) Gentle rounded-corner seam interpenetration (`r0 = 0.08`) + `--round-press`!
        [
            "--rect-dx",
            repr(dx),
            "--rect-dy",
            repr(dy),
            "--seed",
            str(int(seed0) + 37),
            "--round-r0",
            "0.08",
            "--round-steps",
            "14",
            "--round-press",
            "--round-press-outer",
            "8",
            "--round-kick",
            "0.0045",
            "--round-kick-peak",
            "0.06",
            "--loosen",
            "1.005",
            "--outer",
            "52",
            "--pit",
            "280",
            "--flip-top",
            "20",
            "--finish",
            "--stag-w",
            "0",
            "--deadline",
            "20.0",
        ],
        # 3) Deeper Nakajima seam melt (`r0 = 0.16`) + `--round-press`!
        [
            "--rect-dx",
            repr(dx),
            "--rect-dy",
            repr(dy),
            "--seed",
            str(int(seed0) + 79),
            "--round-r0",
            "0.16",
            "--round-steps",
            "16",
            "--round-press",
            "--round-press-outer",
            "8",
            "--round-kick",
            "0.0075",
            "--round-kick-peak",
            "0.11",
            "--loosen",
            "1.006",
            "--outer",
            "52",
            "--pit",
            "280",
            "--flip-top",
            "20",
            "--finish",
            "--stag-w",
            "0",
            "--deadline",
            "21.0",
        ],
    ]
    for extra in schedules:
        rq = RE.run_fq_quench_args(P0, extra, s_ref=s_in, timeout_s=24.0)
        if rq is None:
            continue
        Qg = RE.normalize_to_origin(rq[1])
        fc = RE.gadget_overhang(Qg, w, h)
        if fc < best_f - 1e-8:
            best_f, best_Q = fc, Qg
    return key, float(f_ref), best_f, best_Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--global-gadgets-npz", required=True)
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1840)
    ap.add_argument("--seed-off", type=int, default=2450000)
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

    donors = []
    for n in range(5, 1001):
        K = math.isqrt(n)
        if K * K != n and cur[n][0] < K + 1 - 1e-6 and cur[n][1] is not None:
            donors.append(cur[n][1])

    gad_files = sorted(set(glob.glob("pool/gadgets/**/*.npz", recursive=True)))
    if (
        os.path.exists(args.global_gadgets_npz)
        and args.global_gadgets_npz not in gad_files
    ):
        gad_files.append(args.global_gadgets_npz)
    for gf in gad_files:
        try:
            with np.load(gf) as zg:
                if "keys" in zg.files:
                    for i in range(len(zg["keys"])):
                        donors.append(np.asarray(zg[f"sq_{i}"], dtype=np.float64))
        except Exception:
            pass

    G = fdp8.extract_gadgets(donors)
    dp0, PG0, PM0, PQ0 = fdp8.solve_dp(G)

    all_targets = []
    for n in range(90, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D <= 0:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gl > 5.0e-5:
            all_targets.append((gl / math.sqrt(n), n, K, D))
    all_targets.sort(reverse=True)

    seg_all = collect_merged_segment_jobs(
        dp0, PG0, PM0, PQ0, G, donors, all_targets[:240]
    )
    my_segs = seg_all[args.shard :: args.num_shards][:11]
    for top_s in seg_all[:3]:
        if all(top_s[0] != x[0] for x in my_segs):
            my_segs.append(top_s)

    seg_jobs = [
        (k, f_r, q_s, args.seed_off + args.shard * 4391 + si * 113)
        for si, (k, f_r, q_s) in enumerate(my_segs)
    ]

    print(
        f"[W67S-MergedSeamSegDP s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"gad_files={len(gad_files)} G={len(G)} total_segs={len(seg_all)} my_segs={len(seg_jobs)} ({time.time() - t0:.1f}s)",
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
                f"  [W67S MERGED-SEG WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    upgraded_segs = []
    ex_sg = ProcessPoolExecutor(max_workers=args.procs)
    try:
        futs = [ex_sg.submit(quench_merged_seg_job, jb) for jb in seg_jobs]
        try:
            for f in as_completed(futs, timeout=440.0):
                key, f_ref, best_f, best_Q = f.result()
                if best_Q is not None and best_f < f_ref - 1e-7:
                    upgraded_segs.append((key, f_ref, best_f, best_Q))
                    print(
                        f"  [W67S SEG UPGRADE!!] {key[0]}x{key[1]}+{key[2]}: {f_ref:.6f} -> {best_f:.6f} (df={best_f - f_ref:+.6f})!",
                        flush=True,
                    )
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_sg)

    if upgraded_segs:
        gad_pl = {"keys": np.array([k for k, _, _, _ in upgraded_segs], dtype=np.int32)}
        for gi, (_, _, _, sq1) in enumerate(upgraded_segs):
            gad_pl[f"sq_{gi}"] = np.asarray(sq1, dtype=np.float64)
        np.savez_compressed(
            os.path.join(
                os.path.dirname(args.out_npz), f"w67s_gadgets_s{args.shard}.npz"
            ),
            **gad_pl,
        )
        new_dons = [sq1 for _, _, _, sq1 in upgraded_segs]
        Gn = fdp8.extract_gadgets(new_dons)
        if len(Gn):
            Gn[:, 6] += len(donors)
            G = np.vstack([G, Gn])
            donors = donors + new_dons
            dp0, PG0, PM0, PQ0 = fdp8.solve_dp(G)

    my_sweep = all_targets[args.shard :: args.num_shards]
    full_cands = []
    for _, n, K, D in my_sweep:
        if time.time() - t0 > 680.0:
            break
        if K > fdp8.WMAX or D > fdp8.DMAX:
            continue
        L0 = wins[n][0] if n in wins else cur[n][0]
        dp_chains = fdp8.trace_and_compose(dp0, PG0, PM0, PQ0, G, donors, K, D, top_m=6)
        for L_est, dtag, P_comp in dp_chains:
            if L_est < L0 + 0.020:
                rh = w56g.quench_composed_chain(P_comp, L0)
                if rh is not None:
                    Qc, Sc = rh
                    if Sc < L0 + 0.00030:
                        Qp, Sp = w51d.deep_contact_polish(
                            n, Qc, deadline, slp_sec=38.0, fq_sec=24
                        )
                        record_win_n(n, Qp, Sp, dtag + "_segdp", 0)
                        Qc, Sc = Qp, Sp
                    ad = w53h.topology_angle_dist(Qc, cur[n][1])
                    if Sc < L0 + 0.0130 and ad > 0.035:
                        full_cands.append((Sc - L0, n, dtag, Sc, ad, Qc))

    if upgraded_segs:
        for _, n, K, D in all_targets:
            if time.time() - t0 > 730.0:
                break
            if K > fdp8.WMAX or D > fdp8.DMAX:
                continue
            L0 = wins[n][0] if n in wins else cur[n][0]
            dp_chains = fdp8.trace_and_compose(
                dp0, PG0, PM0, PQ0, G, donors, K, D, top_m=2
            )
            for L_est, dtag, P_comp in dp_chains:
                if L_est < L0 + 0.0020:
                    rh = w56g.quench_composed_chain(P_comp, L0)
                    if rh is not None:
                        Qc, Sc = rh
                        if Sc < L0 + 0.00025:
                            Qp, Sp = w51d.deep_contact_polish(
                                n, Qc, deadline, slp_sec=36.0, fq_sec=22
                            )
                            record_win_n(n, Qp, Sp, dtag + "_glbseg", 0)
                            Qc, Sc = Qp, Sp
                        ad = w53h.topology_angle_dist(Qc, cur[n][1])
                        if Sc < L0 + 0.0090 and ad > 0.035:
                            full_cands.append((Sc - L0, n, dtag + "_gs", Sc, ad, Qc))

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
        f"[W67S s={args.shard}] upgraded_segs={len(upgraded_segs)} screened={len(full_cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
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
                f"  [W67S done] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W67S s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
