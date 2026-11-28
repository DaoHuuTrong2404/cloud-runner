#!/usr/bin/env python3
"""Wave 65F (`w65f_hydraulic_vise_channel_and_large_strip.py`): Hydraulic-Vise Frozen-Crystal
Nakajima `--round-press` Channel Remelt + Large-M (`M = 126..620`) Active Strip Gadget Forge + Sym-Md Harden!

Directly targets the exact physical anatomy uncovered across our Top-12 R35 gaps (`234, 367, 491, 333, 738,
456, 730, 261, 904, 962, 785, 842`):
  1. Every Top-12 gap consists of ONE connected tilted channel (`36..304` squares) wedged between `156..789`
     upright crystal squares.
  2. Engine 1 (Hydraulic-Vise Frozen-Crystal `--round-press` Remelt):
     - Freezes deep crystal components with `--freeze` + `--anchor-x` + `--anchor-y` and activates Evan's
       `--round-press --round-press-outer 8 --round-r0 {0.13, 0.20, 0.28, 0.35} --round-steps 18 --round-kick-peak 0.12`
       (never previously combined with wall-anchored crystal blocks!), turning the crystal blocks into hydraulic
       vise jaws that compress the rounded-square channel before unfrozen two-pass `kkt_sqp_polish_fq`!
  3. Engine 2 (Large-M `126..620` Active Strip/Sub-Rectangle Gadget Forge + Multi-K L-Extend):
     - Forges the large active sub-rectangle gadgets (`22x24+1`, `28x17+1`, `20x29+1`, `19x18+25`, `22x21+29`, etc.)
       skipped by `W64G` (`M <= 125`) via `--rect-dx` / `--rect-dy` + `--round-press`, and pads every improved
       `(w, h, dm)` gadget onto upright grids across ALL `K = max(w, h) .. 31` (`n = K^2 + dm <= 1000`)!
  4. Engine 3 (`harden reharden --sym-md 1 --sym-band {0.0, 2.2, 3.6} --hold-move {0, 1}`):
     - Diagonal-mirror subspace momentum re-hardening (`harden.rs` lines 363-417) + `kkt_sqp_polish_fq`!
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
import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53g_rect_quench_8sym_faultdp as w53g
import w53h_nakajima_reharden_and_softinsert as w53h
import w56f_frozen_grid_channel_remelt as w56f
import w63k_1puzzle_per_shard_killshot as w63k


def pad_gadget_to_square_K(Q_gad, w, h, K):
    """Pad an `M = w * h + dm` sub-rectangle gadget `[0, w + f] x [0, h + f]` onto a `K x K` square
    by placing `(K - w) x h` upright squares on the right and `K x (K - h)` upright squares on top,
    producing `K^2 + dm` squares inside `[0, K + f]^2`!
    """
    P = RE.normalize_to_origin(Q_gad)
    f = RE.gadget_overhang(P, w, h)
    # Place gadget in bottom-left `[0, w + f] x [0, h + f]`, right grid in `[w + f, K + f] x [0, h]`,
    # top grid in `[0, K] x [h + f, K + f]` so ZERO overlap is mathematically guaranteed!
    blocks = [P]
    if K > w:
        rx = np.array(
            [
                (float(w) + f + 0.5 + ix, 0.5 + iy, 0.0)
                for ix in range(K - w)
                for iy in range(h)
            ],
            dtype=np.float64,
        )
        blocks.append(rx)
    if K > h:
        ty = np.array(
            [
                (0.5 + ix, float(h) + f + 0.5 + iy, 0.0)
                for ix in range(K)
                for iy in range(K - h)
            ],
            dtype=np.float64,
        )
        blocks.append(ty)
    return RE.normalize_to_origin(np.vstack(blocks))


def eval_hydraulic_vise_job(job_tup):
    n, tag, Pn, Sn, fs, axs, ays, seed, r0, rk, sym_band, use_sym_harden = job_tup
    try:
        if use_sym_harden:
            rh = RE.run_harden_reharden(
                Pn,
                s_ref=Sn,
                seed=int(seed),
                r0=r0,
                stiff0=0.28,
                pre=2000,
                hold=8500,
                release=4000,
                settle=3000,
                hold_f=1.0004,
                noise=rk,
                mu=0.022,
                sym_md=1,
                sym_band=sym_band,
                hold_move=int(seed % 2),
                timeout_s=22.0,
            )
            if rh is None:
                return n, tag, 1e9, None
            S_fin, Q_fin = RE.kkt_sqp_polish_fq(rh[1], timeout_each=6.5)
            return n, tag, float(S_fin), Q_fin

        extra = ["--freeze", fs, "--seed", str(int(seed))]
        if axs:
            extra += ["--anchor-x", axs]
        if ays:
            extra += ["--anchor-y", ays]
        extra += [
            "--round-r0",
            repr(float(r0)),
            "--round-steps",
            "18",
            "--round-press",
            "--round-press-outer",
            "8",
            "--round-kick",
            repr(float(rk)),
            "--round-kick-peak",
            "0.12",
            "--round-kick-w",
            "0.05",
            "--loosen",
            "1.006",
            "--outer",
            "48",
            "--pit",
            "240",
            "--flip-top",
            "18",
            "--stag-w",
            "0",
            "--deadline",
            "20.5",
        ]
        rq1 = RE.run_fq_quench_args(Pn, extra, s_ref=Sn, timeout_s=23.5)
        if rq1 is None:
            return n, tag, 1e9, None
        S_fin, Q_fin = RE.kkt_sqp_polish_fq(rq1[1], timeout_each=6.5)
        return n, tag, float(S_fin), Q_fin
    except Exception:
        return n, tag, 1e9, None


def eval_large_strip_gadget_job(job_tup):
    key, f_ref, sq_g, seed0 = job_tup
    w, h, dm = int(key[0]), int(key[1]), int(key[2])
    m_min = min(w, h)
    dx, dy = float(w - m_min), float(h - m_min)
    P0 = RE.normalize_to_origin(sq_g)
    s_in = float(m_min) + float(f_ref) + 1e-8

    best_f = float(f_ref)
    best_Q = None
    cands = []
    for ci, (r0, rk) in enumerate([(0.11, 0.005), (0.18, 0.008), (0.26, 0.011)]):
        extra = [
            "--rect-dx",
            repr(dx),
            "--rect-dy",
            repr(dy),
            "--seed",
            str(int(seed0) + ci * 41),
            "--round-r0",
            repr(float(r0)),
            "--round-steps",
            "16",
            "--round-press",
            "--round-press-outer",
            "8",
            "--round-kick",
            repr(float(rk)),
            "--round-kick-peak",
            "0.11",
            "--loosen",
            "1.005",
            "--outer",
            "50",
            "--pit",
            "260",
            "--flip-top",
            "20",
            "--finish",
            "--stag-w",
            "0",
            "--deadline",
            "21.0",
        ]
        rq = RE.run_fq_quench_args(P0, extra, s_ref=s_in, timeout_s=24.5)
        if rq is None:
            continue
        Qg = RE.normalize_to_origin(rq[1])
        fc = RE.gadget_overhang(Qg, w, h)
        if fc < f_ref + 0.011:
            cands.append((fc, f"lgad_{w}x{h}+{dm}_r{r0:.2f}", Qg))
        if fc < best_f - 1e-8:
            best_f, best_Q = fc, Qg
    cands.sort(key=lambda z: z[0])
    return key, float(f_ref), best_f, best_Q, cands[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--global-gadgets-npz", required=True)
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=18)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1800)
    ap.add_argument("--seed-off", type=int, default=1920000)
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
    import glob as _glob

    gad_files = sorted(set(_glob.glob("pool/gadgets/**/*.npz", recursive=True)))
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
    for n in range(150, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D <= 0:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gl > 6.0e-5:
            all_targets.append((gl / math.sqrt(n), n, K, D))
    all_targets.sort(reverse=True)

    my_puzzles = [x[1] for x in all_targets[: args.num_shards * 4]][
        args.shard :: args.num_shards
    ]

    # Collect large active `(w, h, dm)` gadgets (`126 <= M <= 620`) skipped by W64G!
    dp_t = [t for t in all_targets[:220] if t[2] <= fdp8.WMAX and t[3] <= fdp8.DMAX]
    chain_keys = w53g.collect_chain_gadget_keys(dp0, PG0, PM0, PQ0, G, dp_t)
    # Also directly include minimal strict sub-rectangles `(w, h, D)` (`max(w,h) < K` or `min(w,h) < K`) from G!
    direct_sub = []
    for _, n, K, D in all_targets[:120]:
        if D > fdp8.DMAX:
            continue
        rows = np.where(
            (G[:, 2].astype(int) == D)
            & (G[:, 0].astype(int) <= K)
            & (G[:, 1].astype(int) <= K)
            & ((G[:, 0].astype(int) < K) | (G[:, 1].astype(int) < K))
            & (G[:, 3] < (cur[n][0] - K) + 0.004)
        )[0]
        if len(rows):
            r_best = G[rows[int(np.argmin(G[rows, 3]))]]
            k_sub = (int(r_best[0]), int(r_best[1]), int(r_best[2]))
            if k_sub not in direct_sub:
                direct_sub.append(k_sub)

    cand_keys = []
    for k in direct_sub + chain_keys:
        M = k[0] * k[1] + k[2]
        if 126 <= M <= 620 and k not in cand_keys:
            cand_keys.append(k)
    my_large_keys = cand_keys[args.shard :: args.num_shards][:6]

    print(
        f"[W65F-HydraulicVise+LargeStrip s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_puzzles={my_puzzles} large_keys={my_large_keys} ({time.time() - t0:.1f}s)",
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
                f"  [W65F VISE/STRIP WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    vise_jobs = []
    for pi, n in enumerate(my_puzzles):
        P0 = cur[n][1]
        anc_t = w56f.compute_crystal_anchors(P0, buffer_dist=1.75)
        anc_w = w56f.compute_crystal_anchors(P0, buffer_dist=2.65)
        anc = anc_t or anc_w
        bseed = args.seed_off + args.shard * 3301 + pi * 127
        if anc is not None:
            for ci, (ac, r0, rk) in enumerate(
                [
                    (anc, 0.13, 0.0055),
                    (anc, 0.20, 0.0075),
                    (anc_w or anc, 0.27, 0.0095),
                    (anc_w or anc, 0.34, 0.0115),
                ]
            ):
                Pn, Sn, fs, axs, ays = ac
                vise_jobs.append(
                    (
                        n,
                        f"vise_r{r0:.2f}",
                        Pn,
                        Sn,
                        fs,
                        axs,
                        ays,
                        bseed + ci * 19,
                        r0,
                        rk,
                        0.0,
                        False,
                    )
                )
        Pn0 = RE.normalize_to_origin(P0)
        Sn0 = RE.side_exact(Pn0)
        for si, (r0s, sb) in enumerate([(0.15, 2.2), (0.21, 3.6)]):
            vise_jobs.append(
                (
                    n,
                    f"symmd_b{sb:.1f}",
                    Pn0,
                    Sn0,
                    "",
                    "",
                    "",
                    bseed + 200 + si * 23,
                    r0s,
                    0.0045,
                    sb,
                    True,
                )
            )

    gad_jobs = []
    for gi, k in enumerate(my_large_keys):
        rows = np.where(
            (G[:, 0].astype(int) == k[0])
            & (G[:, 1].astype(int) == k[1])
            & (G[:, 2].astype(int) == k[2])
        )[0]
        if not len(rows):
            continue
        best_r = G[rows[int(np.argmin(G[rows, 3]))]]
        sq_g, _ = fdp8.crop_gadget(
            donors[int(best_r[6])], int(best_r[7]), k[0], k[1], k[2]
        )
        if sq_g is not None:
            gad_jobs.append(
                (
                    k,
                    float(best_r[3]),
                    sq_g,
                    args.seed_off + 40000 + args.shard * 2903 + gi * 83,
                )
            )

    screened = []
    ex_sc = ProcessPoolExecutor(max_workers=args.procs)
    try:
        futs_v = [ex_sc.submit(eval_hydraulic_vise_job, jb) for jb in vise_jobs]
        futs_g = [ex_sc.submit(eval_large_strip_gadget_job, jb) for jb in gad_jobs]
        try:
            for f in as_completed(futs_v, timeout=420.0):
                nn, tg, Sq, Qq = f.result()
                if Qq is None:
                    continue
                L0 = wins[nn][0] if nn in wins else cur[nn][0]
                if Sq < L0 - 1e-9:
                    Qp0, Sp0 = w51d.deep_contact_polish(
                        nn, Qq, deadline, slp_sec=40.0, fq_sec=24
                    )
                    record_win_n(nn, Qp0, Sp0, tg + "_dir", 0)
                    Qq, Sq = Qp0, Sp0
                ad = w53h.topology_angle_dist(Qq, cur[nn][1])
                if Sq < L0 + 0.0135 and ad > 0.030:
                    screened.append((Sq - L0, nn, tg, Sq, ad, Qq))
        except TimeoutError:
            pass

        try:
            for f in as_completed(futs_g, timeout=220.0):
                key, f_ref, best_f, best_Q, cands = f.result()
                w, h, dm = key
                if best_Q is not None and best_f < f_ref - 1e-7:
                    print(
                        f"  [W65F LARGE GADGET UPGRADE!!] {w}x{h}+{dm}: {f_ref:.6f} -> {best_f:.6f}!",
                        flush=True,
                    )
                for fc, gtag, Qg in cands:
                    for K in range(max(w, h), 32):
                        nn = K * K + dm
                        if nn > 1000 or math.isqrt(nn) != K:
                            continue
                        L0 = wins[nn][0] if nn in wins else cur[nn][0]
                        if float(K) + fc < L0 + 0.0135:
                            Q_pad = pad_gadget_to_square_K(Qg, w, h, K)
                            if len(Q_pad) != nn:
                                continue
                            Spad = RE.side_exact(Q_pad)
                            if Spad < L0 - 1e-9:
                                Qp0, Sp0 = w51d.deep_contact_polish(
                                    nn, Q_pad, deadline, slp_sec=38.0, fq_sec=22
                                )
                                record_win_n(nn, Qp0, Sp0, f"{gtag}_K{K}", 0)
                                Q_pad, Spad = Qp0, Sp0
                            ad = w53h.topology_angle_dist(Q_pad, cur[nn][1])
                            if Spad < L0 + 0.0130 and ad > 0.030:
                                screened.append(
                                    (Spad - L0, nn, f"{gtag}_K{K}", Spad, ad, Q_pad)
                                )
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_sc)

    screened.sort(key=lambda z: z[0])
    queued = []
    seen_n = {}
    for item in screened:
        nn = item[1]
        if seen_n.get(nn, 0) >= 2:
            continue
        seen_n[nn] = seen_n.get(nn, 0) + 1
        queued.append(item)
        if len(queued) >= 4:
            break

    for fb_i, nn in enumerate(my_puzzles):
        if len(queued) >= 3:
            break
        if seen_n.get(nn, 0) >= 1:
            continue
        L0_fb = wins[nn][0] if nn in wins else cur[nn][0]
        P0_fb = cur[nn][1]
        fz = RE.build_far_grid_freeze_mask(P0_fb, buffer_dist=3.5, min_free=85)
        rhp = RE.run_packer_hop(
            P0_fb,
            L0_fb - 2.5e-4,
            seed=args.seed_off + 75000 + args.shard * 907 + fb_i * 37,
            iters=35000,
            frozen_mask=fz,
            timeout_s=28.0,
        )
        if rhp is not None and rhp[0] < L0_fb + 0.012:
            ad_fb = w53h.topology_angle_dist(rhp[1], P0_fb)
            queued.append((rhp[0] - L0_fb, nn, rhp[2], rhp[0], ad_fb, rhp[1]))
        else:
            queued.append((0.0, nn, "cur_direct", L0_fb, 0.0, P0_fb))
        seen_n[nn] = 1

    print(
        f"[W65F s={args.shard}] screened={len(screened)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
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
                args.seed_off + 90000 + args.shard * 8009 + i * 401,
                sub_b,
                deadline,
                lambda Qc, Sc, stg, wc, _n=n, _tg=tag: record_win_n(
                    _n, Qc, Sc, f"{_tg}->{stg}", wc
                ),
            )
            record_win_n(n, Qb, Sb, f"{tag}->whFin", wt)
            print(
                f"  [W65F done] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W65F s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
