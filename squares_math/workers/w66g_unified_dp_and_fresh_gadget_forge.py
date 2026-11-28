#!/usr/bin/env python3
"""Wave 66G (`w66g_unified_dp_and_fresh_gadget_forge.py`): Unified Global 8-Sym Fault-DP (`top_m=6`)
+ Wave-2 Ab-Initio & Large-Strip Nakajima `--round-press` Gadget Forge + Full 6-Round WideHopper!

Builds directly on Wave 64G's double breakthrough (`7` Pareto gadget upgrades up to `df = -0.02287` + strict R35 wins
on `n = 912` via `fdp8_m2->whR1` and `n = 748` via `fdp8_m1->whR2`) and Wave 65F's `11` large gadget upgrades:
  1. Loads ALL global + `W64G` upgraded gadget `.npz` files into `G` from the start.
  2. Forges the untested `[10:20]` slice of `14 <= M <= 145` chain gadgets (ab-initio disk gas + half-grid + melt)
     AND `3` large `146 <= M <= 560` active strip gadgets, saving all upgrades to `w66g_gadgets_s{shard}.npz`!
  3. Sweeps `fdp8.trace_and_compose(..., top_m=6)` with full `L_est < L0 + 0.020` margin (fixing `W64G`'s `0.002`
     throttle when `upgraded_gadgets` was non-empty!).
  4. Dedicates full `~550s` (`6` uninterrupted rounds of Continuous-Momentum WideHopper) to EACH of the top `2`
     novel multi-gadget basins (`max 2` queued so rounds 3..6 are never cut off!).
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
import w53g_rect_quench_8sym_faultdp as w53g
import w53h_nakajima_reharden_and_softinsert as w53h
import w56g_unified_global_faultdp as w56g
import w63k_1puzzle_per_shard_killshot as w63k
import w64g_abinitio_nakajima_rect_strip_forge as w64g
import w65f_hydraulic_vise_channel_and_large_strip as w65f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--global-gadgets-npz", required=True)
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=22)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1800)
    ap.add_argument("--seed-off", type=int, default=2080000)
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
        gap_ldr = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gap_ldr > 6.0e-5:
            all_targets.append((gap_ldr / math.sqrt(n), n, K, D))
    all_targets.sort(reverse=True)

    dp_targets = [
        t for t in all_targets[:240] if t[2] <= fdp8.WMAX and t[3] <= fdp8.DMAX
    ]
    chain_keys = w53g.collect_chain_gadget_keys(dp0, PG0, PM0, PQ0, G, dp_targets)

    # Wave-2 slice `[10:18]` of small/medium chain keys (`12 <= M <= 145`) + top-4 global chain keys with fresh seeds!
    small_keys = [k for k in chain_keys if 12 <= (k[0] * k[1] + k[2]) <= 145]
    my_small = small_keys[args.shard :: args.num_shards][8:16]
    for gk in small_keys[:4]:
        if gk not in my_small:
            my_small.append(gk)

    large_keys = [k for k in chain_keys if 146 <= (k[0] * k[1] + k[2]) <= 560]
    my_large = large_keys[args.shard :: args.num_shards][:3]

    ab_jobs = []
    for ji, k in enumerate(my_small):
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
        ab_jobs.append(
            (
                k,
                float(best_r[3]),
                sq_g,
                args.seed_off + args.shard * 5107 + ji * 137,
            )
        )

    lg_jobs = []
    for li, k in enumerate(my_large):
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
            lg_jobs.append(
                (
                    k,
                    float(best_r[3]),
                    sq_g,
                    args.seed_off + 30000 + args.shard * 3911 + li * 89,
                )
            )

    print(
        f"[W66G-UnifiedDP+Forge2 s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"gad_files={len(gad_files)} G={len(G)} ab_jobs={len(ab_jobs)} lg_jobs={len(lg_jobs)} ({time.time() - t0:.1f}s)",
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
                f"  [W66G UNIFIED DP WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    upgraded_gadgets = []
    ex_fg = ProcessPoolExecutor(max_workers=args.procs)
    try:
        futs_a = [ex_fg.submit(w64g.forge_abinitio_rect_job, jb) for jb in ab_jobs]
        futs_l = [ex_fg.submit(w65f.eval_large_strip_gadget_job, jb) for jb in lg_jobs]
        for fut_grp, t_lim in ((futs_a, 340.0), (futs_l, 160.0)):
            try:
                for f in as_completed(fut_grp, timeout=t_lim):
                    key, f_ref, best_f, best_sq, _ = f.result()
                    if best_sq is not None and best_f < f_ref - 1e-7:
                        upgraded_gadgets.append((key, f_ref, best_f, best_sq))
                        print(
                            f"  [W66G GADGET UPGRADE!!] {key[0]}x{key[1]}+{key[2]}: {f_ref:.6f} -> {best_f:.6f} (df={best_f - f_ref:+.6f})!",
                            flush=True,
                        )
            except TimeoutError:
                pass
    finally:
        w51d.terminate_pool(ex_fg)

    if upgraded_gadgets:
        gad_pl = {
            "keys": np.array([k for k, _, _, _ in upgraded_gadgets], dtype=np.int32)
        }
        for gi, (_, _, _, sq1) in enumerate(upgraded_gadgets):
            gad_pl[f"sq_{gi}"] = np.asarray(sq1, dtype=np.float64)
        np.savez_compressed(
            os.path.join(
                os.path.dirname(args.out_npz), f"w66g_gadgets_s{args.shard}.npz"
            ),
            **gad_pl,
        )
        new_dons = [sq1 for _, _, _, sq1 in upgraded_gadgets]
        Gn = fdp8.extract_gadgets(new_dons)
        if len(Gn):
            Gn[:, 6] += len(donors)
            G = np.vstack([G, Gn])
            donors = donors + new_dons
            dp0, PG0, PM0, PQ0 = fdp8.solve_dp(G)

    # Sweep shard's full target slice with `top_m=6` and full `L_est < L0 + 0.020`!
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
                        record_win_n(n, Qp, Sp, dtag + "_dp", 0)
                        Qc, Sc = Qp, Sp
                    ad = w53h.topology_angle_dist(Qc, cur[n][1])
                    if Sc < L0 + 0.0130 and ad > 0.040:
                        full_cands.append((Sc - L0, n, dtag, Sc, ad, Qc))

    # Also quick zero-quench check across ALL targets if we upgraded a gadget (`L_est < L0 + 0.0015`)!
    if upgraded_gadgets:
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
                if L_est < L0 + 0.0015:
                    rh = w56g.quench_composed_chain(P_comp, L0)
                    if rh is not None:
                        Qc, Sc = rh
                        if Sc < L0 + 0.00025:
                            Qp, Sp = w51d.deep_contact_polish(
                                n, Qc, deadline, slp_sec=36.0, fq_sec=22
                            )
                            record_win_n(n, Qp, Sp, dtag + "_glbdp", 0)
                            Qc, Sc = Qp, Sp
                        ad = w53h.topology_angle_dist(Qc, cur[n][1])
                        if Sc < L0 + 0.0080 and ad > 0.040:
                            full_cands.append((Sc - L0, n, dtag + "_g", Sc, ad, Qc))

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

    # Zero-idle guard: ensure `2` full-budget (`~540s` each!) basins per shard!
    for _, nn, _, _ in my_sweep:
        if len(queued) >= 2:
            break
        if seen_n.get(nn, 0) >= 1:
            continue
        L0_fb = wins[nn][0] if nn in wins else cur[nn][0]
        P0_fb = cur[nn][1]
        fz = RE.build_far_grid_freeze_mask(P0_fb, buffer_dist=3.5, min_free=85)
        rhp = RE.run_packer_hop(
            P0_fb,
            L0_fb - 2.5e-4,
            seed=args.seed_off + 77000 + args.shard * 907 + nn,
            iters=35000,
            frozen_mask=fz,
            timeout_s=28.0,
        )
        if rhp is not None and rhp[0] < L0_fb + 0.012:
            queued.append(
                (
                    rhp[0] - L0_fb,
                    nn,
                    rhp[2],
                    rhp[0],
                    w53h.topology_angle_dist(rhp[1], P0_fb),
                    rhp[1],
                )
            )
        else:
            queued.append((0.0, nn, "cur_direct", L0_fb, 0.0, P0_fb))
        seen_n[nn] = 1

    print(
        f"[W66G s={args.shard}] upgraded={len(upgraded_gadgets)} screened={len(full_cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
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
                f"  [W66G done] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W66G s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
