#!/usr/bin/env python3
"""Wave 64G (`w64g_abinitio_nakajima_rect_strip_forge.py`): True Ab-Initio Liquid-to-Square
Nakajima Pressure-Hardening (`--round-r0 0.42 --round-press --round-kick-peak 0.16`) Strip & Gadget Forge!

Architectural pivot after `W59..W63` proved all self-frontier transplants (`cur -> cur`) converge to `dEnd = +0.000000`
at algebraic gadget floors (e.g. `L_234 = 14.5 + sqrt(7)/2 = 15.82287566`):
  - Synthesizes `(w, h, dm)` sub-rectangle gadgets AND `(K, h_strip, D)` strips **FROM SCRATCH (`AB-INITIO`)**
    without inheriting `cur[n]`'s trapped fault-line topology!
  - Uses Evan's compiled Rust Nakajima pressure-driven disk-to-square engine (`fq.rs` lines 2038-2069:
    `--rect-dx (w-m) --rect-dy (h-m) --round-r0 0.42 --round-steps 20 --round-press --round-press-outer 10
    --round-kick 0.014 --round-kick-peak 0.16 --round-kick-w 0.05 --finish --stag-w 0`) starting from:
    (1) pure uniform disk gas (`r0 = 0.42`),
    (2) corner sub-grid + liquid disk channel, and
    (3) high-temperature melted existing gadget (`--kick 0.16`)!
  - Directly glues `(K, h_strip, D)` strip breakthroughs onto `K x (K - h_strip)` upright blocks AND updates
    Global 4D 8-Sym `fault_dp_8sym.solve_dp`, escalating all novel `< L0 + 0.012` basins via Continuous-Momentum
    `eta <= 0.28` LP WideHopper + KKT `fq --sqp --finish` + `Decimal64`!
"""

import argparse
import json
import math
import os
import random
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


def make_abinitio_rect_start(w, h, dm, f_guess, mode, rng):
    """Generate an ab-initio `M = w * h + dm` configuration inside `[0, w + f_guess] x [0, h + f_guess]`."""
    M = int(w) * int(h) + int(dm)
    W_box = float(w) + float(f_guess)
    H_box = float(h) + float(f_guess)
    rows = []
    if mode == "half_grid" and w >= 6 and h >= 5:
        gw = max(2, int(w) - 5)
        gh = max(1, int(h) - 3)
        for ix in range(gw):
            for iy in range(gh):
                if len(rows) < M - max(8, dm + 6):
                    rows.append((0.5 + ix, 0.5 + iy, 0.0))
    while len(rows) < M:
        rx = rng.uniform(0.52, max(0.6, W_box - 0.52))
        ry = rng.uniform(0.52, max(0.6, H_box - 0.52))
        rt = rng.uniform(0.0, 0.5 * math.pi)
        rows.append((rx, ry, rt))
    return np.asarray(rows[:M], dtype=np.float64)


def forge_abinitio_rect_job(job_tup):
    key, f_ref, sq_existing, seed0 = job_tup
    w, h, dm = int(key[0]), int(key[1]), int(key[2])
    M = w * h + dm
    m_min = min(w, h)
    dx, dy = float(w - m_min), float(h - m_min)
    rng = random.Random(int(seed0))

    best_f = float(f_ref)
    best_sq = None
    cands_out = []

    starts = [
        (
            "gas0",
            make_abinitio_rect_start(w, h, dm, min(0.85, f_ref), "gas", rng),
            0.43,
            0.015,
        ),
        (
            "gas1",
            make_abinitio_rect_start(w, h, dm, min(0.85, f_ref), "gas", rng),
            0.38,
            0.012,
        ),
        (
            "hgrid",
            make_abinitio_rect_start(w, h, dm, min(0.85, f_ref), "half_grid", rng),
            0.40,
            0.013,
        ),
    ]
    if sq_existing is not None and len(sq_existing) == M:
        P_ex = RE.normalize_to_origin(sq_existing)
        # Melt with large perturbation so it escapes its old algebraic minimum!
        P_melt = P_ex.copy()
        for i in range(M):
            P_melt[i, 0] += rng.gauss(0.0, 0.14)
            P_melt[i, 1] += rng.gauss(0.0, 0.14)
            P_melt[i, 2] += rng.gauss(0.0, 0.35)
        starts.append(("melt", P_melt, 0.34, 0.011))

    for si, (stag, P_init, r0_val, rk_val) in enumerate(starts):
        s_in = float(m_min) + min(0.88, float(f_ref) + 0.04)
        extra = [
            "--rect-dx",
            repr(dx),
            "--rect-dy",
            repr(dy),
            "--seed",
            str(int(seed0) + si * 73),
            "--round-r0",
            repr(float(r0_val)),
            "--round-steps",
            "20",
            "--round-press",
            "--round-press-outer",
            "10",
            "--round-kick",
            repr(float(rk_val)),
            "--round-kick-peak",
            "0.16",
            "--round-kick-w",
            "0.05",
            "--loosen",
            "1.008",
            "--outer",
            "55",
            "--pit",
            "280",
            "--flip-top",
            "22",
            "--finish",
            "--stag-w",
            "0",
            "--deadline",
            "24.0",
        ]
        rq = RE.run_fq_quench_args(P_init, extra, s_ref=s_in, timeout_s=27.5)
        if rq is None:
            continue
        _, Q_cand, _ = rq
        if len(Q_cand) != M:
            continue
        Q_cand = RE.normalize_to_origin(Q_cand)
        fc = RE.gadget_overhang(Q_cand, w, h)
        if fc < f_ref + 0.013:
            cands_out.append((fc, stag, Q_cand))
        if fc < best_f - 1e-8:
            best_f, best_sq = fc, Q_cand

    cands_out.sort(key=lambda z: z[0])
    return key, float(f_ref), best_f, best_sq, cands_out[:2]


def glue_strip_onto_grid(K, h_strip, Q_strip):
    """Glue a `(K, h_strip, D)` strip (`[0, K+f] x [0, h_strip+f]`) onto an upright `K x (K - h_strip)` bottom block!"""
    rows_g = K - h_strip
    if rows_g <= 0:
        return RE.normalize_to_origin(Q_strip)
    grid = np.array(
        [(0.5 + ix, 0.5 + iy, 0.0) for ix in range(K) for iy in range(rows_g)],
        dtype=np.float64,
    )
    top = RE.normalize_to_origin(Q_strip)
    top[:, 1] += float(rows_g)
    return np.vstack([grid, top])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--global-gadgets-npz", required=True)
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=22)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1820)
    ap.add_argument("--seed-off", type=int, default=1580000)
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
    if os.path.exists(args.global_gadgets_npz):
        with np.load(args.global_gadgets_npz) as zg:
            if "keys" in zg.files:
                for i in range(len(zg["keys"])):
                    donors.append(np.asarray(zg[f"sq_{i}"], dtype=np.float64))

    G = fdp8.extract_gadgets(donors)
    dp0, PG0, PM0, PQ0 = fdp8.solve_dp(G)

    all_targets = []
    for n in range(90, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D <= 0:
            continue
        gap_ldr = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gap_ldr > 8.0e-5:
            all_targets.append((gap_ldr / math.sqrt(n), n, K, D))
    all_targets.sort(reverse=True)

    # 1) Build ab-initio strip jobs `(K, h_strip, D)` for moderate M <= 165 + active sub-rectangle DP gadgets!
    forge_jobs = []
    my_strip_targets = all_targets[args.shard :: args.num_shards][:5]
    j_id = 0
    for _, n, K, D in my_strip_targets:
        f_cur = cur[n][0] - K
        for hs in (5, 6, 7):
            if hs < K and (K * hs + D) <= 170:
                forge_jobs.append(
                    (
                        (K, hs, D),
                        f_cur,
                        None,
                        args.seed_off + args.shard * 4111 + j_id * 131,
                    )
                )
                j_id += 1

    dp_targets = [
        t for t in all_targets[:220] if t[2] <= fdp8.WMAX and t[3] <= fdp8.DMAX
    ]
    chain_keys = w53g.collect_chain_gadget_keys(dp0, PG0, PM0, PQ0, G, dp_targets)
    small_keys = [k for k in chain_keys if 14 <= (k[0] * k[1] + k[2]) <= 125]
    for k in small_keys[args.shard :: args.num_shards][:10]:
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
        forge_jobs.append(
            (
                k,
                float(best_r[3]),
                sq_g,
                args.seed_off + 20000 + args.shard * 5003 + j_id * 97,
            )
        )
        j_id += 1

    print(
        f"[W64G-AbInitioNakajima s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"forge_jobs={len(forge_jobs)} ({time.time() - t0:.1f}s)",
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
                f"  [W64G AB-INITIO WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    upgraded_gadgets = []
    full_cands = []
    ex_fg = ProcessPoolExecutor(max_workers=args.procs)
    try:
        futs = [ex_fg.submit(forge_abinitio_rect_job, jb) for jb in forge_jobs]
        try:
            for f in as_completed(futs, timeout=580.0):
                key, f_ref, best_f, best_sq, c_list = f.result()
                w, h, dm = key
                if best_sq is not None and best_f < f_ref - 1e-7:
                    upgraded_gadgets.append((key, f_ref, best_f, best_sq))
                    print(
                        f"  [W64G GADGET UPGRADE!!] {w}x{h}+{dm}: {f_ref:.6f} -> {best_f:.6f} (df={best_f - f_ref:+.6f})!",
                        flush=True,
                    )
                # If `(w, h, dm)` is a `(K, h_strip, D)` strip for `n = K^2 + D <= 1000`, glue onto `K x (K - h)`!
                n_glued = w * w + dm
                if w > h and n_glued <= 1000 and math.isqrt(n_glued) == w:
                    L0_n = wins[n_glued][0] if n_glued in wins else cur[n_glued][0]
                    for fc, stag, Q_st in c_list:
                        Q_full = glue_strip_onto_grid(w, h, Q_st)
                        if len(Q_full) == n_glued:
                            S_full = RE.side_exact(Q_full)
                            if S_full < L0_n - 1e-9:
                                Qp0, Sp0 = w51d.deep_contact_polish(
                                    n_glued, Q_full, deadline, slp_sec=40.0, fq_sec=24
                                )
                                record_win_n(
                                    n_glued, Qp0, Sp0, f"strip_{w}x{h}_{stag}", 0
                                )
                            ad = w53h.topology_angle_dist(Q_full, cur[n_glued][1])
                            if S_full < L0_n + 0.0135 and ad > 0.035:
                                full_cands.append(
                                    (
                                        S_full - L0_n,
                                        n_glued,
                                        f"strip_{w}x{h}_{stag}",
                                        S_full,
                                        ad,
                                        Q_full,
                                    )
                                )
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_fg)

    # Re-solve Global 8-Sym Fault-DP if any gadget upgraded (plus check shard's novel DP chains)!
    if upgraded_gadgets:
        gad_pl = {
            "keys": np.array([k for k, _, _, _ in upgraded_gadgets], dtype=np.int32)
        }
        for gi, (_, _, _, sq1) in enumerate(upgraded_gadgets):
            gad_pl[f"sq_{gi}"] = np.asarray(sq1, dtype=np.float64)
        np.savez_compressed(
            os.path.join(
                os.path.dirname(args.out_npz), f"w64g_gadgets_s{args.shard}.npz"
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

    sweep_list = (
        all_targets if upgraded_gadgets else all_targets[args.shard :: args.num_shards]
    )
    for _, n, K, D in sweep_list:
        if time.time() - t0 > 740.0:
            break
        if K > fdp8.WMAX or D > fdp8.DMAX:
            continue
        L0 = wins[n][0] if n in wins else cur[n][0]
        dp_chains = fdp8.trace_and_compose(dp0, PG0, PM0, PQ0, G, donors, K, D, top_m=3)
        for L_est, dtag, P_comp in dp_chains:
            if L_est < L0 + (
                0.002 if upgraded_gadgets and len(sweep_list) > 60 else 0.016
            ):
                rh = w56g.quench_composed_chain(P_comp, L0)
                if rh is not None:
                    Qc, Sc = rh
                    if Sc < L0 + 0.00025:
                        Qp, Sp = w51d.deep_contact_polish(
                            n, Qc, deadline, slp_sec=36.0, fq_sec=22
                        )
                        record_win_n(n, Qp, Sp, dtag + "_dp", 0)
                        Qc, Sc = Qp, Sp
                    ad = w53h.topology_angle_dist(Qc, cur[n][1])
                    if Sc < L0 + 0.0125 and ad > 0.045:
                        full_cands.append((Sc - L0, n, dtag, Sc, ad, Qc))

    full_cands.sort(key=lambda z: z[0])
    queued = []
    seen_n = {}
    for item in full_cands:
        nn = item[1]
        if seen_n.get(nn, 0) >= 2:
            continue
        seen_n[nn] = seen_n.get(nn, 0) + 1
        queued.append(item)
        if len(queued) >= 4:
            break

    print(
        f"[W64G s={args.shard}] upgraded_gadgets={len(upgraded_gadgets)} queued_basins={len(queued)} ({time.time() - t0:.1f}s)",
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
                f"  [W64G done] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W64G s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
