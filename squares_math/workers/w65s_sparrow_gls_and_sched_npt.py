#!/usr/bin/env python3
"""Wave 65S (`w65s_sparrow_gls_and_sched_npt.py`): Sparrow Guided Local Search (`anneal gls`)
+ Lag-Controlled Regional NPT Monte Carlo (`anneal sched`) + Continuous WideHopper!

Unlocks TWO compiled Rust engines in `fq_crate/src/anneal.rs` NEVER invoked in Waves 1..64:
  1. `anneal gls` (Gardeyn & Wauters Sparrow Guided Local Search, lines 587-835):
     - Cuts along two random seams (`cut_x, cut_y`) to target side `S = L0 * (1 - shrink)`, concentrating
       all compression overlap cleanly along one cross-seam.
     - Alternates analytical penetration-gradient descent (`--relax 24`) + global/local repositioning
       (`--ng 44 --nl 24`) + `--ring 1` neighbor slack propagation + pairwise GLS penalty bumping (`1.2..2.0x`).
     - Any `feasible == true` trajectory is an INSTANT strict puzzle win at `S < L0` (iteratively chained!).
     - Any low-overlap (`overlap <= 0.045`) trajectory is healed via BOTH `fq quench --loosen 1.007` and
       Ostashev `harden reharden --hold-move 1` into novel basins (`adist > 0.03`)!
  2. `anneal sched` (Schedule-Driven Regional Rounded-Square NPT Monte Carlo, lines 423-585):
     - Uses `--lag-tol 0.0015 --lag-grow 0.0006` to eliminate residual rounding blowup at the end of the
       schedule while melting `(cx, cy, rad)` tilted-channel & step-transition pockets under high NPT pressure.
  3. Escalates top distinct `< L0 + 0.013` basins via 6-Round Continuous-Momentum `eta <= 0.28` WideHopper!
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

import rust_engines as RE
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w63k_1puzzle_per_shard_killshot as w63k


def eval_sparrow_gls_job(job_tup):
    n, P0, L0, shrink, seed0 = job_tup
    out_cands = []
    cur_P = RE.normalize_to_origin(P0)
    cur_L = float(L0)

    for step_i in range(3):
        rg = RE.run_anneal_gls(
            cur_P,
            shrink=shrink,
            ms=5200.0,
            seed=int(seed0) + step_i * 97,
            ng=44,
            nl=24,
            ring=1,
            relax=24,
            strike_iters=8,
            timeout_s=9.5,
        )
        if rg is None:
            break
        s_tgt, Q_raw, is_feas, ov = rg
        if is_feas:
            rq = RE.run_fq_quench_args(
                Q_raw,
                [
                    "--loosen",
                    "1.0003",
                    "--outer",
                    "25",
                    "--pit",
                    "450",
                    "--finish",
                    "--stag-w",
                    "0",
                    "--rmin",
                    "1e-12",
                    "--deadline",
                    "9.0",
                ],
                s_ref=s_tgt,
                timeout_s=11.5,
            )
            if rq is not None:
                Sf, Qf = float(rq[0]), rq[1]
                out_cands.append((n, Sf, f"gls_feas_d{shrink:.1e}_s{step_i}", Qf))
                if Sf < cur_L - 1e-8:
                    cur_P, cur_L = Qf, Sf
                    continue
        elif ov <= 0.048:
            # Heal near-feasible Sparrow GLS seam state via both `fq` and `harden reharden`!
            rq1 = RE.run_fq_quench_args(
                Q_raw,
                [
                    "--seed",
                    str(int(seed0) + 31),
                    "--loosen",
                    "1.007",
                    "--outer",
                    "48",
                    "--pit",
                    "280",
                    "--flip-top",
                    "20",
                    "--finish",
                    "--stag-w",
                    "0",
                    "--deadline",
                    "13.0",
                ],
                s_ref=s_tgt,
                timeout_s=15.5,
            )
            if rq1 is not None and float(rq1[0]) < L0 + 0.014:
                Sf1, Qf1 = RE.kkt_sqp_polish_fq(rq1[1], timeout_each=5.5)
                out_cands.append((n, float(Sf1), f"gls_fq_d{shrink:.1e}", Qf1))

            rh = RE.run_harden_reharden(
                Q_raw,
                s_ref=s_tgt,
                seed=int(seed0) + 59,
                r0=0.14,
                stiff0=0.30,
                pre=1600,
                hold=6500,
                release=3200,
                settle=2500,
                hold_f=1.0005,
                noise=0.004,
                mu=0.022,
                hold_move=1,
                timeout_s=16.0,
            )
            if rh is not None and float(rh[0]) < L0 + 0.018:
                Sf2, Qf2 = RE.kkt_sqp_polish_fq(rh[1], timeout_each=6.0)
                if Sf2 < L0 + 0.014:
                    out_cands.append((n, float(Sf2), f"gls_hrd_d{shrink:.1e}", Qf2))
        break
    return out_cands


def eval_sched_regional_job(job_tup):
    n, P0, L0, cx, cy, rad, r_peak, seed0 = job_tup
    rs = RE.run_anneal_sched_regional(
        P0,
        cx=cx,
        cy=cy,
        rad=rad,
        r_peak=r_peak,
        sweeps=6500,
        bp_curve="0:1400,0.5:3800,1:8500",
        lag_tol=0.0015,
        lag_grow=0.0006,
        seed=int(seed0),
        timeout_s=17.5,
    )
    if rs is None:
        return []
    S_mid, Q_mid, _ = rs
    if S_mid > L0 + 0.045:
        return []
    rq = RE.run_fq_quench_args(
        Q_mid,
        [
            "--seed",
            str(int(seed0) + 13),
            "--loosen",
            "1.003",
            "--outer",
            "42",
            "--pit",
            "260",
            "--flip-top",
            "20",
            "--finish",
            "--stag-w",
            "0",
            "--deadline",
            "11.5",
        ],
        s_ref=min(S_mid, L0 + 0.01),
        timeout_s=14.0,
    )
    if rq is None:
        return []
    Sf, Qf = RE.kkt_sqp_polish_fq(rq[1], timeout_each=5.5)
    if Sf < L0 + 0.014:
        return [(n, float(Sf), f"sched_r{rad:.1f}_p{r_peak:.2f}", Qf)]
    return []


def pick_channel_centers(P_in, rng):
    P = RE.normalize_to_origin(P_in)
    S = RE.side_exact(P)
    am = np.mod(P[:, 2], 0.5 * math.pi)
    tl = np.minimum(am, 0.5 * math.pi - am)
    til = np.where(tl > 0.012)[0]
    if len(til) == 0:
        return [(0.5 * S, 0.5 * S), (0.3 * S, 0.7 * S)]
    # Pick centroid + 2 extremal/random tilted centers near step transitions!
    tx, ty = P[til, 0], P[til, 1]
    pts = [(float(np.median(tx)), float(np.median(ty)))]
    order = np.argsort(tx + ty)
    pts.append((float(tx[order[len(order) // 5]]), float(ty[order[len(order) // 5]])))
    pts.append(
        (float(tx[order[4 * len(order) // 5]]), float(ty[order[4 * len(order) // 5]]))
    )
    idx_r = int(rng.choice(til))
    pts.append((float(P[idx_r, 0]), float(P[idx_r, 1])))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=22)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1800)
    ap.add_argument("--seed-off", type=int, default=1820000)
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

    ranked = []
    for n in range(150, 1001):
        K = math.isqrt(n)
        if K * K == n or cur[n][1] is None:
            continue
        gl = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gl > 5.0e-5:
            ranked.append((gl / math.sqrt(n), n))
    ranked.sort(reverse=True)
    my_targets = [x[1] for x in ranked[: args.num_shards * 4]][
        args.shard :: args.num_shards
    ]

    print(
        f"[W65S-SparrowGLS+SchedNPT s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_targets={my_targets} ({time.time() - t0:.1f}s)",
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
                f"  [W65S SPARROW/NPT WIN!!] n={n_w:4d} [{src_tag}] {cur[n_w][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={w_cnt})!",
                flush=True,
            )
            save_ckpt()

    rng = np.random.default_rng(args.seed_off + args.shard * 997)
    gls_jobs = []
    sch_jobs = []
    for ti, n in enumerate(my_targets):
        P0 = cur[n][1]
        L0 = cur[n][0]
        bseed = args.seed_off + args.shard * 4211 + ti * 211
        for si, sh in enumerate((1.5e-4, 4.5e-4, 1.1e-3, 2.4e-3, 6.5e-4, 1.6e-3)):
            gls_jobs.append((n, P0, L0, sh, bseed + si * 17))
        centers = pick_channel_centers(P0, rng)
        for ci, ((cx, cy), (rad, rp)) in enumerate(
            zip(centers[:4], [(3.0, 0.13), (4.2, 0.17), (3.4, 0.15), (4.8, 0.20)])
        ):
            sch_jobs.append((n, P0, L0, cx, cy, rad, rp, bseed + 500 + ci * 29))

    screened = []
    ex_sc = ProcessPoolExecutor(max_workers=args.procs)
    try:
        futs = [ex_sc.submit(eval_sparrow_gls_job, jb) for jb in gls_jobs] + [
            ex_sc.submit(eval_sched_regional_job, jb) for jb in sch_jobs
        ]
        try:
            for f in as_completed(futs, timeout=560.0):
                for nn, Sq, tg, Qq in f.result():
                    L0 = wins[nn][0] if nn in wins else cur[nn][0]
                    if Sq < L0 - 1e-9:
                        Qp0, Sp0 = w51d.deep_contact_polish(
                            nn, Qq, deadline, slp_sec=40.0, fq_sec=24
                        )
                        record_win_n(nn, Qp0, Sp0, tg + "_dir", 0)
                        Qq, Sq = Qp0, Sp0
                    ad = w53h.topology_angle_dist(Qq, cur[nn][1])
                    if Sq < L0 + 0.0135 and ad > 0.028:
                        screened.append((Sq - L0, nn, tg, Sq, ad, Qq))
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

    # Zero-idle guard: top up `queued` to >= 3 using frozen-grid `packer hop` on `my_targets`!
    for fb_i, nn in enumerate(my_targets):
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
            seed=args.seed_off + 70000 + args.shard * 907 + fb_i * 37,
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
        f"[W65S s={args.shard}] screened={len(screened)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
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
                f"  [W65S done] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} r={nr} w={wt}",
                flush=True,
            )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W65S s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
