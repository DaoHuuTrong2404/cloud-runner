#!/usr/bin/env python3
"""Wave 55H: Multi-Physics Native-Rust Soft-to-Rigid Sniper (`n=234, 333` at `+8e-6`!)
+ High-N (`n=370..999`) Zero-Timeout Unjammer -> KKT-Polished 3R Wider-Hopper!

Solves BOTH findings from Wave 53H telemetry:
  1. Shards 0..5 (24 Sniper Cores): Dedicated `~950s` multi-seed escalations (`r1=320, r2=300, r3=300`)
     on the exact W53H ultra-near-misses (`234` +8e-6 x2 shards, `333` +8e-6 x2 shards, `303` +7.1e-5,
     `295` +7.2e-5, `338` +1.66e-4, `209` +0.0, `401`, `332`, `261`, `367`).
  2. Shards 6..15 (40 High-N Cores): Replaces W53H's slow Python `squeeze2 + compact_deleted` inside
     Phase A with native compiled Rust `fq quench --finish --stag-w 0 --deadline 14.0`, running 12
     diverse physics trajectories (`harden reharden` + `fq --round-r0 --round-press` + `fq --soft-a0`)
     in `<20s` per puzzle even at `N = 976` (zero Phase-A timeouts across all 60 top `n >= 370` targets!).
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
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r


SNIPER_SLOTS = [
    [234, 303],
    [333, 295],
    [234, 338],
    [333, 209],
    [401, 332],
    [261, 367],
]


def topology_angle_dist(P_a, P_b):
    aa = np.sort(
        np.abs(((P_a[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0)
    )
    ab = np.sort(
        np.abs(((P_b[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0)
    )
    return float(np.sum(np.abs(aa - ab)))


def eval_rust_soft_job(job_tup):
    """Run one soft-to-rigid physics trajectory entirely in compiled Rust (`~1.5 - 6.0s`!)."""
    kind, tag, P0, L0, seed, p1, p2, p3, p4 = job_tup
    try:
        if kind == "harden":
            # Ostashev momentum soft-disk hardening -> fast Rust fq KKT quench
            rh = RE.run_harden_reharden(
                P0,
                s_ref=L0,
                seed=seed,
                r0=p1,
                stiff0=p2,
                pre=1500,
                hold=6000,
                release=2800,
                settle=2000,
                hold_f=p3,
                noise=p4,
                sym_md=int(seed % 2),
                timeout_s=10.0,
            )
            if rh is None:
                return tag, 1e9, None
            _, Q_h = rh
            rq = RE.run_fq_quench_args(
                Q_h,
                [
                    "--loosen",
                    "1.0015",
                    "--outer",
                    "36",
                    "--pit",
                    "220",
                    "--flip-top",
                    "16",
                    "--finish",
                    "--stag-w",
                    "0",
                    "--deadline",
                    "11.0",
                ],
                s_ref=RE.side_exact(Q_h),
                timeout_s=13.0,
            )
            if rq is not None:
                return tag, float(rq[0]), rq[1]
            return tag, float(RE.side_exact(Q_h)), Q_h

        if kind == "fq_round":
            # Nakajima rounded-square ALM pressure/held hardening inside Rust fq
            extra = [
                "--seed",
                str(int(seed)),
                "--round-r0",
                repr(float(p1)),
                "--round-steps",
                "14",
                "--round-p",
                repr(float(p2)),
                "--hold-f",
                repr(float(p3)),
                "--round-kick",
                repr(float(p4)),
                "--round-kick-peak",
                "0.055",
                "--round-kick-w",
                "0.03",
                "--loosen",
                "1.002",
                "--outer",
                "38",
                "--pit",
                "220",
                "--flip-top",
                "16",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "13.5",
            ]
            if int(seed) % 2 == 0:
                extra += ["--round-press", "--round-press-outer", "14"]
            else:
                extra += ["--round-maxit", "1200", "--round-gtol", "1e-9"]
            rq = RE.run_fq_quench_args(P0, extra, s_ref=L0, timeout_s=15.5)
            if rq is None:
                return tag, 1e9, None
            return tag, float(rq[0]), rq[1]

        if kind == "fq_soft":
            # Deformable unit-area aspect-ratio `a_i = eps tanh(b_i)` soft squares -> rigid KKT
            extra = [
                "--seed",
                str(int(seed)),
                "--soft-a0",
                repr(float(p1)),
                "--soft-steps",
                "14",
                "--soft-p",
                repr(float(p2)),
                "--hold-f",
                repr(float(p3)),
                "--soft-kick",
                repr(float(p4)),
                "--soft-compress",
                "--loosen",
                "1.002",
                "--outer",
                "38",
                "--pit",
                "220",
                "--flip-top",
                "16",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "13.5",
            ]
            rq = RE.run_fq_quench_args(P0, extra, s_ref=L0, timeout_s=15.5)
            if rq is None:
                return tag, 1e9, None
            return tag, float(rq[0]), rq[1]
    except Exception:
        pass
    return tag, 1e9, None


def build_jobs_for_puzzle(n, P0, L0, bseed, sniper=False):
    jobs = []
    if sniper:
        hr_cfgs = [
            (0.09, 0.48, 0.9995, 0.0035),
            (0.12, 0.40, 0.9998, 0.0042),
            (0.15, 0.34, 1.0002, 0.0050),
            (0.18, 0.28, 1.0006, 0.0058),
            (0.21, 0.24, 1.0010, 0.0065),
            (0.14, 0.36, 1.0000, 0.0078),
        ]
        rd_cfgs = [
            (0.08, 1.0, 0.9998, 0.0030),
            (0.12, 1.2, 1.0002, 0.0040),
            (0.16, 1.0, 1.0006, 0.0050),
            (0.20, 1.3, 1.0012, 0.0055),
            (0.14, 0.9, 1.0000, 0.0065),
            (0.18, 1.1, 1.0008, 0.0045),
        ]
        sf_cfgs = [
            (0.05, 1.0, 1.0000, 0.0025),
            (0.09, 1.1, 1.0004, 0.0035),
            (0.13, 1.2, 1.0008, 0.0045),
            (0.17, 1.0, 1.0012, 0.0050),
        ]
    else:
        hr_cfgs = [
            (0.10, 0.45, 0.9996, 0.0038),
            (0.14, 0.35, 1.0000, 0.0048),
            (0.18, 0.28, 1.0006, 0.0058),
            (0.22, 0.22, 1.0012, 0.0065),
        ]
        rd_cfgs = [
            (0.09, 1.0, 0.9998, 0.0032),
            (0.14, 1.1, 1.0003, 0.0042),
            (0.18, 1.2, 1.0008, 0.0052),
            (0.12, 1.0, 1.0000, 0.0060),
        ]
        sf_cfgs = [
            (0.06, 1.0, 1.0000, 0.0028),
            (0.11, 1.1, 1.0005, 0.0038),
            (0.15, 1.2, 1.0010, 0.0045),
            (0.08, 1.0, 1.0002, 0.0035),
        ]

    for i, (r0, st0, hf, ns) in enumerate(hr_cfgs):
        jobs.append(("harden", f"rehard_{i}", P0, L0, bseed + 100 + i, r0, st0, hf, ns))
    for i, (r0, pw, hf, rk) in enumerate(rd_cfgs):
        jobs.append(
            ("fq_round", f"fqround_{i}", P0, L0, bseed + 200 + i, r0, pw, hf, rk)
        )
    for i, (a0, pw, hf, sk) in enumerate(sf_cfgs):
        jobs.append(("fq_soft", f"fqsoft_{i}", P0, L0, bseed + 300 + i, a0, pw, hf, sk))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2220)
    ap.add_argument("--seed-off", type=int, default=620000)
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

    is_sniper = args.shard < len(SNIPER_SLOTS)
    if is_sniper:
        my_slice = SNIPER_SLOTS[args.shard]
        r1_cfg, r2_cfg, r3_cfg = 320, 300, 300
        max_queued = 4
    else:
        hi_shard = args.shard - len(SNIPER_SLOTS)
        hi_num = max(1, args.num_shards - len(SNIPER_SLOTS))
        targets = []
        for n in range(370, 1001):
            K = math.isqrt(n)
            if K * K == n:
                continue
            gap_ldr = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
            if gap_ldr > 1.2e-4:
                targets.append((gap_ldr / math.sqrt(n), n))
        targets.sort(reverse=True)
        my_slice = [x[1] for x in targets[: hi_num * 6]][hi_shard::hi_num]
        r1_cfg, r2_cfg, r3_cfg = 250, 230, 230
        max_queued = 7

    mode_lbl = "SNIPER" if is_sniper else "HIGH-N"
    print(
        f"[W55H-{mode_lbl} s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_slice({len(my_slice)})={my_slice} ({time.time() - t0:.1f}s)",
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

    screened = []
    ex_rh = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_slice):
            if time.time() - t0 > 340.0:
                break
            L0 = cur[n][0]
            P0 = cur[n][1]
            bseed = args.seed_off + args.shard * 2311 + pi * 97
            jobs = build_jobs_for_puzzle(n, P0, L0, bseed, sniper=is_sniper)
            ts0 = time.time()
            futs = [ex_rh.submit(eval_rust_soft_job, jb) for jb in jobs]
            hits = []
            try:
                for f in as_completed(futs, timeout=105.0):
                    tg, S_q, Q_q = f.result()
                    if Q_q is None:
                        continue
                    gap0 = S_q - L0
                    adist = topology_angle_dist(Q_q, P0)
                    min_ad = 0.05 if is_sniper else 0.06
                    if gap0 < -1e-9 or (gap0 < 0.0135 and adist > min_ad):
                        hits.append((gap0, S_q, adist, tg, Q_q))
            except TimeoutError:
                w51d.terminate_pool(ex_rh)
                ex_rh = ProcessPoolExecutor(max_workers=args.procs)

            hits.sort(key=lambda z: z[0])
            b_str = (
                f"{hits[0][0]:+.6f}(dA={hits[0][2]:.2f})@{hits[0][3]}"
                if hits
                else "none_distinct"
            )
            print(
                f"  [W55H screen] n={n:4d} distinct_hits={len(hits)} best={b_str} ({time.time() - ts0:.1f}s)",
                flush=True,
            )
            picked = []
            for gap0, S_q, adist, tg, Q_q in hits:
                if all(abs(S_q - ps) > 1e-5 for _, ps, _, _, _ in picked):
                    picked.append((gap0, S_q, adist, tg, Q_q))
                if len(picked) >= 2:
                    break
            for gap0, S_q, adist, tg, Q_q in picked:
                screened.append((gap0, n, tg, S_q, adist, Q_q))
    finally:
        w51d.terminate_pool(ex_rh)

    screened.sort(key=lambda z: z[0])
    queued = screened[:max_queued]
    print(
        f"[W55H s={args.shard}] Phase B: hunting {len(queued)}/{len(screened)} distinct soft-unjammed basins ({time.time() - t0:.1f}s)!",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, tag, S_init, adist, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 50.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            if S_init < L0_cur + 0.00045:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    n, Q_init, deadline, slp_sec=42.0, fq_sec=28
                )
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        tot_ds += (Lf0 - L0_cur) / math.sqrt(n)
                        wins[n] = (float(Lf0), xf0, yf0, tf0)
                        L0_cur = float(Lf0)
                        print(
                            f"  [W55H DIRECT SOFT WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf0:.7f}!",
                            flush=True,
                        )
                        save_ckpt()

            r1 = min(r1_cfg, max(35, int(0.35 * (rem - 20.0))))
            r2 = min(r2_cfg, max(30, int(0.30 * (rem - 20.0))))
            r3 = min(r3_cfg, max(30, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + 40000 + args.shard * 11003 + i * 709
            Qb, Sb, (tr, ft, wn) = w51d.deep_3round_widehopper(
                ex_hop, n, Q_init, L0_cur, bseed, r1, r2, r3, deadline
            )
            if Sb < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, Qb)
                if ok and Lf < L0_cur - 1e-9:
                    ds = (Lf - L0_cur) / math.sqrt(n)
                    tot_ds += ds
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W55H HOP WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"(dL={Lf - cur[n][0]:+.7f}, dS={ds:+.8f}, dA={adist:.2f}, w={wn})",
                        flush=True,
                    )
                    save_ckpt()
                    if n - 1 >= 80:
                        L_m1 = wins[n - 1][0] if (n - 1) in wins else cur[n - 1][0]
                        if Lf < L_m1 + 0.020:
                            for didx in w32a.pick_near_band(Qb, max_cand=8)[:3]:
                                Qd, Sd = w32a.fast_delete_and_compact(Qb, didx)
                                if Qd is not None and Sd < L_m1 + 0.008:
                                    Qd_b, Sd_b, _ = w51d.deep_3round_widehopper(
                                        ex_hop,
                                        n - 1,
                                        Qd.astype(np.float64),
                                        L_m1,
                                        bseed + 91 + didx,
                                        95,
                                        80,
                                        75,
                                        deadline,
                                    )
                                    if Sd_b < L_m1 - 1e-9:
                                        ok2, Lf2, x2, y2, t2 = f2d.stabilize_decimal(
                                            n - 1, Qd_b
                                        )
                                        if ok2 and Lf2 < L_m1 - 1e-9:
                                            tot_ds += (Lf2 - L_m1) / math.sqrt(n - 1)
                                            wins[n - 1] = (float(Lf2), x2, y2, t2)
                                            L_m1 = float(Lf2)
                                            print(
                                                f"  [W55H CASCADE WIN!!] n={n - 1:4d} {cur[n - 1][0]:.7f} -> {Lf2:.7f}!",
                                                flush=True,
                                            )
                                            save_ckpt()
                else:
                    print(
                        f"  [W55H miss] n={n:4d} [{tag}] dA={adist:.2f} dStart={S_init - L0_cur:+.6f} -> dEnd={Sb - L0_cur:+.6f}",
                        flush=True,
                    )
            else:
                print(
                    f"  [W55H hunt] n={n:4d} [{tag}] dA={adist:.2f} dStart={S_init - L0_cur:+.6f} -> dEnd={Sb - L0_cur:+.6f} (tr={tr}, w={wn})",
                    flush=True,
                )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W55H s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
