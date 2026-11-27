#!/usr/bin/env python3
"""Wave 55S: Ultra-Sniper CC-Bar Cut Escalation (`418 +3e-6`, `756 +7e-6`, `924 +1.0e-5`, `465 +3.6e-5`!)
+ `bandflip` Mixed<->Same-Tilt & Wall-Relocation Macro-Junction Surgery!

80 Free Public Cores (`20` shards x `4` cores):
  - Shards 0..11 (48 Cut-Sniper Cores): Dedicated `~950s/pair` (`r1=320, r2=300, r3=300`) KKT-polished
    Wider-Hopper on the exact ultra-near-miss `(want_n, donor_n)` cut jumpers from W52S/W53S + R33/W54 winners!
  - Shards 12..19 (32 BandFlip Surgery Cores): Applies `bandflip.flip` (mixed-tilt <-> same-tilt junction
    reflection) and `bandflip.relocate` (180-deg opposite-wall swap) quenched via Rust `fq --round-r0 0.07
    --finish --stag-w 0` across the Top 64 two-band high-gap puzzles -> 3R KKT Wider-Hopper!
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

import bandflip as BF
import rust_engines as RE
import squeeze2 as sq2
import sq_engine as E
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w52s_staggered_band_quench_transplanter as w52s
import w53s_ccbar_reharden_splicer as w53s
import w54s_fast_stag_splicer_and_sniper as w54s


# 12 Cut-Sniper shards x 2 pairs/shard = 24 ultra-hot cut targets!
# (want_n == 0 means take the sharpest cut of any target size produced by donor_n)
SNIPER_CUT_SLOTS = [
    [(418, 461), (652, 705)],
    [(756, 813), (871, 932)],
    [(924, 987), (754, 811)],
    [(465, 510), (811, 931)],
    [(418, 461), (0, 380)],
    [(756, 813), (0, 702)],
    [(652, 705), (0, 758)],
    [(924, 987), (870, 931)],
    [(465, 510), (812, 932)],
    [(701, 813), (929, 992)],
    [(549, 598), (376, 417)],
    [(689, 744), (377, 461)],
]


def topology_angle_dist(P_a, P_b):
    aa = np.sort(
        np.abs(((P_a[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0)
    )
    ab = np.sort(
        np.abs(((P_b[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0)
    )
    return float(np.sum(np.abs(aa - ab)))


def eval_raw_cut_multi(args_tuple):
    """Quench a raw spliced packing via BOTH fast Numba `squeeze2` AND Rust `harden + fq`!"""
    tag, P_raw, L_ref, seed = args_tuple
    best_Q = None
    best_S = 1e9
    try:
        rq = w54s.fast_quench_v2(P_raw, L_ref)
        if rq is not None and rq[1] < best_S:
            best_Q, best_S = rq[0], float(rq[1])
    except Exception:
        pass
    try:
        rh = RE.run_harden_reharden(
            P_raw,
            s_ref=max(RE.side_exact(P_raw), L_ref * 1.002),
            seed=seed,
            r0=0.14,
            stiff0=0.32,
            pre=1400,
            hold=5200,
            release=2400,
            settle=1800,
            hold_f=1.0005,
            noise=0.004,
            timeout_s=9.5,
        )
        if rh is not None:
            rfq = RE.run_fq_quench_args(
                rh[1],
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
                    "10.5",
                ],
                s_ref=rh[0],
                timeout_s=12.5,
            )
            if rfq is not None and rfq[0] < best_S:
                best_Q, best_S = rfq[1], float(rfq[0])
    except Exception:
        pass
    return tag, best_S, best_Q


def collect_sniper_cuts(ex, want_n, donor_n, P_donor, cur_L_map, bseed):
    """Generate both standard `generate_fast_splices` AND targeted interior bar cuts matching `want_n`!"""
    cands = []
    # 1) Fast multi-mode splices from `w54s`
    base_list = w54s.generate_fast_splices((donor_n, P_donor, cur_L_map))
    for nn, S, tag, Q in base_list:
        if want_n == 0 or nn == want_n:
            cands.append((S - cur_L_map.get(nn, S), nn, tag, S, Q))

    # 2) Targeted interior bar-pair cuts aimed specifically at `want_n` (or `want_n + 1`)
    if want_n > 0:
        P = E.to_origin(P_donor)
        ang = ((P[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
        ip = w53s.extract_bars_cc(P, np.where(ang > 0.03)[0])
        ineg = w53s.extract_bars_cc(P, np.where(ang < -0.03)[0])
        raw_jobs = []
        if ip is not None and ineg is not None and ip[0] != ineg[0]:
            bp, bn = ip[2], ineg[2]
            for kp in range(2, len(bp) - 2):
                for kn in range(2, len(bn) - 2):
                    P_wc = w54s.exact_wall_cut_fast(P, ip, ineg, kp, kn)
                    if P_wc is not None and len(P_wc) in (want_n, want_n + 1):
                        raw_jobs.append(
                            (
                                f"snipecut_{donor_n}_k{kp}x{kn}",
                                P_wc,
                                cur_L_map.get(len(P_wc), math.sqrt(len(P_wc)) + 0.6),
                                bseed + kp * 31 + kn,
                            )
                        )
                    if len(raw_jobs) >= 12:
                        break
                if len(raw_jobs) >= 12:
                    break
            for kp in range(2, len(bp) - 2):
                for kn in range(2, len(bn) - 2):
                    P_tb = w52s.cut_two_band_staggered(P, ip, ineg, kp, kn)
                    if P_tb is not None and len(P_tb) in (want_n, want_n + 1):
                        raw_jobs.append(
                            (
                                f"snipetb_{donor_n}_k{kp}x{kn}",
                                w54s.relocate_crushed_fast(P_tb),
                                cur_L_map.get(len(P_tb), math.sqrt(len(P_tb)) + 0.6),
                                bseed + 500 + kp * 29 + kn,
                            )
                        )
                    if len(raw_jobs) >= 20:
                        break
                if len(raw_jobs) >= 20:
                    break
        if raw_jobs:
            futs = [ex.submit(eval_raw_cut_multi, jb) for jb in raw_jobs[:16]]
            try:
                for f in as_completed(futs, timeout=75.0):
                    tg, Sq, Qq = f.result()
                    if Qq is None:
                        continue
                    nn = len(Qq)
                    if nn == want_n and Sq < cur_L_map.get(nn, Sq) + 0.014:
                        cands.append((Sq - cur_L_map[nn], nn, tg, Sq, Qq))
                    elif nn == want_n + 1 and Sq < cur_L_map.get(nn, Sq) + 0.015:
                        for didx in w32a.pick_near_band(Qq, max_cand=4)[:2]:
                            Qd, Sd = w32a.fast_delete_and_compact(Qq, didx)
                            if Qd is not None and float(Sd) < cur_L_map[want_n] + 0.014:
                                cands.append(
                                    (
                                        float(Sd) - cur_L_map[want_n],
                                        want_n,
                                        tg + "_d1",
                                        float(Sd),
                                        Qd.astype(np.float64),
                                    )
                                )
            except TimeoutError:
                pass

    cands.sort(key=lambda z: z[0])
    picked = []
    for item in cands:
        if all(not (item[1] == p[1] and abs(item[3] - p[3]) < 1e-5) for p in picked):
            picked.append(item)
        if len(picked) >= 2:
            break
    return picked


def eval_bandflip_job(args_tuple):
    """Run one `bandflip.flip` or `bandflip.relocate` surgery + native Rust `fq` soft-prestage quench!"""
    op, bi, P0_rad, L0, seed = args_tuple
    try:
        P_norm = RE.normalize_to_origin(P0_rad)
        S_norm = RE.side_exact(P_norm)
        P_deg = P_norm.copy()
        P_deg[:, 2] = np.degrees(P_deg[:, 2]) % 90.0
        B = BF.bands(P_deg, min_size=12)
        if len(B) <= bi:
            return f"{op}{bi}", 1e9, None
        rng = random.Random(int(seed))
        if op == "flip":
            sq_deg, _ = BF.flip(S_norm, P_deg, B[bi], rng, B)
        else:
            sq_deg, _ = BF.relocate(S_norm, P_deg, B[bi], rng, B)
        Q_raw = np.asarray(sq_deg, dtype=np.float64)
        Q_raw[:, 2] = np.radians(Q_raw[:, 2])
        Q_raw = w54s.relocate_crushed_fast(Q_raw)
        rq = RE.run_fq_quench_args(
            Q_raw,
            [
                "--seed",
                str(int(seed)),
                "--round-r0",
                "0.075",
                "--round-steps",
                "8",
                "--round-maxit",
                "1000",
                "--round-gtol",
                "1e-8",
                "--hold-f",
                "1.004",
                "--loosen",
                "1.022",
                "--outer",
                "44",
                "--pit",
                "240",
                "--flip-top",
                "16",
                "--finish",
                "--stag-w",
                "0",
                "--deadline",
                "13.5",
            ],
            s_ref=L0,
            timeout_s=15.5,
        )
        if rq is None:
            return f"{op}{bi}", 1e9, None
        return f"{op}{bi}_s{seed % 100}", float(rq[0]), rq[1]
    except Exception:
        return f"{op}{bi}", 1e9, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2220)
    ap.add_argument("--seed-off", type=int, default=635000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    _dm = np.array(
        [[0.5, 0.5, 0.0], [1.6, 0.5, 0.1], [0.5, 1.6, 0.0]], dtype=np.float64
    )
    sq2.squeeze(_dm[:, 0], _dm[:, 1], _dm[:, 2], max_iter=20, max_tries=3)

    cur, fresh, tot_score, _ = w51r.load_full_closed_frontier(
        args.union_npz, args.win_glob
    )
    cur_L_map = {n: cur[n][0] for n in range(1, 1001)}
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

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

    is_cut_sniper = args.shard < len(SNIPER_CUT_SLOTS)
    queued = []

    if is_cut_sniper:
        pairs = list(SNIPER_CUT_SLOTS[args.shard])
        print(
            f"[W55S-CutSniper s={args.shard}/{args.num_shards}] score={tot_score:.10f} pairs={pairs} ({time.time() - t0:.1f}s)",
            flush=True,
        )
        ex_sc = ProcessPoolExecutor(max_workers=args.procs)
        try:
            for pi, (want_n, donor_n) in enumerate(pairs):
                if cur[donor_n][1] is None:
                    continue
                bseed = args.seed_off + args.shard * 1709 + pi * 313
                found = collect_sniper_cuts(
                    ex_sc, want_n, donor_n, cur[donor_n][1], cur_L_map, bseed
                )
                print(
                    f"  [W55S cut-screen] pair=({want_n}<-{donor_n}) found={[(x[1], round(x[0], 6), x[2]) for x in found]}",
                    flush=True,
                )
                queued.extend(found)
        finally:
            w51d.terminate_pool(ex_sc)
        r1_cfg, r2_cfg, r3_cfg = 320, 300, 300
    else:
        bf_shard = args.shard - len(SNIPER_CUT_SLOTS)
        bf_num = max(1, args.num_shards - len(SNIPER_CUT_SLOTS))
        two_band_targets = []
        for n in range(180, 1001):
            if math.isqrt(n) ** 2 == n or cur[n][1] is None:
                continue
            gap_l = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
            if gap_l < 1.2e-4:
                continue
            P_deg = RE.normalize_to_origin(cur[n][1])
            P_deg[:, 2] = np.degrees(P_deg[:, 2]) % 90.0
            if len(BF.bands(P_deg, min_size=12)) >= 2:
                two_band_targets.append((gap_l / math.sqrt(n), n))
        two_band_targets.sort(reverse=True)
        my_slice = [x[1] for x in two_band_targets[: bf_num * 8]][bf_shard::bf_num]
        print(
            f"[W55S-BandFlip s={args.shard}/{args.num_shards}] score={tot_score:.10f} my_slice={my_slice} ({time.time() - t0:.1f}s)",
            flush=True,
        )
        ex_bf = ProcessPoolExecutor(max_workers=args.procs)
        screened = []
        try:
            for pi, n in enumerate(my_slice):
                if time.time() - t0 > 350.0:
                    break
                P0 = cur[n][1]
                L0 = cur[n][0]
                bseed = args.seed_off + args.shard * 1901 + pi * 83
                jobs = []
                for bi in (0, 1):
                    for rep in range(4):
                        jobs.append(("flip", bi, P0, L0, bseed + bi * 20 + rep))
                    for rep in range(2):
                        jobs.append(("reloc", bi, P0, L0, bseed + 60 + bi * 10 + rep))
                ts0 = time.time()
                futs = [ex_bf.submit(eval_bandflip_job, jb) for jb in jobs]
                hits = []
                try:
                    for f in as_completed(futs, timeout=95.0):
                        tg, Sq, Qq = f.result()
                        if Qq is None:
                            continue
                        gap0 = Sq - L0
                        ad = topology_angle_dist(Qq, P0)
                        if gap0 < -1e-9 or (gap0 < 0.0135 and ad > 0.08):
                            hits.append((gap0, n, tg, Sq, Qq))
                except TimeoutError:
                    w51d.terminate_pool(ex_bf)
                    ex_bf = ProcessPoolExecutor(max_workers=args.procs)
                hits.sort(key=lambda z: z[0])
                print(
                    f"  [W55S bandflip] n={n:4d} distinct_hits={len(hits)} best={hits[0][0]:+.6f}@{hits[0][2] if hits else 'none'} ({time.time() - ts0:.1f}s)"
                    if hits
                    else f"  [W55S bandflip] n={n:4d} distinct_hits=0 ({time.time() - ts0:.1f}s)",
                    flush=True,
                )
                for h in hits[:2]:
                    screened.append(h)
        finally:
            w51d.terminate_pool(ex_bf)
        screened.sort(key=lambda z: z[0])
        queued = screened[:7]
        r1_cfg, r2_cfg, r3_cfg = 250, 230, 230

    queued.sort(key=lambda z: z[0])
    print(
        f"[W55S s={args.shard}] Phase B: hunting {len(queued)} targets ({time.time() - t0:.1f}s)!",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, nn, tag, S_init, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 50.0:
                break
            L0_cur = wins[nn][0] if nn in wins else cur[nn][0]
            if S_init < L0_cur + 0.00045:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    nn, Q_init, deadline, slp_sec=44.0, fq_sec=28
                )
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(nn, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        tot_ds += (Lf0 - L0_cur) / math.sqrt(nn)
                        wins[nn] = (float(Lf0), xf0, yf0, tf0)
                        L0_cur = float(Lf0)
                        print(
                            f"  [W55S DIRECT WIN!!] n={nn:4d} [{tag}] {cur[nn][0]:.7f} -> {Lf0:.7f}!",
                            flush=True,
                        )
                        save_ckpt()

            r1 = min(r1_cfg, max(35, int(0.35 * (rem - 20.0))))
            r2 = min(r2_cfg, max(30, int(0.30 * (rem - 20.0))))
            r3 = min(r3_cfg, max(30, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + 50000 + args.shard * 12007 + i * 811
            Qb, Sb, (tr, ft, wn) = w51d.deep_3round_widehopper(
                ex_hop, nn, Q_init, L0_cur, bseed, r1, r2, r3, deadline
            )
            if Sb < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(nn, Qb)
                if ok and Lf < L0_cur - 1e-9:
                    ds = (Lf - L0_cur) / math.sqrt(nn)
                    tot_ds += ds
                    wins[nn] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W55S HOP WIN!!] n={nn:4d} [{tag}] {cur[nn][0]:.7f} -> {Lf:.7f} "
                        f"(dL={Lf - cur[nn][0]:+.7f}, dS={ds:+.8f}, w={wn})",
                        flush=True,
                    )
                    save_ckpt()
                    if nn - 1 >= 80:
                        L_m1 = wins[nn - 1][0] if (nn - 1) in wins else cur[nn - 1][0]
                        if Lf < L_m1 + 0.020:
                            for didx in w32a.pick_near_band(Qb, max_cand=8)[:3]:
                                Qd, Sd = w32a.fast_delete_and_compact(Qb, didx)
                                if Qd is not None and Sd < L_m1 + 0.008:
                                    Qd_b, Sd_b, _ = w51d.deep_3round_widehopper(
                                        ex_hop,
                                        nn - 1,
                                        Qd.astype(np.float64),
                                        L_m1,
                                        bseed + 71 + didx,
                                        95,
                                        80,
                                        75,
                                        deadline,
                                    )
                                    if Sd_b < L_m1 - 1e-9:
                                        ok2, Lf2, x2, y2, t2 = f2d.stabilize_decimal(
                                            nn - 1, Qd_b
                                        )
                                        if ok2 and Lf2 < L_m1 - 1e-9:
                                            tot_ds += (Lf2 - L_m1) / math.sqrt(nn - 1)
                                            wins[nn - 1] = (float(Lf2), x2, y2, t2)
                                            L_m1 = float(Lf2)
                                            print(
                                                f"  [W55S CASCADE WIN!!] n={nn - 1:4d} {cur[nn - 1][0]:.7f} -> {Lf2:.7f}!",
                                                flush=True,
                                            )
                                            save_ckpt()
                else:
                    print(
                        f"  [W55S miss] n={nn:4d} [{tag}] dStart={S_init - L0_cur:+.6f} -> dEnd={Sb - L0_cur:+.6f}",
                        flush=True,
                    )
            else:
                print(
                    f"  [W55S hunt] n={nn:4d} [{tag}] dStart={S_init - L0_cur:+.6f} -> dEnd={Sb - L0_cur:+.6f} (tr={tr}, w={wn})",
                    flush=True,
                )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W55S s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
