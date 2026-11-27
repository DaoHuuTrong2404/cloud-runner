#!/usr/bin/env python3
"""Wave 56F (`w56f_frozen_grid_channel_remelt.py`): Connected-Component Wall-Anchored
Frozen-Grid Channel Remelt (`--freeze` + `--anchor-x` + `--anchor-y`) -> Unfrozen KKT Finish -> 3R Hopper!

Uses the exact crystal-anchoring mechanism built into `fq.rs` (lines 58-104 & 1996-2040):
  1. Finds axis-aligned crystal components (`|tilt| < 0.008`), keeps a 2-layer free buffer (`d > 2.30`)
     next to the tilted bands so step-transitions & corner pockets can rearrange freely, and freezes
     the deep crystal interior (`--freeze SPEC`).
  2. Automatically detects whether each frozen crystal component touches the far right (`x = S`) or
     far top (`y = S`) container wall without touching `0`, setting `--anchor-x` / `--anchor-y` so
     the crystal slides rigidly with `s` while ONLY the tilted channel + boundary melts as rounded
     squares (`--round-r0 R0 --round-steps 14 --round-kick RK`, strictly WITHOUT `--sqp/--finish` per
     `fq.rs` line 2002!).
  3. Immediately unfreezes ALL `N` squares and runs two-pass second-order KKT `newton_finish` + SQP
     (`RE.kkt_sqp_polish_fq`)!
  4. Feeds distinct unjammed channel topologies (`adist > 0.04` or immediate wins) into KKT Wider-Hopper!
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

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
import w53h_nakajima_reharden_and_softinsert as w53h


PRIORITY_CHANNEL_TARGETS = [
    652,
    418,
    756,
    871,
    754,
    924,
    465,
    234,
    333,
    303,
    295,
    338,
    811,
    870,
    812,
    701,
    689,
    549,
    376,
    377,
    491,
    738,
    456,
    904,
    804,
    914,
    748,
    530,
    976,
    500,
    632,
    499,
    540,
    449,
    544,
    590,
]


def compress_idx_spec(indices):
    idx = sorted(set(int(i) for i in indices))
    if not idx:
        return ""
    parts = []
    s = e = idx[0]
    for x in idx[1:]:
        if x == e + 1:
            e = x
        else:
            parts.append(f"{s}-{e}" if s < e else f"{s}")
            s = e = x
    parts.append(f"{s}-{e}" if s < e else f"{s}")
    return ",".join(parts)


def compute_crystal_anchors(P_in, buffer_dist=2.30):
    """Return `(P_norm, S_norm, freeze_spec, anchor_x_spec, anchor_y_spec)` or `None` if no crystal."""
    P = RE.normalize_to_origin(P_in)
    S = RE.side_exact(P)
    tilt = np.abs(((P[:, 2] + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0)
    tilted = np.where(tilt > 0.015)[0]
    upright = np.where(tilt < 0.008)[0]
    if len(tilted) < 8 or len(upright) < 20:
        return None

    d_to_tilt = cdist(P[:, :2], P[tilted, :2]).min(axis=1)
    deep_mask = (tilt < 0.008) & (d_to_tilt > float(buffer_dist))
    if int(deep_mask.sum()) < 12:
        return None

    # Connected components among ALL upright squares to know which wall each crystal block touches
    U_xy = P[upright, :2]
    adj = cdist(U_xy, U_xy) < 1.09
    nc, lab = connected_components(adj, directed=False)

    fr_list, ax_list, ay_list = [], [], []
    for cid in range(nc):
        comp_idx = upright[lab == cid]
        if len(comp_idx) < 8:
            continue
        cx = P[comp_idx, 0]
        cy = P[comp_idx, 1]
        touches_left = float(cx.min()) < 0.56
        touches_right = float(cx.max()) > S - 0.56
        touches_bot = float(cy.min()) < 0.56
        touches_top = float(cy.max()) > S - 0.56

        use_ax = touches_right and not touches_left
        use_ay = touches_top and not touches_bot

        deep_comp = [int(i) for i in comp_idx if deep_mask[i]]
        if not deep_comp:
            continue
        fr_list.extend(deep_comp)
        if use_ax:
            ax_list.extend(deep_comp)
        if use_ay:
            ay_list.extend(deep_comp)

    if len(fr_list) < 12 or len(fr_list) > len(P) - 10:
        return None
    return (
        P,
        S,
        compress_idx_spec(fr_list),
        compress_idx_spec(ax_list),
        compress_idx_spec(ay_list),
    )


def eval_channel_remelt_job(job_tup):
    tag, P_norm, S_norm, fr_spec, ax_spec, ay_spec, seed, r0, rk, hf, use_soft = job_tup
    try:
        # Step 1: Frozen-Grid Wall-Anchored Channel Remelt (NO --sqp / --finish / --shrink allowed with anchors!)
        extra = ["--freeze", fr_spec, "--seed", str(int(seed))]
        if ax_spec:
            extra += ["--anchor-x", ax_spec]
        if ay_spec:
            extra += ["--anchor-y", ay_spec]
        if use_soft:
            extra += [
                "--soft-a0",
                repr(float(r0)),
                "--soft-steps",
                "12",
                "--soft-kick",
                repr(float(rk)),
                "--soft-compress",
                "--hold-f",
                repr(float(hf)),
            ]
        else:
            extra += [
                "--round-r0",
                repr(float(r0)),
                "--round-steps",
                "14",
                "--round-maxit",
                "1100",
                "--round-gtol",
                "1e-9",
                "--round-kick",
                repr(float(rk)),
                "--round-kick-peak",
                "0.055",
                "--hold-f",
                repr(float(hf)),
            ]
        extra += [
            "--loosen",
            "1.006",
            "--outer",
            "44",
            "--pit",
            "220",
            "--flip-top",
            "16",
            "--stag-w",
            "0",
            "--deadline",
            "12.5",
        ]
        rq1 = RE.run_fq_quench_args(P_norm, extra, s_ref=S_norm, timeout_s=14.5)
        if rq1 is None:
            return tag, 1e9, None
        _, Q_mid, _ = rq1

        # Step 2: Unfreeze ALL squares and project onto exact KKT + SQP manifold!
        S_fin, Q_fin = RE.kkt_sqp_polish_fq(Q_mid, timeout_each=6.5)
        return tag, float(S_fin), Q_fin
    except Exception:
        return tag, 1e9, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=18)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=1880)
    ap.add_argument("--r1-sec", type=int, default=240)
    ap.add_argument("--r2-sec", type=int, default=220)
    ap.add_argument("--r3-sec", type=int, default=220)
    ap.add_argument("--seed-off", type=int, default=740000)
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

    ordered = []
    seen = set()
    for n in PRIORITY_CHANNEL_TARGETS:
        if 100 <= n <= 1000 and math.isqrt(n) ** 2 != n and n not in seen:
            seen.add(n)
            ordered.append(n)
    rem_t = []
    for n in range(140, 1001):
        if n in seen or math.isqrt(n) ** 2 == n:
            continue
        gap_l = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gap_l > 1.2e-4:
            rem_t.append((gap_l / math.sqrt(n), n))
    rem_t.sort(reverse=True)
    ordered.extend([n for _, n in rem_t])

    my_slice = ordered[args.shard :: args.num_shards][:6]
    print(
        f"[W56F-FrozenChannel s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_slice={my_slice} ({time.time() - t0:.1f}s)",
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
    ex_sc = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for pi, n in enumerate(my_slice):
            if time.time() - t0 > 330.0 or cur[n][1] is None:
                break
            L0 = cur[n][0]
            P0 = cur[n][1]
            anc_tight = compute_crystal_anchors(P0, buffer_dist=1.85)
            anc_wide = compute_crystal_anchors(P0, buffer_dist=2.75)
            anc = anc_tight or anc_wide
            if anc is None:
                continue

            bseed = args.seed_off + args.shard * 2707 + pi * 113
            jobs = []
            cfgs = [
                (anc, 0.10, 0.0040, 1.0002, False),
                (anc, 0.14, 0.0055, 1.0008, False),
                (anc, 0.18, 0.0070, 1.0015, False),
                (anc, 0.22, 0.0080, 1.0020, False),
                (anc_wide or anc, 0.13, 0.0050, 1.0005, False),
                (anc_wide or anc, 0.17, 0.0065, 1.0012, False),
                (anc, 0.08, 0.0035, 1.0005, True),
                (anc_wide or anc, 0.13, 0.0045, 1.0010, True),
            ]
            for ci, (ac, r0, rk, hf, us) in enumerate(cfgs):
                Pn, Sn, fs, axs, ays = ac
                jobs.append(
                    (
                        f"chmelt_{ci}",
                        Pn,
                        Sn,
                        fs,
                        axs,
                        ays,
                        bseed + ci * 19,
                        r0,
                        rk,
                        hf,
                        us,
                    )
                )

            ts0 = time.time()
            futs = [ex_sc.submit(eval_channel_remelt_job, jb) for jb in jobs]
            hits = []
            try:
                for f in as_completed(futs, timeout=90.0):
                    tg, Sq, Qq = f.result()
                    if Qq is None:
                        continue
                    gap0 = Sq - L0
                    ad = w53h.topology_angle_dist(Qq, P0)
                    if gap0 < -1e-9 or (gap0 < 0.0130 and ad > 0.045):
                        hits.append((gap0, n, tg, Sq, ad, Qq))
            except TimeoutError:
                w51d.terminate_pool(ex_sc)
                ex_sc = ProcessPoolExecutor(max_workers=args.procs)

            hits.sort(key=lambda z: z[0])
            b_str = (
                f"{hits[0][0]:+.6f}(dA={hits[0][4]:.2f})@{hits[0][2]}"
                if hits
                else "none"
            )
            print(
                f"  [W56F screen] n={n:4d} distinct_hits={len(hits)} best={b_str} ({time.time() - ts0:.1f}s)",
                flush=True,
            )
            for h in hits[:2]:
                screened.append(h)
    finally:
        w51d.terminate_pool(ex_sc)

    screened.sort(key=lambda z: z[0])
    queued = screened[:6]
    print(
        f"[W56F s={args.shard}] Phase B: hunting {len(queued)}/{len(screened)} channel-remelted basins ({time.time() - t0:.1f}s)!",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, tag, S_init, adist, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 50.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            if S_init < L0_cur + 0.00035:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    n, Q_init, deadline, slp_sec=38.0, fq_sec=24
                )
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        tot_ds += (Lf0 - L0_cur) / math.sqrt(n)
                        wins[n] = (float(Lf0), xf0, yf0, tf0)
                        L0_cur = float(Lf0)
                        print(
                            f"  [W56F DIRECT CHANNEL WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf0:.7f}!",
                            flush=True,
                        )
                        save_ckpt()

            r1 = min(args.r1_sec, max(35, int(0.35 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(30, int(0.30 * (rem - 20.0))))
            r3 = min(args.r3_sec, max(30, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + 30000 + args.shard * 11027 + i * 757
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
                        f"  [W56F HOP WIN!!] n={n:4d} [{tag}] dA={adist:.2f} {cur[n][0]:.7f} -> {Lf:.7f} (dS={ds:+.8f}, w={wn})",
                        flush=True,
                    )
                    save_ckpt()
                else:
                    print(
                        f"  [W56F miss] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f}",
                        flush=True,
                    )
            else:
                print(
                    f"  [W56F done] n={n:4d} [{tag}] dA={adist:.2f} dStart={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} w={wn}",
                    flush=True,
                )
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W56F s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
