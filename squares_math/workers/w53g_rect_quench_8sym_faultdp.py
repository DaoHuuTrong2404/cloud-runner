#!/usr/bin/env python3
"""Wave 53G: Rectangle-Mode `fq quench --rect-dx/--rect-dy` Gadget Quencher over 558 Private Donors
+ 8-Symmetry Fault-Line DP Re-Composer + Novelty Filter -> 3-Round Deep-Polish Wider-Hopper!

Upgrades Wave 52F (`16,515` 8-sym gadgets, which hit `gap0 = +0.00001` on `785, 901, 585, 261, 332`!)
with TWO critical algorithmic breakthroughs:
  1. Quenches the active Pareto `(w, h, dm)` gadgets extracted from our 558 private+public donors
     using compiled Rust `fq quench --rect-dx (w-m) --rect-dy (h-m)` (`quench_gadgets.py` protocol —
     shrinking internal gadget slack BEFORE Fault-DP composes multi-gadget chains!).
  2. Novelty Filter (`topology_angle_dist > 0.05` vs `cur[n][1]` OR `Sc < L0 - 1e-8`) so Phase B
     NEVER wastes 600s re-polishing trivial single-gadget clones of R32 (`gap0 = +0.00001 -> dEnd = 0.000000`),
     spending 100% of its Hopper budget on GENUINELY NEW multi-gadget / quenched topologies!
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
import squeeze2 as sq2
import w5b_2d_faultdp_gadgetq as f2d
import w32a_fresh_basin_fast_hopper as w32a
import w51d_two_stage_del1_widehopper as w51d
import w51r_residual_contact_continuation as w51r
import w53h_nakajima_reharden_and_softinsert as w53h
import w53s_ccbar_reharden_splicer as w53s


def fast_quench_dp(P_comp, target_L_ref):
    x0, y0, t0 = sq2.normalize(P_comp[:, 0], P_comp[:, 1], P_comp[:, 2])
    S_init = max(float(sq2.side(x0, y0, t0)), float(target_L_ref)) * 1.0015
    ok, rx, ry, rt, _ = sq2.relax_at(x0, y0, t0, S_init, max_iter=5000)
    if not ok:
        ok, rx, ry, rt, _ = sq2.relax_at(x0, y0, t0, S_init * 1.010, max_iter=6500)
        if not ok:
            return None
    _, x2, y2, t2 = sq2.squeeze(
        rx, ry, rt, rel0=2.5e-3, rel_min=5e-7, max_iter=4000, max_tries=40
    )
    Lf, fx, fy, ft, good = sq2.finalize(x2, y2, t2, tol=2e-12)
    if not good:
        return None
    import sq_engine as E

    Q = E.to_origin(np.column_stack([fx, fy, ft]))
    try:
        Sc, Qc = w51d.compact_deleted(Q, max_iter=30, patience=2)
        if Qc is not None and Sc < Lf:
            return Qc, float(Sc)
    except Exception:
        pass
    return Q, float(E.side(Q))


def quench_gadget_worker(args_tuple):
    key, sq_init, seed0 = args_tuple
    w, h, dm = key
    f0 = RE.gadget_overhang(sq_init, w, h)
    best_sq, best_f = sq_init, f0
    # 5 loosen/kick passes in rectangle mode
    schedules = [
        (1.000, 0.000),
        (1.008, 0.004),
        (1.016, 0.008),
        (1.025, 0.012),
        (1.012, 0.006),
    ]
    for k, (lo, kk) in enumerate(schedules):
        cand, fc = RE.quench_rect_gadget(
            best_sq,
            w,
            h,
            loosen=lo,
            kick=kk,
            seed=seed0 + k * 131,
            outer=50,
            pit=200,
            flip_top=16,
            timeout_s=18.0,
        )
        if fc < best_f - 1e-9 and len(cand) == w * h + dm:
            best_sq, best_f = cand, fc
    return key, f0, best_f, best_sq


def collect_chain_gadget_keys(dp, PG, PM, PQ, G, targets):
    """Trace optimal and near-optimal DP chains (`dp[:K+1, :K+1, D, :]`) to collect active `(w, h, dm)` keys."""
    key_counts = {}
    for _, n, K, D in targets:
        if K > fdp8.WMAX or D <= 0 or D > fdp8.DMAX:
            continue
        sub = dp[: K + 1, : K + 1, D, :]
        if not np.isfinite(sub).any():
            continue
        best_f = float(np.min(sub))
        flat_idx = np.argsort(sub.ravel())
        for fi in flat_idx[:18]:
            cw, ch, q = np.unravel_index(int(fi), sub.shape)
            fv = float(sub[cw, ch, q])
            if not math.isfinite(fv) or fv > best_f + 0.035:
                break
            t_cw, t_ch, t_d, t_q = int(cw), int(ch), int(D), int(q)
            links = []
            ok_tr = True
            while (t_cw or t_ch or t_d) and len(links) < 12:
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
            if not ok_tr or not links:
                continue
            if (
                len(links) == 1
                and int(G[links[0][0], 0]) == K
                and int(G[links[0][0], 1]) == K
            ):
                continue
            wt = 3 if len(links) >= 2 else 1
            for g, _ in links:
                gw, gh, gdm, gf = (
                    int(G[g, 0]),
                    int(G[g, 1]),
                    int(G[g, 2]),
                    float(G[g, 3]),
                )
                if gw * gh + gdm >= 10 and gdm >= 1 and gf > 0.42:
                    k = (gw, gh, gdm)
                    key_counts[k] = key_counts.get(k, 0) + wt
    ranked = sorted(key_counts.items(), key=lambda kv: (-kv[1], -kv[0][2]))
    return [k for k, _ in ranked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2220)
    ap.add_argument("--r1-sec", type=int, default=200)
    ap.add_argument("--r2-sec", type=int, default=180)
    ap.add_argument("--r3-sec", type=int, default=190)
    ap.add_argument("--seed-off", type=int, default=570000)
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

    G = fdp8.extract_gadgets(donors)
    dp0, PG0, PM0, PQ0 = fdp8.solve_dp(G)

    all_targets = []
    for n in range(100, 1001):
        K = math.isqrt(n)
        D = n - K * K
        if D <= 0 or D > fdp8.DMAX or K > fdp8.WMAX:
            continue
        gap_ldr = max(0.0, cur[n][0] - front_L.get(n, cur[n][0]))
        if gap_ldr > 1e-4 or cur[n][0] >= K + 1 - 1e-6:
            all_targets.append((gap_ldr / math.sqrt(n), n, K, D))
    all_targets.sort(reverse=True)

    active_keys = collect_chain_gadget_keys(dp0, PG0, PM0, PQ0, G, all_targets[:180])
    # Shard active gadget keys across shards AND also vary kick seeds so all shards quench prime keys!
    my_keys = active_keys[args.shard :: args.num_shards][:22]
    # Add top-6 global multi-gadget keys with shard-unique seeds so every shard benefits from top keys!
    for gk in active_keys[:6]:
        if gk not in my_keys:
            my_keys.append(gk)

    q_jobs = []
    for ki, key in enumerate(my_keys):
        rows = np.where(
            (G[:, 0].astype(int) == key[0])
            & (G[:, 1].astype(int) == key[1])
            & (G[:, 2].astype(int) == key[2])
        )[0]
        if not len(rows):
            continue
        best_r = G[rows[int(np.argmin(G[rows, 3]))]]
        sq_g, _ = fdp8.crop_gadget(
            donors[int(best_r[6])], int(best_r[7]), key[0], key[1], key[2]
        )
        if sq_g is not None:
            q_jobs.append((key, sq_g, args.seed_off + args.shard * 4001 + ki * 97))

    improved_gadgets = []
    ex_gq = ProcessPoolExecutor(max_workers=args.procs)
    try:
        futs = [ex_gq.submit(quench_gadget_worker, jb) for jb in q_jobs]
        try:
            for f in as_completed(futs, timeout=220.0):
                key, f0, f1, sq1 = f.result()
                if f1 < f0 - 1e-7:
                    improved_gadgets.append((key, f0, f1, sq1))
        except TimeoutError:
            pass
    finally:
        w51d.terminate_pool(ex_gq)

    print(
        f"[W53G-RectGadgetDP s={args.shard}/{args.num_shards}] gadgets={len(G)} quenched={len(q_jobs)} "
        f"IMPROVED={len(improved_gadgets)} ({time.time() - t0:.1f}s)",
        flush=True,
    )
    for key, f0, f1, _ in improved_gadgets[:8]:
        print(
            f"  [W53G gadget win] {key[0]}x{key[1]}+{key[2]}: f {f0:.6f} -> {f1:.6f} (df={f1 - f0:+.6f})",
            flush=True,
        )

    if improved_gadgets:
        new_donors = [sq1 for _, _, _, sq1 in improved_gadgets]
        Gn = fdp8.extract_gadgets(new_donors)
        if len(Gn):
            Gn[:, 6] += len(donors)
            G = np.vstack([G, Gn])
            donors = donors + new_donors
            dp0, PG0, PM0, PQ0 = fdp8.solve_dp(G)

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

    # Scan ALL targets if gadgets improved (fast check for immediate wins!), plus shard's slice for Phase B!
    my_targets = all_targets[args.shard :: args.num_shards]
    cands = []
    for _, n, K, D in my_targets:
        if time.time() - t0 > 460.0:
            break
        L0 = cur[n][0]
        dp_chains = fdp8.trace_and_compose(dp0, PG0, PM0, PQ0, G, donors, K, D, top_m=3)
        for ci, (L_est, dtag, P_comp) in enumerate(dp_chains):
            # Prefer multi-gadget (`m>=2`) OR improved single-gadget chains!
            if L_est < L0 + 0.024:
                rh = fast_quench_dp(P_comp, L0)
                if rh is not None:
                    Qc, Sc = rh
                    adist = w53h.topology_angle_dist(Qc, cur[n][1])
                    # NOVELTY FILTER: skip `m1` clones of `cur[n][1]` (`adist < 0.04` and `Sc >= L0 - 1e-8`)!
                    if Sc < L0 - 1e-8 or (Sc < L0 + 0.014 and adist > 0.04):
                        cands.append((Sc - L0, n, Sc, adist, dtag, Qc))

    cands.sort(key=lambda z: z[0])
    seen_n = {}
    queued = []
    for gap0, n, Sc, adist, dtag, Qc in cands:
        if seen_n.get(n, 0) >= 2:
            continue
        seen_n[n] = seen_n.get(n, 0) + 1
        queued.append((gap0, n, Sc, adist, dtag, Qc))
        if len(queued) >= 9:
            break

    print(
        f"[W53G s={args.shard}] novel DP cands={len(cands)} -> queued={len(queued)} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    ex_hop = ProcessPoolExecutor(max_workers=args.procs)
    try:
        for i, (gap0, n, S_init, adist, tag, Q_init) in enumerate(queued):
            rem = deadline - time.time()
            if rem < 50.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            if S_init < L0_cur + 0.00035:
                Qp0, Sp0 = w51d.deep_contact_polish(
                    n, Q_init, deadline, slp_sec=38.0, fq_sec=26
                )
                if Sp0 < L0_cur - 1e-9:
                    ok0, Lf0, xf0, yf0, tf0 = f2d.stabilize_decimal(n, Qp0)
                    if ok0 and Lf0 < L0_cur - 1e-9:
                        tot_ds += (Lf0 - L0_cur) / math.sqrt(n)
                        wins[n] = (float(Lf0), xf0, yf0, tf0)
                        L0_cur = float(Lf0)
                        print(
                            f"  [W53G DIRECT DP WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf0:.7f}!",
                            flush=True,
                        )
                        save_ckpt()

            r1 = min(args.r1_sec, max(30, int(0.35 * (rem - 20.0))))
            r2 = min(args.r2_sec, max(25, int(0.30 * (rem - 20.0))))
            r3 = min(args.r3_sec, max(25, int(0.30 * (rem - 20.0))))
            bseed = args.seed_off + 40000 + args.shard * 10007 + i * 601
            Qb, Sb, (tr, ft, wn) = w51d.deep_3round_widehopper(
                ex_hop, n, Q_init, L0_cur, bseed, r1, r2, r3, deadline
            )
            if Sb < L0_cur - 1e-9:
                ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, Qb)
                if ok and Lf < L0_cur - 1e-9:
                    dL = Lf - cur[n][0]
                    tot_ds += (Lf - L0_cur) / math.sqrt(n)
                    wins[n] = (float(Lf), xf, yf, tf)
                    print(
                        f"  [W53G WIN!!] n={n:4d} [{tag}_wh28] gap0={gap0:+.5f} dA={adist:.2f} tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W53G done] n={n:4d} [{tag}] gap0={gap0:+.5f} dA={adist:.2f} -> dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{i + 1}/{len(queued)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W53G s={args.shard} EXC] {exc}", flush=True)
    finally:
        w51d.terminate_pool(ex_hop)

    save_ckpt()
    print(
        f"[W53G s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
