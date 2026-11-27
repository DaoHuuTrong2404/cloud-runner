#!/usr/bin/env python3
"""Wave 50D: Exhaustive All-Near-Band Delete-1 Compaction Map -> 4-Seed True `eta<=0.28` Wider-Move Hopper!

Implements the EXACT mechanism proven in goodpjw2008 v45/v49/v53 (655->654, 817->816, 492->491)
PLUS the newly uncovered `eta <= 0.28` (>0.2071 45-deg flip threshold!) + interleaved `SLP.polish`
across ALL ~120 high-value `n+1 -> n` pairs — ESPECIALLY our 103 exclusive private R32 donors
(`915->914`, `817->816`, `921->920`, `795->794`, `593->592`, `642->641`, `500->499`, `499->498`,
`683->682`, `703->702`, `804->803`, `748->747`, `556->555`, ...) that NO public notebook possesses!

Two-Phase Shard Pipeline (4 cores):
  - Phase A (Exhaustive LP Removal Map): Screens EVERY near-band / tilted / loaded square of `P[n+1]`
    (up to 300 squares, `max_iter=60, patience=2`, zero premature 30s cutoff!) in parallel across all
    4 cores to find the 1-2 magic squares that compact within `< +0.015` of `L_cur[n]`.
  - Phase B (4-Seed Parallel True `eta<=0.28` Wider-Move Hopper + Interleaved `SLP.polish` Round 2):
    Runs 4 parallel seeds x `r1_sec` on the best start -> `SLP.polish(iters=110)` -> 4 parallel seeds
    x `r2_sec` if Round 1 improved or landed within `+0.0025` of `L_cur[n]` -> `fq` + `lp_rot` +
    exact `Decimal64 >= 1.05e-9`!
"""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
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

import sq_engine as E
import slp as SLP
import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12
import w32a_fresh_basin_fast_hopper as w32a
import w49_fault_open_sa_engine as w49a


def all_near_band_indices(P1, max_cand=300):
    """Collect ALL tilted squares + 2-hop neighbors around tilted bands + high-load squares!"""
    P1 = E.to_origin(P1)
    n1 = len(P1)
    tilts = np.abs(E.fold(P1[:, 2]))
    layer0 = tilts > 0.015
    body, _, _, size = E.find_bodies(P1, 1e-3)
    layer0 = layer0 | (size[body] <= 2)
    pairs = E.neighbour_pairs(np.ascontiguousarray(P1), 1.65)
    layer1 = layer0.copy()
    for u, v in pairs:
        if layer0[u]:
            layer1[v] = True
        if layer0[v]:
            layer1[u] = True
    layer2 = layer1.copy()
    for u, v in pairs:
        if layer1[u]:
            layer2[v] = True
        if layer1[v]:
            layer2[u] = True
    cand = np.where(layer2)[0]
    if len(cand) <= max_cand:
        return [int(i) for i in cand]
    # Rank by tilted + layer1 priority if > max_cand
    prio = layer0.astype(np.float64) * 2.0 + layer1.astype(np.float64) * 1.0
    order = cand[np.argsort(-prio[cand])[:max_cand]]
    return [int(i) for i in sorted(order)]


def eval_single_delete(args_tuple):
    """Exact sis_live.delete_and_compact (`max_iter=60, patience=2`)."""
    P1, idx = args_tuple
    try:
        P = E.to_origin(np.delete(P1, idx, axis=0))
        body, _, _, size = E.find_bodies(P, 1e-3)
        core = size[body] <= 2
        free_ = core.copy()
        for u, v in E.neighbour_pairs(np.ascontiguousarray(P), 1.45):
            if core[u] and size[body[v]] <= 12:
                free_[v] = True
            if core[v] and size[body[u]] <= 12:
                free_[u] = True
        Q, S, ov = E.Compactor(P, 1e-3, singles=np.where(free_)[0]).compact(
            P, max_iter=60, patience=2
        )
        if ov > 5e-8:
            return idx, 1e9, None
        return idx, float(S), Q.astype(np.float64)
    except Exception:
        return idx, 1e9, None


def hop_seed_worker(args_tuple):
    P_in, seed, sec, deadline = args_tuple
    rem = deadline - time.time() - 8.0
    run_s = min(float(sec), rem)
    if run_s < 4.0:
        P0 = w32a.safe_open(P_in)
        return seed, P0, float(E.side(P0)), (0, 0, 0)
    try:
        H = E.Hopper(w32a.safe_open(P_in), seed=int(seed), wide=True)
        if not H.applicable():
            P0 = w32a.safe_open(P_in)
            return seed, P0, float(E.side(P0)), (0, 0, 0)
        Q, S = H.run(run_s)
        Q = w32a.safe_open(Q)
        return (
            seed,
            Q,
            float(E.side(Q)),
            (H.stats["trials"], H.stats["fit"], H.stats["wins"]),
        )
    except Exception:
        P0 = w32a.safe_open(P_in)
        return seed, P0, float(E.side(P0)), (0, 0, 0)


def slp_contact_polish(P_in):
    cur_a = w32a.safe_open(P_in)
    cur_s = float(E.side(cur_a))
    try:
        xs, ys, ts, _ = SLP.polish(
            cur_a[:, 0],
            cur_a[:, 1],
            cur_a[:, 2],
            iters=115,
            r0=2.5e-3,
            rmin=1e-10,
            tol_sep=-1e-13,
            time_limit=22.0,
        )
        Qs = w32a.safe_open(np.column_stack([xs, ys, ts]))
        Ss = float(E.side(Qs))
        if Ss < cur_s - 1e-10 and E.min_gap(Qs) >= -1e-12:
            return Qs, Ss
    except Exception:
        pass
    return cur_a, cur_s


def certify_winner(n, P_best, L0, deadline):
    cur_a, cur_s = slp_contact_polish(P_best)
    if os.path.exists(f2d.FQ_BIN) and (deadline - time.time()) > 12.0:
        with tempfile.TemporaryDirectory(prefix=f"w50_{n}_") as tmp:
            r_fin = f2d.run_fq(
                cur_s,
                cur_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
                timeout=min(50, max(12, int(deadline - time.time() - 4))),
            )
            if r_fin is not None and r_fin[0] < cur_s:
                cur_s, cur_a = r_fin[0], r_fin[1]
    try:
        cur_a = w12.lp_rot_polish(cur_a)
    except Exception:
        pass
    ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_a)
    if ok and Lf < L0 - 1e-9:
        return (float(Lf), xf, yf, tf)
    return None


def run_4seed_2round_widehopper(ex, P_start, L0, base_seed, r1_sec, r2_sec, deadline):
    """4 parallel seeds x r1_sec -> SLP contact polish -> 4 parallel seeds x r2_sec -> SLP!"""
    P_cur = w32a.safe_open(P_start)
    S_start = float(E.side(P_cur))
    best_P, best_S = P_cur, S_start
    tot_tr, tot_fit, tot_w = 0, 0, 0

    # Round 1: 4 parallel seeds!
    futs1 = [
        ex.submit(hop_seed_worker, (P_cur, base_seed + s * 1009, r1_sec, deadline))
        for s in range(4)
    ]
    for f in as_completed(futs1, timeout=max(20.0, r1_sec + 45.0)):
        _, Q, S, (tr, ft, wn) = f.result()
        tot_tr += tr
        tot_fit += ft
        tot_w += wn
        if S < best_S - 1e-10 and E.min_gap(Q) >= -1e-12:
            best_S, best_P = S, Q

    # Interleaved SLP Contact Polish (unlocks jammed contacts for Round 2 jumps — NOTES.md v48/v50!)
    if best_S < S_start - 1e-8 or best_S < L0 + 0.004:
        best_P, best_S = slp_contact_polish(best_P)

    # Round 2: 4 new parallel seeds if Round 1 moved significantly OR within +0.0028 of L0!
    if (
        r2_sec >= 20
        and (deadline - time.time()) > 35.0
        and (best_S < L0 + 0.0028 or (S_start - best_S) > 3e-4)
    ):
        futs2 = [
            ex.submit(
                hop_seed_worker,
                (best_P, base_seed + 50000 + s * 1009, r2_sec, deadline),
            )
            for s in range(4)
        ]
        for f in as_completed(futs2, timeout=max(20.0, r2_sec + 45.0)):
            _, Q, S, (tr, ft, wn) = f.result()
            tot_tr += tr
            tot_fit += ft
            tot_w += wn
            if S < best_S - 1e-10 and E.min_gap(Q) >= -1e-12:
                best_S, best_P = S, Q
        best_P, best_S = slp_contact_polish(best_P)

    return best_P, best_S, (tot_tr, tot_fit, tot_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--targets-csv", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2300)
    ap.add_argument("--r1-sec", type=int, default=210)
    ap.add_argument("--r2-sec", type=int, default=170)
    ap.add_argument("--max-puzzles", type=int, default=120)
    ap.add_argument("--seed-off", type=int, default=500000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    # Load base v5 alone first so we know which n+1 packings are OUR private/fresh advances!
    base_only, _, _, _ = w12.load_frontier_with_wins(args.union_npz, [])
    cur, fresh, tot_score, _ = w49a.load_closed_frontier(args.union_npz, args.win_glob)
    front_rows = (
        json.load(open(args.frontier_json))
        if os.path.exists(args.frontier_json)
        else []
    )
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    tg_hints = {}
    if os.path.exists(args.targets_csv):
        with open(args.targets_csv) as f:
            for row in csv.DictReader(f):
                tg_hints.setdefault(int(row["n"]), []).append(int(row["removed"]))

    # Rank n in [180..999] where n+1 is non-grid and L(n+1) - L(n) < 0.034!
    cand_ns = []
    for n in range(180, 1000):
        K = math.isqrt(n)
        K1 = math.isqrt(n + 1)
        if K * K == n or K1 * K1 == (n + 1):
            continue
        if cur[n + 1][0] >= K1 + 1 - 1e-6:
            continue
        step_up = cur[n + 1][0] - cur[n][0]
        if step_up > 0.034:
            continue
        fl = front_L.get(n, cur[n][0])
        gap_wr = cur[n][0] - fl
        if gap_wr < 5e-6 and step_up > 0.010:
            continue
        # Huge priority boost when P[n+1] improved over pub_base_v5 (our private R32 donors!)
        priv_adv = max(0.0, base_only[n + 1][0] - cur[n + 1][0])
        prio = (
            (gap_wr / math.sqrt(n))
            + (0.00060 if priv_adv > 1e-4 else 0.0)
            + max(0.0, 0.024 - step_up) * 0.025
            + (0.00015 if n in tg_hints else 0.0)
        )
        cand_ns.append((prio, n, step_up, priv_adv))

    cand_ns.sort(reverse=True)
    selected = [x[1] for x in cand_ns[: args.max_puzzles]]
    my_slice = selected[args.shard :: args.num_shards]
    deadline = t0 + args.budget_sec

    print(
        f"[W50D-ExhaustiveDel1 s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_slice({len(my_slice)})={my_slice} ({time.time() - t0:.1f}s)",
        flush=True,
    )

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

    ex = ProcessPoolExecutor(max_workers=args.procs)
    try:
        # Phase A: Screen ALL near-band squares of P[n+1] for each n in my_slice!
        screened_starts = []
        for n in my_slice:
            if time.time() > deadline - 120.0:
                break
            P1 = cur[n + 1][1]
            L0 = cur[n][0]
            idxs = list(
                dict.fromkeys(
                    list(tg_hints.get(n, [])) + all_near_band_indices(P1, max_cand=280)
                )
            )
            idxs = [i for i in idxs if 0 <= i < len(P1)]
            ts0 = time.time()
            futs = [ex.submit(eval_single_delete, (P1, idx)) for idx in idxs]
            holes = []
            try:
                for f in as_completed(futs, timeout=150.0):
                    idx, S_del, Q_del = f.result()
                    if Q_del is not None and S_del < L0 + 0.015:
                        holes.append((S_del - L0, S_del, idx, Q_del))
            except TimeoutError:
                pass
            holes.sort(key=lambda z: z[0])
            best_gap_str = (
                f"{holes[0][0]:+.6f}@sq{holes[0][2]}" if holes else "none(<+0.015)"
            )
            print(
                f"  [W50D screen] n={n:4d}<-{n + 1:4d} tested={len(idxs)} holes(<+0.015)={len(holes)} best={best_gap_str} ({time.time() - ts0:.1f}s)",
                flush=True,
            )
            # Keep up to 2 distinct hole starts per n!
            picked_n = []
            for gap0, S_del, idx, Q_del in holes:
                if all(abs(S_del - ps) > 3e-6 for _, ps, _, _ in picked_n):
                    picked_n.append((gap0, S_del, idx, Q_del))
                if len(picked_n) >= 2:
                    break
            for gap0, S_del, idx, Q_del in picked_n:
                screened_starts.append((gap0, n, idx, S_del, Q_del))

        # Sort ALL screened starts on this shard by smallest post-compaction gap first!
        screened_starts.sort(key=lambda z: z[0])
        print(
            f"[W50D s={args.shard}] Phase B: hunting {len(screened_starts)} best delete-1 starts with 4-seed eta<=0.28 Wider-Hopper!",
            flush=True,
        )

        for rank_i, (gap0, n, idx, S_del, Q_del) in enumerate(screened_starts):
            rem = deadline - time.time()
            if rem < 40.0:
                break
            L0_cur = wins[n][0] if n in wins else cur[n][0]
            # Check if compaction ALONE already beat L0_cur!
            if S_del < L0_cur - 1e-8:
                res0 = certify_winner(n, Q_del, L0_cur, deadline)
                if res0 is not None:
                    Lf = res0[0]
                    tot_ds += (Lf - L0_cur) / math.sqrt(n)
                    wins[n] = res0
                    L0_cur = Lf
                    print(
                        f"  [W50D DIRECT COMPACT WIN!!] n={n:4d} [del_{idx}] {cur[n][0]:.7f} -> {Lf:.7f}!",
                        flush=True,
                    )
                    save_ckpt()

            r1 = min(args.r1_sec, max(30, int(0.42 * (rem - 15.0))))
            r2 = min(args.r2_sec, max(20, int(0.35 * (rem - 15.0))))
            bseed = args.seed_off + args.shard * 10007 + rank_i * 401
            Qb, Sb, (tr, ft, wn) = run_4seed_2round_widehopper(
                ex, Q_del, L0_cur, bseed, r1, r2, deadline
            )
            if Sb < L0_cur - 1e-9:
                res = certify_winner(n, Qb, L0_cur, deadline)
                if res is not None:
                    Lf = res[0]
                    dL = Lf - cur[n][0]
                    tot_ds += (Lf - L0_cur) / math.sqrt(n)
                    wins[n] = res
                    print(
                        f"  [W50D WIN!!] n={n:4d} [del_{idx}_wh28] gap0={gap0:+.5f} tr={tr} fit={ft} w={wn} "
                        f"{cur[n][0]:.7f} -> {Lf:.7f} (dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{rank_i + 1}/{len(screened_starts)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
                    continue
            print(
                f"  [W50D done] n={n:4d} [del_{idx}] gap0={gap0:+.5f} -> dEnd={Sb - L0_cur:+.6f} tr={tr} fit={ft} w={wn} [{rank_i + 1}/{len(screened_starts)}, {time.time() - t0:.0f}s]",
                flush=True,
            )
    except Exception as exc:
        print(f"[W50D s={args.shard} EXC] {exc}", flush=True)

    save_ckpt()
    print(
        f"[W50D s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
        flush=True,
    )
    for proc in list(getattr(ex, "_processes", {}).values()):
        try:
            proc.terminate()
        except Exception:
            pass
    os._exit(0)


if __name__ == "__main__":
    main()
