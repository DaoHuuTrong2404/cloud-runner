#!/usr/bin/env python3
"""Wave 32B: Constructive Exact-Angle Erdős-Graham Single-Band MILP (band_gen) + Split-Region L-Band MILP (lband2) -> Rust fq -> Fast Stock Hopper!

De-novo constructive topology engine for breaking structural traps that local search cannot escape:
  1. Single-Band MILP (band_gen.assemble2 exact sliding + band_gen.assemble rich=True):
     Targets k^2 + 1..15 (especially 730, 785, 842, 901, 962 breaking f < 0.50, PLUS giant gaps
     234, 367, 491, 333, 738, 261, 332, 904, 371, 295, 632, 540, 911, 638).
  2. Split-Region Two-Band L-Band MILP (lband2.assemble_split with right_rot & right_ref, rich=0.18,
     tol in {1e-7, 0.015} for near-feasible assemblies resolved cleanly by Rust fq!):
     Targets A_GRID_BREAK (605, 762, 819, 877, 1000) + F_LARGE_MID_F / G_LARGE_HIGH_F giant gaps
     (456, 804, 748, 414, 914, 976, 500, 303, 377, 378, 341, 273, 499, 692, 694, 864, 376, 418, ...).
  3. Every MILP assembly with cnt >= n is trimmed to n, quenched with verified Rust fq (--loosen 1.03 -> 1.00),
     chained into FAST Stock sq_engine.Hopper + SLP polish, and certified to exact Decimal64 >= 1.05e-9!
"""

import argparse
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
import band_gen as BG
import lband2 as LB2
import slp as SLP
import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12
import w32a_fresh_basin_fast_hopper as w32a


def trim_general(sol, n):
    """Trim extra squares (prefer uprights closest to walls/seams) down to exactly n."""
    sol = np.asarray(sol, dtype=np.float64)
    if len(sol) <= n:
        return sol
    ang = np.abs(np.remainder(sol[:, 2] + math.pi / 4.0, math.pi / 2.0) - math.pi / 4.0)
    is_up = ang < 1e-4
    # Drop uprights farthest from origin/edges or lowest interior density first
    # Keep tilted bands intact!
    L = BG.side_L(sol)
    edge_dist = np.minimum.reduce([sol[:, 0], L - sol[:, 0], sol[:, 1], L - sol[:, 1]])
    # Prefer dropping uprights near mid-band edges (score: small if upright & near wall)
    score = (~is_up).astype(np.float64) * 100.0 + edge_dist
    order = np.argsort(score)
    drop = set(order[: len(sol) - n].tolist())
    return sol[[i for i in range(len(sol)) if i not in drop]]


def fq_quench_and_hop(n, sol, L_cur, seed, tmp, allow_overlap=False, hop_sec=160):
    """Quench a MILP assembly via Rust fq (handling near-feasible overlap), then Stock Hopper!"""
    sol = E.to_origin(np.asarray(sol, dtype=np.float64))
    L_init = float(E.side(sol))
    cur_a = sol
    cur_s = L_init

    if os.path.exists(f2d.FQ_BIN):
        # Stage 1: Resolve overlaps / relax ends with gentle loosen + round-press
        loosen = "1.035" if allow_overlap else "1.012"
        r1 = f2d.run_fq(
            cur_s * (1.008 if allow_overlap else 1.002),
            cur_a,
            tmp,
            [
                "--loosen",
                loosen,
                "--round-r0",
                "0.05",
                "--round-compress",
                "--round-press",
                "--round-steps",
                "8",
                "--round-press-outer",
                "14",
                "--seed",
                str(seed),
                "--flip-top",
                "48",
                "--pit",
                "380",
            ],
            timeout=55,
        )
        if r1 is not None:
            cur_s, cur_a = r1[0], r1[1]
        # Stage 2: Tight deterministic polish
        r2 = f2d.run_fq(
            cur_s,
            cur_a,
            tmp,
            ["--loosen", "1.0", "--no-alm", "--flip-top", "56", "--pit", "500"],
            timeout=50,
        )
        if r2 is not None and r2[0] < cur_s:
            cur_s, cur_a = r2[0], r2[1]
    else:
        return None, 1e9

    if E.min_gap(cur_a) < -1e-6:
        return None, 1e9

    cur_a = w32a.safe_open(cur_a)
    cur_s = float(E.side(cur_a))

    # Stage 3: If quenched result is competitive (< L_cur + 0.012), unleash FAST Stock Hopper!
    if cur_s < L_cur + 0.012:
        Qh, Sh, _ = w32a.run_stock_hopper_chain(
            cur_a, seed + 1337, first_sec=hop_sec, cascade_sec=140, max_cascades=3
        )
        if Sh < cur_s:
            cur_a, cur_s = Qh, Sh
    return cur_a, cur_s


def worker_milp(job):
    n, K, D, L0, L_front, mode_kind, seed, hop_sec = job
    best_S = L0
    best_P = None
    best_tag = ""
    f_cur = L0 - K
    f_front = max(0.42, L_front - K)
    # Note: for A_GRID_BREAK where L0 == K+1 (integer grid), K_eff = K, f_cur = 1.0!
    if L0 >= K + 1.0 - 1e-6:
        f_cur = 1.0
        f_front = max(0.985, L_front - K)

    assemblies = []  # (f_try, tag, sol, near_feas)
    t_milp = time.time()

    if mode_kind in ("band", "both"):
        # Single-band Erdos-Graham MILP (exact-angle solve_theta(N, N-1, f))
        # Scan f from f_front - 0.004 up to f_cur + 0.004
        f_vals = np.linspace(max(0.45, f_front - 0.003), min(0.999, f_cur + 0.004), 7)
        # Choose N range matching K and f
        if f_cur < 0.62:
            Ns = [N for N in range(7, min(18, K)) if N - 1 <= K]
        else:
            Ns = [N for N in range(5, min(15, K)) if N - 1 <= K]
        for f_try in f_vals:
            if time.time() - t_milp > 260.0 or len(assemblies) >= 6:
                break
            hit_f = False
            for N in Ns:
                if time.time() - t_milp > 260.0 or len(assemblies) >= 6:
                    break
                if BG.solve_theta(N, N - 1, float(f_try)) is None:
                    continue
                y0_opts = sorted({0, (K - N + 1) // 2})
                for y0 in y0_opts:
                    for phase in (0.0, 0.25, 0.5, 0.75):
                        if time.time() - t_milp > 260.0:
                            break
                        try:
                            cnt, sol = BG.assemble2(
                                K,
                                float(f_try),
                                N,
                                float(y0),
                                float(phase),
                                time_limit=8.0,
                            )
                            if cnt >= n and sol is not None:
                                sol_t = BG.trim(sol, n, y0, y0 + N - 1 + float(f_try))
                                assemblies.append(
                                    (
                                        float(f_try),
                                        f"bg2_N{N}_y{y0}_f{f_try:.3f}",
                                        sol_t,
                                        False,
                                    )
                                )
                                hit_f = True
                                break
                        except Exception:
                            continue
                    if hit_f:
                        break
                # Also try rich=True on best N if assemble2 didn't hit at this f
                if not hit_f and time.time() - t_milp < 220.0:
                    try:
                        y0 = (K - N + 1) // 2
                        cnt, sol = BG.assemble(
                            K,
                            float(f_try),
                            N,
                            float(y0),
                            0.25,
                            ec=4,
                            phases=6,
                            time_limit=10.0,
                            rich=True,
                        )
                        if cnt >= n and sol is not None:
                            sol_t = BG.trim(sol, n, y0, y0 + N - 1 + float(f_try))
                            assemblies.append(
                                (float(f_try), f"bgr_N{N}_f{f_try:.3f}", sol_t, False)
                            )
                            hit_f = True
                    except Exception:
                        pass
            if hit_f and len(assemblies) >= 3:
                break

    if mode_kind in ("lband", "both"):
        # Split-region Two-Band L-Band MILP (lband2.assemble_split) — both exact & tol=0.015 near-feasible!
        f_vals = np.linspace(max(0.55, f_front - 0.002), min(0.9985, f_cur + 0.004), 6)
        Ns = [N for N in range(4, min(13, K - 2)) if N - 1 <= K]
        for f_try in f_vals:
            if time.time() - t_milp > 380.0 or len(assemblies) >= 8:
                break
            hit_l = False
            for N in Ns:
                if time.time() - t_milp > 380.0 or len(assemblies) >= 8:
                    break
                if BG.solve_theta(N, N - 1, float(f_try)) is None:
                    continue
                for mode in ("right_ref", "right_rot"):
                    for tol_use in (1e-7, 0.014):
                        if time.time() - t_milp > 380.0:
                            break
                        try:
                            cnt, sol = LB2.assemble_split(
                                K,
                                float(f_try),
                                N,
                                0.25,
                                0.25,
                                mode,
                                ec=4,
                                phases=5,
                                jm=2.5,
                                rich=0.20,
                                time_limit=8.0,
                                tol=tol_use,
                            )
                            if cnt >= n and sol is not None:
                                sol_t = trim_general(sol, n)
                                assemblies.append(
                                    (
                                        float(f_try),
                                        f"lb2_N{N}_{mode[:4]}_t{int(tol_use * 1000)}",
                                        sol_t,
                                        tol_use > 1e-5,
                                    )
                                )
                                hit_l = True
                                break
                        except Exception:
                            continue
                    if hit_l:
                        break

    # Quench top MILP assemblies with Rust fq + Stock Hopper!
    assemblies.sort(key=lambda z: z[0])
    with tempfile.TemporaryDirectory(prefix=f"w32b_{n}_") as tmp:
        for ai, (f_try, atag, sol_t, near_feas) in enumerate(assemblies[:4]):
            Qa, Sa = fq_quench_and_hop(
                n,
                sol_t,
                best_S,
                seed + ai * 4099,
                tmp,
                allow_overlap=near_feas,
                hop_sec=hop_sec,
            )
            if Qa is not None and Sa < best_S - 1e-8:
                best_S, best_P, best_tag = Sa, Qa, atag

    if best_P is not None and best_S < L0 - 1e-9:
        cur_a = w32a.safe_open(best_P)
        cur_s = float(E.side(cur_a))
        try:
            xs, ys, ts, Ls = SLP.polish(
                cur_a[:, 0],
                cur_a[:, 1],
                cur_a[:, 2],
                iters=120,
                r0=2e-3,
                rmin=1e-10,
                tol_sep=-1e-13,
                time_limit=25.0,
            )
            Qs = w32a.safe_open(np.column_stack([xs, ys, ts]))
            Ss = float(E.side(Qs))
            if Ss < cur_s - 1e-10 and E.min_gap(Qs) >= -1e-12:
                cur_a, cur_s = Qs, Ss
        except Exception:
            pass
        try:
            cur_a = w12.lp_rot_polish(cur_a)
        except Exception:
            pass
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_a)
        if ok and Lf < L0 - 1e-9:
            return n, best_tag, (Lf, xf, yf, tf)
    return n, best_tag, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2700)
    ap.add_argument("--hop-sec", type=int, default=160)
    ap.add_argument("--max-targets", type=int, default=128)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    front_rows = json.load(open(args.frontier_json))
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    grid_breaks = {605, 762, 819, 877, 1000}
    k2p1_vip = {
        401,
        530,
        577,
        677,
        730,
        785,
        842,
        901,
        962,
        731,
        786,
        843,
        902,
        963,
        904,
    }
    scored = []
    for n in range(150, 1001):
        K = math.isqrt(n)
        if K * K == n:
            continue
        is_gb = n in grid_breaks and cur[n][0] >= K + 1 - 1e-6
        if not is_gb and cur[n][0] >= K + 1 - 1e-6:
            continue
        Lf = front_L.get(n, cur[n][0])
        gap_ds = max(0.0, (cur[n][0] - Lf) / math.sqrt(n))
        if gap_ds > 1e-6 or is_gb or n in k2p1_vip:
            D = n - K * K
            if is_gb:
                mk = "lband"
            elif n in k2p1_vip or (cur[n][0] - K) < 0.68:
                mk = "band"
            elif (cur[n][0] - K) > 0.80:
                mk = "lband"
            else:
                mk = "both"
            boost = 0.01 if is_gb else (0.003 if n in k2p1_vip else 0.0)
            scored.append((gap_ds + boost, n, K, D, Lf, mk))
    scored.sort(reverse=True)
    picked = scored[: args.max_targets]
    my_slice = picked[args.shard :: args.num_shards]
    jobs = [
        (
            n,
            K,
            D,
            cur[n][0],
            Lf,
            mk,
            3220001 + args.shard * 10169 + i * 331,
            args.hop_sec,
        )
        for i, (_, n, K, D, Lf, mk) in enumerate(my_slice)
    ]

    print(
        f"[W32B-MILP s={args.shard}/{args.num_shards}] Base={tot_score:.10f} "
        f"(loaded {n_files} win files), my_targets({len(my_slice)})={[(x[1], x[5]) for x in my_slice]} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    wins = {}
    tot_ds = 0.0
    done = 0

    def save_ckpt():
        payload = {}
        for k_n, (Lf, xf, yf, tf) in sorted(wins.items()):
            payload[f"L_{k_n}"] = np.array([Lf], dtype=np.float64)
            payload[f"x_{k_n}"] = np.asarray(xf, dtype=np.float64)
            payload[f"y_{k_n}"] = np.asarray(yf, dtype=np.float64)
            payload[f"t_{k_n}"] = np.asarray(tf, dtype=np.float64)
        np.savez_compressed(args.out_npz, **payload)

    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(worker_milp, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W32B ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                dL = Lf - cur[n][0]
                if dL < -1e-9:
                    dS = dL / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W32B-MILP WIN!!] n={n:4d} {tag:22s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W32B s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
