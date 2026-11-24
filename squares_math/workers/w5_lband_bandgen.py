#!/usr/bin/env python3
"""Wave 5 Engine A (Cloud Cluster): Exact-Angle Two-Arm L-Band (lband2.assemble_split)
+ Slid-Column Single-Band (band_gen.assemble2) + Step-Up-Then-fq-Quench + Deep Polish + Decimal64 Certification.
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
import numpy as np
import scipy.optimize._milp as _milp

# 18x speedup on multi-core workers: prevent HiGHS thread-pool spinning inside ProcessPoolExecutor!
if hasattr(_milp, "_highs_wrapper"):
    _hw = _milp._highs_wrapper
    _milp._highs_wrapper = lambda *a: _hw(*a[:-1], dict(a[-1], threads=1))

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
sys.path.insert(0, DS_DIR)

import band_gen as bg
import lband2 as lb2
from decimal_verifier import verify_puzzle_decimal

FQ_BIN = os.environ.get("FQ_BIN", "/tmp/sq_bins/fq")
TARGET_GAP = Decimal("1e-9")
SAFE_GAP = Decimal("1.05e-9")


def load_union(npz_path):
    d = np.load(npz_path)
    x, y, t, L = d["x"], d["y"], d["t"], d["L"]
    offs = [0]
    for n in range(1, 1001):
        offs.append(offs[-1] + n)
    return x, y, t, L, offs


def get_rad_array(n, x, y, t, L, offs):
    s0, e0 = offs[n - 1], offs[n]
    xs = x[s0:e0].copy()
    ys = y[s0:e0].copy()
    ts = t[s0:e0].copy()
    hx = 0.5 * (np.abs(np.cos(ts)) + np.abs(np.sin(ts)))
    xs -= np.min(xs - hx)
    ys -= np.min(ys - hx)
    return float(L[n - 1]), np.column_stack([xs, ys, ts])


def write_fq_cfg(path, s, arr_rad):
    degs = np.degrees(arr_rad[:, 2]) % 90.0
    with open(path, "w") as f:
        f.write(f"{len(arr_rad)} {float(s)!r}\n")
        for (x, y), deg in zip(arr_rad[:, :2], degs):
            f.write(f"{float(x)!r} {float(y)!r} {float(deg)!r}\n")


def read_fq_cfg_rad(path):
    with open(path) as f:
        lines = [l.split() for l in f if l.strip() and not l.startswith("#")]
    n, s = int(lines[0][0]), float(lines[0][1])
    arr = np.array(
        [
            [float(a), float(b), math.radians(float(c) % 90.0)]
            for a, b, c, *_ in lines[1 : n + 1]
        ],
        dtype=np.float64,
    )
    return s, arr


def run_fq(s, arr_rad, tmp, extra_args, timeout=90):
    a = os.path.join(tmp, "in.txt")
    b = os.path.join(tmp, "out.txt")
    write_fq_cfg(a, s, arr_rad)
    if os.path.exists(b):
        os.remove(b)
    try:
        r = subprocess.run(
            [FQ_BIN, "quench", "--in", a, "--out", b, *extra_args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not os.path.exists(b):
        return None
    lines = r.stdout.strip().splitlines()
    if not lines:
        return None
    try:
        d = json.loads(lines[-1])
    except Exception:
        return None
    mg, mw = d.get("min_gap"), d.get("min_wall")
    if mg is None or mw is None or not (mg >= -1e-12 and mw >= -1e-12):
        return None
    s2, arr2 = read_fq_cfg_rad(b)
    if len(arr2) != len(arr_rad):
        return None
    return s2, arr2


def verify_and_stabilize(n, arr_rad):
    xs, ys, ts = arr_rad[:, 0].copy(), arr_rad[:, 1].copy(), arr_rad[:, 2].copy()
    hx = 0.5 * (np.abs(np.cos(ts)) + np.abs(np.sin(ts)))
    xs -= 0.5 * (float(np.min(xs - hx)) + float(np.max(xs + hx)))
    ys -= 0.5 * (float(np.min(ys - hx)) + float(np.max(ys + hx)))
    rows = [
        (Decimal(repr(float(a))), Decimal(repr(float(b))), Decimal(repr(float(c))))
        for a, b, c in zip(xs, ys, ts)
    ]
    pv = verify_puzzle_decimal(n, rows, min_required_gap=TARGET_GAP)
    if pv.valid and pv.min_separation_gap >= TARGET_GAP:
        return True, float(pv.L), xs, ys, ts
    pv0 = verify_puzzle_decimal(n, rows, min_required_gap=Decimal("-1e-11"))
    if not pv0.valid:
        return False, 1e18, xs, ys, ts
    gap0 = max(0.0, float(pv0.min_separation_gap))
    delta = max(1e-11, float(SAFE_GAP) - gap0)
    for _ in range(24):
        sc = 1.0 + delta
        xn, yn = xs * sc, ys * sc
        rows_n = [
            (Decimal(repr(float(a))), Decimal(repr(float(b))), Decimal(repr(float(c))))
            for a, b, c in zip(xn, yn, ts)
        ]
        pvn = verify_puzzle_decimal(n, rows_n, min_required_gap=TARGET_GAP)
        if pvn.valid and pvn.min_separation_gap >= TARGET_GAP:
            return True, float(pvn.L), xn, yn, ts
        delta *= 2.0
    return False, 1e18, xs, ys, ts


def trim_to_n(sol, n):
    if len(sol) <= n:
        return sol
    # Drop surplus squares furthest from box interior (prefer dropping uprights near diagonal/corners)
    keep = np.argsort(-np.abs(sol[:, 0] - sol[:, 1]))[:n]
    return sol[np.sort(keep)]


def quench_chain(sol_rad, s_init, best_s, rng, tmp, t_stop, tag_prefix):
    """Step-up-then-quench + multi-loosen follow-up chain on a fresh L-band / Single-band assembly."""
    cur_best_s = best_s
    cur_best_a = None
    best_tag = None

    schedules = [
        (["--loosen", "1.0", "--flip-top", "32", "--pit", "200"], "q0"),
        (
            [
                "--loosen",
                "1.015",
                "--kick",
                "0.004",
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "32",
                "--pit",
                "200",
            ],
            "qk1",
        ),
        (["--loosen", "1.02", "--finish"], "qfin"),
    ]
    best_local = None
    for args_fq, stg in schedules:
        if time.time() >= t_stop:
            break
        r = run_fq(
            s_init,
            sol_rad,
            tmp,
            args_fq,
            timeout=max(10, int(min(60, t_stop - time.time()))),
        )
        if r is None:
            continue
        if best_local is None or r[0] < best_local[0]:
            best_local = r
        if r[0] < cur_best_s - 1e-10:
            cur_best_s, cur_best_a, best_tag = r[0], r[1], f"{tag_prefix}_{stg}"

    # If within +0.012 of best_s (or already improved), run 3 extra kick+loosen hops from best_local!
    if best_local is not None and best_local[0] < cur_best_s + 0.012:
        src_s, src_a = best_local
        for rep in range(3):
            if time.time() >= t_stop:
                break
            loosen = rng.choice(("1.005", "1.012", "1.025", "1.04"))
            kick = rng.choice(("0.002", "0.005", "0.012"))
            r = run_fq(
                src_s,
                src_a,
                tmp,
                [
                    "--loosen",
                    loosen,
                    "--kick",
                    kick,
                    "--seed",
                    str(rng.randrange(1 << 30)),
                    "--flip-top",
                    "48",
                    "--pit",
                    "300",
                ],
                timeout=max(10, int(min(60, t_stop - time.time()))),
            )
            if r and r[0] < src_s:
                src_s, src_a = r[0], r[1]
            if r and r[0] < cur_best_s - 1e-10:
                cur_best_s, cur_best_a, best_tag = r[0], r[1], f"{tag_prefix}_hop{rep}"
    return cur_best_s, cur_best_a, best_tag


def solve_target(job):
    n, s0, arr0, lb_best, gap_hint, budget_sec, seed = job
    rng = random.Random(seed)
    K = math.isqrt(n)
    D = n - K * K
    f0 = s0 - K
    fb = lb_best - K
    t_start = time.time()
    t_end = t_start + budget_sec
    best_s, best_a = s0, arr0
    wins = []

    # Candidate N values sorted by proximity to hint angle (if any) or classical optimal band thickness
    hint_angs = [
        float(v) for v in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*°", gap_hint or "")
    ]
    n_lo = max(2, min(4, K // 5))
    n_hi = min(15, K - 4)
    cand_Ns = list(range(n_lo, n_hi + 1))

    def n_priority(N_val):
        th = bg.solve_theta(N_val, N_val - 1, f0)
        if th is None:
            return 999.0
        deg = math.degrees(th)
        if hint_angs:
            return min(abs(deg - h) for h in hint_angs)
        return abs(deg - 26.0)

    cand_Ns.sort(key=n_priority)
    # Keep top 8 most geometrically relevant N values
    cand_Ns = cand_Ns[:8]

    with tempfile.TemporaryDirectory() as tmp:
        # =====================================================================
        # TRACK A: Two-Arm Exact-Angle L-Band (lband2.assemble_split) for K>=18
        # =====================================================================
        if K >= 18:
            t_a = t_start + 0.50 * budget_sec
            # Step-up f grid: try below f0, at f0, and slightly above f0 (up to +0.014, where fq drops 0.005..0.015!)
            f_grid = [
                round(f, 4)
                for f in (
                    max(0.42, fb + 0.001),
                    max(0.42, 0.5 * (fb + f0)),
                    f0 + 0.002,
                    f0 + 0.007,
                    f0 + 0.013,
                )
                if 0.40 <= f < 0.998
            ]
            # Deduplicate preserving ascending order
            f_grid = sorted(set(f_grid))
            hits_L = []
            for f_try in f_grid:
                if time.time() >= t_a - 25 or len(hits_L) >= 4:
                    break
                for N_val in cand_Ns[:6]:
                    if time.time() >= t_a - 25 or len(hits_L) >= 4:
                        break
                    if K - N_val + 1 < 4:
                        continue
                    for mode in ("right_rot", "right_ref"):
                        if time.time() >= t_a - 25:
                            break
                        # Fast phase sweep (5 phases along diagonal + offsets)
                        phases = [
                            (0.0, 0.0),
                            (0.2, 0.2),
                            (0.5, 0.5),
                            (0.8, 0.8),
                            (0.0, 0.5),
                            (0.5, 0.0),
                            (0.3, 0.7),
                        ]
                        for pt, pr in phases:
                            if time.time() >= t_a - 25:
                                break
                            try:
                                cnt, sol = lb2.assemble_split(
                                    K,
                                    f_try,
                                    N_val,
                                    pt,
                                    pr,
                                    mode,
                                    ec=4,
                                    phases=6,
                                    jm=3.0,
                                    rich=0.0,
                                    time_limit=6.0,
                                    tol=1e-7,
                                )
                            except Exception:
                                continue
                            if cnt >= n:
                                hits_L.append((f_try, cnt, N_val, mode, pt, pr, sol))
                                break
                            elif cnt >= n - 2:
                                # Retry exact hit with rich=0.15 junction candidates!
                                try:
                                    cnt2, sol2 = lb2.assemble_split(
                                        K,
                                        f_try,
                                        N_val,
                                        pt,
                                        pr,
                                        mode,
                                        ec=4,
                                        phases=6,
                                        jm=3.0,
                                        rich=0.15,
                                        time_limit=10.0,
                                        tol=1e-7,
                                    )
                                    if cnt2 >= n:
                                        hits_L.append(
                                            (
                                                f_try,
                                                cnt2,
                                                N_val,
                                                mode + "_rich",
                                                pt,
                                                pr,
                                                sol2,
                                            )
                                        )
                                        break
                                except Exception:
                                    pass
            hits_L.sort(key=lambda z: (z[0], -z[1]))
            for f_try, cnt, N_val, mode, pt, pr, sol in hits_L[:3]:
                if time.time() >= t_a:
                    break
                s_sol = trim_to_n(sol, n)
                L_asm = bg.side_L(s_sol)
                ns, na, tag = quench_chain(
                    s_sol, L_asm, best_s, rng, tmp, t_a, f"lband_N{N_val}_{mode}"
                )
                if na is not None and ns < best_s - 1e-10:
                    best_s, best_a = ns, na
                    wins.append((tag, best_s))

        # =====================================================================
        # TRACK B: Slid-Column Single-Band Erdos-Graham (band_gen.assemble2)
        # =====================================================================
        t_b = t_start + 0.85 * budget_sec
        f_grid_b = sorted(
            {
                round(f, 4)
                for f in (max(0.42, fb + 0.002), f0 + 0.001, f0 + 0.007, f0 + 0.014)
                if 0.40 <= f < 0.998
            }
        )
        hits_B = []
        for f_try in f_grid_b:
            if time.time() >= t_b - 20 or len(hits_B) >= 4:
                break
            for N_val in cand_Ns[:6]:
                if time.time() >= t_b - 20 or len(hits_B) >= 4:
                    break
                y0_list = sorted({0.0, 1.0, float(max(0, (K - N_val) // 2))})
                for y0 in y0_list:
                    for ph in (0.0, 0.25, 0.5, 0.75):
                        if time.time() >= t_b - 20:
                            break
                        try:
                            cnt, sol = bg.assemble2(
                                K, f_try, N_val, y0, ph, time_limit=5.0
                            )
                        except Exception:
                            continue
                        if cnt >= n:
                            hits_B.append((f_try, cnt, N_val, y0, ph, sol))
                            break
        hits_B.sort(key=lambda z: (z[0], -z[1]))
        for f_try, cnt, N_val, y0, ph, sol in hits_B[:3]:
            if time.time() >= t_b:
                break
            s_sol = trim_to_n(sol, n)
            L_asm = bg.side_L(s_sol)
            ns, na, tag = quench_chain(
                s_sol, L_asm, best_s, rng, tmp, t_b, f"sband_N{N_val}_y{int(y0)}"
            )
            if na is not None and ns < best_s - 1e-10:
                best_s, best_a = ns, na
                wins.append((tag, best_s))

        # =====================================================================
        # TRACK C: Multi-loosen / kick re-quench on best_a (remaining 15% budget)
        # =====================================================================
        while time.time() < t_end - 4:
            loosen = rng.choice(("1.005", "1.012", "1.02", "1.035", "1.05"))
            kick = rng.choice(("0.0", "0.002", "0.006", "0.015"))
            args_fq = [
                "--loosen",
                loosen,
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "32",
                "--pit",
                "200",
            ]
            if kick != "0.0":
                args_fq += ["--kick", kick]
            r = run_fq(
                best_s,
                best_a,
                tmp,
                args_fq,
                timeout=max(8, int(min(45, t_end - time.time()))),
            )
            if r and r[0] < best_s - 1e-10:
                best_s, best_a = r[0], r[1]
                wins.append((f"req_L{loosen}_k{kick}", best_s))

        # Final deep polish + exact Decimal64 >= 1.05e-9 certification
        if best_s < s0 - 1e-10:
            r_fin = run_fq(
                best_s,
                best_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
                timeout=90,
            )
            if r_fin and r_fin[0] < best_s:
                best_s, best_a = r_fin[0], r_fin[1]
            ok, Lf, xf, yf, tf = verify_and_stabilize(n, best_a)
            if ok and Lf < s0 - 1e-10:
                return n, (Lf, xf, yf, tf), wins
    return n, None, wins


def save_ckpt(path, improved):
    payload = {}
    for n, (Lf, xf, yf, tf) in improved.items():
        payload[f"L_{n}"] = np.array([Lf], dtype=np.float64)
        payload[f"x_{n}"] = xf
        payload[f"y_{n}"] = yf
        payload[f"t_{n}"] = tf
    np.savez_compressed(path, **payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--lags-json", required=True)
    ap.add_argument("--gaps-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=6)
    ap.add_argument("--max-targets", type=int, default=240)
    ap.add_argument("--minutes-total", type=float, default=42.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    x, y, t, L, offs = load_union(args.union_npz)
    lags = json.load(open(args.lags_json))
    gaps = {g["n"]: g["kind"] for g in json.load(open(args.gaps_json))}

    all_t = lags[: args.max_targets]
    my_t = [r for idx, r in enumerate(all_t) if idx % args.num_shards == args.shard]
    per_sec = max(45.0, (args.minutes_total * 60.0 * args.procs) / max(1, len(my_t)))
    print(
        f"W5 LBand+BandGen Shard {args.shard}/{args.num_shards}: {len(my_t)} targets, {per_sec:.1f}s/target on {args.procs} CPUs -> {[r['n'] for r in my_t[:12]]}...",
        flush=True,
    )

    jobs = []
    for r in my_t:
        n = r["n"]
        s0, arr0 = get_rad_array(n, x, y, t, L, offs)
        jobs.append(
            (
                n,
                s0,
                arr0,
                r["Lb"],
                gaps.get(n, ""),
                per_sec,
                20261125 + 1009 * n + args.shard,
            )
        )

    improved = {}
    tot_gain = 0.0
    done = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(solve_target, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            n, res, wins = fut.result()
            s0 = float(L[n - 1])
            if res is not None:
                Lf, xf, yf, tf = res
                gain = (s0 - Lf) / math.sqrt(n)
                tot_gain += gain
                improved[n] = (Lf, xf, yf, tf)
                print(
                    f"W5_LBAND_WIN [{done}/{len(jobs)}] n={n:4d} {s0:.9f} -> {Lf:.9f} dL={s0 - Lf:.6f} dS={gain:.6f} tot={tot_gain:.6f} via={[w[0] for w in wins]}",
                    flush=True,
                )
            else:
                print(
                    f"W5_LBAND_DRY [{done}/{len(jobs)}] n={n:4d} s0={s0:.9f}",
                    flush=True,
                )
            if done % 6 == 0 and improved:
                save_ckpt(args.out_npz, improved)

    save_ckpt(args.out_npz, improved)
    print(
        f"DONE W5 LBand Shard {args.shard}: {len(improved)} wins, total_gain={tot_gain:.6f} -> {args.out_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
