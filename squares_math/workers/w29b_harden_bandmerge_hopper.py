#!/usr/bin/env python3
"""Wave 29B: Nakajima/Ostashev Soft-to-Rigid Momentum Physics (`harden` Rust binary) -> `fq` -> `sq_engine.Hopper`.

Brand-new untouched engine compiled from fq_crate/src/harden.rs:
  - Mode `reharden` (Shards 0..7, 32 cores, Top-128 frontier gaps):
      Softens every square (Minkowski radius r0=0.08..0.28, stiffness ramp --stiff0 0.25..1.0,
      optional --hold-move 1 & --sym-md 1), runs 10^4 momentum steps with transient overlap tunneling
      to trigger band merges impossible for quasi-static ALM (`fq`), then quenches via `fq` and
      finishes promising basins with Ross Wade's Force-Guided LP Hopper!
  - Mode `fresh` (Shards 8..11, 16 cores, Top-24 medium-N topology traps like n=234, 367, 333, 261, 332, 456, 491):
      Runs `harden fresh` (hot disk soup -> square morph under momentum physics) + `anneal run` (NPT MC)
      from scratch -> `fq` -> `sq_engine.Hopper` to break decades-old L-extended topology traps!
"""

import argparse
import json
import math
import os
import subprocess
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
import w5b_2d_faultdp_gadgetq as f2d
import w12a_super_bfmelt_rp as w12

HARDEN_BIN = os.environ.get("HARDEN_BIN", "/tmp/sq_bins/harden")
ANNEAL_BIN = os.environ.get("ANNEAL_BIN", "/tmp/sq_bins/anneal")


def safe_open(P):
    try:
        return E.to_origin(E.open_contacts(P))
    except Exception:
        return E.to_origin(P)


def warm_numba():
    P = E.to_origin(
        np.array(
            [
                [0.5, 0.5, 0.0],
                [1.5, 0.5, 0.0],
                [0.5, 1.5, 0.0],
                [1.5, 1.5, 0.0],
                [2.5, 0.5, 0.15],
            ],
            dtype=np.float64,
        )
    )
    E.min_gap(P)
    E.side(P)
    safe_open(P)
    H = E.Hopper(P, seed=0)
    if H.applicable():
        H.run(1)


def finish_candidate(n, L0, arr_rad, tmp, seed, hop_sec=130):
    cur_P = safe_open(arr_rad)
    cur_S = float(E.side(cur_P))

    # Quick fq quench first to snap contacts cleanly
    if os.path.exists(f2d.FQ_BIN):
        r_q = f2d.run_fq(
            cur_S,
            cur_P,
            tmp,
            ["--loosen", "1.0", "--flip-top", "40", "--pit", "260"],
            timeout=50,
        )
        if r_q is not None and r_q[0] < cur_S:
            cur_S, cur_P = r_q[0], safe_open(r_q[1])

    if cur_S > L0 + 0.008:
        return cur_S, None

    # Ping-pong: slam promising new momentum basin with Force-Guided LP Hopper!
    try:
        H = E.Hopper(cur_P, seed=seed)
        if H.applicable():
            Q2, S2 = H.run(hop_sec)
            Q2 = safe_open(Q2)
            S2 = float(E.side(Q2))
            if S2 < cur_S and E.min_gap(Q2) >= -1e-12:
                cur_P, cur_S = Q2, S2
    except Exception:
        pass

    if cur_S < L0 - 1e-9:
        if os.path.exists(f2d.FQ_BIN):
            r_fin = f2d.run_fq(
                cur_S,
                cur_P,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "56", "--pit", "500"],
                timeout=60,
            )
            if r_fin is not None and r_fin[0] < cur_S:
                cur_S, cur_P = r_fin[0], r_fin[1]
        try:
            cur_P = w12.lp_rot_polish(cur_P)
        except Exception:
            pass
        ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_P)
        if ok and Lf < L0 - 1e-9:
            return Lf, (Lf, xf, yf, tf)
    return cur_S, None


def worker_reharden(job):
    n, L0, arr0, seed = job
    best_res = None
    best_L = L0
    best_tag = ""

    # 6 diverse Nakajima/Ostashev soft-to-rigid momentum re-hardening protocols!
    protocols = [
        # (tag, r0, pre, hold, release, settle, hold_f, noise, stiff0, hold_move, sym_md)
        (
            "rh_soft",
            "0.10",
            "1800",
            "7000",
            "3500",
            "2500",
            "1.000",
            "0.004",
            "1.0",
            "0",
            "0",
        ),
        (
            "rh_tun",
            "0.16",
            "2200",
            "9000",
            "4000",
            "2500",
            "1.002",
            "0.006",
            "0.35",
            "0",
            "0",
        ),
        (
            "rh_deep",
            "0.24",
            "2500",
            "10000",
            "4500",
            "3000",
            "1.005",
            "0.008",
            "0.25",
            "0",
            "0",
        ),
        (
            "rh_mv1",
            "0.14",
            "1800",
            "8000",
            "4000",
            "2500",
            "0.998",
            "0.005",
            "0.50",
            "1",
            "0",
        ),
        (
            "rh_mv2",
            "0.22",
            "2000",
            "9000",
            "4500",
            "2500",
            "1.000",
            "0.007",
            "0.30",
            "1",
            "0",
        ),
        (
            "rh_sym",
            "0.16",
            "2000",
            "8500",
            "4000",
            "2500",
            "1.001",
            "0.005",
            "0.40",
            "0",
            "1",
        ),
    ]

    with tempfile.TemporaryDirectory(prefix=f"w29b_rh_{n}_") as tmp:
        pin = os.path.join(tmp, "in.txt")
        pout = os.path.join(tmp, "out.txt")
        for pi, (tag, r0, pre, hold, rel, sett, hf, noise, k0, hmv, sym) in enumerate(
            protocols
        ):
            cur_arr = (
                np.column_stack([best_res[1], best_res[2], best_res[3]])
                if best_res is not None
                else arr0
            )
            f2d.write_fq_cfg(pin, best_L, cur_arr)
            if os.path.exists(pout):
                os.remove(pout)
            cmd = [
                HARDEN_BIN,
                "reharden",
                "--in",
                pin,
                "--out",
                pout,
                "--seed",
                str(seed + pi * 1009),
                "--r0",
                r0,
                "--pre",
                pre,
                "--hold",
                hold,
                "--release",
                rel,
                "--settle",
                sett,
                "--hold-f",
                hf,
                "--noise",
                noise,
                "--mu",
                "0.02",
                "--stiff0",
                k0,
                "--hold-move",
                hmv,
                "--sym-md",
                sym,
            ]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=110)
            except subprocess.TimeoutExpired:
                continue
            if r.returncode != 0 or not os.path.exists(pout):
                continue
            try:
                s_out, arr_out = f2d.read_fq_cfg_rad(pout)
            except Exception:
                continue
            if s_out > best_L + 0.030:
                continue
            s_fin, res = finish_candidate(
                n, best_L, arr_out, tmp, seed + 50000 + pi * 313, hop_sec=120
            )
            if res is not None and res[0] < best_L - 1e-9:
                best_L = res[0]
                best_res = res
                best_tag = tag
    return n, best_tag, best_res


def worker_fresh(job):
    n, L0, seed, trials = job
    best_res = None
    best_L = L0
    best_tag = ""

    with tempfile.TemporaryDirectory(prefix=f"w29b_fr_{n}_") as tmp:
        pout = os.path.join(tmp, "out.txt")
        for ti in range(trials):
            if os.path.exists(pout):
                os.remove(pout)
            if ti % 3 != 2 or not os.path.exists(ANNEAL_BIN):
                # Nakajima fresh soft-to-rigid disk-to-square momentum morph!
                mu = "0.02" if ti % 2 == 0 else "0.012"
                cmd = [
                    HARDEN_BIN,
                    "fresh",
                    "--n",
                    str(n),
                    "--seed",
                    str(seed + ti * 2027),
                    "--compact",
                    "4000",
                    "--morph",
                    "32000",
                    "--settle",
                    "3000",
                    "--mu",
                    mu,
                    "--anneal",
                    "1",
                    "--out",
                    pout,
                ]
                tag = f"fresh_h{ti}"
            else:
                # NPT hard rounded-square MC from hot disk soup!
                cmd = [
                    ANNEAL_BIN,
                    "run",
                    "--n",
                    str(n),
                    "--seed",
                    str(seed + ti * 3037),
                    "--sweeps",
                    "18000",
                    "--bp0",
                    "1",
                    "--bp1",
                    "3000",
                    "--t0",
                    "0.18",
                    "--t1",
                    "0.82",
                    "--p",
                    "1.0",
                    "--out",
                    pout,
                ]
                tag = f"fresh_a{ti}"
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=140)
            except subprocess.TimeoutExpired:
                continue
            if r.returncode != 0 or not os.path.exists(pout):
                continue
            try:
                s_out, arr_out = f2d.read_fq_cfg_rad(pout)
            except Exception:
                continue
            if s_out > best_L + 0.045:
                continue
            s_fin, res = finish_candidate(
                n, best_L, arr_out, tmp, seed + 70000 + ti * 409, hop_sec=120
            )
            if res is not None and res[0] < best_L - 1e-9:
                best_L = res[0]
                best_res = res
                best_tag = tag
    return n, best_tag, best_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--frontier-json", required=True)
    ap.add_argument("--mode", choices=["reharden", "fresh"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2750)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    warm_numba()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )
    front_rows = json.load(open(args.frontier_json))
    front_L = {int(r["n"]): float(r["L_front"]) for r in front_rows}

    scored = []
    for n in range(150, 1000):
        K = math.isqrt(n)
        if K * K == n or cur[n][0] >= K + 1 - 1e-6:
            continue
        gap_ds = max(0.0, (cur[n][0] - front_L.get(n, cur[n][0])) / math.sqrt(n))
        if gap_ds > 1e-7:
            scored.append((gap_ds, n))
    scored.sort(reverse=True)

    if args.mode == "reharden":
        targets = [n for _, n in scored[:128]]
        my_slice = targets[args.shard :: args.num_shards]
        jobs = [
            (n, cur[n][0], cur[n][1], 2920001 + args.shard * 10133 + i * 263)
            for i, n in enumerate(my_slice)
        ]
        fn_worker = worker_reharden
    else:
        # Medium-N topology traps (n <= 520, largest frontier gaps!)
        med = [n for _, n in scored if n <= 520][:24]
        my_slice = med[args.shard :: args.num_shards]
        jobs = [
            (n, cur[n][0], 2930001 + args.shard * 10141 + i * 271, 14)
            for i, n in enumerate(my_slice)
        ]
        fn_worker = worker_fresh

    print(
        f"[W29B-Harden({args.mode}) s={args.shard}/{args.num_shards}] Base={tot_score:.10f}, "
        f"my_targets({len(my_slice)})={my_slice} ({time.time() - t0:.1f}s)",
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
        futs = [ex.submit(fn_worker, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W29B ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                dL = Lf - cur[n][0]
                if dL < -1e-9:
                    dS = dL / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W29B HARDEN WIN!!] n={n:4d} {tag:14s} {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"dL={dL:.7f} dS={dS:.8f} ({done}/{len(jobs)}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() - t0 > args.budget_sec:
                print(
                    f"[W29B s={args.shard}] Reached graceful budget {args.budget_sec}s; cancelling remaining.",
                    flush=True,
                )
                for f_rem in futs:
                    f_rem.cancel()
                break

    save_ckpt()
    print(
        f"[W29B({args.mode}) s={args.shard} DONE] {len(wins)} certified wins, total_dS={tot_ds:.9f} -> {args.out_npz} ({time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
