#!/usr/bin/env python3
"""Wave 5 Engine B (Cloud Cluster): Exhaustive Arm-Stretch (stretch.py +-1, +-2 steps across all 1000 packings)
+ Near-Miss Kick/Loosen Chains + Multi-Schedule Re-Quench Rounds + Decimal64 Certification.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.environ.get("SQ_DS_DIR", HERE)
sys.path.insert(0, DS_DIR)

import stretch as st_mod
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


def side_rad(arr):
    hx = 0.5 * (np.abs(np.cos(arr[:, 2])) + np.abs(np.sin(arr[:, 2])))
    lx = float(np.max(arr[:, 0] + hx) - np.min(arr[:, 0] - hx))
    ly = float(np.max(arr[:, 1] + hx) - np.min(arr[:, 1] - hx))
    return max(lx, ly)


def write_fq_cfg(path, s, arr_rad):
    hx = 0.5 * (np.abs(np.cos(arr_rad[:, 2])) + np.abs(np.sin(arr_rad[:, 2])))
    xs = arr_rad[:, 0] - np.min(arr_rad[:, 0] - hx)
    ys = arr_rad[:, 1] - np.min(arr_rad[:, 1] - hx)
    degs = np.degrees(arr_rad[:, 2]) % 90.0
    s_use = max(float(s), float(np.max(xs + hx)), float(np.max(ys + hx)))
    with open(path, "w") as f:
        f.write(f"{len(arr_rad)} {s_use!r}\n")
        for x, y, deg in zip(xs, ys, degs):
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


def solve_stretch_target(job):
    n_tgt, s0, arr0, cand_list, budget_sec, seed = job
    rng = random.Random(seed)
    t_start = time.time()
    t_end = t_start + budget_sec
    best_s, best_a = s0, arr0
    wins = []

    with tempfile.TemporaryDirectory() as tmp:
        # 1) Try stretched donor candidates (sorted by lowest estimated side!)
        for est_L, src_n, steps, drop_k, s_arr in cand_list[:6]:
            if time.time() >= t_end - 8:
                break
            cur_a = s_arr
            if drop_k > 0:
                # Drop drop_k outermost squares cleanly
                L_tmp = side_rad(cur_a)
                dwall = np.minimum.reduce(
                    [cur_a[:, 0], L_tmp - cur_a[:, 0], cur_a[:, 1], L_tmp - cur_a[:, 1]]
                )
                drop_idx = set(np.argsort(dwall)[:drop_k].tolist())
                cur_a = np.delete(cur_a, sorted(drop_idx), axis=0)
            if len(cur_a) != n_tgt:
                continue
            L_in = side_rad(cur_a)
            local_best = None
            for sched in (
                ["--loosen", "1.0", "--flip-top", "32", "--pit", "200"],
                ["--loosen", "1.015", "--flip-top", "32", "--pit", "200"],
                [
                    "--loosen",
                    "1.0",
                    "--round-r0",
                    "0.12",
                    "--round-compress",
                    "--round-press",
                    "--round-steps",
                    "6",
                    "--round-press-outer",
                    "10",
                    "--flip-top",
                    "16",
                    "--pit",
                    "80",
                ],
            ):
                if time.time() >= t_end - 4:
                    break
                r = run_fq(
                    L_in,
                    cur_a,
                    tmp,
                    sched,
                    timeout=max(10, int(min(55, t_end - time.time()))),
                )
                if r is None:
                    continue
                if local_best is None or r[0] < local_best[0]:
                    local_best = r
                if r[0] < best_s - 1e-10:
                    best_s, best_a = r[0], r[1]
                    wins.append((f"st_{src_n}s{steps:+d}d{drop_k}", best_s))

            # Near-miss kick+loosen chain (from az05192000: cracks stretches within +0.015 of s0!)
            if local_best is not None and local_best[0] < best_s + 0.015:
                ls, la = local_best
                for rep in range(5):
                    if time.time() >= t_end - 4:
                        break
                    loosen = rng.choice(("1.005", "1.012", "1.02", "1.035"))
                    kick = rng.choice(("0.002", "0.005", "0.010", "0.018"))
                    r = run_fq(
                        ls,
                        la,
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
                            "250",
                        ],
                        timeout=max(10, int(min(55, t_end - time.time()))),
                    )
                    if r and r[0] < ls:
                        ls, la = r[0], r[1]
                    if r and r[0] < best_s - 1e-10:
                        best_s, best_a = r[0], r[1]
                        wins.append((f"stchain_{src_n}_{rep}", best_s))

        # 2) Multi-schedule re-quench on best_a with remaining budget
        while time.time() < t_end - 4:
            loosen = rng.choice(("1.005", "1.01", "1.02", "1.03", "1.05"))
            kick = rng.choice(("0.0", "0.0015", "0.004", "0.01", "0.02"))
            args_fq = [
                "--loosen",
                loosen,
                "--seed",
                str(rng.randrange(1 << 30)),
                "--flip-top",
                "48",
                "--pit",
                "250",
            ]
            if kick != "0.0":
                args_fq += ["--kick", kick]
            r = run_fq(
                best_s,
                best_a,
                tmp,
                args_fq,
                timeout=max(8, int(min(50, t_end - time.time()))),
            )
            if r and r[0] < best_s - 1e-10:
                best_s, best_a = r[0], r[1]
                wins.append((f"req_{loosen}_{kick}", best_s))

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
            ok, Lf, xf, yf, tf = verify_and_stabilize(n_tgt, best_a)
            if ok and Lf < s0 - 1e-10:
                return n_tgt, (Lf, xf, yf, tf), wins
    return n_tgt, None, wins


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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--minutes-total", type=float, default=42.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    x, y, t, L, offs = load_union(args.union_npz)
    lags = json.load(open(args.lags_json))
    lag_map = {r["n"]: r["dS"] for r in lags}

    # Enumerate ALL arm-stretches across all 1000 packings in ~2-4 seconds!
    per_target = {}
    stretched_ok = 0
    for src_n in range(80, 1001):
        s_src, a_src = get_rad_array(src_n, x, y, t, L, offs)
        for steps in (-2, -1, 1, 2):
            try:
                b, _ = st_mod.stretch(a_src, steps)
            except Exception:
                continue
            if b is None:
                continue
            m = len(b)
            if not (50 <= m <= 1002):
                continue
            L_est = side_rad(b)
            stretched_ok += 1
            for drop_k in (0, 1, 2):
                n_tgt = m - drop_k
                if 50 <= n_tgt <= 1000 and L_est < float(L[n_tgt - 1]) + 0.055:
                    per_target.setdefault(n_tgt, []).append(
                        (L_est, src_n, steps, drop_k, b)
                    )

    # Prioritize targets with highest lag dS + stretched candidates, plus top remaining lags for re-quench
    all_targets = sorted(
        set(per_target.keys()) | {r["n"] for r in lags[:240]},
        key=lambda n: (-lag_map.get(n, 0.0), -len(per_target.get(n, []))),
    )
    my_targets = [
        n for idx, n in enumerate(all_targets) if idx % args.num_shards == args.shard
    ]
    per_sec = max(
        35.0, (args.minutes_total * 60.0 * args.procs) / max(1, len(my_targets))
    )
    print(
        f"W5 Stretch+ReQuench Shard {args.shard}/{args.num_shards}: generated {stretched_ok} stretches across {len(per_target)} targets ({time.time() - t0:.1f}s); running {len(my_targets)} targets at {per_sec:.1f}s/target",
        flush=True,
    )

    jobs = []
    for n in my_targets:
        s0, a0 = get_rad_array(n, x, y, t, L, offs)
        cands = sorted(per_target.get(n, []), key=lambda z: z[0])
        jobs.append((n, s0, a0, cands, per_sec, 20261126 + 977 * n + args.shard))

    improved = {}
    tot_gain = 0.0
    done = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(solve_stretch_target, j) for j in jobs]
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
                    f"W5_STRETCH_WIN [{done}/{len(jobs)}] n={n:4d} {s0:.9f} -> {Lf:.9f} dL={s0 - Lf:.6f} dS={gain:.6f} tot={tot_gain:.6f} via={[w[0] for w in wins]}",
                    flush=True,
                )
            else:
                if done % 15 == 0:
                    print(
                        f"W5_STRETCH_PROG [{done}/{len(jobs)}] wins={len(improved)} tot={tot_gain:.6f}",
                        flush=True,
                    )
            if done % 10 == 0 and improved:
                save_ckpt(args.out_npz, improved)

    save_ckpt(args.out_npz, improved)
    print(
        f"DONE W5 Stretch Shard {args.shard}: {len(improved)} wins, total_gain={tot_gain:.6f} -> {args.out_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
