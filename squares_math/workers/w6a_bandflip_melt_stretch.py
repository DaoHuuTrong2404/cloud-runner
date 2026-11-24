#!/usr/bin/env python3
"""Wave 6A Engine 2 (16 Cores / 4 Shards): Laser-Focused Proven Winners Engine:
1) Bandflip + Anneal Melt + fq (the exact combo that broke n=485 (22^2+1) by -0.000651!)
2) Arm-Stretch (+-1, +-2) + Near-Miss Kick/Loosen Chain (broke n=410 by -0.001098!)
3) Remove-1/2 from (n+1, n+2) + Deep Kick/Melt Hopping (Ross Wade's exact n=992 win!)
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

import bandflip as bf
import stretch as st_mod
from decimal_verifier import verify_puzzle_decimal

FQ_BIN = os.environ.get("FQ_BIN", "/tmp/sq_bins/fq")
ANNEAL_BIN = os.environ.get("ANNEAL_BIN", "/tmp/sq_bins/anneal")
TARGET_GAP = Decimal("1e-9")
SAFE_GAP = Decimal("1.05e-9")


def load_union(npz_path):
    d = np.load(npz_path)
    x, y, t, L = d["x"], d["y"], d["t"], d["L"]
    offs = [0]
    for n in range(1, 1001):
        offs.append(offs[-1] + n)
    return x, y, t, L, offs


def get_deg_pts(n, x, y, t, L, offs):
    s0, e0 = offs[n - 1], offs[n]
    xs, ys, ts = x[s0:e0].copy(), y[s0:e0].copy(), t[s0:e0].copy()
    hx = 0.5 * (np.abs(np.cos(ts)) + np.abs(np.sin(ts)))
    xs -= np.min(xs - hx)
    ys -= np.min(ys - hx)
    degs = np.degrees(ts) % 90.0
    return float(L[n - 1]), list(zip(xs.tolist(), ys.tolist(), degs.tolist()))


def write_cfg(path, s, pts):
    arr = np.array(pts, dtype=np.float64)
    ts = np.radians(arr[:, 2])
    hx = 0.5 * (np.abs(np.cos(ts)) + np.abs(np.sin(ts)))
    xs = arr[:, 0] - np.min(arr[:, 0] - hx)
    ys = arr[:, 1] - np.min(arr[:, 1] - hx)
    s_use = max(float(s), float(np.max(xs + hx)), float(np.max(ys + hx)))
    with open(path, "w") as f:
        f.write(f"{len(pts)} {s_use!r}\n")
        for a, b, c in zip(xs, ys, arr[:, 2]):
            f.write(f"{float(a)!r} {float(b)!r} {float(c) % 90.0!r}\n")


def read_cfg(path):
    with open(path) as f:
        lines = [l.split() for l in f if l.strip() and not l.startswith("#")]
    n, s = int(lines[0][0]), float(lines[0][1])
    pts = [(float(a), float(b), float(c) % 90.0) for a, b, c, *_ in lines[1 : n + 1]]
    return s, pts


def run_fq(s, pts, tmp, extra_args, timeout=60):
    a = os.path.join(tmp, "in.txt")
    b = os.path.join(tmp, "out.txt")
    write_cfg(a, s, pts)
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
    s2, pts2 = read_cfg(b)
    return (s2, pts2) if len(pts2) == len(pts) else None


def run_melt(s, pts, tmp, rng, timeout=40):
    a = os.path.join(tmp, "min.txt")
    b = os.path.join(tmp, "mout.txt")
    write_cfg(a, s, pts)
    if os.path.exists(b):
        os.remove(b)
    cx, cy = pts[rng.randrange(len(pts))][:2]
    try:
        r = subprocess.run(
            [
                ANNEAL_BIN,
                "melt",
                "--in",
                a,
                "--out",
                b,
                "--cx",
                repr(float(cx)),
                "--cy",
                repr(float(cy)),
                "--rad",
                f"{rng.uniform(1.8, 4.2):.3f}",
                "--rmax",
                f"{rng.uniform(0.06, 0.16):.3f}",
                "--sweeps",
                "2800",
                "--bp",
                "2200.0",
                "--seed",
                str(rng.randrange(1 << 30)),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if r.returncode != 0 or not os.path.exists(b):
        return None
    return read_cfg(b)


def stabilize_decimal(n, pts):
    arr = np.array(pts, dtype=np.float64)
    xs, ys, ts = arr[:, 0], arr[:, 1], np.radians(arr[:, 2])
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


def solve_one(job):
    n, s0, pts0, p_plus1, s_plus1, stretch_cands, budget_sec, seed = job
    rng = random.Random(seed)
    t_end = time.time() + budget_sec
    best_s, best_p = s0, pts0
    wins = []

    with tempfile.TemporaryDirectory() as tmp:
        # 1) Stretch candidates (if any!) + near-miss chain
        for sc_pts in stretch_cands[:3]:
            if time.time() >= t_end - 12:
                break
            r = run_fq(
                best_s + 0.04,
                sc_pts,
                tmp,
                ["--loosen", "1.015", "--flip-top", "32", "--pit", "200"],
                timeout=45,
            )
            if r is None:
                continue
            if r[0] < best_s - 1e-10:
                best_s, best_p = r[0], r[1]
                wins.append(("stretch", best_s))
            if r[0] < best_s + 0.015:
                ls, lp = r
                for rep in range(3):
                    if time.time() >= t_end - 8:
                        break
                    r2 = run_fq(
                        ls,
                        lp,
                        tmp,
                        [
                            "--loosen",
                            rng.choice(("1.008", "1.02")),
                            "--kick",
                            rng.choice(("0.003", "0.01")),
                            "--seed",
                            str(rng.randrange(1 << 30)),
                            "--flip-top",
                            "32",
                            "--pit",
                            "200",
                        ],
                        timeout=45,
                    )
                    if r2 and r2[0] < ls:
                        ls, lp = r2
                    if r2 and r2[0] < best_s - 1e-10:
                        best_s, best_p = r2[0], r2[1]
                        wins.append(("st_chain", best_s))

        # 2) Ross Wade's remove-1 from (n+1) + melt/hop!
        if (
            p_plus1 is not None
            and s_plus1 < best_s + 0.045
            and time.time() < t_end - 10
        ):
            arr1 = np.array(p_plus1, dtype=np.float64)
            # Try removing tilted-end squares AND outermost squares
            a_fold = np.abs((arr1[:, 2] + 45.0) % 90.0 - 45.0)
            tilt_idx = np.where(a_fold > 3.0)[0].tolist()
            dwall = np.minimum.reduce(
                [arr1[:, 0], s_plus1 - arr1[:, 0], arr1[:, 1], s_plus1 - arr1[:, 1]]
            )
            edge_idx = np.argsort(dwall)[:18].tolist()
            picks = []
            if tilt_idx:
                picks += rng.sample(tilt_idx, min(3, len(tilt_idx)))
            picks += rng.sample(edge_idx, min(3, len(edge_idx)))
            for drop_i in picks[:5]:
                if time.time() >= t_end - 8:
                    break
                cand = [p_plus1[i] for i in range(n + 1) if i != drop_i]
                r = run_fq(
                    s_plus1,
                    cand,
                    tmp,
                    ["--loosen", "1.02", "--flip-top", "32", "--pit", "180"],
                    timeout=40,
                )
                if r and r[0] < best_s - 1e-10:
                    best_s, best_p = r[0], r[1]
                    wins.append(("rem1", best_s))

        # 3) Main loop: Bandflip + Anneal Melt + Local Kick-Quench (proven n=485 killer!)
        while time.time() < t_end - 6:
            arr = np.array(best_p, dtype=np.float64)
            mode = rng.random()
            cand = None
            tag = "melt"
            if mode < 0.45:
                # Regional NPT hard-particle melt -> fq!
                m = run_melt(
                    best_s,
                    best_p,
                    tmp,
                    rng,
                    timeout=max(8, int(min(35, t_end - time.time()))),
                )
                if m is not None:
                    cand, tag = m[1], "melt"
            elif mode < 0.78:
                # Bandflip surgery -> optional melt -> fq!
                B = bf.bands(arr, min_size=8)
                if B:
                    cand, _ = bf.flip(best_s, best_p, B[rng.randrange(len(B))], rng, B)
                    tag = "bandflip"
                    if rng.random() < 0.35 and time.time() < t_end - 12:
                        m2 = run_melt(best_s + 0.02, cand, tmp, rng, timeout=25)
                        if m2 is not None:
                            cand, tag = m2[1], "bf_melt"
            else:
                # Strong cluster rotation + kick
                c0 = np.array(
                    [rng.uniform(0.5, best_s - 0.5), rng.uniform(0.5, best_s - 0.5)]
                )
                rad = rng.uniform(1.6, 3.8)
                phi = rng.choice([-1, 1]) * rng.uniform(4.0, 40.0)
                idx = np.hypot(arr[:, 0] - c0[0], arr[:, 1] - c0[1]) < rad
                c, sn = math.cos(math.radians(phi)), math.sin(math.radians(phi))
                out = arr.copy()
                d = out[idx, :2] - c0
                out[idx, 0] = c0[0] + c * d[:, 0] - sn * d[:, 1]
                out[idx, 1] = c0[1] + sn * d[:, 0] + c * d[:, 1]
                out[idx, 2] += phi
                cand, tag = [tuple(rw) for rw in out], "crot"

            if cand is None or time.time() >= t_end - 4:
                continue
            r = run_fq(
                best_s,
                cand,
                tmp,
                [
                    "--loosen",
                    rng.choice(("1.01", "1.02", "1.03")),
                    "--flip-top",
                    "32",
                    "--pit",
                    "180",
                ],
                timeout=max(8, int(min(45, t_end - time.time()))),
            )
            if r and r[0] < best_s - 1e-10:
                best_s, best_p = r[0], r[1]
                wins.append((tag, best_s))

        if best_s < s0 - 1e-10:
            r_fin = run_fq(
                best_s,
                best_p,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
                timeout=80,
            )
            if r_fin and r_fin[0] < best_s:
                best_s, best_p = r_fin[0], r_fin[1]
            ok, Lf, xf, yf, tf = stabilize_decimal(n, best_p)
            if ok and Lf < s0 - 1e-10:
                return n, (Lf, xf, yf, tf), wins
    return n, None, wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--lags-json", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--max-targets", type=int, default=160)
    ap.add_argument("--minutes-total", type=float, default=26.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    x, y, t, L, offs = load_union(args.union_npz)
    lags = json.load(open(args.lags_json))

    # Pre-enumerate stretches across all 1000 packings (~3s)
    st_map = {}
    for sn in range(100, 1001):
        s0, e0 = offs[sn - 1], offs[sn]
        a_rad = np.column_stack([x[s0:e0], y[s0:e0], t[s0:e0]])
        for steps in (-2, -1, 1, 2):
            try:
                b, _ = st_mod.stretch(a_rad, steps)
            except Exception:
                continue
            if b is not None and 50 <= len(b) <= 1000:
                m = len(b)
                pts_deg = list(
                    zip(
                        b[:, 0].tolist(),
                        b[:, 1].tolist(),
                        (np.degrees(b[:, 2]) % 90.0).tolist(),
                    )
                )
                st_map.setdefault(m, []).append(pts_deg)

    # Prioritize Top-max_targets lagging puzzles + ALL K^2+1 (K=18..31)
    prio = [r["n"] for r in lags[: args.max_targets]]
    for K in range(18, 32):
        kn = K * K + 1
        if kn not in prio:
            prio.append(kn)
    my_ns = [n for idx, n in enumerate(prio) if idx % args.num_shards == args.shard]
    per_sec = max(35.0, (args.minutes_total * 60.0 * args.procs) / max(1, len(my_ns)))
    print(
        f"W6A BF+Melt+Stretch Shard {args.shard}/{args.num_shards}: {len(my_ns)} targets at {per_sec:.1f}s/target",
        flush=True,
    )

    jobs = []
    for n in my_ns:
        s0, p0 = get_deg_pts(n, x, y, t, L, offs)
        s1, p1 = get_deg_pts(n + 1, x, y, t, L, offs) if n + 1 <= 1000 else (1e9, None)
        jobs.append(
            (
                n,
                s0,
                p0,
                p1,
                s1,
                st_map.get(n, []),
                per_sec,
                20261128 + 811 * n + args.shard,
            )
        )

    improved = {}
    tot_gain = 0.0
    done = 0
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(solve_one, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            n, res, wins = fut.result()
            if res is not None:
                Lf, xf, yf, tf = res
                s0 = float(L[n - 1])
                gain = (s0 - Lf) / math.sqrt(n)
                tot_gain += gain
                improved[n] = (Lf, xf, yf, tf)
                print(
                    f"W6A_BFM_WIN [{done}/{len(jobs)}] n={n:4d} {s0:.9f} -> {Lf:.9f} dL={s0 - Lf:.6f} dS={gain:.6f} tot={tot_gain:.6f} via={[w[0] for w in wins]}",
                    flush=True,
                )

    payload = {}
    for n, (Lf, xf, yf, tf) in improved.items():
        payload[f"L_{n}"] = np.array([Lf], dtype=np.float64)
        payload[f"x_{n}"] = xf
        payload[f"y_{n}"] = yf
        payload[f"t_{n}"] = tf
    np.savez_compressed(args.out_npz, **payload)
    print(
        f"DONE W6A BFM Shard {args.shard}: {len(improved)} wins, total_gain={tot_gain:.6f} -> {args.out_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
