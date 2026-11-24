#!/usr/bin/env python3
"""Wave 4 Engine 2 (Kaggle Private CPU / Cloud):
Full Quenched & Grown Gadget Bank (gadgetq + faultdp-grow + seg) -> 1D Knapsack Fault-Line DP -> Rust fq Cross-Quench.
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS_DIR = None
for cand in [
    os.environ.get("SQ_DS_DIR", ""),
    "/kaggle/input/sq-wave3-union",
    "/kaggle/input/datasets/daohuutrong2404/sq-wave3-union",
    HERE,
]:
    if cand and os.path.exists(os.path.join(cand, "nov24_union.npz")):
        DS_DIR = cand
        break
if DS_DIR is None and os.path.exists("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        if "nov24_union.npz" in files:
            DS_DIR = root
            break
assert DS_DIR is not None, "Could not locate nov24_union.npz"
sys.path.insert(0, DS_DIR)

from decimal_verifier import verify_puzzle_decimal

BIN_DIR = "/tmp/sq_w4_bins"
os.makedirs(BIN_DIR, exist_ok=True)
for bname in ("fq", "rect_quench"):
    src = os.path.join(DS_DIR, bname)
    dst = os.path.join(BIN_DIR, bname)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)

FQ_BIN = os.environ.get("FQ_BIN", os.path.join(BIN_DIR, "fq"))
RQ_BIN = os.environ.get("RQ_BIN", os.path.join(BIN_DIR, "rect_quench"))

TARGET_GAP = Decimal("1e-9")
SAFE_GAP = Decimal("1.05e-9")


def load_union():
    d = np.load(os.path.join(DS_DIR, "nov24_union.npz"))
    x, y, t, L = d["x"], d["y"], d["t"], d["L"]
    offs = [0]
    for n in range(1, 1001):
        offs.append(offs[-1] + n)
    return x, y, t, L, offs


def get_arrays(n, x, y, t, L, offs):
    s0, e0 = offs[n - 1], offs[n]
    xs = x[s0:e0].copy()
    ys = y[s0:e0].copy()
    ts = t[s0:e0].copy()
    hx = 0.5 * (np.abs(np.cos(ts)) + np.abs(np.sin(ts)))
    xs -= np.min(xs - hx)
    ys -= np.min(ys - hx)
    return float(L[n - 1]), xs, ys, ts


def write_cfg(path, s, xs, ys, degs):
    with open(path, "w") as f:
        f.write(f"{len(xs)} {float(s)!r}\n")
        for a, b, c in zip(xs, ys, degs):
            f.write(f"{float(a)!r} {float(b)!r} {float(c) % 90.0!r}\n")


def read_cfg(path):
    with open(path) as f:
        lines = [l.split() for l in f if l.strip() and not l.startswith("#")]
    n, s = int(lines[0][0]), float(lines[0][1])
    arr = np.array(
        [[float(a), float(b), float(c) % 90.0] for a, b, c, *_ in lines[1 : n + 1]],
        dtype=np.float64,
    )
    return s, arr[:, 0], arr[:, 1], arr[:, 2]


def run_fq(s, xs, ys, degs, tmp, extra_args, timeout=150):
    a = os.path.join(tmp, "in.txt")
    b = os.path.join(tmp, "out.txt")
    write_cfg(a, s, xs, ys, degs)
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
    return read_cfg(b)


def verify_and_stabilize(n, xs, ys, degs):
    ts = np.radians(degs)
    hx = 0.5 * (np.abs(np.cos(ts)) + np.abs(np.sin(ts)))
    xs = xs - 0.5 * (float(np.min(xs - hx)) + float(np.max(xs + hx)))
    ys = ys - 0.5 * (float(np.min(ys - hx)) + float(np.max(ys + hx)))
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
    for _ in range(20):
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


def find_fault_cuts(lo, hi):
    order = np.argsort(lo)
    lo_s, hi_s = lo[order], hi[order]
    cuts = []
    cur_max = -1e18
    for idx in range(len(order)):
        if idx > 0 and lo_s[idx] >= cur_max - 1e-12:
            cuts.append(0.5 * (cur_max + lo_s[idx]))
        cur_max = max(cur_max, hi_s[idx])
    return cuts


def extract_slabs(L, xs, ys, ts):
    hx = 0.5 * (np.abs(np.cos(ts)) + np.abs(np.sin(ts)))
    degs = np.degrees(ts) % 90.0
    slabs = []
    for axis in (0, 1):
        if axis == 0:
            ax, ay, ad = xs, ys, degs
        else:
            ax, ay, ad = ys, L - xs, (90.0 - degs) % 90.0
        lo, hi = ay - hx, ay + hx
        cuts = [0.0] + find_fault_cuts(lo, hi) + [L]
        if len(cuts) < 3:
            continue
        # Extract both single slabs AND 2-adjacent-slab merged segments!
        for span in (1, 2):
            for idx in range(len(cuts) - span):
                c0, c1 = cuts[idx], cuts[idx + span]
                mask = (ay >= c0) & (ay <= c1)
                m = int(np.sum(mask))
                if m <= 0 or m > 420:
                    continue
                sx, sy, sd = ax[mask], ay[mask], ad[mask]
                shx = hx[mask]
                x0, x1 = float(np.min(sx - shx)), float(np.max(sx + shx))
                y0, y1 = float(np.min(sy - shx)), float(np.max(sy + shx))
                slabs.append((y1 - y0, x1 - x0, sx - x0, sy - y0, sd))
    return slabs


def quench_gadget_batch(jobs):
    """Worker: quench each (H, W, sx, sy, sd, W_try, grow_k) via static C++20 rect_quench."""
    out = []
    rng = np.random.default_rng(12345 + os.getpid())
    with tempfile.TemporaryDirectory() as tmp:
        rin = os.path.join(tmp, "in.txt")
        rout = os.path.join(tmp, "out.txt")
        for H, W, sx, sy, sd, W_try, grow_k in jobs:
            cx, cy, cd = sx.copy(), sy.copy(), sd.copy()
            if W_try < W:
                cx = cx * (W_try / max(W, 1e-9))
            H_init = H
            if grow_k > 0:
                # Insert grow_k squares at max-clearance holes inside [0.5, W_try-0.5] x [0.5, H+0.3]
                H_init = H + 0.15 * grow_k
                pts_g = np.column_stack(
                    [
                        rng.uniform(0.5, max(0.6, W_try - 0.5), 400),
                        rng.uniform(0.5, max(0.6, H_init - 0.5), 400),
                    ]
                )
                dmin = np.min(
                    np.hypot(
                        pts_g[:, None, 0] - cx[None, :], pts_g[:, None, 1] - cy[None, :]
                    ),
                    axis=1,
                )
                top = np.argsort(-dmin)[:grow_k]
                tilt_med = float(np.median(cd)) if len(cd) else 0.0
                cx = np.concatenate([cx, pts_g[top, 0]])
                cy = np.concatenate([cy, pts_g[top, 1]])
                cd = np.concatenate([cd, np.full(grow_k, tilt_med)])
            write_cfg(rin, H_init, cx, cy, cd)
            if os.path.exists(rout):
                os.remove(rout)
            try:
                res = subprocess.run(
                    [RQ_BIN, rin, rout, f"{W_try:.12f}", f"{H_init:.12f}"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except Exception:
                continue
            if res.returncode != 0 or not os.path.exists(rout):
                continue
            try:
                info = json.loads(res.stdout.strip().splitlines()[-1])
                Wn, Hn = float(info["W"]), float(info["H"])
                if Wn <= W_try + 1e-7 and (
                    Hn < H - 1e-5 or grow_k > 0 or W_try < W - 1e-5
                ):
                    _, qx, qy, qd = read_cfg(rout)
                    out.append((Hn, Wn, qx, qy, qd))
            except Exception:
                continue
    return out


def fault_dp_assemble(n_target, W_limit, gadgets, max_overshoot=2):
    valid = [g for g in gadgets if g[1] <= W_limit + 1e-8]
    best_per_m = {}
    for g in valid:
        H, W, sx, sy, sd = g
        m = len(sx)
        if m not in best_per_m or H < best_per_m[m][0]:
            best_per_m[m] = g
    items = sorted(best_per_m.items())
    INF = 1e18
    cap = n_target + max_overshoot
    dp = [INF] * (cap + 1)
    prev = [None] * (cap + 1)
    dp[0] = 0.0
    for c in range(1, cap + 1):
        bc = INF
        bp = None
        for m, g in items:
            if m > c:
                break
            cand_H = dp[c - m] + g[0]
            if cand_H < bc:
                bc = cand_H
                bp = (c - m, m)
        dp[c] = bc
        prev[c] = bp

    out = []
    for c in range(n_target, cap + 1):
        if dp[c] < W_limit + 0.025:
            used = []
            cur = c
            while cur > 0 and prev[cur] is not None:
                pc, m = prev[cur]
                used.append(best_per_m[m])
                cur = pc
            if cur == 0:
                xs, ys, ds = [], [], []
                y_off = 0.0
                max_w = 0.0
                for H, W, sx, sy, sd in used:
                    xs.append(sx)
                    ys.append(sy + y_off)
                    ds.append(sd)
                    y_off += H
                    max_w = max(max_w, W)
                ax, ay, ad = np.concatenate(xs), np.concatenate(ys), np.concatenate(ds)
                if c > n_target:
                    # Drop outermost surplus squares
                    dist = np.minimum(max_w - ax, y_off - ay)
                    keep = np.argsort(dist)[c - n_target :]
                    ax, ay, ad = ax[keep], ay[keep], ad[keep]
                out.append((max(max_w, y_off), ax, ay, ad))
    return out


def solve_dp_target(job):
    n, s0, gadgets = job
    best_s = s0
    best_res = None
    with tempfile.TemporaryDirectory() as tmp:
        w_grid = [
            s0 - dw
            for dw in (0.024, 0.018, 0.014, 0.010, 0.007, 0.004, 0.002, 0.0005, -0.002)
        ]
        for w_try in w_grid:
            for s_asm, ax, ay, ad in fault_dp_assemble(
                n, w_try, gadgets, max_overshoot=2
            ):
                if s_asm < s0 + 0.022:
                    r = run_fq(
                        s_asm,
                        ax,
                        ay,
                        ad,
                        tmp,
                        ["--loosen", "1.015", "--flip-top", "48", "--pit", "500"],
                    )
                    if r and r[0] < best_s - 1e-10:
                        best_s = r[0]
                        best_res = (r[1], r[2], r[3])
        if best_res is not None and best_s < s0 - 1e-10:
            r_fin = run_fq(
                best_s,
                best_res[0],
                best_res[1],
                best_res[2],
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "800"],
            )
            if r_fin and r_fin[0] < best_s:
                best_s = r_fin[0]
                best_res = (r_fin[1], r_fin[2], r_fin[3])
            ok, Lf, xf, yf, tf = verify_and_stabilize(
                n, best_res[0], best_res[1], best_res[2]
            )
            if ok and Lf < s0 - 1e-10:
                return n, (Lf, xf, yf, tf)
    return n, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=2)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out-npz", default="/kaggle/working/w4_faultdp_wins.npz")
    args = ap.parse_args()

    t0 = time.time()
    x, y, t, L, offs = load_union()
    lags = json.load(open(os.path.join(DS_DIR, "lag_targets.json")))
    all_targets = [r["n"] for r in lags[:260]]
    my_targets = [
        n for idx, n in enumerate(all_targets) if idx % args.num_shards == args.shard
    ]
    target_widths = sorted({round(r["Lb"], 2) for r in lags[:180]})

    # Current best pool
    cur_pack = {n: list(get_arrays(n, x, y, t, L, offs)) for n in range(1, 1001)}
    base_L = {n: cur_pack[n][0] for n in range(1, 1001)}

    improved = {}
    tot_gain = 0.0

    for rnd in range(args.rounds):
        # 1) Extract raw slabs from all 1000 packings
        bank = {}
        for n in range(1, 1001):
            Ln, xs, ys, ts = cur_pack[n]
            for H, W, sx, sy, sd in extract_slabs(Ln, xs, ys, ts):
                key = (len(sx), round(W, 3))
                if key not in bank or H < bank[key][0] - 1e-8:
                    bank[key] = (H, W, sx, sy, sd)
            degs = np.degrees(ts) % 90.0
            key = (n, round(Ln, 3))
            if key not in bank or Ln < bank[key][0]:
                bank[key] = (Ln, Ln, xs, ys, degs)
        raw_slabs = list(bank.values())
        print(
            f"[Round {rnd}] Extracted {len(raw_slabs)} raw slabs ({time.time() - t0:.1f}s)",
            flush=True,
        )

        # 2) Quench & grow non-trivial slabs at target widths via C++20 rect_quench
        q_jobs = []
        for H, W, sx, sy, sd in raw_slabs:
            m = len(sx)
            if m < 4 or m > 260 or H < 1.05:
                continue
            has_tilt = bool(np.any(np.minimum(sd % 90.0, 90.0 - (sd % 90.0)) > 2.0))
            if not has_tilt and H < 2.0:
                continue
            # Find nearby target widths from lagging puzzles
            near_ws = [wt for wt in target_widths if W - 0.14 <= wt <= W + 0.03]
            if not near_ws:
                near_ws = [W - 0.015, W]
            for wt in near_ws[:3]:
                q_jobs.append((H, W, sx, sy, sd, wt, 0))
            if has_tilt and m <= 160 and rnd == 0:
                q_jobs.append((H, W, sx, sy, sd, W + 0.01, 1))
                q_jobs.append((H, W, sx, sy, sd, W + 0.02, 2))
        random.Random(42 + rnd + args.shard).shuffle(q_jobs)
        q_jobs = q_jobs[:3600]
        chunks = [q_jobs[i :: args.procs] for i in range(args.procs)]
        print(
            f"[Round {rnd}] Quenching/growing {len(q_jobs)} gadgets across {args.procs} CPUs...",
            flush=True,
        )
        shrank = 0
        with ProcessPoolExecutor(max_workers=args.procs) as ex:
            for res_list in ex.map(quench_gadget_batch, chunks):
                for Hn, Wn, qx, qy, qd in res_list:
                    key = (len(qx), round(Wn, 3))
                    if key not in bank or Hn < bank[key][0] - 1e-8:
                        bank[key] = (Hn, Wn, qx, qy, qd)
                        shrank += 1
        gadgets = list(bank.values())
        print(
            f"[Round {rnd}] Bank upgraded with {shrank} new/tighter gadgets (total {len(gadgets)}, {time.time() - t0:.1f}s)",
            flush=True,
        )

        # 3) Run Fault-Line DP + fq cross-quench across shard targets
        dp_jobs = [(n, cur_pack[n][0], gadgets) for n in my_targets]
        rnd_wins = 0
        with ProcessPoolExecutor(max_workers=args.procs) as ex:
            futs = [ex.submit(solve_dp_target, j) for j in dp_jobs]
            for fut in as_completed(futs):
                n, res = fut.result()
                if res is not None:
                    Lf, xf, yf, tf = res
                    s_old = cur_pack[n][0]
                    if Lf < s_old - 1e-10:
                        gain = (base_L[n] - Lf) / math.sqrt(n)
                        tot_gain += (s_old - Lf) / math.sqrt(n)
                        cur_pack[n] = [Lf, xf, yf, tf]
                        improved[n] = (Lf, xf, yf, tf)
                        rnd_wins += 1
                        print(
                            f"W4_FDP_WIN r={rnd} n={n:4d} {s_old:.9f} -> {Lf:.9f} dL={s_old - Lf:.6f} tot={tot_gain:.6f}",
                            flush=True,
                        )
        print(
            f"[Round {rnd}] Complete: {rnd_wins} wins, elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        if rnd_wins == 0 and rnd > 0:
            break

    payload = {}
    for n, (Lf, xf, yf, tf) in improved.items():
        payload[f"L_{n}"] = np.array([Lf], dtype=np.float64)
        payload[f"x_{n}"] = xf
        payload[f"y_{n}"] = yf
        payload[f"t_{n}"] = tf
    np.savez_compressed(args.out_npz, **payload)
    print(
        f"DONE W4 FaultDP Shard {args.shard}: {len(improved)} wins, total_gain={tot_gain:.6f} -> {args.out_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
