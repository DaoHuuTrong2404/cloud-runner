#!/usr/bin/env python3
"""Pure-NumPy precomputation of the global 8-Symmetry Fault-Line DP table
`(G, dp, PG, PM, PQ, donors, donor_n)` once inside `consolidate_w68_megapool`!
"""

import argparse
import glob
import math
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fault_dp_8sym as fdp8


def _span_np(sq):
    h = 0.5 * (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2])))
    return float(
        max(
            (sq[:, 0] + h).max() - (sq[:, 0] - h).min(),
            (sq[:, 1] + h).max() - (sq[:, 1] - h).min(),
        )
    )


def load_closed_frontier_pure_np(union_npz, win_globs):
    z = np.load(union_npz)
    ux, uy, ut = z["x"], z["y"], z["t"]
    cur = [None] * 1001
    off = 0
    for n in range(1, 1001):
        sq = np.column_stack(
            [ux[off : off + n], uy[off : off + n], ut[off : off + n]]
        ).astype(np.float64)
        off += n
        cur[n] = (_span_np(sq), sq)

    wfiles = []
    for pat in win_globs:
        wfiles.extend(sorted(glob.glob(pat, recursive=True)))
    for wf in sorted(set(wfiles)):
        try:
            with np.load(wf) as zw:
                for lk in [k for k in zw.files if k.startswith("L_")]:
                    n = int(lk.split("_")[1])
                    if not (1 <= n <= 1000):
                        continue
                    xs, ys, ts = zw[f"x_{n}"], zw[f"y_{n}"], zw[f"t_{n}"]
                    if len(xs) != n:
                        continue
                    ts = np.asarray(ts, dtype=np.float64)
                    if float(np.max(np.abs(ts))) > 2.0:
                        ts = np.radians(ts)
                    sq = np.column_stack([xs, ys, ts]).astype(np.float64)
                    Ls = _span_np(sq)
                    if Ls < cur[n][0] - 1e-9:
                        cur[n] = (Ls, sq)
        except Exception:
            pass

    for _ in range(3):
        moved = 0
        for n in range(999, 10, -1):
            Lp1, sq = cur[n + 1]
            if Lp1 < cur[n][0] - 1e-8:
                sub = sq[:-1].copy()
                Ls = _span_np(sub)
                if Ls < cur[n][0] - 1e-8:
                    cur[n] = (Ls, sub)
                    moved += 1
        for n in range(10, 1000):
            Ln, sq = cur[n]
            k = int(math.floor(Ln))
            n2 = n + 2 * k + 1
            if n2 <= 1000 and Ln + 1.0 + 1e-7 < cur[n2][0] - 1e-7:
                he = 0.5 * (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2])))
                xs = sq[:, 0] - np.min(sq[:, 0] - he)
                ys = sq[:, 1] - np.min(sq[:, 1] - he)
                L_act = max(float(np.max(xs + he)), float(np.max(ys + he)))
                L_new = L_act + 1.0 + 1e-7
                extra = []
                for i in range(k):
                    extra.append((L_new - 0.5, i + 0.5, 0.0))
                    extra.append((i + 0.5, L_new - 0.5, 0.0))
                extra.append((L_new - 0.5, L_new - 0.5, 0.0))
                sq2 = np.vstack(
                    [
                        np.column_stack([xs, ys, sq[:, 2]]),
                        np.array(extra, dtype=np.float64),
                    ]
                )
                Ls2 = _span_np(sq2)
                if Ls2 < cur[n2][0] - 1e-8:
                    cur[n2] = (Ls2, sq2)
                    moved += 1
        if moved == 0:
            break

    tot_score = sum(cur[n][0] / math.sqrt(n) - 1.0 for n in range(1, 1001))
    return cur, tot_score


def build_dp8_cache(union_npz, win_globs, gad_globs, out_npz):
    t0 = time.time()
    cur, tot_score = load_closed_frontier_pure_np(union_npz, win_globs)
    donors = []
    donor_n = []
    for n in range(5, 1001):
        K = math.isqrt(n)
        if K * K != n and cur[n][0] < K + 1 - 1e-6 and cur[n][1] is not None:
            donors.append(np.asarray(cur[n][1], dtype=np.float64))
            donor_n.append(int(n))
    n_front = len(donors)

    gad_files = []
    for gp in gad_globs:
        gad_files.extend(sorted(glob.glob(gp, recursive=True)))
    gad_files = sorted(set(gad_files))

    for gf in gad_files:
        try:
            with np.load(gf) as zg:
                if "keys" in zg.files:
                    for i in range(len(zg["keys"])):
                        donors.append(np.asarray(zg[f"sq_{i}"], dtype=np.float64))
                        donor_n.append(-1)
        except Exception:
            pass

    G = fdp8.extract_gadgets(donors)
    dp, PG, PM, PQ = fdp8.solve_dp(G)

    offs = [0]
    for sq in donors:
        offs.append(offs[-1] + len(sq))
    don_sq = (
        np.concatenate(donors, axis=0) if donors else np.zeros((0, 3), dtype=np.float64)
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    np.savez_compressed(
        out_npz,
        G=G,
        dp=dp,
        PG=PG,
        PM=PM,
        PQ=PQ,
        donor_n=np.asarray(donor_n, dtype=np.int32),
        don_offs=np.asarray(offs, dtype=np.int32),
        don_sq=don_sq,
    )
    print(
        f"[dp8_cache_w68] score={tot_score:.10f} donors={len(donors)} (frontier={n_front}, "
        f"gadgets={len(donors) - n_front} from {len(gad_files)} files) -> G={len(G)} saved to {out_npz} in {time.time() - t0:.1f}s!",
        flush=True,
    )


def load_dp8_cache(cache_npz):
    with np.load(cache_npz) as z:
        G = np.asarray(z["G"], dtype=np.float64)
        dp = np.asarray(z["dp"], dtype=np.float64)
        PG = np.asarray(z["PG"], dtype=np.int32)
        PM = np.asarray(z["PM"], dtype=np.int8)
        PQ = np.asarray(z["PQ"], dtype=np.int8)
        donor_n = np.asarray(z["donor_n"], dtype=np.int32)
        offs = np.asarray(z["don_offs"], dtype=np.int32)
        sq_all = np.asarray(z["don_sq"], dtype=np.float64)
    donors = [sq_all[offs[i] : offs[i + 1]] for i in range(len(offs) - 1)]
    return donors, donor_n, G, dp, PG, PM, PQ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--gad-glob", action="append", default=[])
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()
    build_dp8_cache(args.union_npz, args.win_glob, args.gad_glob, args.out_npz)


if __name__ == "__main__":
    main()
