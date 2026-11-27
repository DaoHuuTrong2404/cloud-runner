#!/usr/bin/env python3
"""Wave 45G: Deep 450s x 2 Wider-Move Hopper (max_cl=18) on Fresh W44/pub_v9/pub_v10 Designs + goodpjw2008 Open Near-Miss Targets!

Two ultra-high-ROI lanes based directly on empirical Wave 42-44 + goodpjw2008 v52 evidence:
  Lane 1 (Freshly-Born Soft Packings):
    Any packing that just jumped in W44/pub_v9/pub_v10 (n=921 from gpeel/htpD -0.0021!, 920, 922,
    451, 635, 599, 804, 699, 748, 500, 378, 758, 875, 868, ...) is STILL SOFT -> 450s x 2 Wide-Hopper
    + SLP triggers second-wave jumps (-3e-5 to -3e-4, exact v50/v52 mechanism!).
  Lane 2 (goodpjw2008 targets.csv Open Near-Miss n+1 -> n Deletions):
    Tests the 43+ pre-identified near-miss sizes (685, 234, 904, 694, 491, 692, ...) by screening
    both the exact target index AND all near-band squares of our latest cur[n+1], then running
    deep 450s x 2 Wider-Move Hopper (max_cl=18) + SLP + Rust fq + lp_rot + Decimal64 >= 1.05e-9!
"""

import argparse
import csv
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


def _slp_polish(P):
    cur_a = w32a.safe_open(P)
    cur_s = float(E.side(cur_a))
    try:
        xs, ys, ts, Ls = SLP.polish(
            cur_a[:, 0],
            cur_a[:, 1],
            cur_a[:, 2],
            iters=120,
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


def _certify_win(n, L0, best_P):
    cur_a, cur_s = _slp_polish(best_P)
    if os.path.exists(f2d.FQ_BIN):
        with tempfile.TemporaryDirectory(prefix=f"w45g_{n}_") as tmp:
            r_fin = f2d.run_fq(
                cur_s,
                cur_a,
                tmp,
                ["--loosen", "1.0", "--no-alm", "--flip-top", "64", "--pit", "600"],
                timeout=65,
            )
            if r_fin is not None and r_fin[0] < cur_s:
                cur_s, cur_a = r_fin[0], r_fin[1]
    try:
        cur_a = w12.lp_rot_polish(cur_a)
    except Exception:
        pass
    ok, Lf, xf, yf, tf = f2d.stabilize_decimal(n, cur_a)
    if ok and Lf < L0 - 1e-9:
        return (Lf, xf, yf, tf)
    return None


def worker_fresh_or_target(job):
    kind, n, L0, arr0, arr_p1, hint_idxs, seed, round_sec, n_rounds, deadline = job
    best_S = L0
    best_P = None
    best_tag = ""

    starts = []
    if kind == "fresh":
        starts.append((w32a.safe_open(arr0), "fresh"))
        # Also test quick best n+1 -> n removal on freshly-born neighbors (e.g. 921 -> 920!)
        if arr_p1 is not None and (deadline - time.time()) > 120.0:
            cands = list(
                dict.fromkeys(
                    list(hint_idxs) + list(w32a.pick_near_band(arr_p1, max_cand=120))
                )
            )
            best_h = None
            t_scr = time.time()
            for idx in cands:
                if time.time() - t_scr > 60.0 or (deadline - time.time()) < 60.0:
                    break
                if not (0 <= idx < len(arr_p1)):
                    continue
                try:
                    Qh, Sh = w32a.fast_delete_and_compact(arr_p1, idx)
                    if (
                        Qh is not None
                        and Sh < L0 + 0.014
                        and (best_h is None or Sh < best_h[0])
                    ):
                        best_h = (Sh, idx, Qh)
                except Exception:
                    continue
            if best_h is not None:
                starts.append((w32a.safe_open(best_h[2]), f"fdel_{best_h[1]}"))
    else:
        # Target near-miss lane: screen hint_idxs first + ALL near-band squares of arr_p1!
        if arr_p1 is not None:
            cands = list(
                dict.fromkeys(
                    list(hint_idxs) + list(w32a.pick_near_band(arr_p1, max_cand=300))
                )
            )
            holes = []
            t_scr = time.time()
            for idx in cands:
                if time.time() - t_scr > 95.0 or (deadline - time.time()) < 60.0:
                    break
                if not (0 <= idx < len(arr_p1)):
                    continue
                try:
                    Qh, Sh = w32a.fast_delete_and_compact(arr_p1, idx)
                    if Qh is not None and Sh < L0 + 0.016:
                        holes.append((Sh, idx, Qh))
                except Exception:
                    continue
            holes.sort(key=lambda z: z[0])
            picked = []
            for Sh, idx, Qh in holes:
                if all(abs(Sh - ps) > 1e-5 for ps, _, _ in picked):
                    picked.append((Sh, idx, Qh))
                if len(picked) >= 2:
                    break
            for Sh, idx, Qh in picked:
                if Sh < best_S - 1e-9 and E.min_gap(Qh) >= -1e-12:
                    best_S, best_P, best_tag = Sh, Qh, f"tdel_{idx}_raw"
                starts.append((w32a.safe_open(Qh), f"tdel_{idx}"))

    for si, (P_init, stag) in enumerate(starts):
        cur_P = P_init
        for r in range(n_rounds):
            rem = deadline - time.time()
            if rem < 20.0:
                break
            sec = min(round_sec, rem - 10.0)
            try:
                H = E.Hopper(cur_P, seed=seed + si * 15013 + r * 7919, wide=True)
                if not H.applicable():
                    break
                Q, S = H.run(sec)
                Q, S = _slp_polish(Q)
                if S < best_S - 1e-9 and E.min_gap(Q) >= -1e-12:
                    best_S, best_P, best_tag = S, Q, f"{stag}_wh_r{r}"
                    cur_P = Q
                elif S < float(E.side(cur_P)) - 1e-9:
                    cur_P = Q
            except Exception:
                break

    if best_P is not None and best_S < L0 - 1e-9:
        cert = _certify_win(n, L0, best_P)
        if cert is not None:
            return n, best_tag, cert
    return n, best_tag, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-npz", required=True)
    ap.add_argument("--win-glob", action="append", default=[])
    ap.add_argument("--targets-csv", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget-sec", type=int, default=2350)
    ap.add_argument("--round-sec", type=int, default=420)
    ap.add_argument("--n-rounds", type=int, default=2)
    ap.add_argument("--seed-off", type=int, default=455000)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    t0 = time.time()
    w32a.warm_numba()
    cur, fresh, tot_score, n_files = w12.load_frontier_with_wins(
        args.union_npz, args.win_glob
    )

    # Parse goodpjw2008 targets.csv (exact open near-miss n -> [removed_idxs])
    tg_hints = {}
    tg_order = []
    if os.path.exists(args.targets_csv):
        with open(args.targets_csv) as f:
            for row in csv.DictReader(f):
                tn = int(row["n"])
                rm = int(row["removed"])
                if tn not in tg_hints:
                    tg_hints[tn] = []
                    tg_order.append(tn)
                tg_hints[tn].append(rm)

    # Lane 1: Brand-new freshly-born soft packings (W45 703/921/451 + pub_v11 556 + W44 + pub_v9/v10!)
    fresh_ns = [
        703,
        702,
        704,
        556,
        555,
        557,
        921,
        920,
        922,
        451,
        450,
        376,
        635,
        634,
        804,
        803,
        699,
        698,
        748,
        747,
        599,
        598,
        500,
        499,
        378,
        377,
        758,
        757,
        875,
        874,
        868,
        991,
        910,
    ]
    queue = []
    seen_q = set()
    for n in fresh_ns:
        K = math.isqrt(n)
        if (
            80 <= n <= 1000
            and n != K * K
            and cur[n][0] < K + 1 - 1e-6
            and n not in seen_q
        ):
            queue.append(("fresh", n))
            seen_q.add(n)
    # Lane 2: ALL open near-miss n+1 -> n targets from goodpjw2008 targets.csv!
    for n in tg_order:
        K = math.isqrt(n)
        if (
            80 <= n < 1000
            and n != K * K
            and cur[n][0] < K + 1 - 1e-6
            and n not in seen_q
        ):
            queue.append(("target", n))
            seen_q.add(n)

    my_slice = queue[args.shard :: args.num_shards]
    deadline = t0 + args.budget_sec
    jobs = []
    for i, (kind, n) in enumerate(my_slice):
        arr_p1 = None
        if n + 1 <= 1000:
            k1 = math.isqrt(n + 1)
            if k1 * k1 != (n + 1) and cur[n + 1][0] < k1 + 1 - 1e-6:
                arr_p1 = cur[n + 1][1]
        jobs.append(
            (
                kind,
                n,
                cur[n][0],
                cur[n][1],
                arr_p1,
                tg_hints.get(n, []),
                args.seed_off + args.shard * 10429 + i * 733,
                args.round_sec,
                args.n_rounds,
                deadline,
            )
        )

    print(
        f"[W45G-Fresh+TargetsWideHop s={args.shard}/{args.num_shards}] score={tot_score:.10f} "
        f"my_jobs({len(my_slice)})={my_slice} round={args.round_sec}sx{args.n_rounds} ({time.time() - t0:.1f}s)",
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

    done = 0
    ex = ProcessPoolExecutor(max_workers=args.procs)
    futs = [ex.submit(worker_fresh_or_target, j) for j in jobs]
    try:
        for fut in as_completed(futs, timeout=max(30.0, deadline - time.time() + 90.0)):
            done += 1
            try:
                n, tag, res = fut.result()
            except Exception as exc:
                print(f"  [W45G ERR] {exc}", flush=True)
                continue
            if res is not None:
                Lf, xf, yf, tf = res
                prev_L = wins[n][0] if n in wins else cur[n][0]
                if Lf < prev_L - 1e-10:
                    dL = Lf - cur[n][0]
                    dS = (Lf - prev_L) / math.sqrt(n)
                    tot_ds += dS
                    wins[n] = res
                    print(
                        f"  [W45G WIN!!] n={n:4d} [{tag}] {cur[n][0]:.7f} -> {Lf:.7f} "
                        f"(dL={dL:.7f}, dS={dL / math.sqrt(n):.8f}) [{done}/{len(jobs)}, {time.time() - t0:.0f}s]",
                        flush=True,
                    )
                    save_ckpt()
            if time.time() > deadline + 45.0:
                print(
                    f"  [W45G s={args.shard}] Hard deadline reached after {done}/{len(jobs)} jobs.",
                    flush=True,
                )
                break
    except TimeoutError:
        print(
            f"  [W45G s={args.shard}] as_completed timeout after {done}/{len(jobs)} jobs — exiting cleanly!",
            flush=True,
        )

    save_ckpt()
    print(
        f"[W45G s={args.shard}] DONE in {time.time() - t0:.1f}s | wins={len(wins)} | dS={tot_ds:.8f}",
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
