#!/usr/bin/env python3
"""Wave 56 Stage 2 (`w56_merge_gadgets.py`): Unifies all 16 Forge shards' upgraded
rectangle gadgets into ONE global `global_rect_gadgets.npz`!
"""

import argparse
import glob
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.abspath(os.path.join(HERE, "..", "tools"))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rust_engines as RE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", required=True)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    best_by_key = {}
    files = sorted(glob.glob(args.in_glob, recursive=True))
    tot_raw = 0
    for fp in files:
        try:
            with np.load(fp) as z:
                if "keys" not in z.files:
                    continue
                keys = z["keys"].copy()
                f_old = z["f_old"].copy()
                f_new = z["f_new"].copy()
                sqs = [
                    np.asarray(z[f"sq_{i}"], dtype=np.float64) for i in range(len(keys))
                ]
        except Exception:
            continue
        for i in range(len(keys)):
            w, h, dm = int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])
            sq = sqs[i]
            if len(sq) != w * h + dm:
                continue
            f_chk = RE.gadget_overhang(sq, w, h)
            fn = max(float(f_new[i]), f_chk)
            if fn >= float(f_old[i]) - 1e-8:
                continue
            tot_raw += 1
            k = (w, h, dm)
            if k not in best_by_key or fn < best_by_key[k][1] - 1e-12:
                best_by_key[k] = (float(f_old[i]), fn, sq)

    items = sorted(best_by_key.items(), key=lambda kv: kv[1][1] - kv[1][0])
    print(
        f"[W56-MergeGadgets] files={len(files)} raw_improved={tot_raw} unique_upgraded={len(items)}!",
        flush=True,
    )
    for k, (f0, fn, _) in items[:20]:
        print(
            f"  [Global gadget] {k[0]}x{k[1]}+{k[2]}: {f0:.6f} -> {fn:.6f} (df={fn - f0:+.6f})",
            flush=True,
        )

    pl = {
        "keys": np.array([[k[0], k[1], k[2]] for k, _ in items], dtype=np.int32),
        "f_old": np.array([v[0] for _, v in items], dtype=np.float64),
        "f_new": np.array([v[1] for _, v in items], dtype=np.float64),
    }
    for idx, (_, (_, _, sq)) in enumerate(items):
        pl[f"sq_{idx}"] = sq
    np.savez_compressed(args.out_npz, **pl)


if __name__ == "__main__":
    main()
