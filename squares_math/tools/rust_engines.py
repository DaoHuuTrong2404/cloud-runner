#!/usr/bin/env python3
"""Clean Python wrappers for compiled Rust `harden` (Nakajima / Ostashev soft-to-rigid)
and `fq` (rectangle-mode `--rect-dx` / `--rect-dy` gadget quench + square quench).
"""

import json
import math
import os
import subprocess
import tempfile
import numpy as np


def get_fq_bin():
    return os.environ.get("FQ_BIN", "/tmp/sq_bins/fq")


def get_harden_bin():
    return os.environ.get("HARDEN_BIN", "/tmp/sq_bins/harden")


def side_exact(P):
    h = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    return float(
        max(
            (P[:, 0] + h).max() - (P[:, 0] - h).min(),
            (P[:, 1] + h).max() - (P[:, 1] - h).min(),
        )
    )


def normalize_to_origin(P):
    P = np.asarray(P, dtype=np.float64).copy()
    h = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    P[:, 0] -= (P[:, 0] - h).min()
    P[:, 1] -= (P[:, 1] - h).min()
    return P


def write_packer_cfg(path, s, P):
    P = normalize_to_origin(P)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{len(P)} {float(s):.17f}\n")
        for x, y, t in P:
            deg = math.degrees(float(t)) % 90.0
            fh.write(f"{float(x):.17f} {float(y):.17f} {deg:.15f}\n")


def read_packer_cfg(path):
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    if not lines:
        return None, None
    hd = lines[0].split()
    n = int(hd[0])
    s = float(hd[1])
    rows = []
    for ln in lines[1 : n + 1]:
        vals = ln.split()
        rows.append((float(vals[0]), float(vals[1]), math.radians(float(vals[2]))))
    if len(rows) != n:
        return None, None
    return s, np.asarray(rows, dtype=np.float64)


def run_harden_reharden(
    P_in,
    s_ref=None,
    seed=1,
    r0=0.16,
    stiff0=0.35,
    pre=1800,
    hold=7500,
    release=3500,
    settle=2800,
    hold_f=1.001,
    noise=0.0045,
    mu=0.02,
    sym_md=0,
    timeout_s=25.0,
):
    """Run Ostashev/Nakajima `harden reharden` in compiled Rust (~0.3-1.2s per call!).
    Allows overlapping/inserted/staggered-cut squares to slide past angular barriers as soft disks.
    Returns `(S_legal, Q_legal)` or `None`.
    """
    hbin = get_harden_bin()
    if not os.path.exists(hbin):
        return None
    P = normalize_to_origin(P_in)
    s0 = float(s_ref) if s_ref is not None else side_exact(P)
    with tempfile.TemporaryDirectory(prefix="hrd_") as tmp:
        f_in = os.path.join(tmp, "in.txt")
        f_out = os.path.join(tmp, "out.txt")
        write_packer_cfg(f_in, s0, P)
        cmd = [
            hbin,
            "reharden",
            "--in",
            f_in,
            "--out",
            f_out,
            "--seed",
            str(int(seed)),
            "--r0",
            repr(float(r0)),
            "--stiff0",
            repr(float(stiff0)),
            "--pre",
            str(int(pre)),
            "--hold",
            str(int(hold)),
            "--release",
            str(int(release)),
            "--settle",
            str(int(settle)),
            "--hold-f",
            repr(float(hold_f)),
            "--noise",
            repr(float(noise)),
            "--mu",
            repr(float(mu)),
            "--sym-md",
            str(int(sym_md)),
        ]
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=float(timeout_s)
            )
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0 or not os.path.exists(f_out):
            return None
        s_out, Q_out = read_packer_cfg(f_out)
        if Q_out is None or len(Q_out) != len(P) or not np.isfinite(Q_out).all():
            return None
        Q_out = normalize_to_origin(Q_out)
        return side_exact(Q_out), Q_out


def gadget_overhang(sq, w, h):
    P = normalize_to_origin(sq)
    he = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    xmax = float((P[:, 0] + he).max())
    ymax = float((P[:, 1] + he).max())
    return float(max(xmax - float(w), ymax - float(h), 0.0))


def quench_rect_gadget(
    sq,
    w,
    h,
    loosen=1.015,
    kick=0.008,
    seed=1,
    outer=55,
    pit=220,
    flip_top=16,
    timeout_s=25.0,
):
    """Rectangle-mode `fq quench --rect-dx (w-m) --rect-dy (h-m)` on a `(w, h, dm)` gadget!
    Minimizes overhang `f` so all `len(sq)` unit squares fit inside `[0, w + f] x [0, h + f]`.
    Returns `(Q_out, f_out)` or `(sq, inf)` on failure.
    """
    fqbin = get_fq_bin()
    if not os.path.exists(fqbin):
        return sq, math.inf
    P = normalize_to_origin(sq)
    m = min(int(w), int(h))
    f0 = gadget_overhang(P, w, h)
    s_in = float(m) + f0 + 1e-9
    with tempfile.TemporaryDirectory(prefix="fqr_") as tmp:
        f_in = os.path.join(tmp, "a.txt")
        f_out = os.path.join(tmp, "b.txt")
        write_packer_cfg(f_in, s_in, P)
        cmd = [
            fqbin,
            "quench",
            "--in",
            f_in,
            "--out",
            f_out,
            "--rect-dx",
            repr(float(w - m)),
            "--rect-dy",
            repr(float(h - m)),
            "--loosen",
            repr(float(loosen)),
            "--kick",
            repr(float(kick)),
            "--seed",
            str(int(seed)),
            "--outer",
            str(int(outer)),
            "--pit",
            str(int(pit)),
            "--flip-top",
            str(int(flip_top)),
        ]
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=float(timeout_s)
            )
        except subprocess.TimeoutExpired:
            return sq, math.inf
        if r.returncode != 0 or not os.path.exists(f_out):
            return sq, math.inf
        # Verify min_gap >= 0 and min_wall >= 0 from JSON summary
        try:
            js = json.loads(r.stdout.strip().splitlines()[-1])
            mg = js.get("min_gap")
            mw = js.get("min_wall")
            if mg is None or mw is None or mg < -1e-12 or mw < -1e-12:
                return sq, math.inf
        except Exception:
            pass
        _, Q_out = read_packer_cfg(f_out)
        if Q_out is None or Q_out.shape != P.shape or not np.isfinite(Q_out).all():
            return sq, math.inf
        Q_out = normalize_to_origin(Q_out)
        return Q_out, gadget_overhang(Q_out, w, h)


def run_fq_quench_args(P_in, extra_args, s_ref=None, timeout_s=20.0):
    """Run compiled Rust `fq quench` with arbitrary CLI flags (e.g. `--round-r0`, `--soft-a0`,
    `--sqp`, `--finish`, `--shrink`, `--stag-w 0`).
    Returns `(S_exact, Q_out, js_dict)` if feasible in f64 SAT, else `None`.
    """
    fqbin = get_fq_bin()
    if not os.path.exists(fqbin):
        return None
    P = normalize_to_origin(P_in)
    s0 = float(s_ref) if s_ref is not None else side_exact(P)
    with tempfile.TemporaryDirectory(prefix="fqq_") as tmp:
        f_in = os.path.join(tmp, "a.txt")
        f_out = os.path.join(tmp, "b.txt")
        write_packer_cfg(f_in, s0, P)
        cmd = [fqbin, "quench", "--in", f_in, "--out", f_out] + [
            str(x) for x in extra_args
        ]
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=float(timeout_s)
            )
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0 or not os.path.exists(f_out):
            return None
        js = {}
        try:
            js = json.loads(r.stdout.strip().splitlines()[-1])
            mg = js.get("min_gap")
            mw = js.get("min_wall")
            if mg is None or mw is None or mg < -1e-12 or mw < -1e-12:
                return None
        except Exception:
            return None
        _, Q_out = read_packer_cfg(f_out)
        if Q_out is None or Q_out.shape != P.shape or not np.isfinite(Q_out).all():
            return None
        Q_out = normalize_to_origin(Q_out)
        return side_exact(Q_out), Q_out, js


def kkt_sqp_polish_fq(P_in, timeout_each=18.0):
    """Two-pass exact Rust KKT finisher (`newton_finish` + `--sqp` Hessian + `--stag-w 0`).
    Projects curved tilted-band contact networks onto `{g_A = 0}` down to `1e-13`.
    Returns `(S_best, Q_best)`.
    """
    best_Q = normalize_to_origin(P_in)
    best_S = side_exact(best_Q)
    dl = max(4.0, float(timeout_each) - 2.0)
    r1 = run_fq_quench_args(
        best_Q,
        [
            "--loosen",
            "1.0",
            "--no-alm",
            "--finish",
            "--stag-w",
            "0",
            "--rmin",
            "1e-12",
            "--flip-top",
            "64",
            "--pit",
            "900",
            "--deadline",
            f"{dl:.1f}",
        ],
        s_ref=best_S,
        timeout_s=timeout_each,
    )
    if r1 is not None and r1[0] < best_S - 1e-12:
        best_S, best_Q = r1[0], r1[1]
    r2 = run_fq_quench_args(
        best_Q,
        [
            "--loosen",
            "1.0",
            "--no-alm",
            "--sqp",
            "--finish",
            "--stag-w",
            "0",
            "--rmin",
            "1e-12",
            "--flip-top",
            "32",
            "--pit",
            "500",
            "--deadline",
            f"{dl:.1f}",
        ],
        s_ref=best_S,
        timeout_s=timeout_each,
    )
    if r2 is not None and r2[0] < best_S - 1e-12:
        best_S, best_Q = r2[0], r2[1]
    return best_S, best_Q
