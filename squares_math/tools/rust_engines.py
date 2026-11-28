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


def get_packer_bin():
    return os.environ.get("PACKER_BIN", "/tmp/sq_bins/packer")


def get_anneal_bin():
    return os.environ.get("ANNEAL_BIN", "/tmp/sq_bins/anneal")


def get_santa_ga_bin():
    return os.environ.get("SANTA_GA_BIN", "/tmp/sq_bins/santa_ga")


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


def write_packer_cfg(path, s, P, frozen_mask=None):
    P = normalize_to_origin(P)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{len(P)} {float(s):.17f}\n")
        for idx, (x, y, t) in enumerate(P):
            deg = math.degrees(float(t)) % 90.0
            fz_tag = (
                " F"
                if (
                    frozen_mask is not None
                    and idx < len(frozen_mask)
                    and bool(frozen_mask[idx])
                )
                else ""
            )
            fh.write(f"{float(x):.17f} {float(y):.17f} {deg:.15f}{fz_tag}\n")


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
    sym_band=0.0,
    hold_move=0,
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
            "--sym-band",
            repr(float(sym_band)),
            "--hold-move",
            str(int(hold_move)),
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


def build_far_grid_freeze_mask(P_in, buffer_dist=3.6, min_free=90):
    """Freeze strictly axis-aligned grid squares farther than `buffer_dist` from every tilted square.
    Orientation-independent (works for any D4 corner/channel layout) and keeps >= `min_free` squares free.
    """
    P = normalize_to_origin(P_in)
    n = len(P)
    if n <= min_free + 25:
        return None
    half_pi = 0.5 * math.pi
    a_mod = np.mod(P[:, 2], half_pi)
    tilts = np.minimum(a_mod, half_pi - a_mod)
    tilted_idx = np.where(tilts > 0.012)[0]
    if len(tilted_idx) == 0:
        return None
    tx = P[tilted_idx, 0][None, :]
    ty = P[tilted_idx, 1][None, :]
    d2 = (P[:, 0:1] - tx) ** 2 + (P[:, 1:2] - ty) ** 2
    min_d = np.sqrt(d2.min(axis=1))
    fz = (tilts < 1e-3) & (min_d > float(buffer_dist))
    n_free = int((~fz).sum())
    if n_free < min_free:
        # Keep the `min_free` closest squares to the tilted band unfrozen
        order = np.argsort(min_d)
        fz[:] = True
        fz[order[:min_free]] = False
        fz[tilted_idx] = False
    if int(fz.sum()) < 15:
        return None
    return fz


def run_packer_hop(
    P_in,
    target_s,
    seed=1,
    iters=90000,
    temp=0.26,
    amp=1.0,
    maxit=2800,
    descend=5e-5,
    linepen=0.01,
    stall=1e-5,
    frozen_mask=None,
    timeout_s=180.0,
):
    """Run compiled Rust `packer hop` (descending-target region-move + `best_hole` + `jam_lines` row-break).
    `packer` writes `--out` atomically on every new feasible side AND writes `--out.inf` on best infeasible energy.
    Harvests both even when terminated at `timeout_s`, then certifies f64 SAT + KKT via `fq`.
    Returns `(S_exact, Q_best, kind_str)` or `None`.
    """
    pbin = get_packer_bin()
    if not os.path.exists(pbin):
        return None
    P = normalize_to_origin(P_in)
    s0 = side_exact(P)
    ts = float(target_s)
    P_start = P.copy()
    if ts < s0 and frozen_mask is None:
        ratio = ts / max(s0, 1e-9)
        P_start[:, 0] = np.clip(P_start[:, 0] * ratio, 0.5, ts - 0.5)
        P_start[:, 1] = np.clip(P_start[:, 1] * ratio, 0.5, ts - 0.5)

    with tempfile.TemporaryDirectory(prefix="pkhop_") as tmp:
        f_in = os.path.join(tmp, "in.txt")
        f_out = os.path.join(tmp, "out.txt")
        write_packer_cfg(f_in, ts, P_start, frozen_mask=frozen_mask)
        cmd = [
            pbin,
            "hop",
            "--in",
            f_in,
            "--s",
            repr(ts),
            "--seed",
            str(int(seed)),
            "--iters",
            str(int(iters)),
            "--temp",
            repr(float(temp)),
            "--amp",
            repr(float(amp)),
            "--maxit",
            str(int(maxit)),
            "--descend",
            repr(float(descend)),
            "--linepen",
            repr(float(linepen)),
            "--stall",
            repr(float(stall)),
            "--quiet",
            "--out",
            f_out,
        ]
        try:
            subprocess.run(
                cmd, capture_output=True, text=True, timeout=float(timeout_s)
            )
        except subprocess.TimeoutExpired:
            pass

        best_res = None
        if os.path.exists(f_out):
            _, Q_f = read_packer_cfg(f_out)
            if Q_f is not None and Q_f.shape == P.shape and np.isfinite(Q_f).all():
                rq = run_fq_quench_args(
                    Q_f,
                    [
                        "--loosen",
                        "1.0004",
                        "--outer",
                        "25",
                        "--pit",
                        "450",
                        "--finish",
                        "--stag-w",
                        "0",
                        "--rmin",
                        "1e-12",
                        "--deadline",
                        "10.0",
                    ],
                    timeout_s=12.5,
                )
                if rq is not None:
                    best_res = (float(rq[0]), rq[1], "hop_feas")

        f_inf = f_out + ".inf"
        if best_res is None and os.path.exists(f_inf):
            _, Q_inf = read_packer_cfg(f_inf)
            if (
                Q_inf is not None
                and Q_inf.shape == P.shape
                and np.isfinite(Q_inf).all()
            ):
                rq2 = run_fq_quench_args(
                    Q_inf,
                    [
                        "--loosen",
                        "1.005",
                        "--outer",
                        "45",
                        "--pit",
                        "450",
                        "--finish",
                        "--stag-w",
                        "0",
                        "--rmin",
                        "1e-12",
                        "--deadline",
                        "13.0",
                    ],
                    s_ref=ts,
                    timeout_s=15.5,
                )
                if rq2 is not None:
                    best_res = (float(rq2[0]), rq2[1], "hop_inf_fq")

        return best_res


def run_packer_hhop(
    P_in,
    seed=1,
    iters=85000,
    temp=1.2e-4,
    chains=8,
    chainlen=1.05,
    gmax=0.018,
    soft=0.18,
    forbid_lines=0,
    frozen_mask=None,
    timeout_s=180.0,
):
    """Run compiled Rust `packer hhop` (hard-square collision event-chains + `--soft` penalty-squeeze).
    Starts at `side_exact(P) + 1e-8` so `assert!(e < ETOL())` always holds, then harvests atomic `--out`
    even on `TimeoutExpired`, followed by `fq` SAT + KKT finish.
    Returns `(S_exact, Q_best)` or `None`.
    """
    pbin = get_packer_bin()
    if not os.path.exists(pbin):
        return None
    P = normalize_to_origin(P_in)
    s0 = side_exact(P) + 1e-8
    with tempfile.TemporaryDirectory(prefix="pkhhop_") as tmp:
        f_in = os.path.join(tmp, "in.txt")
        f_out = os.path.join(tmp, "out.txt")
        write_packer_cfg(f_in, s0, P, frozen_mask=frozen_mask)
        cmd = [
            pbin,
            "hhop",
            "--in",
            f_in,
            "--seed",
            str(int(seed)),
            "--iters",
            str(int(iters)),
            "--temp",
            repr(float(temp)),
            "--chains",
            str(int(chains)),
            "--chainlen",
            repr(float(chainlen)),
            "--gmax",
            repr(float(gmax)),
            "--soft",
            repr(float(soft)),
            "--forbid-lines",
            str(int(forbid_lines)),
            "--out",
            f_out,
        ]
        try:
            subprocess.run(
                cmd, capture_output=True, text=True, timeout=float(timeout_s)
            )
        except subprocess.TimeoutExpired:
            pass

        if not os.path.exists(f_out):
            return None
        _, Q_h = read_packer_cfg(f_out)
        if Q_h is None or Q_h.shape != P.shape or not np.isfinite(Q_h).all():
            return None
        rq = run_fq_quench_args(
            Q_h,
            [
                "--loosen",
                "1.0003",
                "--outer",
                "20",
                "--pit",
                "450",
                "--finish",
                "--stag-w",
                "0",
                "--rmin",
                "1e-12",
                "--deadline",
                "10.0",
            ],
            timeout_s=12.5,
        )
        if rq is None:
            return None
        return float(rq[0]), rq[1]


def run_anneal_gls(
    P_in,
    shrink=6e-4,
    ms=4500.0,
    seed=1,
    ng=40,
    nl=24,
    ring=1,
    relax=24,
    strike_iters=8,
    timeout_s=12.0,
):
    """Run Gardeyn & Wauters Sparrow-style Guided Local Search (`anneal gls` in `anneal.rs` lines 587-835).
    Cuts along two random seams to target side `S = s0 * (1 - shrink)`, then alternates continuous penetration-gradient
    relaxation (`--relax`) + global/local repositioning (`--ng/--nl`) + `--ring 1` neighbor slack + GLS pair-weight bumps.
    Returns `(s_target, Q_out, is_feasible, raw_overlap)` or `None`.
    """
    abin = get_anneal_bin()
    if not os.path.exists(abin):
        return None
    P = normalize_to_origin(P_in)
    s0 = side_exact(P)
    with tempfile.TemporaryDirectory(prefix="angls_") as tmp:
        f_in = os.path.join(tmp, "in.txt")
        f_out = os.path.join(tmp, "out.txt")
        write_packer_cfg(f_in, s0, P)
        cmd = [
            abin,
            "gls",
            "--in",
            f_in,
            "--out",
            f_out,
            "--shrink",
            repr(float(shrink)),
            "--ms",
            repr(float(ms)),
            "--seed",
            str(int(seed)),
            "--ng",
            str(int(ng)),
            "--nl",
            str(int(nl)),
            "--ring",
            str(int(ring)),
            "--relax",
            str(int(relax)),
            "--strike-iters",
            str(int(strike_iters)),
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
        if Q_out is None or Q_out.shape != P.shape or not np.isfinite(Q_out).all():
            return None
        feas = False
        ov = 1e9
        try:
            js = json.loads(r.stdout.strip().splitlines()[-1])
            feas = bool(js.get("feasible", False))
            ov = float(js.get("overlap", 1e9))
        except Exception:
            pass
        return float(s_out), Q_out, feas, ov


def run_anneal_sched_regional(
    P_in,
    cx,
    cy,
    rad=3.4,
    r_peak=0.15,
    sweeps=6500,
    bp_curve="0:1400,0.5:3800,1:8500",
    lag_tol=0.0015,
    lag_grow=0.0006,
    seed=1,
    timeout_s=18.0,
):
    """Run schedule-driven rounded-square regional NPT Monte Carlo (`anneal sched` in `anneal.rs` lines 423-585).
    Uses `--lag-tol` to eliminate end-of-schedule residual rounding blowup while melting a `(cx, cy, rad)` pocket.
    Returns `(S_exact, Q_legal, js_dict)` or `None`.
    """
    abin = get_anneal_bin()
    if not os.path.exists(abin):
        return None
    P = normalize_to_origin(P_in)
    s0 = side_exact(P)
    r_curve = f"0:0,0.22:{float(r_peak):.4f},0.55:{float(r_peak):.4f},0.88:0,1:0"
    with tempfile.TemporaryDirectory(prefix="ansch_") as tmp:
        f_in = os.path.join(tmp, "in.txt")
        f_out = os.path.join(tmp, "out.txt")
        write_packer_cfg(f_in, s0, P)
        cmd = [
            abin,
            "sched",
            "--in",
            f_in,
            "--out",
            f_out,
            "--sweeps",
            str(int(sweeps)),
            "--seed",
            str(int(seed)),
            "--bp",
            str(bp_curve),
            "--r",
            r_curve,
            "--cx",
            repr(float(cx)),
            "--cy",
            repr(float(cy)),
            "--rad",
            repr(float(rad)),
            "--lag-tol",
            repr(float(lag_tol)),
            "--lag-grow",
            repr(float(lag_grow)),
        ]
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=float(timeout_s)
            )
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0 or not os.path.exists(f_out):
            return None
        _, Q_out = read_packer_cfg(f_out)
        if Q_out is None or Q_out.shape != P.shape or not np.isfinite(Q_out).all():
            return None
        js = {}
        try:
            js = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            pass
        Q_out = normalize_to_origin(Q_out)
        return side_exact(Q_out), Q_out, js


def run_santa_ga(
    P_in,
    nbr_packings=None,
    seed=1,
    sec=520.0,
    islands=6,
    pop=28,
    offspring=112,
    steps=260,
    threads=4,
    timeout_s=555.0,
):
    """Run compiled Rust 1st-Place Santa 2025 Island Genetic Algorithm (`santa_ga`).
    Uses `std::thread::scope` across `threads` OS threads (`arm_extend` + `mirror_seed` + `released_mirror`
    + `row_slide` + `band_reangle` + `twist` + `crossover` + `OVERLAP_TARGETING=0.80`).
    Returns `(Q_win_or_None, Q_champ_or_None, js_dict)`.
    """
    gbin = get_santa_ga_bin()
    if not os.path.exists(gbin):
        return None, None, {}
    P = normalize_to_origin(P_in)
    s0 = side_exact(P)
    with tempfile.TemporaryDirectory(prefix="sga_") as tmp:
        f_in = os.path.join(tmp, "in.txt")
        f_out = os.path.join(tmp, "out.txt")
        write_packer_cfg(f_in, s0, P)
        cmd = [
            gbin,
            "--in",
            f_in,
            "--out",
            f_out,
            "--seed",
            str(int(seed)),
            "--sec",
            repr(float(sec)),
            "--islands",
            str(int(islands)),
            "--pop",
            str(int(pop)),
            "--offspring",
            str(int(offspring)),
            "--steps",
            str(int(steps)),
            "--threads",
            str(int(threads)),
        ]
        for ni, Pn in enumerate(nbr_packings or []):
            if Pn is not None and len(Pn) >= len(P):
                fn = os.path.join(tmp, f"nbr_{ni}.txt")
                write_packer_cfg(fn, side_exact(Pn), Pn)
                cmd += ["--nbr", fn]
        js = {}
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=float(timeout_s)
            )
            if r.stdout.strip():
                js = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            pass

        Q_win = None
        if os.path.exists(f_out):
            _, Qw = read_packer_cfg(f_out)
            if Qw is not None and Qw.shape == P.shape and np.isfinite(Qw).all():
                Q_win = normalize_to_origin(Qw)

        Q_champ = None
        f_ch = f_out + ".champ"
        if os.path.exists(f_ch):
            _, Qc = read_packer_cfg(f_ch)
            if Qc is not None and Qc.shape == P.shape and np.isfinite(Qc).all():
                Q_champ = normalize_to_origin(Qc)

        return Q_win, Q_champ, js
