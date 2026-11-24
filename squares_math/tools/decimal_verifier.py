"""60-Digit Decimal Exact SAT Collision & Containment Certifier (immune to Float64 rounding traps)."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

# 60 decimal digits of precision mirrors/exceeds high-precision interval validation
_PREC = 64
getcontext().prec = _PREC

_PI_64 = Decimal("3.1415926535897932384626433832795028841971693993751058209749445923")
_TWO_PI = _PI_64 * 2
_HALF_PI = _PI_64 / 2
_HALF = Decimal("0.5")
_ZERO = Decimal("0")
_ONE = Decimal("1")


def _dec_sin_cos_small(x: Decimal) -> tuple[Decimal, Decimal]:
    """Compute (cos(x), sin(x)) for |x| <= pi/4 using exact Taylor series at 64-digit precision."""
    x2 = x * x
    # cos(x) = 1 - x^2/2! + x^4/4! - ...
    cos_val = _ONE
    term_c = _ONE
    sin_val = x
    term_s = x
    for k in range(1, 28):
        term_c = -term_c * x2 / Decimal((2 * k - 1) * (2 * k))
        cos_val += term_c
        term_s = -term_s * x2 / Decimal((2 * k) * (2 * k + 1))
        sin_val += term_s
        if abs(term_c) < Decimal("1e-61") and abs(term_s) < Decimal("1e-61"):
            break
    return cos_val, sin_val


@lru_cache(maxsize=8192)
def dec_cos_sin(theta: Decimal) -> tuple[Decimal, Decimal]:
    """Compute exact (cos(theta), sin(theta)) at 64-digit Decimal precision for any radian angle."""
    t = theta % _TWO_PI
    if t < _ZERO:
        t += _TWO_PI
    quadrant = int((t / _HALF_PI).to_integral_value())
    rem = t - Decimal(quadrant) * _HALF_PI
    # Clamp rem into [-pi/4, pi/4]
    if rem > _PI_64 / 4:
        rem -= _HALF_PI
        quadrant += 1
    elif rem < -_PI_64 / 4:
        rem += _HALF_PI
        quadrant -= 1
    c0, s0 = _dec_sin_cos_small(rem)
    q = quadrant % 4
    if q == 0:
        return c0, s0
    if q == 1:
        return -s0, c0
    if q == 2:
        return -c0, -s0
    return s0, -c0


@dataclass(frozen=True)
class PuzzleVerification:
    """Exact 64-digit Decimal verification report for a single puzzle N."""

    n: int
    valid: bool
    L: Decimal
    score: Decimal
    min_separation_gap: Decimal
    overlap_pair: tuple[int, int] | None = None
    reason: str = ""


@dataclass
class SubmissionVerificationReport:
    """Full verification summary across all 1,000 puzzles / 500,500 rows."""

    valid: bool
    puzzles_checked: int
    total_rows: int
    total_score: Decimal
    baseline_grid_score: Decimal
    improved_count: int
    min_global_gap: Decimal
    failed_puzzles: list[PuzzleVerification] = field(default_factory=list)


def compute_exact_puzzle_L(rows: Sequence[tuple[Decimal, Decimal, Decimal]]) -> Decimal:
    """Compute exact enclosing square side L in 64-digit Decimal."""
    min_x = None
    max_x = None
    min_y = None
    max_y = None
    for x, y, ang in rows:
        c, s = dec_cos_sin(ang)
        hx = _HALF * (abs(c) + abs(s))
        lo_x, hi_x = x - hx, x + hx
        lo_y, hi_y = y - hx, y + hx
        min_x = lo_x if min_x is None or lo_x < min_x else min_x
        max_x = hi_x if max_x is None or hi_x > max_x else max_x
        min_y = lo_y if min_y is None or lo_y < min_y else min_y
        max_y = hi_y if max_y is None or hi_y > max_y else max_y
    if min_x is None:
        return _ZERO
    return max(max_x - min_x, max_y - min_y)


def verify_puzzle_decimal(
    n: int,
    rows: Sequence[tuple[Decimal, Decimal, Decimal]],
    min_required_gap: Decimal = Decimal("0"),
) -> PuzzleVerification:
    """Strictly verify zero overlaps for puzzle n using spatial grid + 64-digit Decimal SAT."""
    if len(rows) != n:
        return PuzzleVerification(
            n=n,
            valid=False,
            L=_ZERO,
            score=Decimal("999"),
            min_separation_gap=Decimal("-1"),
            reason=f"Expected {n} rows, got {len(rows)}",
        )

    precomp: list[tuple[Decimal, Decimal, Decimal, Decimal, Decimal, float, float]] = []
    min_x = max_x = min_y = max_y = None
    for x, y, ang in rows:
        if not (x.is_finite() and y.is_finite() and ang.is_finite()):
            return PuzzleVerification(
                n=n,
                valid=False,
                L=_ZERO,
                score=Decimal("999"),
                min_separation_gap=Decimal("-1"),
                reason="Non-finite coordinate",
            )
        c, s = dec_cos_sin(ang)
        hx = _HALF * (abs(c) + abs(s))
        lo_x, hi_x = x - hx, x + hx
        lo_y, hi_y = y - hx, y + hx
        min_x = lo_x if min_x is None or lo_x < min_x else min_x
        max_x = hi_x if max_x is None or hi_x > max_x else max_x
        min_y = lo_y if min_y is None or lo_y < min_y else min_y
        max_y = hi_y if max_y is None or hi_y > max_y else max_y
        precomp.append((x, y, c, s, hx, float(x), float(y)))

    L = max(max_x - min_x, max_y - min_y) if min_x is not None else _ZERO
    score = (L / Decimal(n).sqrt()) - _ONE

    if n <= 1:
        return PuzzleVerification(
            n=n, valid=True, L=L, score=score, min_separation_gap=Decimal("1")
        )

    # Spatial grid cell size 1.5 guarantees any pair within max possible center distance sqrt(2) is adjacent
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (_, _, _, _, _, fx, fy) in enumerate(precomp):
        gx = math.floor(fx / 1.5)
        gy = math.floor(fy / 1.5)
        grid[(gx, gy)].append(idx)

    eps10 = Decimal("1e-10")
    min_gap = Decimal("1e9")
    for (gx, gy), idxs in grid.items():
        neighbors: list[int] = []
        for dx_cell in (-1, 0, 1):
            for dy_cell in (-1, 0, 1):
                neighbors.extend(grid.get((gx + dx_cell, gy + dy_cell), ()))
        for i in idxs:
            xi, yi, ci, si, hxi, fxi, fyi = precomp[i]
            for j in neighbors:
                if j <= i:
                    continue
                xj, yj, cj, sj, hxj, fxj, fyj = precomp[j]
                if abs(fxi - fxj) > 1.45 or abs(fyi - fyj) > 1.45:
                    continue
                dx = xi - xj
                dy = yi - yj
                max_h = hxi + hxj + eps10
                if abs(dx) >= max_h or abs(dy) >= max_h:
                    continue

                cc = abs(ci * cj + si * sj)
                cs = abs(-ci * sj + si * cj)
                hs = _HALF + _HALF * (cc + cs)

                # Separation along each of the 4 candidate axes (positive => strictly disjoint)
                gap_1 = abs(dx * ci + dy * si) - hs
                gap_2 = abs(-dx * si + dy * ci) - hs
                gap_3 = abs(dx * cj + dy * sj) - hs
                gap_4 = abs(-dx * sj + dy * cj) - hs
                pair_sep = max(gap_1, gap_2, gap_3, gap_4)
                if pair_sep < min_gap:
                    min_gap = pair_sep
                if pair_sep <= min_required_gap:
                    return PuzzleVerification(
                        n=n,
                        valid=False,
                        L=L,
                        score=score,
                        min_separation_gap=pair_sep,
                        overlap_pair=(i, j),
                        reason=f"Overlap or insufficient gap ({pair_sep:.3e}) between squares {i} and {j}",
                    )

    return PuzzleVerification(
        n=n, valid=True, L=L, score=score, min_separation_gap=min_gap
    )


def verify_csv_submission(
    csv_path: str | Path,
    expected_max_n: int = 1000,
    min_required_gap: Decimal = Decimal("0"),
) -> SubmissionVerificationReport:
    """Parse emitted submission.csv strings directly into 64-digit Decimal and verify every puzzle."""
    path = Path(csv_path)
    puzzles: dict[int, list[tuple[int, Decimal, Decimal, Decimal]]] = defaultdict(list)
    total_rows = 0
    expected_id = 0

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != ["id", "n", "square_id", "x", "y", "angle"]:
            raise ValueError(f"Invalid CSV header: {header!r}")
        for row in reader:
            if len(row) != 6:
                raise ValueError(f"Malformed CSV row at id={expected_id}: {row!r}")
            rid = int(row[0])
            n_val = int(row[1])
            sid = int(row[2])
            if rid != expected_id:
                raise ValueError(
                    f"Non-sequential id: expected {expected_id}, got {rid}"
                )
            puzzles[n_val].append(
                (sid, Decimal(row[3]), Decimal(row[4]), Decimal(row[5]))
            )
            expected_id += 1
            total_rows += 1

    total_score = _ZERO
    base_score = _ZERO
    improved_cnt = 0
    min_global_gap = Decimal("1e9")
    failed: list[PuzzleVerification] = []

    for n in range(1, expected_max_n + 1):
        items = sorted(puzzles.get(n, ()), key=lambda t: t[0])
        if [t[0] for t in items] != list(range(n)):
            pv = PuzzleVerification(
                n=n,
                valid=False,
                L=_ZERO,
                score=Decimal("999"),
                min_separation_gap=Decimal("-1"),
                reason=f"Missing or invalid square_id sequence for n={n}",
            )
            failed.append(pv)
            continue
        coords = [(x, y, ang) for _, x, y, ang in items]
        pv = verify_puzzle_decimal(n, coords, min_required_gap=min_required_gap)
        base_L = Decimal(math.ceil(math.sqrt(n)))
        base_n_score = base_L / Decimal(n).sqrt() - _ONE
        base_score += base_n_score
        if not pv.valid:
            failed.append(pv)
            total_score += base_n_score
        else:
            total_score += pv.score
            if pv.L < base_L - Decimal("1e-6"):
                improved_cnt += 1
            if pv.min_separation_gap < min_global_gap:
                min_global_gap = pv.min_separation_gap

    return SubmissionVerificationReport(
        valid=(
            len(failed) == 0
            and total_rows == expected_max_n * (expected_max_n + 1) // 2
        ),
        puzzles_checked=expected_max_n,
        total_rows=total_rows,
        total_score=total_score,
        baseline_grid_score=base_score,
        improved_count=improved_cnt,
        min_global_gap=min_global_gap,
        failed_puzzles=failed,
    )
