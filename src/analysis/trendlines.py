"""Trendline detection algorithm.

Detects sloped support and resistance lines from OHLCV data.
A valid trendline must:
  1. Originate from a clear local extreme (swing high/low)
  2. Have 3+ price touches
  3. Span a minimum number of candles
"""

from __future__ import annotations

import logging
import math
from itertools import combinations

import numpy as np
import pandas as pd

from src.models.config import TrendlineConfig
from src.models.domain import Trendline, TrendDirection

logger = logging.getLogger(__name__)


def detect_trendlines(
    df: pd.DataFrame,
    config: TrendlineConfig,
) -> list[Trendline]:
    """Detect valid trendlines from OHLCV DataFrame.

    Returns a list of Trendline objects sorted by touch count descending.
    """
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n = len(df)

    # 1. Find swing highs and swing lows (local extremes)
    swing_high_idx = _find_swing_highs(highs, order=5)
    swing_low_idx = _find_swing_lows(lows, order=5)

    trendlines: list[Trendline] = []

    # 2. Build resistance lines from swing highs (descending or flat)
    trendlines.extend(
        _build_lines_from_pivots(
            pivots=swing_high_idx,
            prices=highs,
            all_highs=highs,
            all_lows=lows,
            direction=TrendDirection.DOWN,
            config=config,
            n=n,
        )
    )

    # 3. Build support lines from swing lows (ascending or flat)
    trendlines.extend(
        _build_lines_from_pivots(
            pivots=swing_low_idx,
            prices=lows,
            all_highs=highs,
            all_lows=lows,
            direction=TrendDirection.UP,
            config=config,
            n=n,
        )
    )

    # Deduplicate very similar lines
    trendlines = _deduplicate(trendlines, config)

    # Sort by touch count (best first)
    trendlines.sort(key=lambda t: t.touch_count, reverse=True)
    return trendlines


# ------------------------------------------------------------------
# Swing detection
# ------------------------------------------------------------------

def _find_swing_highs(highs: np.ndarray, order: int = 5) -> list[int]:
    """Return indices of local maxima (swing highs)."""
    indices = []
    for i in range(order, len(highs) - order):
        window = highs[i - order : i + order + 1]
        if highs[i] == window.max() and np.sum(window == highs[i]) == 1:
            indices.append(i)
    return indices


def _find_swing_lows(lows: np.ndarray, order: int = 5) -> list[int]:
    """Return indices of local minima (swing lows)."""
    indices = []
    for i in range(order, len(lows) - order):
        window = lows[i - order : i + order + 1]
        if lows[i] == window.min() and np.sum(window == lows[i]) == 1:
            indices.append(i)
    return indices


# ------------------------------------------------------------------
# Line construction
# ------------------------------------------------------------------

def _build_lines_from_pivots(
    pivots: list[int],
    prices: np.ndarray,
    all_highs: np.ndarray,
    all_lows: np.ndarray,
    direction: TrendDirection,
    config: TrendlineConfig,
    n: int,
) -> list[Trendline]:
    """Try all pairs of pivot points, build a line, and count touches."""
    results: list[Trendline] = []

    if len(pivots) < 2:
        return results

    for i, j in combinations(range(len(pivots)), 2):
        idx_a, idx_b = pivots[i], pivots[j]
        if abs(idx_b - idx_a) < config.min_candles_span:
            continue

        p_a = prices[idx_a]
        p_b = prices[idx_b]
        slope = (p_b - p_a) / (idx_b - idx_a)

        # Check slope angle isn't too steep
        angle_deg = abs(math.degrees(math.atan(slope / p_a * 100)))  # normalized
        if angle_deg > config.max_slope_angle:
            continue

        # For support (UP), slope should be >= 0 (ascending or flat)
        # For resistance (DOWN), slope should be <= 0 (descending or flat)
        if direction == TrendDirection.UP and slope < -0.0001 * p_a:
            continue
        if direction == TrendDirection.DOWN and slope > 0.0001 * p_a:
            continue

        intercept = p_a - slope * idx_a

        # Count touches across all candles
        touches = _count_touches(
            slope, intercept, all_highs, all_lows, direction, config, n
        )

        if len(touches) < config.min_touches:
            continue

        tl = Trendline(
            direction=direction,
            slope=slope,
            intercept=intercept,
            touch_indices=touches,
            start_index=min(touches),
            end_index=max(touches),
        )
        results.append(tl)

    return results


def _count_touches(
    slope: float,
    intercept: float,
    highs: np.ndarray,
    lows: np.ndarray,
    direction: TrendDirection,
    config: TrendlineConfig,
    n: int,
) -> list[int]:
    """Count how many candles 'touch' the trendline within tolerance."""
    touches: list[int] = []
    for k in range(n):
        line_price = intercept + slope * k
        if line_price <= 0:
            continue
        tol = line_price * config.touch_tolerance_pct / 100

        if direction == TrendDirection.UP:
            # Support line — lows should touch from above
            if abs(lows[k] - line_price) <= tol and lows[k] >= line_price - tol:
                touches.append(k)
        else:
            # Resistance line — highs should touch from below
            if abs(highs[k] - line_price) <= tol and highs[k] <= line_price + tol:
                touches.append(k)
    return touches


# ------------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------------

def _deduplicate(trendlines: list[Trendline], config: TrendlineConfig) -> list[Trendline]:
    """Remove near-duplicate trendlines (similar slope and intercept)."""
    if not trendlines:
        return []

    kept: list[Trendline] = []
    for tl in trendlines:
        is_dup = False
        for existing in kept:
            if existing.direction != tl.direction:
                continue
            price_ref = max(abs(tl.intercept), 1e-9)
            slope_diff = abs(tl.slope - existing.slope) / max(abs(existing.slope), 1e-9)
            intercept_diff = abs(tl.intercept - existing.intercept) / price_ref
            if slope_diff < 0.1 and intercept_diff < 0.01:
                # Keep the one with more touches
                if tl.touch_count > existing.touch_count:
                    kept.remove(existing)
                    kept.append(tl)
                is_dup = True
                break
        if not is_dup:
            kept.append(tl)
    return kept
