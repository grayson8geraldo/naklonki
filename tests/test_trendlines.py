"""Tests for trendline detection."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.trendlines import detect_trendlines, _find_swing_highs, _find_swing_lows
from src.models.config import TrendlineConfig


def _make_df(closes: list[float], noise: float = 0.0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from close prices."""
    n = len(closes)
    arr = np.array(closes)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-noise, noise, n) if noise else np.zeros(n)
    highs = arr + abs(jitter) + 0.5
    lows = arr - abs(jitter) - 0.5
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": arr,
        "high": highs,
        "low": lows,
        "close": arr,
        "volume": np.ones(n) * 1000,
    })


class TestSwingDetection:
    def test_swing_high(self):
        prices = [1, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1]
        highs = np.array(prices, dtype=float)
        result = _find_swing_highs(highs, order=5)
        assert 5 in result or 15 in result

    def test_swing_low(self):
        prices = [10, 8, 6, 4, 2, 1, 2, 4, 6, 8, 10, 8, 6, 4, 2, 1, 2, 4, 6, 8, 10]
        lows = np.array(prices, dtype=float)
        result = _find_swing_lows(lows, order=5)
        assert 5 in result or 15 in result


class TestTrendlineDetection:
    def test_ascending_support_detected(self):
        """Create an ascending pattern with clear swing lows on a line."""
        # Build price data where lows oscillate with troughs on an ascending line.
        # Troughs occur at roughly indices 10, 31, 52, 73 (period ~21).
        n = 100
        x = np.arange(n, dtype=float)
        support = 100 + 0.5 * x  # ascending support line
        # Oscillation: lows dip down to the support line at troughs,
        # then bounce well above it.
        osc = 8 * np.abs(np.sin(x * np.pi / 21))  # always >= 0, zero at troughs
        lows = support + osc  # at troughs, lows == support
        closes = lows + 3
        highs = closes + 3

        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n) * 1000,
        })

        config = TrendlineConfig(
            min_touches=3,
            touch_tolerance_pct=1.5,
            min_candles_span=20,
            max_slope_angle=60,
        )
        lines = detect_trendlines(df, config)
        # Should find at least one ascending support line
        assert len(lines) > 0

    def test_no_trendline_in_random_data(self):
        """Random noise should produce fewer high-quality trendlines."""
        rng = np.random.default_rng(123)
        n = 100
        closes = rng.uniform(90, 110, n)
        df = _make_df(closes.tolist())

        config = TrendlineConfig(
            min_touches=5,
            touch_tolerance_pct=0.1,
            min_candles_span=20,
            max_slope_angle=60,
        )
        lines = detect_trendlines(df, config)
        # With tight tolerance and high touch requirement, random data
        # should yield few lines
        assert len(lines) <= 5
