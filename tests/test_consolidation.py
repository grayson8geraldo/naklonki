"""Tests for consolidation zone detection."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.consolidation import detect_consolidation_zones, find_nearest_consolidation
from src.models.config import ConsolidationConfig
from src.models.domain import ConsolidationZone


def _make_df(highs: list[float], lows: list[float]) -> pd.DataFrame:
    n = len(highs)
    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.ones(n) * 1000,
    })


class TestConsolidation:
    def test_detects_flat_zone(self):
        """A series of candles with tight range should be detected."""
        # 10 candles in a tight range, then a breakout
        highs = [101.0] * 10 + [110.0] * 5
        lows = [99.0] * 10 + [100.0] * 5
        df = _make_df(highs, lows)

        config = ConsolidationConfig(min_candles=5, max_range_pct=3.0)
        zones = detect_consolidation_zones(df, config)
        assert len(zones) >= 1
        assert zones[0].start_index == 0
        assert zones[0].end_index >= 4

    def test_no_zone_in_volatile_data(self):
        """Highly volatile data should not form a consolidation zone."""
        highs = [100 + i * 10 for i in range(20)]
        lows = [90 + i * 10 for i in range(20)]
        df = _make_df(highs, lows)

        config = ConsolidationConfig(min_candles=5, max_range_pct=2.0)
        zones = detect_consolidation_zones(df, config)
        assert len(zones) == 0

    def test_find_nearest(self):
        zones = [
            ConsolidationZone(high=101, low=99, start_index=10, end_index=20),
            ConsolidationZone(high=105, low=103, start_index=50, end_index=60),
        ]
        # Index 25 should find the first zone (distance 5)
        result = find_nearest_consolidation(zones, 25)
        assert result is not None
        assert result.start_index == 10

        # Index inside zone
        result = find_nearest_consolidation(zones, 55)
        assert result is not None
        assert result.start_index == 50
