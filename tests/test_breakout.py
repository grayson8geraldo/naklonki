"""Tests for breakout detection."""

import numpy as np
import pandas as pd
import pytest

from src.entry.breakout import BreakoutDetector
from src.models.config import EntryConfig
from src.models.domain import (
    ConsolidationZone,
    Setup,
    SetupQuality,
    TradeDirection,
    Trendline,
    TrendDirection,
)


def _make_setup(direction: TradeDirection = TradeDirection.LONG) -> Setup:
    tl_dir = TrendDirection.DOWN if direction == TradeDirection.LONG else TrendDirection.UP
    slope = -0.1 if tl_dir == TrendDirection.DOWN else 0.1
    tl = Trendline(
        direction=tl_dir,
        slope=slope,
        intercept=100.0,
        touch_indices=[10, 20, 30],
        start_index=10,
        end_index=30,
    )
    consol = ConsolidationZone(high=98, low=95, start_index=25, end_index=30)
    return Setup(
        symbol="TEST/USDT",
        trendline=tl,
        consolidation=consol,
        direction=direction,
    )


def _make_breakout_df(n: int = 50, breakout_at: int = 45) -> pd.DataFrame:
    """Create OHLCV data with a breakout at given index.

    Trendline: intercept=100, slope=-0.1 → descending resistance.
    At index 45, line_price = 100 - 0.1*45 = 95.5.
    We make closes above that from breakout_at onwards.
    """
    closes = []
    for i in range(n):
        line = 100 - 0.1 * i
        if i >= breakout_at:
            closes.append(line + 2)  # above the line
        else:
            closes.append(line - 2)  # below the line
    arr = np.array(closes)

    # Volume spike at breakout
    volumes = np.ones(n) * 100
    volumes[breakout_at:] = 300  # 3x spike

    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": arr,
        "high": arr + 1,
        "low": arr - 1,
        "close": arr,
        "volume": volumes,
    })


class TestBreakoutDetector:
    def test_detects_confirmed_breakout(self):
        config = EntryConfig(
            breakout_confirm_candles=2,
            volume_spike_multiplier=1.5,
            volume_avg_period=20,
        )
        detector = BreakoutDetector(config)
        setup = _make_setup(TradeDirection.LONG)
        df = _make_breakout_df(n=50, breakout_at=45)

        signal = detector.check_breakout(df, setup)
        assert signal is not None
        assert signal.entry_price > 0
        assert signal.stop_loss < signal.entry_price
        assert signal.take_profit > signal.entry_price
        assert signal.risk_reward >= 3.0

    def test_no_breakout_without_volume(self):
        config = EntryConfig(
            breakout_confirm_candles=2,
            volume_spike_multiplier=1.5,
            volume_avg_period=20,
        )
        detector = BreakoutDetector(config)
        setup = _make_setup(TradeDirection.LONG)
        df = _make_breakout_df(n=50, breakout_at=45)
        # Kill volume spike
        df["volume"] = 100.0

        signal = detector.check_breakout(df, setup)
        assert signal is None

    def test_reentry_limit(self):
        config = EntryConfig(allow_reentry=True, reentry_window_candles=10)
        detector = BreakoutDetector(config)

        # First stopout
        detector.record_stopout("TEST/USDT", candle_index=40)

        setup = _make_setup(TradeDirection.LONG)
        df = _make_breakout_df(n=50, breakout_at=45)

        # First re-entry should work
        signal = detector.check_breakout(df, setup)
        assert signal is not None
        assert signal.is_reentry is True

        # Second stopout
        detector.record_stopout("TEST/USDT", candle_index=48)
        signal2 = detector.check_breakout(df, setup)
        assert signal2 is None  # exceeded re-entry limit

    def test_reentry_disabled(self):
        config = EntryConfig(allow_reentry=False)
        detector = BreakoutDetector(config)
        detector.record_stopout("TEST/USDT", candle_index=40)

        setup = _make_setup(TradeDirection.LONG)
        df = _make_breakout_df(n=50, breakout_at=45)

        signal = detector.check_breakout(df, setup)
        assert signal is None
