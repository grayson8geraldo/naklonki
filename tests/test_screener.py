"""Tests for asset screener logic."""

import numpy as np
import pandas as pd
import pytest

from src.models.config import ScreenerConfig
from src.models.domain import CoinCategory
from src.screener.screener import AssetScreener


class TestScreenerTrendComputation:
    """Test the static trend computation method without exchange calls."""

    def test_uptrend_detected(self):
        """A linearly increasing series should have high R^2 and positive direction."""
        closes = [100 + i * 2 for i in range(20)]
        df = pd.DataFrame({"close": closes})
        strength, direction = AssetScreener._compute_trend(df)
        assert strength > 0.9
        assert direction > 0

    def test_downtrend_detected(self):
        closes = [200 - i * 3 for i in range(20)]
        df = pd.DataFrame({"close": closes})
        strength, direction = AssetScreener._compute_trend(df)
        assert strength > 0.9
        assert direction < 0

    def test_flat_market(self):
        """Flat market should have near-zero strength."""
        closes = [100.0] * 20
        df = pd.DataFrame({"close": closes})
        strength, direction = AssetScreener._compute_trend(df)
        assert strength < 0.01

    def test_noisy_no_trend(self):
        rng = np.random.default_rng(42)
        closes = rng.uniform(95, 105, 20).tolist()
        df = pd.DataFrame({"close": closes})
        strength, _ = AssetScreener._compute_trend(df)
        # Noisy data — strength should be low
        assert strength < 0.5


class TestScreenerTickerEvaluation:
    def test_volume_filter(self):
        config = ScreenerConfig(min_volume_24h=20_000_000, pump_threshold_pct=30)
        screener = AssetScreener(exchange=None, config=config)  # type: ignore[arg-type]

        # Below threshold
        result = screener._evaluate_ticker("LOW/USDT", {
            "quoteVolume": 10_000_000, "last": 50, "percentage": 5,
        })
        assert result is None

        # Above threshold
        result = screener._evaluate_ticker("HIGH/USDT", {
            "quoteVolume": 30_000_000, "last": 50, "percentage": 5,
        })
        assert result is not None
        assert result.symbol == "HIGH/USDT"

    def test_pump_classification(self):
        config = ScreenerConfig(pump_threshold_pct=30)
        screener = AssetScreener(exchange=None, config=config)  # type: ignore[arg-type]

        result = screener._evaluate_ticker("PUMP/USDT", {
            "quoteVolume": 50_000_000, "last": 1.5, "percentage": 45,
        })
        assert result is not None
        assert CoinCategory.PUMP in result.categories

    def test_no_pump_below_threshold(self):
        config = ScreenerConfig(pump_threshold_pct=30)
        screener = AssetScreener(exchange=None, config=config)  # type: ignore[arg-type]

        result = screener._evaluate_ticker("NORMAL/USDT", {
            "quoteVolume": 50_000_000, "last": 2.0, "percentage": 10,
        })
        assert result is not None
        assert CoinCategory.PUMP not in result.categories
