"""Setup finder — combines trendlines and consolidation to produce Setups.

A valid setup requires:
  - A trendline with 3+ touches
  - Preferably a consolidation zone near the latest touch area
  - Enough room for a 1:3 R/R move
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models.config import AnalysisConfig
from src.models.domain import (
    CoinInfo,
    ConsolidationZone,
    Setup,
    SetupQuality,
    TradeDirection,
    Trendline,
    TrendDirection,
)
from src.analysis.trendlines import detect_trendlines
from src.analysis.consolidation import detect_consolidation_zones, find_nearest_consolidation

logger = logging.getLogger(__name__)


class SetupFinder:
    """Scans OHLCV data for tradeable trendline-breakout setups."""

    def __init__(self, config: AnalysisConfig) -> None:
        self._cfg = config

    def find_setups(self, df: pd.DataFrame, coin: CoinInfo) -> list[Setup]:
        """Return a list of valid setups for the given coin's candle data."""
        trendlines = detect_trendlines(df, self._cfg.trendline)
        consolidation_zones = detect_consolidation_zones(df, self._cfg.consolidation)

        setups: list[Setup] = []
        last_idx = len(df) - 1

        for tl in trendlines:
            # Trendline should extend to recent candles (still relevant)
            if tl.end_index < last_idx - 20:
                continue

            # Determine trade direction from the trendline breakout
            direction = self._breakout_direction(tl)

            # Find consolidation near the end of the trendline
            consol = find_nearest_consolidation(
                consolidation_zones, tl.end_index, max_distance=15
            )

            # Quality assessment
            quality = self._assess_quality(tl, consol, coin, direction)

            # Is this counter-trend?
            is_counter = self._is_counter_trend(coin, direction)

            # Counter-trend needs higher quality
            if is_counter and quality == SetupQuality.LOW:
                continue

            setup = Setup(
                symbol=coin.symbol,
                trendline=tl,
                consolidation=consol,
                direction=direction,
                quality=quality,
                is_counter_trend=is_counter,
            )
            setups.append(setup)

        # Sort: trend-following first, then by quality
        setups.sort(key=lambda s: (not s.is_counter_trend, _quality_rank(s.quality)), reverse=True)
        return setups

    # ------------------------------------------------------------------

    @staticmethod
    def _breakout_direction(tl: Trendline) -> TradeDirection:
        """Determine trade direction when price breaks the trendline.

        - Breaking above a descending resistance → LONG
        - Breaking below an ascending support → SHORT
        """
        if tl.direction == TrendDirection.DOWN:
            return TradeDirection.LONG
        return TradeDirection.SHORT

    @staticmethod
    def _assess_quality(
        tl: Trendline,
        consol: ConsolidationZone | None,
        coin: CoinInfo,
        direction: TradeDirection,
    ) -> SetupQuality:
        """Score the setup quality."""
        score = 0

        # More touches → better
        if tl.touch_count >= 4:
            score += 2
        elif tl.touch_count >= 3:
            score += 1

        # Consolidation present → better
        if consol is not None:
            score += 2

        # Higher volume → better
        if coin.volume_24h >= 50_000_000:
            score += 1

        if score >= 4:
            return SetupQuality.HIGH
        if score >= 2:
            return SetupQuality.MEDIUM
        return SetupQuality.LOW

    @staticmethod
    def _is_counter_trend(coin: CoinInfo, direction: TradeDirection) -> bool:
        """Check if the trade goes against the dominant trend."""
        from src.models.domain import CoinCategory

        if direction == TradeDirection.LONG:
            return CoinCategory.STRONG_DOWNTREND in coin.categories
        if direction == TradeDirection.SHORT:
            return CoinCategory.STRONG_UPTREND in coin.categories
        return False


def _quality_rank(q: SetupQuality) -> int:
    return {SetupQuality.HIGH: 3, SetupQuality.MEDIUM: 2, SetupQuality.LOW: 1}[q]
