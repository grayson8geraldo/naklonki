"""Block 3 — Trade Entry (Breakout Detection & Triggers).

Monitors setups for trendline breakout with volume confirmation.
Supports one re-entry after false breakout if the formation is still intact.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from src.models.config import EntryConfig
from src.models.domain import (
    Setup,
    TradeDirection,
    TradeSignal,
    TrendDirection,
)

logger = logging.getLogger(__name__)


class BreakoutDetector:
    """Detects confirmed breakouts with volume spikes."""

    def __init__(self, config: EntryConfig) -> None:
        self._cfg = config
        # Track stop-outs for re-entry logic: symbol -> count
        self._stopout_count: dict[str, int] = {}
        # Track last breakout candle index per symbol for re-entry window
        self._last_stopout_index: dict[str, int] = {}

    def check_breakout(self, df: pd.DataFrame, setup: Setup) -> TradeSignal | None:
        """Check if the latest candles confirm a breakout of the setup's trendline.

        Returns a TradeSignal if confirmed, otherwise None.
        """
        n = len(df)
        if n < self._cfg.breakout_confirm_candles + self._cfg.volume_avg_period:
            return None

        tl = setup.trendline
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        volumes = df["volume"].values.astype(float)

        # Check the last N candles for breakout confirmation
        confirm = self._cfg.breakout_confirm_candles
        breakout_confirmed = True

        for offset in range(1, confirm + 1):
            idx = n - offset
            line_price = tl.price_at(idx)

            if setup.direction == TradeDirection.LONG:
                # Price must close above the trendline
                if closes[idx] <= line_price:
                    breakout_confirmed = False
                    break
            else:
                # Price must close below the trendline
                if closes[idx] >= line_price:
                    breakout_confirmed = False
                    break

        if not breakout_confirmed:
            return None

        # Volume confirmation: breakout candle volume vs average
        breakout_idx = n - 1
        avg_vol = volumes[breakout_idx - self._cfg.volume_avg_period : breakout_idx].mean()
        breakout_vol = volumes[breakout_idx]

        if avg_vol <= 0:
            return None

        if breakout_vol < avg_vol * self._cfg.volume_spike_multiplier:
            logger.debug(
                "%s: breakout without volume spike (%.0f vs avg %.0f)",
                setup.symbol, breakout_vol, avg_vol,
            )
            return None

        # Check re-entry eligibility
        is_reentry = False
        if setup.symbol in self._stopout_count:
            if not self._cfg.allow_reentry:
                logger.info("%s: re-entry disabled, skipping", setup.symbol)
                return None
            if self._stopout_count[setup.symbol] > 1:
                logger.info("%s: already used re-entry, skipping", setup.symbol)
                return None
            # Check re-entry window
            last_stop = self._last_stopout_index.get(setup.symbol, 0)
            if breakout_idx - last_stop > self._cfg.reentry_window_candles:
                logger.info("%s: re-entry window expired", setup.symbol)
                return None
            is_reentry = True

        # Calculate entry, SL, TP
        entry_price = closes[breakout_idx]
        stop_loss = self._calculate_stop_loss(setup, df, breakout_idx)
        take_profit = self._calculate_take_profit(entry_price, stop_loss, setup.direction)

        return TradeSignal(
            setup=setup,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volume_at_breakout=breakout_vol,
            avg_volume=avg_vol,
            is_reentry=is_reentry,
        )

    def record_stopout(self, symbol: str, candle_index: int) -> None:
        """Record that a position was stopped out (for re-entry logic)."""
        self._stopout_count[symbol] = self._stopout_count.get(symbol, 0) + 1
        self._last_stopout_index[symbol] = candle_index

    def reset_symbol(self, symbol: str) -> None:
        """Clear stopout tracking for a symbol (after successful trade or new setup)."""
        self._stopout_count.pop(symbol, None)
        self._last_stopout_index.pop(symbol, None)

    # ------------------------------------------------------------------
    # SL / TP Calculation
    # ------------------------------------------------------------------

    def _calculate_stop_loss(
        self, setup: Setup, df: pd.DataFrame, breakout_idx: int
    ) -> float:
        """Place stop-loss behind the consolidation zone or recent swing."""
        # Prefer consolidation zone boundary
        if setup.consolidation is not None:
            if setup.direction == TradeDirection.LONG:
                return setup.consolidation.low * 0.998  # small buffer below
            return setup.consolidation.high * 1.002  # small buffer above

        # Fallback: use recent swing extreme
        lookback = min(20, breakout_idx)
        lows = df["low"].values.astype(float)
        highs = df["high"].values.astype(float)

        if setup.direction == TradeDirection.LONG:
            recent_low = lows[breakout_idx - lookback : breakout_idx].min()
            return recent_low * 0.998
        recent_high = highs[breakout_idx - lookback : breakout_idx].max()
        return recent_high * 1.002

    @staticmethod
    def _calculate_take_profit(
        entry: float, stop_loss: float, direction: TradeDirection
    ) -> float:
        """Calculate TP at 3.5 R/R from entry (beyond minimum 3.0 requirement)."""
        risk = abs(entry - stop_loss)
        target_rr = 3.5

        if direction == TradeDirection.LONG:
            return entry + risk * target_rr
        return entry - risk * target_rr
