"""Block 1 — Asset Screener.

Builds a daily watchlist by filtering exchange tickers against strict criteria:
  - Minimum 24h volume ($20M default)
  - Classifies coins into categories: strong uptrend, strong downtrend, pump, recent listing
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.exchange.client import ExchangeClient
from src.models.config import ScreenerConfig
from src.models.domain import CoinCategory, CoinInfo, Watchlist

logger = logging.getLogger(__name__)


class AssetScreener:
    """Filters and classifies crypto assets for the trading watchlist."""

    def __init__(self, exchange: ExchangeClient, config: ScreenerConfig) -> None:
        self._exchange = exchange
        self._cfg = config

    async def build_watchlist(self) -> Watchlist:
        """Run the full screening pipeline and return a Watchlist."""
        tickers = await self._exchange.fetch_tickers()

        # Only USDT perpetual / spot pairs
        usdt_tickers = {
            sym: t for sym, t in tickers.items()
            if sym.endswith("/USDT") or sym.endswith("/USDT:USDT")
        }

        logger.info("Screening %d USDT pairs", len(usdt_tickers))

        coins: list[CoinInfo] = []
        for symbol, ticker in usdt_tickers.items():
            coin = self._evaluate_ticker(symbol, ticker)
            if coin is not None:
                coins.append(coin)

        # Enrich with trend data (fetch daily candles for trend classification)
        coins = await self._classify_trends(coins)

        # Sort by volume descending, cap at max size
        coins.sort(key=lambda c: c.volume_24h, reverse=True)
        coins = coins[: self._cfg.watchlist_max_size]

        logger.info("Watchlist: %d coins passed filters", len(coins))
        return Watchlist(coins=coins)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_ticker(self, symbol: str, ticker: dict) -> CoinInfo | None:
        """Apply volume filter and basic classification on a single ticker."""
        quote_vol = ticker.get("quoteVolume") or 0
        if quote_vol < self._cfg.min_volume_24h:
            return None

        price = ticker.get("last") or ticker.get("close") or 0
        change_pct = ticker.get("percentage") or 0

        categories: list[CoinCategory] = []

        # Detect pumps (sharp rise)
        if change_pct >= self._cfg.pump_threshold_pct:
            categories.append(CoinCategory.PUMP)

        return CoinInfo(
            symbol=symbol,
            volume_24h=quote_vol,
            price=price,
            change_24h_pct=change_pct,
            categories=categories,
        )

    async def _classify_trends(self, coins: list[CoinInfo]) -> list[CoinInfo]:
        """Fetch daily candles and classify trend direction/strength."""
        enriched: list[CoinInfo] = []
        for coin in coins:
            try:
                df = await self._exchange.fetch_ohlcv(
                    coin.symbol,
                    timeframe="1d",
                    limit=self._cfg.trend_lookback_days,
                )
                strength, direction = self._compute_trend(df)
                coin.trend_strength = strength

                if strength >= self._cfg.trend_strength_threshold:
                    if direction > 0:
                        coin.categories.append(CoinCategory.STRONG_UPTREND)
                    else:
                        coin.categories.append(CoinCategory.STRONG_DOWNTREND)

                enriched.append(coin)
            except Exception:
                logger.warning("Failed to fetch candles for %s, skipping trend", coin.symbol)
                enriched.append(coin)
        return enriched

    @staticmethod
    def _compute_trend(df: pd.DataFrame) -> tuple[float, float]:
        """Return (strength 0..1, direction_sign) using linear‑regression slope.

        Strength is the R^2 of the regression (how linear the trend is).
        Direction sign is +1 for uptrend, -1 for downtrend.
        """
        if len(df) < 5:
            return 0.0, 0.0

        closes = df["close"].values.astype(float)
        x = np.arange(len(closes), dtype=float)

        # Linear regression
        x_mean = x.mean()
        y_mean = closes.mean()
        ss_xy = ((x - x_mean) * (closes - y_mean)).sum()
        ss_xx = ((x - x_mean) ** 2).sum()
        ss_yy = ((closes - y_mean) ** 2).sum()

        if ss_xx == 0 or ss_yy == 0:
            return 0.0, 0.0

        slope = ss_xy / ss_xx
        r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

        direction = 1.0 if slope > 0 else -1.0
        return float(min(r_squared, 1.0)), direction
