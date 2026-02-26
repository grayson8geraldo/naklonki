"""Exchange client wrapping ccxt for async market data and order execution."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ccxt.async_support as ccxt
import pandas as pd

from src.models.config import ExchangeConfig

logger = logging.getLogger(__name__)


class ExchangeClient:
    """Thin async wrapper around a ccxt exchange instance."""

    def __init__(self, config: ExchangeConfig) -> None:
        self._config = config
        self._exchange: ccxt.Exchange | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        exchange_cls = getattr(ccxt, self._config.name)
        opts: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},  # USDT perpetual futures
        }
        # Only pass credentials if provided (paper mode may not need them)
        if self._config.api_key:
            opts["apiKey"] = self._config.api_key
        if self._config.api_secret:
            opts["secret"] = self._config.api_secret
        if self._config.testnet:
            opts["sandbox"] = True
        self._exchange = exchange_cls(opts)
        await self._exchange.load_markets()
        logger.info("Connected to %s (%d markets)", self._config.name, len(self._exchange.markets))

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()

    @property
    def exchange(self) -> ccxt.Exchange:
        assert self._exchange is not None, "Call connect() first"
        return self._exchange

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def fetch_tickers(self) -> dict[str, dict]:
        """Return all tickers (symbol -> ticker dict)."""
        return await self.exchange.fetch_tickers()

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 500
    ) -> pd.DataFrame:
        """Fetch OHLCV candles and return as a DataFrame."""
        raw = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    async def fetch_balance(self) -> dict:
        return await self.exchange.fetch_balance()

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    async def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float | None = None,
        order_type: str = "limit",
        params: dict | None = None,
    ) -> dict:
        """Create an order. Returns the ccxt order dict."""
        params = params or {}
        logger.info("Creating %s %s order: %s qty=%.6f price=%s", order_type, side, symbol, amount, price)
        return await self.exchange.create_order(
            symbol, order_type, side, amount, price, params
        )

    async def create_stop_loss(
        self, symbol: str, side: str, amount: float, stop_price: float
    ) -> dict:
        """Place a stop-loss (stop-market) order."""
        params = {"stopPrice": stop_price, "type": "stop_market"}
        return await self.create_order(symbol, side, amount, order_type="stop_market", params=params)

    async def create_take_profit(
        self, symbol: str, side: str, amount: float, tp_price: float
    ) -> dict:
        """Place a take-profit (limit) order."""
        params = {"stopPrice": tp_price, "type": "take_profit_market"}
        return await self.create_order(symbol, side, amount, order_type="take_profit_market", params=params)

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        return await self.exchange.cancel_order(order_id, symbol)

    async def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        return await self.exchange.fetch_open_orders(symbol)

    async def fetch_positions(self) -> list[dict]:
        return await self.exchange.fetch_positions()
