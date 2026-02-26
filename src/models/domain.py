"""Core domain models for the crypto screener bot."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime


class CoinCategory(enum.Enum):
    """Category assigned to a coin by the screener."""
    STRONG_UPTREND = "strong_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    PUMP = "pump"
    RECENT_LISTING = "recent_listing"


class TrendDirection(enum.Enum):
    UP = "up"
    DOWN = "down"


class TradeDirection(enum.Enum):
    LONG = "long"
    SHORT = "short"


class SetupQuality(enum.Enum):
    """How strong the setup is — affects position sizing confidence."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Screener models
# ---------------------------------------------------------------------------

@dataclass
class CoinInfo:
    """A coin that passed the screener filter."""
    symbol: str                          # e.g. "BTC/USDT"
    volume_24h: float                    # 24h volume in USD
    price: float                         # current price
    change_24h_pct: float                # 24h price change %
    categories: list[CoinCategory] = field(default_factory=list)
    listing_date: datetime | None = None
    trend_strength: float = 0.0          # 0..1 measure of trend intensity


@dataclass
class Watchlist:
    """Daily watchlist produced by the screener."""
    coins: list[CoinInfo] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Technical‑analysis models
# ---------------------------------------------------------------------------

@dataclass
class Trendline:
    """A detected sloped support/resistance trendline."""
    direction: TrendDirection             # up = ascending support, down = descending resistance
    slope: float                          # price change per candle
    intercept: float                      # price at candle index 0
    touch_indices: list[int] = field(default_factory=list)  # candle indices of touches
    start_index: int = 0
    end_index: int = 0

    @property
    def touch_count(self) -> int:
        return len(self.touch_indices)

    def price_at(self, candle_index: int) -> float:
        """Return the trendline price at the given candle index."""
        return self.intercept + self.slope * candle_index


@dataclass
class ConsolidationZone:
    """A horizontal consolidation (range) detected near a trendline."""
    high: float
    low: float
    start_index: int
    end_index: int

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    @property
    def range_pct(self) -> float:
        if self.low == 0:
            return 0.0
        return (self.high - self.low) / self.low * 100


@dataclass
class Setup:
    """A tradeable setup found by the analysis module."""
    symbol: str
    trendline: Trendline
    consolidation: ConsolidationZone | None
    direction: TradeDirection             # long if price breaks above, short if below
    quality: SetupQuality = SetupQuality.MEDIUM
    is_counter_trend: bool = False        # True if trading against the dominant trend


# ---------------------------------------------------------------------------
# Trade / order models
# ---------------------------------------------------------------------------

@dataclass
class TradeSignal:
    """Emitted when a breakout is confirmed."""
    setup: Setup
    entry_price: float
    stop_loss: float
    take_profit: float
    volume_at_breakout: float
    avg_volume: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_reentry: bool = False

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        if risk == 0:
            return 0.0
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk


@dataclass
class Position:
    """An open or closed position."""
    symbol: str
    direction: TradeDirection
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: datetime | None = None
    exit_price: float | None = None
    pnl: float | None = None
    was_stopped: bool = False
    reentry_used: bool = False
