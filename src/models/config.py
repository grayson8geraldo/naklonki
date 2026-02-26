"""Configuration model loaded from settings.yaml."""

from __future__ import annotations

from pydantic import BaseModel


class ExchangeConfig(BaseModel):
    name: str = "bybit"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True


class PaperTradingConfig(BaseModel):
    enabled: bool = True
    initial_balance: float = 10_000.0
    state_file: str = "data/paper_state.json"


class ScreenerConfig(BaseModel):
    min_volume_24h: float = 20_000_000
    recent_listing_days: int = 30
    pump_threshold_pct: float = 30.0
    trend_lookback_days: int = 14
    trend_strength_threshold: float = 0.6
    watchlist_max_size: int = 50


class TrendlineConfig(BaseModel):
    min_touches: int = 3
    touch_tolerance_pct: float = 0.5
    min_candles_span: int = 20
    max_slope_angle: int = 60


class ConsolidationConfig(BaseModel):
    min_candles: int = 5
    max_range_pct: float = 3.0


class AnalysisConfig(BaseModel):
    timeframe: str = "1h"
    candle_lookback: int = 500
    trendline: TrendlineConfig = TrendlineConfig()
    consolidation: ConsolidationConfig = ConsolidationConfig()


class EntryConfig(BaseModel):
    breakout_confirm_candles: int = 2
    volume_spike_multiplier: float = 1.5
    volume_avg_period: int = 20
    allow_reentry: bool = True
    reentry_window_candles: int = 10
    max_breakout_age_candles: int = 5
    max_distance_from_trendline_pct: float = 2.0


class RiskConfig(BaseModel):
    risk_per_trade_pct: float = 1.0
    min_risk_reward: float = 3.0
    max_open_positions: int = 5
    max_daily_loss_pct: float = 3.0
    use_market_orders: bool = False


class BotConfig(BaseModel):
    exchange: ExchangeConfig = ExchangeConfig()
    paper_trading: PaperTradingConfig = PaperTradingConfig()
    screener: ScreenerConfig = ScreenerConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    entry: EntryConfig = EntryConfig()
    risk: RiskConfig = RiskConfig()
