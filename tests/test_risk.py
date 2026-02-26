"""Tests for risk management logic."""

import pytest

from src.models.config import RiskConfig
from src.models.domain import (
    ConsolidationZone,
    Position,
    Setup,
    SetupQuality,
    TradeDirection,
    TradeSignal,
    Trendline,
    TrendDirection,
)
from src.risk.manager import RiskManager


def _make_signal(
    entry: float = 100.0,
    sl: float = 97.0,
    tp: float = 109.0,
    direction: TradeDirection = TradeDirection.LONG,
) -> TradeSignal:
    tl = Trendline(direction=TrendDirection.DOWN, slope=-0.1, intercept=110)
    setup = Setup(
        symbol="BTC/USDT",
        trendline=tl,
        consolidation=None,
        direction=direction,
    )
    return TradeSignal(
        setup=setup,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        volume_at_breakout=5000,
        avg_volume=2000,
    )


class TestRiskManager:
    def _make_manager(self, **overrides) -> RiskManager:
        cfg = RiskConfig(**overrides)
        # No real exchange needed for validation tests
        return RiskManager(cfg, exchange=None)  # type: ignore[arg-type]

    def test_position_sizing_1pct_risk(self):
        mgr = self._make_manager()
        signal = _make_signal(entry=100, sl=97)
        qty = mgr.calculate_position_size(signal, balance=10_000)
        # risk_amount = 10000 * 1% = 100, price_risk = 3, qty = 100/3 ≈ 33.33
        assert abs(qty - 33.333) < 0.01

    def test_rejects_low_rr(self):
        mgr = self._make_manager(min_risk_reward=3.0)
        # R/R = 2/3 ≈ 0.67 — way below 3.0
        signal = _make_signal(entry=100, sl=97, tp=102)
        assert mgr.validate_signal(signal, balance=10_000) is False

    def test_accepts_good_rr(self):
        mgr = self._make_manager(min_risk_reward=3.0)
        # R/R = 10.5/3 = 3.5
        signal = _make_signal(entry=100, sl=97, tp=110.5)
        assert mgr.validate_signal(signal, balance=10_000) is True

    def test_max_positions_enforced(self):
        mgr = self._make_manager(max_open_positions=1)
        signal = _make_signal(entry=100, sl=97, tp=110.5)
        # Add a fake open position
        mgr._open_positions.append(
            Position(
                symbol="ETH/USDT",
                direction=TradeDirection.LONG,
                entry_price=3000,
                quantity=1,
                stop_loss=2900,
                take_profit=3300,
            )
        )
        assert mgr.validate_signal(signal, balance=10_000) is False

    def test_daily_loss_limit(self):
        mgr = self._make_manager(max_daily_loss_pct=3.0)
        # Simulate a daily loss of 4% of 10k = 400
        mgr._daily_pnl = -400
        mgr._daily_reset_date = "9999-12-31"  # prevent reset
        signal = _make_signal(entry=100, sl=97, tp=110.5)
        assert mgr.validate_signal(signal, balance=10_000) is False


class TestPositionSizing:
    def test_zero_risk_returns_zero(self):
        cfg = RiskConfig()
        mgr = RiskManager(cfg, exchange=None)  # type: ignore[arg-type]
        signal = _make_signal(entry=100, sl=100)  # zero risk
        assert mgr.calculate_position_size(signal, balance=10_000) == 0.0

    def test_short_position_sizing(self):
        cfg = RiskConfig(risk_per_trade_pct=2.0)
        mgr = RiskManager(cfg, exchange=None)  # type: ignore[arg-type]
        signal = _make_signal(entry=100, sl=105, tp=85, direction=TradeDirection.SHORT)
        qty = mgr.calculate_position_size(signal, balance=5_000)
        # risk = 5000*2% = 100, price_risk = 5, qty = 20
        assert abs(qty - 20.0) < 0.01
