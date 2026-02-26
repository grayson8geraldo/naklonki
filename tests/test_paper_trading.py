"""Tests for the paper trading engine."""

import json
import os
import tempfile

import pytest

from src.exchange.paper_trading import PaperTradingEngine
from src.models.config import PaperTradingConfig


class TestPaperTradingBalance:
    def _engine(self, balance: float = 10_000) -> PaperTradingEngine:
        cfg = PaperTradingConfig(enabled=True, initial_balance=balance, state_file="")
        return PaperTradingEngine(cfg)

    @pytest.mark.asyncio
    async def test_initial_balance(self):
        engine = self._engine(10_000)
        bal = await engine.fetch_balance()
        assert bal["total"]["USDT"] == 10_000

    @pytest.mark.asyncio
    async def test_market_order_creates_position(self):
        engine = self._engine()
        result = await engine.create_order("BTC/USDT", "buy", 0.1, price=50000.0, order_type="market")
        assert result["status"] == "closed"

        positions = await engine.fetch_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTC/USDT"
        assert positions[0]["contracts"] == 0.1

    @pytest.mark.asyncio
    async def test_sl_tp_orders_stored(self):
        engine = self._engine()
        await engine.create_order("BTC/USDT", "buy", 0.1, price=50000.0, order_type="market")
        await engine.create_stop_loss("BTC/USDT", "sell", 0.1, stop_price=48000.0)
        await engine.create_take_profit("BTC/USDT", "sell", 0.1, tp_price=55000.0)

        orders = await engine.fetch_open_orders("BTC/USDT")
        assert len(orders) == 2


class TestPaperTradingPriceUpdates:
    def _engine(self, balance: float = 10_000) -> PaperTradingEngine:
        cfg = PaperTradingConfig(enabled=True, initial_balance=balance, state_file="")
        return PaperTradingEngine(cfg)

    @pytest.mark.asyncio
    async def test_stop_loss_triggered_long(self):
        engine = self._engine(10_000)
        await engine.create_order("BTC/USDT", "buy", 0.1, price=50000.0, order_type="market")
        await engine.create_stop_loss("BTC/USDT", "sell", 0.1, stop_price=48000.0)
        await engine.create_take_profit("BTC/USDT", "sell", 0.1, tp_price=55000.0)

        # Price drops to SL
        engine.update_prices("BTC/USDT", high=49000, low=47500, close=47800)

        positions = await engine.fetch_positions()
        assert len(positions) == 0  # position closed

        # PnL: (48000 - 50000) * 0.1 = -200
        assert engine.balance == pytest.approx(10_000 - 200, abs=1)

    @pytest.mark.asyncio
    async def test_take_profit_triggered_long(self):
        engine = self._engine(10_000)
        await engine.create_order("BTC/USDT", "buy", 0.1, price=50000.0, order_type="market")
        await engine.create_stop_loss("BTC/USDT", "sell", 0.1, stop_price=48000.0)
        await engine.create_take_profit("BTC/USDT", "sell", 0.1, tp_price=55000.0)

        # Price rises to TP
        engine.update_prices("BTC/USDT", high=55500, low=53000, close=55200)

        positions = await engine.fetch_positions()
        assert len(positions) == 0  # position closed

        # PnL: (55000 - 50000) * 0.1 = 500
        assert engine.balance == pytest.approx(10_000 + 500, abs=1)

    @pytest.mark.asyncio
    async def test_stop_loss_triggered_short(self):
        engine = self._engine(10_000)
        await engine.create_order("ETH/USDT", "sell", 1.0, price=3000.0, order_type="market")
        await engine.create_stop_loss("ETH/USDT", "buy", 1.0, stop_price=3100.0)
        await engine.create_take_profit("ETH/USDT", "buy", 1.0, tp_price=2700.0)

        # Price rises to SL
        engine.update_prices("ETH/USDT", high=3150, low=2980, close=3120)

        positions = await engine.fetch_positions()
        assert len(positions) == 0

        # PnL: (3000 - 3100) * 1.0 = -100
        assert engine.balance == pytest.approx(10_000 - 100, abs=1)

    @pytest.mark.asyncio
    async def test_take_profit_triggered_short(self):
        engine = self._engine(10_000)
        await engine.create_order("ETH/USDT", "sell", 1.0, price=3000.0, order_type="market")
        await engine.create_stop_loss("ETH/USDT", "buy", 1.0, stop_price=3100.0)
        await engine.create_take_profit("ETH/USDT", "buy", 1.0, tp_price=2700.0)

        # Price drops to TP
        engine.update_prices("ETH/USDT", high=2900, low=2650, close=2680)

        positions = await engine.fetch_positions()
        assert len(positions) == 0

        # PnL: (3000 - 2700) * 1.0 = 300
        assert engine.balance == pytest.approx(10_000 + 300, abs=1)

    @pytest.mark.asyncio
    async def test_no_trigger_in_range(self):
        """Price stays between SL and TP — position stays open."""
        engine = self._engine(10_000)
        await engine.create_order("BTC/USDT", "buy", 0.1, price=50000.0, order_type="market")
        await engine.create_stop_loss("BTC/USDT", "sell", 0.1, stop_price=48000.0)
        await engine.create_take_profit("BTC/USDT", "sell", 0.1, tp_price=55000.0)

        engine.update_prices("BTC/USDT", high=52000, low=49000, close=51000)

        positions = await engine.fetch_positions()
        assert len(positions) == 1
        assert engine.balance == 10_000  # unchanged

    @pytest.mark.asyncio
    async def test_unrealized_pnl(self):
        engine = self._engine(10_000)
        await engine.create_order("BTC/USDT", "buy", 0.1, price=50000.0, order_type="market")
        await engine.create_stop_loss("BTC/USDT", "sell", 0.1, stop_price=48000.0)
        await engine.create_take_profit("BTC/USDT", "sell", 0.1, tp_price=55000.0)

        engine.update_prices("BTC/USDT", high=52000, low=49500, close=51000)

        # Equity = balance + unrealized pnl
        # Unrealized: (51000 - 50000) * 0.1 = 100
        assert engine.equity == pytest.approx(10_100, abs=1)


class TestPaperTradingStats:
    @pytest.mark.asyncio
    async def test_stats_after_trades(self):
        cfg = PaperTradingConfig(enabled=True, initial_balance=10_000, state_file="")
        engine = PaperTradingEngine(cfg)

        # Win trade
        await engine.create_order("BTC/USDT", "buy", 0.1, price=50000.0, order_type="market")
        await engine.create_stop_loss("BTC/USDT", "sell", 0.1, stop_price=48000.0)
        await engine.create_take_profit("BTC/USDT", "sell", 0.1, tp_price=55000.0)
        engine.update_prices("BTC/USDT", high=56000, low=53000, close=55500)

        # Loss trade
        await engine.create_order("ETH/USDT", "buy", 1.0, price=3000.0, order_type="market")
        await engine.create_stop_loss("ETH/USDT", "sell", 1.0, stop_price=2900.0)
        await engine.create_take_profit("ETH/USDT", "sell", 1.0, tp_price=3300.0)
        engine.update_prices("ETH/USDT", high=2950, low=2850, close=2870)

        stats = engine.get_stats()
        assert stats["total_trades"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["win_rate"] == 50.0


class TestPaperTradingPersistence:
    @pytest.mark.asyncio
    async def test_save_and_load_state(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            # Create engine, make trades
            cfg = PaperTradingConfig(enabled=True, initial_balance=10_000, state_file=state_file)
            engine = PaperTradingEngine(cfg)

            await engine.create_order("BTC/USDT", "buy", 0.1, price=50000.0, order_type="market")
            await engine.create_stop_loss("BTC/USDT", "sell", 0.1, stop_price=48000.0)
            await engine.create_take_profit("BTC/USDT", "sell", 0.1, tp_price=55000.0)

            # Close with TP
            engine.update_prices("BTC/USDT", high=56000, low=53000, close=55500)

            saved_balance = engine.balance

            # Create new engine from saved state
            engine2 = PaperTradingEngine(cfg)
            assert engine2.balance == pytest.approx(saved_balance, abs=0.01)
            assert len(engine2._trade_history) == 1
        finally:
            os.unlink(state_file)
