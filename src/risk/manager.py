"""Block 4 — Risk Management.

Handles position sizing, R/R validation, daily drawdown tracking,
and order placement with hard SL/TP.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.exchange.client import ExchangeClient
from src.models.config import RiskConfig
from src.models.domain import Position, TradeDirection, TradeSignal

logger = logging.getLogger(__name__)


class RiskManager:
    """Enforces risk rules and calculates position sizes."""

    def __init__(self, config: RiskConfig, exchange: ExchangeClient) -> None:
        self._cfg = config
        self._exchange = exchange
        self._open_positions: list[Position] = []
        self._daily_pnl: float = 0.0
        self._daily_reset_date: str = ""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_signal(self, signal: TradeSignal, balance: float) -> bool:
        """Check if a trade signal passes all risk rules.

        Returns True if the trade is allowed.
        """
        # 1. Check R/R ratio
        rr = signal.risk_reward
        if rr < self._cfg.min_risk_reward:
            logger.info(
                "%s: R/R %.2f below minimum %.2f — skipping",
                signal.setup.symbol, rr, self._cfg.min_risk_reward,
            )
            return False

        # 2. Check if already have a position on this symbol
        for pos in self._open_positions:
            if pos.symbol == signal.setup.symbol:
                logger.info("%s: already have an open position — skipping", signal.setup.symbol)
                return False

        # 3. Check max open positions
        if len(self._open_positions) >= self._cfg.max_open_positions:
            logger.info("Max open positions (%d) reached — skipping", self._cfg.max_open_positions)
            return False

        # 4. Check daily drawdown
        self._maybe_reset_daily_pnl()
        if balance > 0:
            daily_loss_pct = abs(min(self._daily_pnl, 0)) / balance * 100
            if daily_loss_pct >= self._cfg.max_daily_loss_pct:
                logger.warning("Daily loss limit (%.1f%%) reached — halting", self._cfg.max_daily_loss_pct)
                return False

        # 5. Ensure SL and entry make sense
        if signal.stop_loss <= 0 or signal.entry_price <= 0:
            return False

        risk = abs(signal.entry_price - signal.stop_loss)
        if risk == 0:
            return False

        return True

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(self, signal: TradeSignal, balance: float) -> float:
        """Calculate the position quantity based on 1% risk rule.

        risk_amount = balance * risk_per_trade_pct / 100
        quantity = risk_amount / (entry - stop_loss)
        """
        risk_amount = balance * self._cfg.risk_per_trade_pct / 100
        price_risk = abs(signal.entry_price - signal.stop_loss)

        if price_risk == 0:
            return 0.0

        quantity = risk_amount / price_risk
        return quantity

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    async def execute_trade(self, signal: TradeSignal, balance: float) -> Position | None:
        """Open a position with hard SL and TP orders.

        Steps:
          1. Calculate position size
          2. Place entry order
          3. Place stop-loss order
          4. Place take-profit order
        """
        quantity = self.calculate_position_size(signal, balance)
        if quantity <= 0:
            logger.warning("%s: calculated zero quantity", signal.setup.symbol)
            return None

        side = "buy" if signal.setup.direction == TradeDirection.LONG else "sell"
        close_side = "sell" if side == "buy" else "buy"

        try:
            # Entry order — always pass entry_price so paper engine
            # knows the fill price even for market orders
            order_type = "market" if self._cfg.use_market_orders else "limit"

            entry_order = await self._exchange.create_order(
                symbol=signal.setup.symbol,
                side=side,
                amount=quantity,
                price=signal.entry_price,
                order_type=order_type,
            )
            logger.info(
                "%s: entry order placed — %s %.6f @ %s",
                signal.setup.symbol, side, quantity, price or "market",
            )

            # Stop-loss order
            await self._exchange.create_stop_loss(
                symbol=signal.setup.symbol,
                side=close_side,
                amount=quantity,
                stop_price=signal.stop_loss,
            )
            logger.info(
                "%s: SL placed at %.6f", signal.setup.symbol, signal.stop_loss
            )

            # Take-profit order
            await self._exchange.create_take_profit(
                symbol=signal.setup.symbol,
                side=close_side,
                amount=quantity,
                tp_price=signal.take_profit,
            )
            logger.info(
                "%s: TP placed at %.6f", signal.setup.symbol, signal.take_profit
            )

            # Track position
            position = Position(
                symbol=signal.setup.symbol,
                direction=signal.setup.direction,
                entry_price=signal.entry_price,
                quantity=quantity,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reentry_used=signal.is_reentry,
            )
            self._open_positions.append(position)
            return position

        except Exception:
            logger.exception("%s: failed to execute trade", signal.setup.symbol)
            return None

    # ------------------------------------------------------------------
    # Position tracking
    # ------------------------------------------------------------------

    def close_position(self, symbol: str, exit_price: float, was_stopped: bool = False) -> None:
        """Mark a position as closed and update daily P&L."""
        for pos in self._open_positions:
            if pos.symbol == symbol and pos.closed_at is None:
                pos.closed_at = datetime.now(timezone.utc)
                pos.exit_price = exit_price
                pos.was_stopped = was_stopped

                if pos.direction == TradeDirection.LONG:
                    pos.pnl = (exit_price - pos.entry_price) * pos.quantity
                else:
                    pos.pnl = (pos.entry_price - exit_price) * pos.quantity

                self._daily_pnl += pos.pnl
                self._open_positions.remove(pos)
                logger.info(
                    "%s: position closed — PnL=%.2f stopped=%s",
                    symbol, pos.pnl, was_stopped,
                )
                return

    @property
    def open_positions(self) -> list[Position]:
        return list(self._open_positions)

    def _maybe_reset_daily_pnl(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_reset_date < today:
            self._daily_pnl = 0.0
            self._daily_reset_date = today
