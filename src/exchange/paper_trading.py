"""Paper Trading Engine.

Simulates order execution with a virtual balance while using real market data.
Tracks positions, checks SL/TP hits on each candle, maintains trade history
and performance statistics.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.models.config import PaperTradingConfig
from src.models.domain import Position, TradeDirection, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class PaperOrder:
    """A simulated order."""
    id: str
    symbol: str
    side: str                  # "buy" / "sell"
    order_type: str            # "market", "limit", "stop_market", "take_profit_market"
    amount: float
    price: float | None = None
    stop_price: float | None = None
    filled: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PaperPosition:
    """A virtual open position."""
    symbol: str
    side: str                  # "buy" (long) / "sell" (short)
    entry_price: float
    amount: float
    stop_loss: float
    take_profit: float
    sl_order_id: str = ""
    tp_order_id: str = ""
    unrealized_pnl: float = 0.0


@dataclass
class ClosedTrade:
    """Record of a completed trade for history."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    amount: float
    pnl: float
    pnl_pct: float
    reason: str                # "stop_loss", "take_profit", "manual"
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PaperTradingEngine:
    """Virtual exchange engine — real data, fake money.

    Replaces the real exchange order/balance methods while keeping
    market data fetches going to the live Bybit/Binance API.
    """

    def __init__(self, config: PaperTradingConfig) -> None:
        self._cfg = config
        self._balance: float = config.initial_balance
        self._positions: dict[str, PaperPosition] = {}   # symbol -> position
        self._orders: dict[str, PaperOrder] = {}          # order_id -> order
        self._trade_history: list[ClosedTrade] = []
        self._peak_balance: float = config.initial_balance

        # Load saved state if exists
        if config.state_file:
            self._load_state(config.state_file)

        logger.info(
            "Paper trading engine started — balance: $%.2f",
            self._balance,
        )

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def equity(self) -> float:
        """Balance + unrealized PnL from open positions."""
        return self._balance + sum(p.unrealized_pnl for p in self._positions.values())

    async def fetch_balance(self) -> dict:
        """Mimic ccxt balance response."""
        return {
            "total": {"USDT": self._balance},
            "free": {"USDT": self._balance},
            "used": {"USDT": 0.0},
        }

    # ------------------------------------------------------------------
    # Order Simulation
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
        """Simulate placing an order.

        Market orders are filled immediately at the given price.
        Stop/TP orders are stored and checked on each price update.
        """
        params = params or {}
        order_id = str(uuid.uuid4())[:8]
        stop_price = params.get("stopPrice")

        order = PaperOrder(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
            stop_price=stop_price,
        )

        if order_type in ("market", "limit"):
            # Immediate fill for market; limit orders also filled immediately
            # in paper mode (assumes price is reachable)
            fill_price = price if price else 0.0
            self._fill_entry(order, fill_price)
            order.filled = True
            logger.info(
                "[PAPER] %s %s %.6f %s @ %.4f",
                order_type.upper(), side, amount, symbol, fill_price,
            )
        else:
            # Pending stop/TP order
            self._orders[order_id] = order
            logger.info(
                "[PAPER] %s order placed: %s %s %.6f trigger=%.4f",
                order_type, side, symbol, amount, stop_price or 0,
            )

        return {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "price": price,
            "status": "closed" if order.filled else "open",
        }

    async def create_stop_loss(
        self, symbol: str, side: str, amount: float, stop_price: float
    ) -> dict:
        result = await self.create_order(
            symbol, side, amount,
            order_type="stop_market",
            params={"stopPrice": stop_price},
        )
        # Link SL order to position and set stop price
        if symbol in self._positions:
            self._positions[symbol].sl_order_id = result["id"]
            self._positions[symbol].stop_loss = stop_price
        return result

    async def create_take_profit(
        self, symbol: str, side: str, amount: float, tp_price: float
    ) -> dict:
        result = await self.create_order(
            symbol, side, amount,
            order_type="take_profit_market",
            params={"stopPrice": tp_price},
        )
        # Link TP order to position and set TP price
        if symbol in self._positions:
            self._positions[symbol].tp_order_id = result["id"]
            self._positions[symbol].take_profit = tp_price
        return result

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        order = self._orders.pop(order_id, None)
        if order:
            logger.info("[PAPER] Cancelled order %s", order_id)
        return {"id": order_id, "status": "cancelled"}

    async def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        orders = self._orders.values()
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return [{"id": o.id, "symbol": o.symbol, "type": o.order_type} for o in orders]

    async def fetch_positions(self) -> list[dict]:
        return [
            {
                "symbol": p.symbol,
                "side": p.side,
                "contracts": p.amount,
                "entryPrice": p.entry_price,
                "unrealizedPnl": p.unrealized_pnl,
            }
            for p in self._positions.values()
        ]

    # ------------------------------------------------------------------
    # Price Update — check SL / TP on each candle
    # ------------------------------------------------------------------

    def update_prices(self, symbol: str, high: float, low: float, close: float) -> None:
        """Called every candle to check if SL or TP was hit.

        Processes the candle's high/low to see if any pending
        stop or take-profit order should trigger.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return

        # Update unrealized PnL
        if pos.side == "buy":
            pos.unrealized_pnl = (close - pos.entry_price) * pos.amount
        else:
            pos.unrealized_pnl = (pos.entry_price - close) * pos.amount

        # Check SL hit
        if pos.side == "buy" and low <= pos.stop_loss:
            self._close_position(symbol, pos.stop_loss, "stop_loss")
            return
        if pos.side == "sell" and high >= pos.stop_loss:
            self._close_position(symbol, pos.stop_loss, "stop_loss")
            return

        # Check TP hit
        if pos.side == "buy" and high >= pos.take_profit:
            self._close_position(symbol, pos.take_profit, "take_profit")
            return
        if pos.side == "sell" and low <= pos.take_profit:
            self._close_position(symbol, pos.take_profit, "take_profit")
            return

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fill_entry(self, order: PaperOrder, fill_price: float) -> None:
        """Process an entry fill — create or add to position."""
        margin_required = fill_price * order.amount
        # No margin check for shorts in simplified model;
        # we just track P&L vs balance.

        pos = PaperPosition(
            symbol=order.symbol,
            side=order.side,
            entry_price=fill_price,
            amount=order.amount,
            stop_loss=0.0,
            take_profit=0.0,
        )
        self._positions[order.symbol] = pos

    def _close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        """Close a position and record the trade."""
        pos = self._positions.pop(symbol, None)
        if pos is None:
            return

        if pos.side == "buy":
            pnl = (exit_price - pos.entry_price) * pos.amount
        else:
            pnl = (pos.entry_price - exit_price) * pos.amount

        pnl_pct = pnl / (pos.entry_price * pos.amount) * 100 if pos.entry_price else 0

        self._balance += pnl
        self._peak_balance = max(self._peak_balance, self._balance)

        # Remove linked orders
        for oid in (pos.sl_order_id, pos.tp_order_id):
            self._orders.pop(oid, None)

        trade = ClosedTrade(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            amount=pos.amount,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
        )
        self._trade_history.append(trade)

        emoji = "+" if pnl >= 0 else ""
        logger.info(
            "[PAPER] CLOSED %s %s @ %.4f -> %.4f | PnL: %s%.2f (%.2f%%) | reason: %s | balance: $%.2f",
            pos.side.upper(), symbol, pos.entry_price, exit_price,
            emoji, pnl, pnl_pct, reason, self._balance,
        )

        # Auto-save state
        if self._cfg.state_file:
            self._save_state(self._cfg.state_file)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return paper trading performance statistics."""
        if not self._trade_history:
            return {
                "total_trades": 0,
                "balance": self._balance,
                "initial_balance": self._cfg.initial_balance,
                "open_positions": len(self._positions),
            }

        wins = [t for t in self._trade_history if t.pnl > 0]
        losses = [t for t in self._trade_history if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self._trade_history)
        max_drawdown = self._cfg.initial_balance - min(
            self._cfg.initial_balance,
            self._balance,
        )

        return {
            "total_trades": len(self._trade_history),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self._trade_history) * 100,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl / self._cfg.initial_balance * 100,
            "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.pnl for t in losses) / len(losses) if losses else 0,
            "best_trade": max(t.pnl for t in self._trade_history),
            "worst_trade": min(t.pnl for t in self._trade_history),
            "max_drawdown": max_drawdown,
            "balance": self._balance,
            "initial_balance": self._cfg.initial_balance,
            "peak_balance": self._peak_balance,
            "open_positions": len(self._positions),
        }

    def print_stats(self) -> None:
        """Log performance summary."""
        stats = self.get_stats()
        logger.info("=" * 60)
        logger.info("PAPER TRADING STATS")
        logger.info("=" * 60)
        logger.info("Balance:       $%.2f (started $%.2f)", stats["balance"], stats["initial_balance"])
        if stats["total_trades"] > 0:
            logger.info("Total PnL:     $%.2f (%.2f%%)", stats["total_pnl"], stats["total_pnl_pct"])
            logger.info("Trades:        %d (W:%d / L:%d)", stats["total_trades"], stats["wins"], stats["losses"])
            logger.info("Win rate:      %.1f%%", stats["win_rate"])
            logger.info("Avg win:       $%.2f", stats["avg_win"])
            logger.info("Avg loss:      $%.2f", stats["avg_loss"])
            logger.info("Best trade:    $%.2f", stats["best_trade"])
            logger.info("Worst trade:   $%.2f", stats["worst_trade"])
            logger.info("Max drawdown:  $%.2f", stats["max_drawdown"])
            logger.info("Peak balance:  $%.2f", stats["peak_balance"])
        logger.info("Open positions: %d", stats["open_positions"])
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self, path: str) -> None:
        """Save balance, positions, and trade history to JSON."""
        state = {
            "balance": self._balance,
            "peak_balance": self._peak_balance,
            "positions": {
                sym: {
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "amount": p.amount,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                }
                for sym, p in self._positions.items()
            },
            "trade_history": [
                {
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "amount": t.amount,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "reason": t.reason,
                }
                for t in self._trade_history
            ],
        }
        Path(path).write_text(json.dumps(state, indent=2))

    def _load_state(self, path: str) -> None:
        """Load saved state if file exists."""
        p = Path(path)
        if not p.exists():
            return

        try:
            state = json.loads(p.read_text())
            self._balance = state.get("balance", self._cfg.initial_balance)
            self._peak_balance = state.get("peak_balance", self._balance)

            for sym, pos_data in state.get("positions", {}).items():
                self._positions[sym] = PaperPosition(
                    symbol=sym,
                    side=pos_data["side"],
                    entry_price=pos_data["entry_price"],
                    amount=pos_data["amount"],
                    stop_loss=pos_data["stop_loss"],
                    take_profit=pos_data["take_profit"],
                )

            for td in state.get("trade_history", []):
                self._trade_history.append(ClosedTrade(
                    symbol=td["symbol"],
                    side=td["side"],
                    entry_price=td["entry_price"],
                    exit_price=td["exit_price"],
                    amount=td["amount"],
                    pnl=td["pnl"],
                    pnl_pct=td["pnl_pct"],
                    reason=td["reason"],
                ))

            logger.info(
                "[PAPER] Loaded state: balance=$%.2f, %d open positions, %d historical trades",
                self._balance, len(self._positions), len(self._trade_history),
            )
        except Exception:
            logger.warning("[PAPER] Failed to load state from %s, starting fresh", path)
