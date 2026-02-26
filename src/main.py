"""Main orchestrator — ties all four blocks together in a continuous loop.

Supports two modes:
  - Paper trading (default): real market data from Bybit, virtual balance
  - Live trading: real orders on the exchange

Flow:
  1. Screener builds daily watchlist
  2. For each coin in watchlist: fetch candles -> find setups
  3. For each setup: check for breakout with volume confirmation
  4. If breakout confirmed: validate risk -> execute trade
  5. (Paper mode) Check SL/TP hits for open positions on each candle
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

import yaml

from src.exchange.client import ExchangeClient
from src.exchange.paper_trading import PaperTradingEngine
from src.models.config import BotConfig
from src.screener.screener import AssetScreener
from src.analysis.setup_finder import SetupFinder
from src.entry.breakout import BreakoutDetector
from src.risk.manager import RiskManager

logger = logging.getLogger("bot")

# How often to re-scan for breakouts (seconds)
SCAN_INTERVAL = 60
# How often to rebuild the watchlist (seconds)
WATCHLIST_REFRESH = 3600 * 4  # every 4 hours
# How often to print paper trading stats (seconds)
STATS_INTERVAL = 3600  # every hour


def load_config(path: str = "config/settings.yaml") -> BotConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return BotConfig(**raw)


async def run_bot(config: BotConfig) -> None:
    """Main bot loop."""
    # Initialize exchange client (real market data)
    exchange = ExchangeClient(config.exchange)
    await exchange.connect()

    # Decide trading backend: paper or live
    paper_mode = config.paper_trading.enabled
    if paper_mode:
        # Ensure data directory exists for state persistence
        state_dir = Path(config.paper_trading.state_file).parent
        state_dir.mkdir(parents=True, exist_ok=True)

        paper_engine = PaperTradingEngine(config.paper_trading)
        # RiskManager uses paper engine for order execution
        trade_backend = paper_engine
        logger.info(
            "MODE: Paper Trading | Balance: $%.2f | Data: real %s",
            paper_engine.balance, config.exchange.name,
        )
    else:
        paper_engine = None
        trade_backend = exchange
        logger.info("MODE: Live Trading on %s", config.exchange.name)

    screener = AssetScreener(exchange, config.screener)
    setup_finder = SetupFinder(config.analysis)
    breakout_detector = BreakoutDetector(config.entry)
    risk_manager = RiskManager(config.risk, trade_backend)

    watchlist = None
    watchlist_age = 0.0
    stats_age = 0.0

    # Graceful shutdown
    shutdown_event = asyncio.Event()

    def _handle_signal(*_: object) -> None:
        logger.info("Shutdown signal received...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    try:
        while not shutdown_event.is_set():
            # 1. Refresh watchlist periodically
            if watchlist is None or watchlist_age >= WATCHLIST_REFRESH:
                logger.info("Building watchlist...")
                watchlist = await screener.build_watchlist()
                watchlist_age = 0.0
                logger.info("Watchlist: %d coins", len(watchlist.coins))

            # 2. Get current balance
            if paper_mode:
                balance = paper_engine.balance
            else:
                try:
                    balance_data = await exchange.fetch_balance()
                    balance = float(balance_data.get("total", {}).get("USDT", 0))
                except Exception:
                    logger.warning("Failed to fetch balance, using 0")
                    balance = 0.0

            # 3. For each coin — fetch candles, find setups, check breakouts
            coins_processed = 0
            setups_total = 0
            signals_total = 0
            trades_opened = 0

            for coin in watchlist.coins:
                if shutdown_event.is_set():
                    break
                try:
                    df = await exchange.fetch_ohlcv(
                        coin.symbol,
                        timeframe=config.analysis.timeframe,
                        limit=config.analysis.candle_lookback,
                    )
                    coins_processed += 1

                    # Paper mode: check SL/TP for open positions using latest candle
                    if paper_mode and len(df) > 0:
                        last = df.iloc[-1]
                        paper_engine.update_prices(
                            coin.symbol,
                            high=float(last["high"]),
                            low=float(last["low"]),
                            close=float(last["close"]),
                        )

                    setups = setup_finder.find_setups(df, coin)
                    setups_total += len(setups)

                    if setups:
                        logger.debug(
                            "%s: %d setups found", coin.symbol, len(setups),
                        )

                    # Collect all confirmed breakout signals for this coin
                    signals = []
                    for setup in setups:
                        sig = breakout_detector.check_breakout(df, setup)
                        if sig is not None:
                            signals.append(sig)

                    if not signals:
                        continue

                    signals_total += len(signals)

                    # Pick the best signal by composite score
                    best = max(signals, key=lambda s: s.signal_score)

                    if len(signals) > 1:
                        logger.info(
                            "%s: %d breakout signals, best score=%.3f (age=%d, vol=%.1fx)",
                            coin.symbol, len(signals), best.signal_score,
                            best.breakout_age, best.volume_spike,
                        )

                    # 4. Validate and execute only the best signal
                    if not risk_manager.validate_signal(best, balance):
                        continue

                    logger.info(
                        "SIGNAL: %s %s entry=%.4f SL=%.4f TP=%.4f R/R=%.2f score=%.3f",
                        best.setup.symbol,
                        best.setup.direction.value,
                        best.entry_price,
                        best.stop_loss,
                        best.take_profit,
                        best.risk_reward,
                        best.signal_score,
                    )

                    position = await risk_manager.execute_trade(best, balance)
                    if position:
                        trades_opened += 1
                        logger.info(
                            "OPENED: %s %s qty=%.6f",
                            position.symbol,
                            position.direction.value,
                            position.quantity,
                        )

                except Exception:
                    logger.exception("Error processing %s", coin.symbol)

            # Cycle summary
            open_pos = len(risk_manager.open_positions)
            logger.info(
                "Scan done: %d/%d coins | %d setups | %d breakouts | %d new trades | %d open positions",
                coins_processed, len(watchlist.coins),
                setups_total, signals_total, trades_opened, open_pos,
            )

            # Paper mode: periodic stats
            if paper_mode:
                stats_age += SCAN_INTERVAL
                if stats_age >= STATS_INTERVAL:
                    paper_engine.print_stats()
                    stats_age = 0.0

            watchlist_age += SCAN_INTERVAL

            # Wait with interruptible sleep
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=SCAN_INTERVAL)
            except asyncio.TimeoutError:
                pass

    finally:
        if paper_mode:
            logger.info("Final paper trading results:")
            paper_engine.print_stats()
        await exchange.close()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("bot.log"),
        ],
    )


def main() -> None:
    setup_logging()
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/settings.yaml"
    config = load_config(config_path)
    mode = "PAPER" if config.paper_trading.enabled else "LIVE"
    logger.info(
        "Starting crypto screener bot [%s] (exchange=%s)",
        mode, config.exchange.name,
    )
    asyncio.run(run_bot(config))


if __name__ == "__main__":
    main()
