"""Main orchestrator — ties all four blocks together in a continuous loop.

Flow:
  1. Screener builds daily watchlist
  2. For each coin in watchlist: fetch candles → find setups
  3. For each setup: check for breakout with volume confirmation
  4. If breakout confirmed: validate risk → execute trade
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import yaml

from src.exchange.client import ExchangeClient
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


def load_config(path: str = "config/settings.yaml") -> BotConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return BotConfig(**raw)


async def run_bot(config: BotConfig) -> None:
    """Main bot loop."""
    # Initialize components
    exchange = ExchangeClient(config.exchange)
    await exchange.connect()

    screener = AssetScreener(exchange, config.screener)
    setup_finder = SetupFinder(config.analysis)
    breakout_detector = BreakoutDetector(config.entry)
    risk_manager = RiskManager(config.risk, exchange)

    watchlist = None
    watchlist_age = 0.0

    try:
        while True:
            # 1. Refresh watchlist periodically
            if watchlist is None or watchlist_age >= WATCHLIST_REFRESH:
                logger.info("Building watchlist...")
                watchlist = await screener.build_watchlist()
                watchlist_age = 0.0
                logger.info("Watchlist: %d coins", len(watchlist.coins))

            # 2. Get current balance
            try:
                balance_data = await exchange.fetch_balance()
                balance = float(balance_data.get("total", {}).get("USDT", 0))
            except Exception:
                logger.warning("Failed to fetch balance, using 0")
                balance = 0.0

            # 3. For each coin — fetch candles, find setups, check breakouts
            for coin in watchlist.coins:
                try:
                    df = await exchange.fetch_ohlcv(
                        coin.symbol,
                        timeframe=config.analysis.timeframe,
                        limit=config.analysis.candle_lookback,
                    )

                    setups = setup_finder.find_setups(df, coin)

                    for setup in setups:
                        signal = breakout_detector.check_breakout(df, setup)
                        if signal is None:
                            continue

                        # 4. Validate and execute
                        if not risk_manager.validate_signal(signal, balance):
                            continue

                        logger.info(
                            "SIGNAL: %s %s entry=%.4f SL=%.4f TP=%.4f R/R=%.2f",
                            signal.setup.symbol,
                            signal.setup.direction.value,
                            signal.entry_price,
                            signal.stop_loss,
                            signal.take_profit,
                            signal.risk_reward,
                        )

                        position = await risk_manager.execute_trade(signal, balance)
                        if position:
                            logger.info(
                                "OPENED: %s %s qty=%.6f",
                                position.symbol,
                                position.direction.value,
                                position.quantity,
                            )

                except Exception:
                    logger.exception("Error processing %s", coin.symbol)

            watchlist_age += SCAN_INTERVAL
            await asyncio.sleep(SCAN_INTERVAL)

    finally:
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
    logger.info("Starting crypto screener bot (exchange=%s, testnet=%s)", config.exchange.name, config.exchange.testnet)
    asyncio.run(run_bot(config))


if __name__ == "__main__":
    main()
