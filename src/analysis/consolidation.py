"""Consolidation zone detection.

Finds horizontal ranges where price traded in a tight band,
indicating accumulation/distribution before a breakout.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models.config import ConsolidationConfig
from src.models.domain import ConsolidationZone

logger = logging.getLogger(__name__)


def detect_consolidation_zones(
    df: pd.DataFrame,
    config: ConsolidationConfig,
) -> list[ConsolidationZone]:
    """Scan for consolidation zones in OHLCV data.

    A consolidation zone is a contiguous run of candles where
    the high-low range stays within ``max_range_pct`` and the run
    is at least ``min_candles`` long.
    """
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    n = len(df)

    zones: list[ConsolidationZone] = []
    i = 0

    while i < n - config.min_candles:
        # Start a candidate zone
        zone_high = highs[i]
        zone_low = lows[i]
        j = i + 1

        while j < n:
            candidate_high = max(zone_high, highs[j])
            candidate_low = min(zone_low, lows[j])

            if candidate_low == 0:
                break

            range_pct = (candidate_high - candidate_low) / candidate_low * 100
            if range_pct > config.max_range_pct:
                break

            zone_high = candidate_high
            zone_low = candidate_low
            j += 1

        length = j - i
        if length >= config.min_candles:
            zones.append(ConsolidationZone(
                high=zone_high,
                low=zone_low,
                start_index=i,
                end_index=j - 1,
            ))
            i = j  # skip past this zone
        else:
            i += 1

    # Merge overlapping zones
    zones = _merge_overlapping(zones)
    return zones


def find_nearest_consolidation(
    zones: list[ConsolidationZone],
    candle_index: int,
    max_distance: int = 30,
) -> ConsolidationZone | None:
    """Find the consolidation zone closest to a given candle index."""
    best: ConsolidationZone | None = None
    best_dist = max_distance + 1

    for z in zones:
        if z.start_index <= candle_index <= z.end_index:
            return z  # inside the zone
        dist = min(abs(candle_index - z.end_index), abs(candle_index - z.start_index))
        if dist < best_dist:
            best_dist = dist
            best = z
    return best


def _merge_overlapping(zones: list[ConsolidationZone]) -> list[ConsolidationZone]:
    """Merge zones that overlap or are adjacent."""
    if not zones:
        return []

    zones.sort(key=lambda z: z.start_index)
    merged = [zones[0]]

    for z in zones[1:]:
        prev = merged[-1]
        if z.start_index <= prev.end_index + 1:
            merged[-1] = ConsolidationZone(
                high=max(prev.high, z.high),
                low=min(prev.low, z.low),
                start_index=prev.start_index,
                end_index=max(prev.end_index, z.end_index),
            )
        else:
            merged.append(z)
    return merged
