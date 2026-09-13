from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .base import ForecastRequest


@dataclass(frozen=True)
class PriceObservation:
    timestamp: datetime
    close: float


def build_forecast_request(
    symbol: str,
    observations: list[PriceObservation],
    horizon: int,
) -> ForecastRequest:
    """Build a validated causal forecast request from chronological close prices.

    The function does not resample, interpolate, fill missing observations, or
    use any future values. Those decisions belong to the market-data layer.
    """
    if not observations:
        raise ValueError("at least one price observation is required")

    ordered = sorted(observations, key=lambda item: item.timestamp)
    timestamps = [item.timestamp for item in ordered]
    values = [float(item.close) for item in ordered]

    if any(value != value for value in values):
        raise ValueError("close prices must not contain NaN")
    if any(timestamp is None for timestamp in timestamps):
        raise ValueError("timestamps must not contain None")
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("timestamps must be unique")

    return ForecastRequest(
        symbol=symbol,
        values=values,
        timestamps=timestamps,
        horizon=horizon,
    )
