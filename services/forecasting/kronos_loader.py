from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.db.models import PriceBar, Stock


@dataclass(frozen=True)
class OHLCVObservation:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float


def load_ohlcv_observations(
    db: Session,
    symbol: str,
    *,
    interval: str = "1d",
    source: str = "yfinance",
    limit: int = 512,
) -> list[OHLCVObservation]:
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must not be empty")
    if limit < 2:
        raise ValueError("limit must be >= 2")

    stock_id = db.scalar(select(Stock.id).where(Stock.symbol == normalized_symbol))
    if stock_id is None:
        raise ValueError(f"unknown stock symbol: {normalized_symbol}")

    stmt = (
        select(
            PriceBar.timestamp,
            PriceBar.open,
            PriceBar.high,
            PriceBar.low,
            PriceBar.close,
            PriceBar.volume,
            PriceBar.traded_value,
        )
        .where(
            PriceBar.stock_id == stock_id,
            PriceBar.interval == interval,
            PriceBar.source == source,
        )
        .order_by(PriceBar.timestamp.desc())
        .limit(limit)
    )
    rows = list(db.execute(stmt).all())
    rows.reverse()

    result: list[OHLCVObservation] = []
    for row in rows:
        if row.volume is None:
            raise ValueError(f"missing volume for {normalized_symbol} at {row.timestamp}")
        result.append(
            OHLCVObservation(
                timestamp=row.timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                amount=float(row.traded_value or 0),
            )
        )
    return result
