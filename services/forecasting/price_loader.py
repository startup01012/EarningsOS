from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.db.models import PriceBar, Stock

from .price_series import PriceObservation


def load_close_observations(
    db: Session,
    symbol: str,
    *,
    interval: str = "1d",
    source: str = "yfinance",
    limit: int = 512,
    as_of: datetime | None = None,
) -> list[PriceObservation]:
    """Load causal close-price observations for a stock from PostgreSQL.

    ``as_of`` is an information boundary: when supplied, no observation after
    that timestamp is returned. The query intentionally does not interpolate,
    backfill, or otherwise manufacture missing prices.
    """
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must not be empty")
    if limit < 2:
        raise ValueError("limit must be >= 2")

    stock_id = db.scalar(select(Stock.id).where(Stock.symbol == normalized_symbol))
    if stock_id is None:
        raise ValueError(f"unknown stock symbol: {normalized_symbol}")

    stmt = (
        select(PriceBar.timestamp, PriceBar.close)
        .where(
            PriceBar.stock_id == stock_id,
            PriceBar.interval == interval,
            PriceBar.source == source,
        )
        .order_by(PriceBar.timestamp.desc())
        .limit(limit)
    )
    if as_of is not None:
        stmt = stmt.where(PriceBar.timestamp <= as_of)

    rows = list(db.execute(stmt).all())
    rows.reverse()
    return [PriceObservation(timestamp=row.timestamp, close=float(row.close)) for row in rows]
