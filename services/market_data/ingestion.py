from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.db.models import PriceBar, Stock

from .schemas import MarketBar


def get_or_create_stock(db: Session, bar: MarketBar) -> Stock:
    statement = select(Stock).where(
        Stock.symbol == bar.symbol,
        Stock.exchange == bar.exchange,
    )

    stock = db.execute(statement).scalar_one_or_none()

    if stock is None:
        stock = Stock(
            symbol=bar.symbol,
            exchange=bar.exchange,
            company_name=bar.symbol,
            currency="INR",
            active=True,
        )
        db.add(stock)
        db.flush()

    return stock


def ingest_bars(
    db: Session,
    bars: Iterable[MarketBar],
) -> int:
    inserted = 0

    for bar in bars:
        stock = get_or_create_stock(db, bar)

        statement = select(PriceBar).where(
            PriceBar.stock_id == stock.id,
            PriceBar.interval == bar.interval,
            PriceBar.timestamp == bar.timestamp,
            PriceBar.source == bar.source,
        )

        existing = db.execute(statement).scalar_one_or_none()

        if existing is not None:
            continue

        price_bar = PriceBar(
            stock_id=stock.id,
            interval=bar.interval,
            timestamp=bar.timestamp,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            traded_value=bar.traded_value,
            source=bar.source,
        )

        db.add(price_bar)
        inserted += 1

    db.commit()

    return inserted