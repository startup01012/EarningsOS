from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import select

from apps.api.db.models import PriceBar, Stock
from apps.api.db.session import SessionLocal
from services.market_data.providers.yfinance_provider import YFinanceProvider

REFERENCE_PATH = "data/reference/nifty50.csv"
IST = ZoneInfo("Asia/Kolkata")


def ingest_nifty50_history(start: datetime, end: datetime) -> tuple[int, int]:
    reference = pd.read_csv(REFERENCE_PATH)
    if len(reference) != 50 or reference["symbol"].duplicated().any():
        raise ValueError("NIFTY 50 reference must contain exactly 50 unique symbols")

    provider = YFinanceProvider()
    inserted = 0
    skipped = 0
    db = SessionLocal()
    try:
        for symbol in reference["symbol"].astype(str).str.strip().str.upper():
            stock = db.execute(select(Stock).where(Stock.symbol == symbol)).scalar_one_or_none()
            if stock is None:
                raise ValueError(f"Stock {symbol} is missing; run NIFTY 50 stock ingestion first")

            bars = provider.get_historical_bars(symbol, start, end, interval="1d")
            for bar in bars:
                existing = db.execute(
                    select(PriceBar.id).where(
                        PriceBar.stock_id == stock.id,
                        PriceBar.interval == bar.interval,
                        PriceBar.timestamp == bar.timestamp,
                        PriceBar.source == bar.source,
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    skipped += 1
                    continue
                db.add(
                    PriceBar(
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
                )
                inserted += 1
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return inserted, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest historical daily NIFTY 50 OHLCV")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=IST)
    end = datetime.fromisoformat(args.end).replace(tzinfo=IST)
    if start >= end:
        raise SystemExit("--start must be earlier than --end")

    inserted, skipped = ingest_nifty50_history(start, end)
    print("NIFTY 50 historical ingestion complete")
    print(f"Inserted: {inserted}")
    print(f"Skipped existing: {skipped}")
