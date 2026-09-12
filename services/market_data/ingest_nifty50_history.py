from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from apps.api.db.models import PriceBar, Stock
from apps.api.db.session import SessionLocal
from services.market_data.providers.yfinance_provider import YFinanceProvider

REFERENCE_PATH = "data/reference/nifty50.csv"
IST = ZoneInfo("Asia/Kolkata")


def ingest_nifty50_history(start: datetime, end: datetime) -> tuple[int, int]:
    """Download NIFTY 50 daily history and bulk-insert it idempotently."""
    reference = pd.read_csv(REFERENCE_PATH)
    if len(reference) != 50 or reference["symbol"].duplicated().any():
        raise ValueError("NIFTY 50 reference must contain exactly 50 unique symbols")

    symbols = reference["symbol"].astype(str).str.strip().str.upper().tolist()
    provider = YFinanceProvider()
    db = SessionLocal()
    inserted = 0
    skipped = 0
    empty_symbols: list[str] = []

    try:
        stocks = db.execute(
            select(Stock).where(Stock.symbol.in_(symbols), Stock.exchange == "NSE")
        ).scalars().all()
        stock_by_symbol = {stock.symbol: stock for stock in stocks}
        missing_stocks = sorted(set(symbols) - set(stock_by_symbol))
        if missing_stocks:
            raise ValueError(
                "Stocks missing from database; run NIFTY 50 stock ingestion first: "
                f"{missing_stocks}"
            )

        for symbol in symbols:
            bars = provider.get_historical_bars(symbol, start, end, interval="1d")
            if not bars:
                empty_symbols.append(symbol)
                print(f"{symbol}: downloaded=0 inserted=0")
                continue

            rows = [
                {
                    "stock_id": stock_by_symbol[bar.symbol].id,
                    "interval": bar.interval,
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "traded_value": bar.traded_value,
                    "source": bar.source,
                }
                for bar in bars
            ]

            # RETURNING lets us count actual inserts reliably. PostgreSQL/psycopg
            # may report rowcount=-1 for INSERT ... ON CONFLICT statements.
            statement = (
                insert(PriceBar)
                .values(rows)
                .on_conflict_do_nothing(constraint="uq_price_bar")
                .returning(PriceBar.id)
            )
            result = db.execute(statement)
            inserted_now = len(result.fetchall())
            skipped_now = len(rows) - inserted_now

            inserted += inserted_now
            skipped += skipped_now
            db.commit()
            print(f"{symbol}: downloaded={len(rows)} inserted={inserted_now} skipped={skipped_now}")

        if empty_symbols:
            print(f"Symbols with no returned bars: {', '.join(empty_symbols)}")

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
