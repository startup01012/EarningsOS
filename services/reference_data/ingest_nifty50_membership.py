from __future__ import annotations

from datetime import date
from uuid import uuid4

import pandas as pd
from sqlalchemy import select

from apps.api.db.models import IndexMembership, Stock
from apps.api.db.session import SessionLocal

REFERENCE_PATH = "data/reference/nifty50.csv"
INDEX_NAME = "NIFTY 50"


def ingest_current_membership() -> tuple[int, int]:
    df = pd.read_csv(REFERENCE_PATH)
    required = {"symbol", "exchange", "as_of_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(df) != 50:
        raise ValueError(f"Expected 50 NIFTY 50 constituents, got {len(df)}")

    as_of = pd.to_datetime(df["as_of_date"], errors="raise").dt.date.unique()
    if len(as_of) != 1:
        raise ValueError("NIFTY 50 snapshot must have exactly one as_of_date")
    effective_from: date = as_of[0]

    db = SessionLocal()
    inserted = 0
    updated = 0
    try:
        for row in df.to_dict("records"):
            symbol = str(row["symbol"]).strip().upper()
            exchange = str(row["exchange"]).strip().upper()
            stock = db.execute(
                select(Stock).where(Stock.symbol == symbol, Stock.exchange == exchange)
            ).scalar_one_or_none()
            if stock is None:
                raise ValueError(f"Stock {symbol}/{exchange} is not present; run ingest_nifty50.py first")

            membership = db.execute(
                select(IndexMembership).where(
                    IndexMembership.stock_id == stock.id,
                    IndexMembership.index_name == INDEX_NAME,
                    IndexMembership.effective_from == effective_from,
                )
            ).scalar_one_or_none()
            if membership is None:
                db.add(
                    IndexMembership(
                        id=uuid4(),
                        stock_id=stock.id,
                        index_name=INDEX_NAME,
                        effective_from=effective_from,
                        effective_to=None,
                        source="NSE NIFTY 50 constituent snapshot",
                    )
                )
                inserted += 1
            else:
                if membership.effective_to is not None or membership.source != "NSE NIFTY 50 constituent snapshot":
                    membership.effective_to = None
                    membership.source = "NSE NIFTY 50 constituent snapshot"
                    updated += 1
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return inserted, updated


if __name__ == "__main__":
    inserted, updated = ingest_current_membership()
    print("NIFTY 50 membership ingestion complete")
    print(f"Inserted: {inserted}")
    print(f"Updated: {updated}")
