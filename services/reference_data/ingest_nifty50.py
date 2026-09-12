from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from apps.api.db.models import Stock
from apps.api.db.session import SessionLocal

REFERENCE_PATH = "data/reference/nifty50.csv"


def ingest_nifty50() -> tuple[int, int]:
    df = pd.read_csv(REFERENCE_PATH)

    required = {
        "symbol",
        "company_name",
        "industry",
        "series",
        "isin",
        "exchange",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if len(df) != 50:
        raise ValueError(f"Expected 50 NIFTY 50 constituents, got {len(df)}")

    if df["symbol"].duplicated().any():
        raise ValueError("Duplicate NIFTY 50 symbols found")

    db = SessionLocal()

    inserted = 0
    updated = 0

    try:
        for row in df.to_dict("records"):
            symbol = str(row["symbol"]).strip().upper()
            exchange = str(row["exchange"]).strip().upper()

            stock = db.execute(
                select(Stock).where(
                    Stock.symbol == symbol,
                    Stock.exchange == exchange,
                )
            ).scalar_one_or_none()

            if stock is None:
                stock = Stock(
                    symbol=symbol,
                    exchange=exchange,
                    isin=str(row["isin"]).strip().upper()
                    if pd.notna(row["isin"])
                    else None,
                    company_name=str(row["company_name"]).strip(),
                    industry=str(row["industry"]).strip()
                    if pd.notna(row["industry"])
                    else None,
                    currency="INR",
                    active=True,
                )
                db.add(stock)
                inserted += 1
            else:
                stock.company_name = str(row["company_name"]).strip()

                stock.industry = (
                    str(row["industry"]).strip()
                    if pd.notna(row["industry"])
                    else None
                )

                stock.isin = (
                    str(row["isin"]).strip().upper()
                    if pd.notna(row["isin"])
                    else None
                )

                stock.active = True
                updated += 1

        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()

    return inserted, updated


if __name__ == "__main__":
    inserted, updated = ingest_nifty50()

    print("NIFTY 50 ingestion complete")
    print(f"Inserted: {inserted}")
    print(f"Updated:  {updated}")
    print(f"Total processed: {inserted + updated}")
