from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from apps.api.db.models import Stock
from apps.api.db.session import get_session


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize missing NIFTY 50 stocks into EarningsOS.")
    parser.add_argument("--input", default="data/reference/nifty50.csv")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input))
    required = {"Company Name", "Industry", "Symbol", "ISIN Code"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing NIFTY50 reference columns: {sorted(missing)}")

    db = get_session()
    inserted = 0
    try:
        existing_by_symbol = {s.symbol: s for s in db.scalars(select(Stock)).all()}
        existing_isins = {
            s.isin: s for s in existing_by_symbol.values() if s.isin
        }

        for row in df.itertuples(index=False):
            symbol = str(getattr(row, "Symbol")).strip().upper()
            isin = str(getattr(row, "ISIN_Code")).strip() if getattr(row, "ISIN_Code") else None
            company_name = str(getattr(row, "Company_Name")).strip()
            industry = str(getattr(row, "Industry")).strip() or None

            if symbol in existing_by_symbol:
                continue
            if isin and isin in existing_isins:
                # Keep the existing stock identity when the ISIN already exists.
                continue

            stock = Stock(
                symbol=symbol,
                exchange="NSE",
                isin=isin,
                company_name=company_name,
                industry=industry,
                active=True,
            )
            db.add(stock)
            inserted += 1

        db.commit()
        print(f"NIFTY50 reference rows: {len(df)}")
        print(f"stocks already present: {len(existing_by_symbol)}")
        print(f"stocks inserted: {inserted}")
        return 0
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
