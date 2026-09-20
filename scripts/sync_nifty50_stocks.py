from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from apps.api.db.models import Stock
from apps.api.db.session import get_session


def _first(row, *names):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() and str(value).strip().lower() != "nan":
            return str(value).strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synchronize missing NIFTY 50 stocks from NSE announcement source data."
    )
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    path = Path(args.input)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Input must be .parquet or .csv")

    required = {"symbol"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")

    db = get_session()
    inserted = 0
    try:
        existing_by_symbol = {s.symbol: s for s in db.scalars(select(Stock)).all()}
        existing_isins = {s.isin: s for s in existing_by_symbol.values() if s.isin}
        pending_symbols: set[str] = set()
        pending_isins: set[str] = set()

        # NSE financial-results contains multiple filings per symbol/period.
        # Deduplicate identities before inserting so repeated rows cannot violate
        # the unique symbol/ISIN constraints within the same transaction.
        for row in df.drop_duplicates(subset=["symbol"]).to_dict("records"):
            symbol = _first(row, "symbol").upper()
            if not symbol or symbol in existing_by_symbol or symbol in pending_symbols:
                continue

            isin = _first(row, "sm_isin", "isin", "ISIN Code") or None
            if isin and (isin in existing_isins or isin in pending_isins):
                continue

            company_name = _first(row, "sm_name", "company_name", "Company Name") or symbol
            industry = _first(row, "smIndustry", "industry", "Industry") or None

            db.add(
                Stock(
                    symbol=symbol,
                    exchange="NSE",
                    isin=isin,
                    company_name=company_name,
                    industry=industry,
                    active=True,
                )
            )
            pending_symbols.add(symbol)
            if isin:
                pending_isins.add(isin)
            inserted += 1

        db.commit()
        print(f"source rows: {len(df)}")
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
