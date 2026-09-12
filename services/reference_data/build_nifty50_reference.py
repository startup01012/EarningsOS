from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from services.reference_data.nifty50 import (
    Nifty50Constituent,
    validate_constituents,
)

RAW_PATH = Path("data/raw/nifty50/ind_nifty50list.csv")
OUTPUT_PATH = Path("data/reference/nifty50.csv")


def build_reference() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing NSE source file: {RAW_PATH}")

    raw = pd.read_csv(RAW_PATH)

    required = {
        "Company Name",
        "Industry",
        "Symbol",
        "Series",
        "ISIN Code",
    }

    missing = required - set(raw.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    records = []

    for _, row in raw.iterrows():
        records.append(
            Nifty50Constituent(
                symbol=str(row["Symbol"]).strip().upper(),
                company_name=str(row["Company Name"]).strip(),
                industry=(
                    str(row["Industry"]).strip()
                    if pd.notna(row["Industry"])
                    else None
                ),
                series=(
                    str(row["Series"]).strip().upper()
                    if pd.notna(row["Series"])
                    else None
                ),
                isin=(
                    str(row["ISIN Code"]).strip().upper()
                    if pd.notna(row["ISIN Code"])
                    else None
                ),
            )
        )

    validate_constituents(records)

    output = pd.DataFrame(
        [
            {
                "symbol": item.symbol,
                "company_name": item.company_name,
                "industry": item.industry,
                "series": item.series,
                "isin": item.isin,
                "exchange": item.exchange,
                "index": item.index,
                "source": item.source,
                "as_of_date": date.today().isoformat(),
            }
            for item in records
        ]
    )

    if len(output) != 50:
        raise ValueError(
            f"Expected 50 constituents, got {len(output)}"
        )

    if output["symbol"].duplicated().any():
        raise ValueError("Duplicate symbols found.")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    return output


if __name__ == "__main__":
    df = build_reference()

    print(f"Created: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    print(df.head())
