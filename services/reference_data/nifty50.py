from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Nifty50Constituent:
    symbol: str
    company_name: str
    industry: str | None
    series: str | None
    isin: str | None
    exchange: str = "NSE"
    index: str = "NIFTY 50"
    source: str = "NSE"


def validate_constituents(
    constituents: list[Nifty50Constituent],
) -> None:
    """Validate the normalized NIFTY 50 reference records."""

    if not constituents:
        raise ValueError("NIFTY 50 constituent list is empty.")

    symbols = [item.symbol.strip().upper() for item in constituents]

    if len(symbols) != len(set(symbols)):
        raise ValueError("Duplicate NIFTY 50 symbols detected.")

    for item in constituents:
        if not item.symbol.strip():
            raise ValueError("Constituent has an empty symbol.")

        if not item.company_name.strip():
            raise ValueError(
                f"Missing company name for {item.symbol!r}."
            )


def reference_output_path() -> Path:
    return Path("data/reference/nifty50.csv")


def raw_output_directory() -> Path:
    return Path("data/raw/nifty50")


def reference_as_of_date() -> date:
    return date.today()
