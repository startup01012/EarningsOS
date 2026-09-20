"""Download NSE financial-results filings with explicit reporting periods.

The financial-results endpoint is used instead of deriving a reporting period from
announcement timestamps. NSE exposes the quarter covered and broadcast date
separately, which is the correct boundary for canonical earnings events.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from nse import NSE


def _parse_date(value: object) -> date | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.date()


def _parse_datetime(value: object) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True, dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    rows: list[dict] = []
    with NSE("", server=True, timeout=45, use_requests_library=True) as nse:
        try:
            # Fetch the financial-results metadata once for the requested broadcast-date
            # window, then restrict locally to NIFTY 50. The endpoint exposes the
            # reporting-period end separately from the broadcast timestamp.
            records = nse.financial_results(
                segment="equities",
                period="quarterly",
                symbol=None,
                from_date=start,
                to_date=end,
            )
        except Exception as exc:
            raise SystemExit(f"NSE financial-results request failed: {exc}") from exc

        for record in records or []:
            row = dict(record)
            row_symbol = str(row.get("symbol") or "").strip().upper()
            if row_symbol not in symbols:
                continue

            period_ended = _parse_date(
                row.get("toDate")
                or row.get("to_date")
                or row.get("periodEnded")
                or row.get("period_ended")
            )
            announcement_datetime = _parse_datetime(
                row.get("broadcastDate")
                or row.get("broadcastDateTime")
                or row.get("an_dt")
                or row.get("sort_date")
            )
            if period_ended is None:
                continue

            row["symbol"] = row_symbol
            row["period_ended"] = period_ended
            row["announcement_datetime"] = announcement_datetime
            row["document_type"] = str(
                row.get("subject")
                or row.get("relatingTo")
                or "Financial Results"
            )
            row["announcement_text"] = " ".join(
                str(row.get(key) or "").strip()
                for key in ("subject", "relatingTo", "audited", "consolidated", "period")
            ).strip()
            row["filing_url"] = str(
                row.get("xbrl")
                or row.get("xbrlFile")
                or row.get("attchmntFile")
                or ""
            )
            row["source"] = "NSE financial results"
            rows.append(row)

        print(
            f"NSE financial-results records={len(records or [])}, "
            f"NIFTY50 retained={len(rows)}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("NSE returned no financial-results filings for the requested range")

    # Keep distinct filings, including standalone/consolidated variants.
    df = df.drop_duplicates()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"saved {len(df)} rows to {out}")
    print(f"periods={df['period_ended'].nunique()} symbols={df['symbol'].nunique()}")


if __name__ == "__main__":
    main()
