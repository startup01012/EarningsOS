"""Download NSE corporate announcements and recover explicit reporting periods."""
from __future__ import annotations

import argparse
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from jugaad_data.nse import NSELive

NSE_PAGE = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
NSE_API = "https://www.nseindia.com/api/corporate-announcements"
KEYWORDS = (
    "financial result",
    "financial results",
    "quarterly result",
    "quarterly results",
    "audited result",
    "unaudited result",
    "result update",
    "results",
)

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december"
)


def _parse_datetime(value: object) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True, dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _parse_period_end(text: str) -> date | None:
    text = " ".join(text.split())

    patterns = [
        rf"(?:quarter|period|year)(?: and financial year)?\s+ended\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS})\s*,?\s+(\d{{4}})",
        rf"(?:quarter|period|year)(?: and financial year)?\s+ended\s+({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(\d{{4}})",
        r"(?:quarter|period|year)(?: and financial year)?\s+ended\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})",
        r"(?:quarter|period|year)(?: and financial year)?\s+ended\s+on\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})",
    ]

    for index, pattern in enumerate(patterns):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            if index in (0, 1):
                if index == 0:
                    day, month, year = match.groups()
                else:
                    month, day, year = match.groups()
                return pd.to_datetime(
                    f"{day} {month} {year}", dayfirst=True, errors="raise"
                ).date()
            day, month, year = match.groups()
            return date(int(year), int(month), int(day))
        except (TypeError, ValueError):
            continue
    return None


def _client() -> NSELive:
    return NSELive()


def _windows(start: date, end: date, days: int = 7):
    current = start
    while current <= end:
        window_end = min(end, current + timedelta(days=days - 1))
        yield current, window_end
        current = window_end + timedelta(days=1)


def _fetch(client: NSELive, start: date, end: date):
    return client.corporate_announcements(
        from_date=start,
        to_date=end,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    symbols = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}

    client = _client()
    rows: list[dict] = []

    for window_start, window_end in _windows(start, end):
        batch = _fetch(client, window_start, window_end)
        retained = 0

        for raw in batch:
            row = dict(raw)
            symbol = str(
                row.get("symbol") or row.get("sym") or ""
            ).strip().upper()
            subject = str(
                row.get("desc")
                or row.get("subject")
                or row.get("description")
                or ""
            ).strip()

            if symbol not in symbols:
                continue
            if not any(keyword in subject.lower() for keyword in KEYWORDS):
                continue

            attachment_text = str(
                row.get("attchmntText")
                or row.get("attachmentText")
                or ""
            ).strip()
            searchable_text = " ".join(part for part in (subject, attachment_text) if part)
            period_ended = _parse_period_end(searchable_text)

            # Do not infer a reporting period from the announcement timestamp.
            # Rows without an explicit period are retained only as diagnostics.
            announcement_datetime = _parse_datetime(
                row.get("an_dt")
                or row.get("sort_date")
                or row.get("broadCastDate")
                or row.get("broadcastdate")
            )

            row["symbol"] = symbol
            row["announcement_datetime"] = announcement_datetime
            row["period_ended"] = period_ended
            row["document_type"] = subject
            row["announcement_text"] = searchable_text
            row["filing_url"] = str(
                row.get("attchmntFile")
                or row.get("attachmentFile")
                or ""
            )
            row["consolidated_status"] = (
                "Consolidated"
                if "consolidated" in searchable_text.lower()
                and "non-consolidated" not in searchable_text.lower()
                else "Non-Consolidated"
                if "non-consolidated" in searchable_text.lower()
                else ""
            )
            row["source"] = "NSE corporate announcements"
            rows.append(row)
            retained += 1

        print(
            f"{window_start}..{window_end}: "
            f"{len(batch)} announcements, retained={retained}",
            flush=True,
        )

    df = pd.DataFrame(rows).drop_duplicates()
    if df.empty:
        raise SystemExit("NSE returned no earnings-related announcements")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"saved {len(df)} rows to {out}")
    print(
        f"explicit_period_rows={df['period_ended'].notna().sum()} "
        f"symbols={df['symbol'].nunique()}"
    )


if __name__ == "__main__":
    main()
