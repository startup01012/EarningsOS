"""Download NSE corporate announcements for the NIFTY 50 and retain earnings-related records."""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

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
    "board meeting",
)


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": NSE_PAGE,
        }
    )
    s.get(NSE_PAGE, timeout=30)
    return s


def windows(start: date, end: date, days: int = 30):
    cur = start
    while cur <= end:
        nxt = min(end, cur + timedelta(days=days - 1))
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def fetch(s: requests.Session, start: date, end: date):
    params = {
        "index": "equities",
        "from_date": start.strftime("%d-%m-%Y"),
        "to_date": end.strftime("%d-%m-%Y"),
    }
    for attempt in range(4):
        try:
            r = s.get(NSE_API, params=params, timeout=45)
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict):
                payload = payload.get("data", payload.get("results", []))
            return payload if isinstance(payload, list) else []
        except Exception as exc:
            print(
                f"NSE request failed {start}..{end} attempt={attempt + 1}: {exc}",
                flush=True,
            )
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    symbols = {
        s.strip().upper()
        for s in args.symbols.split(",")
        if s.strip()
    }
    s = session()
    rows = []

    for a, b in windows(start, end):
        batch = fetch(s, a, b)
        for row in batch:
            symbol = str(
                row.get("symbol") or row.get("sym") or ""
            ).strip().upper()
            desc = str(
                row.get("desc")
                or row.get("subject")
                or row.get("description")
                or ""
            ).strip()
            details = str(row.get("attchmntText") or "").strip()

            searchable_text = f"{desc} {details}".strip()
            if symbol not in symbols or not any(
                keyword in searchable_text.lower() for keyword in KEYWORDS
            ):
                continue

            announcement_datetime = row.get("an_dt") or row.get("sort_date")
            filing_url = row.get("attchmntFile") or ""

            normalized = dict(row)
            normalized["symbol"] = symbol
            normalized["document_type"] = desc
            normalized["announcement_text"] = searchable_text
            normalized["announcement_datetime"] = announcement_datetime
            normalized["filing_url"] = filing_url
            normalized["source"] = "NSE corporate announcements"
            rows.append(normalized)

        print(
            f"{a}..{b}: {len(batch)} announcements, retained={len(rows)}",
            flush=True,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).drop_duplicates()

    if df.empty:
        raise SystemExit(
            "NSE returned no earnings-related announcements for the requested range"
        )

    df.to_parquet(out, index=False)
    print(f"saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
