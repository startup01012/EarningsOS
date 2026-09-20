"""Download NSE corporate financial-results filings with explicit reporting periods."""
from __future__ import annotations

import argparse
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests


NSE_HOME = "https://www.nseindia.com"
FINANCIAL_RESULTS_URL = f"{NSE_HOME}/api/corporates-financial-results"


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


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 Chrome/153.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{NSE_HOME}/companies-listing/corporate-filings-financial-results",
            "Connection": "keep-alive",
        }
    )
    response = session.get(NSE_HOME, timeout=30)
    response.raise_for_status()
    return session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    rows: list[dict] = []
    session = _session()

    for index, symbol in enumerate(symbols, start=1):
        try:
            response = session.get(
                FINANCIAL_RESULTS_URL,
                params={"index": "equities", "symbol": symbol, "period": "Quarterly"},
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            records = payload.get("data", []) if isinstance(payload, dict) else []

            for record in records:
                row = dict(record)
                period_ended = _parse_date(row.get("toDate"))
                announcement_datetime = _parse_datetime(
                    row.get("broadCastDate") or row.get("filingDate")
                )
                if period_ended is None or announcement_datetime is None:
                    continue
                announcement_date = announcement_datetime.date()
                if announcement_date < start or announcement_date > end:
                    continue

                row["symbol"] = symbol
                row["period_ended"] = period_ended
                row["announcement_datetime"] = announcement_datetime
                row["document_type"] = "Financial Results"
                row["consolidated_status"] = str(row.get("consolidated") or "").strip()
                row["announcement_text"] = " ".join(
                    str(row.get(key) or "").strip()
                    for key in (
                        "relatingTo",
                        "audited",
                        "consolidated",
                        "period",
                        "financialYear",
                    )
                ).strip()
                row["filing_url"] = str(
                    row.get("xbrl") or row.get("xbrl_attachment") or ""
                )
                row["source"] = "NSE financial results"
                rows.append(row)

            print(
                f"[{index}/{len(symbols)}] {symbol}: "
                f"records={len(records)} retained={sum(1 for r in rows if r.get('symbol') == symbol)}",
                flush=True,
            )
        except Exception as exc:
            print(f"[{index}/{len(symbols)}] {symbol}: ERROR {exc}", flush=True)
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("NSE returned no financial-results filings for the requested range")

    df = df.drop_duplicates()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"saved {len(df)} rows to {out}")
    print(f"periods={df['period_ended'].nunique()} symbols={df['symbol'].nunique()}")


if __name__ == "__main__":
    main()
