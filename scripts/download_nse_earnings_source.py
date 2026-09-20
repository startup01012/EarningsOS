"""Download NSE corporate financial-results filings with explicit reporting periods."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
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
    symbols = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}

    rows: list[dict] = []
    window_start = start

    with NSE("", server=True, timeout=45, use_requests_library=True) as nse:
        # The NSE client/API returns only a limited recent slice for a very large
        # date window. Query bounded windows so the full historical range is
        # covered deterministically.
        while window_start <= end:
            window_end = min(window_start + timedelta(days=119), end)
            try:
                records = nse.financial_results(
                    segment="equities",
                    period="quarterly",
                    symbol=None,
                    from_date=window_start,
                    to_date=window_end,
                )
            except Exception as exc:
                print(
                    f"financial-results window failed "
                    f"{window_start.date()}..{window_end.date()}: {exc}",
                    flush=True,
                )
                window_start = window_end + timedelta(days=1)
                continue

            retained_window = 0
            for record in records or []:
                row = dict(record)
                symbol = str(row.get("symbol") or "").strip().upper()
                if symbol not in symbols:
                    continue

                period_ended = _parse_date(row.get("toDate"))
                announcement_datetime = _parse_datetime(
                    row.get("broadCastDate") or row.get("filingDate")
                )
                if period_ended is None or announcement_datetime is None:
                    continue
                if not (start.date() <= announcement_datetime.date() <= end.date()):
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
                retained_window += 1

            print(
                f"window {window_start.date()}..{window_end.date()}: "
                f"records={len(records or [])} retained={retained_window}",
                flush=True,
            )
            window_start = window_end + timedelta(days=1)

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
