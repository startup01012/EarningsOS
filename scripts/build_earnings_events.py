from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, select

from apps.api.db.models import EarningsDocument, EarningsEvent, Stock
from apps.api.db.session import get_session


DATE_PATTERNS = [
    re.compile(
        r"(?:quarter|period|year)(?:\s+and\s+year)?\s+ended"
        r"\s*[:\-]?\s*(\d{1,2}(?:st|nd|rd|th)?[/-]\d{1,2}[/-]\d{4})",
        re.I,
    ),
    re.compile(
        r"(?:quarter|period|year)(?:\s+and\s+year)?\s+ended"
        r"\s*[:\-]?\s*(\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"(?:,)?\s+\d{4})",
        re.I,
    ),
    re.compile(
        r"(?:quarter|period|year)(?:\s+and\s+year)?\s+ended"
        r"\s*[:\-]?\s*((?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4})",
        re.I,
    ),
    re.compile(
        r"(\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"(?:,)?\s+\d{4})\s+(?:quarter|period|year)\s+ended",
        re.I,
    ),
    re.compile(
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4})\s+(?:quarter|period|year)\s+ended",
        re.I,
    ),
]


@dataclass(frozen=True)
class Classification:
    document_type: str
    document_subtype: str | None
    is_primary_result: bool
    is_followup_document: bool


def _first_column(df: pd.DataFrame, *names: str) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _parse_datetime(value: object) -> datetime | None:
    if value is None or _text(value) == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _parse_date(value: object) -> date | None:
    if value is None or _text(value) == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _extract_period_from_text(text: str) -> date | None:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        parsed = pd.to_datetime(match.group(1), dayfirst=True, errors="coerce")
        if not pd.isna(parsed):
            return parsed.date()
    return None


def classify_document(document_type: str, announcement_text: str, filing_url: str) -> Classification:
    raw = " ".join([document_type, announcement_text, filing_url]).lower()
    compact = re.sub(r"[^a-z0-9]+", " ", raw)

    if any(term in compact for term in ("transcript", "transcripts", "concall transcript")):
        return Classification("earnings_call_transcript", "transcript", False, True)
    if any(term in compact for term in ("recording", "earnings call", "conference call", "concall")):
        return Classification("earnings_call", "recording", False, True)
    if any(term in compact for term in ("investor presentation", "investor presentation", "presentation")):
        return Classification("investor_presentation", None, False, True)
    if any(term in compact for term in ("media release", "press release", "media-release")):
        return Classification("media_release", None, False, True)
    if any(term in compact for term in ("revised financial results", "revised results", "rectification of financial results", "correction in financial results")):
        return Classification("financial_results_correction", None, False, True)
    if any(term in compact for term in (
        "financial results",
        "financial result",
        "quarterly results",
        "audited results",
        "unaudited results",
        "results for the quarter",
        "results for the year",
    )):
        subtype = "financial_result_update" if "update" in compact or "outcome" in compact else None
        return Classification("financial_results", subtype, True, False)
    if any(term in compact for term in ("board meeting outcome", "outcome of board meeting", "board meeting")):
        return Classification("board_meeting", None, False, True)
    return Classification("other", None, False, True)


def _document_id(symbol: str, announcement_datetime: datetime | None, filing_url: str, text: str) -> str:
    payload = "|".join(
        [
            symbol,
            announcement_datetime.isoformat() if announcement_datetime else "",
            filing_url.strip(),
            text.strip(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    symbol_col = _first_column(df, "symbol", "Symbol", "TckrSymb")
    period_col = _first_column(df, "period_ended", "period_end", "period_end_date", "event_date", "reporting_period_end")
    announced_col = _first_column(df, "announcement_datetime", "announced_at", "announcement_date_time", "timestamp")
    type_col = _first_column(df, "document_type", "Document Type", "category", "type")
    text_col = _first_column(df, "announcement_text", "announcement", "description", "text")
    url_col = _first_column(df, "filing_url", "url", "link", "Filing URL")
    source_col = _first_column(df, "source", "Source")
    fiscal_col = _first_column(df, "fiscal_period", "quarter", "fiscal_quarter")

    if not symbol_col:
        raise ValueError("Input must contain a symbol column")
    if period_col:
        period_values = [_parse_date(value) for value in df[period_col].tolist()]
    else:
        period_values = [None] * len(df)
    df["__period_ended"] = pd.Series(period_values, index=df.index, dtype="object")

    df["__symbol"] = df[symbol_col].map(lambda x: _text(x).upper())
    announced_values = [_parse_datetime(value) for value in df[announced_col].tolist()] if announced_col else [None] * len(df)
    df["__announcement_datetime"] = pd.Series(announced_values, index=df.index, dtype="object")
    df["__document_type"] = df[type_col].map(_text) if type_col else ""
    df["__announcement_text"] = df[text_col].map(_text) if text_col else ""
    df["__filing_url"] = df[url_col].map(_text) if url_col else ""
    df["__source"] = df[source_col].map(_text) if source_col else "earnings_pipeline"
    df["__fiscal_period"] = df[fiscal_col].map(_text) if fiscal_col else ""

    missing_period = df["__period_ended"].isna()
    df.loc[missing_period, "__period_ended"] = df.loc[missing_period, "__announcement_text"].map(_extract_period_from_text)
    return df


def build_event_documents(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _normalize(df.copy())
    classifications = [
        classify_document(document_type, announcement_text, filing_url)
        for document_type, announcement_text, filing_url
        in df[["__document_type", "__announcement_text", "__filing_url"]].itertuples(index=False, name=None)
    ]
    df["__classified_type"] = [x.document_type for x in classifications]
    df["__subtype"] = [x.document_subtype for x in classifications]
    df["__primary"] = [x.is_primary_result for x in classifications]
    df["__followup"] = [x.is_followup_document for x in classifications]

    df["__document_id"] = [
        _document_id(symbol, dt, url, text)
        for symbol, dt, url, text in zip(
            df["__symbol"], df["__announcement_datetime"], df["__filing_url"], df["__announcement_text"]
        )
    ]

    # Deduplicate only exact re-ingestion of the same deterministic document.
    df = df.drop_duplicates("__document_id", keep="first")

    quality = []
    event_rows = []
    document_rows = []

    for (symbol, period_ended), group in df.groupby(["__symbol", "__period_ended"], dropna=False, sort=True):
        if not symbol or pd.isna(period_ended):
            quality.append({
                "event_key": None,
                "symbol": symbol,
                "period_ended": None if pd.isna(period_ended) else period_ended,
                "document_count": len(group),
                "primary_result_count": int(group["__primary"].sum()),
                "classification_status": "MISSING_PERIOD_END" if pd.isna(period_ended) else "OK",
                "quality_flag": "MISSING_PERIOD_END" if pd.isna(period_ended) else "OK",
            })
            continue

        period_date = period_ended
        event_key = f"{symbol}_{period_date.isoformat()}"

        # Prefer original financial-results filings. A correction stays attached
        # to the same event but must not replace the canonical result timestamp.
        primary_candidates = group[group["__classified_type"].eq("financial_results")].copy()
        if primary_candidates.empty:
            primary_candidates = group[group["__classified_type"].eq("financial_results_correction")].copy()
        primary_candidates = primary_candidates.sort_values("__announcement_datetime", na_position="last")
        primary_row = primary_candidates.iloc[0] if len(primary_candidates) else None
        result_dt = primary_row["__announcement_datetime"] if primary_row is not None else None

        types = set(group["__classified_type"].tolist())
        document_times = [
            value for value in group["__announcement_datetime"].tolist()
            if value is not None and not pd.isna(value)
        ]
        first_document_datetime = min(document_times) if document_times else None
        last_document_datetime = max(document_times) if document_times else None
        quality_flag = "OK"
        if len(primary_candidates) > 1:
            quality_flag = "MULTIPLE_PRIMARY_RESULTS"
        elif result_dt is None:
            quality_flag = "MISSING_RESULT"
        elif result_dt.date() < period_date:
            quality_flag = "SUSPICIOUS_DATE"

        event_rows.append({
            "event_key": event_key,
            "symbol": symbol,
            "period_ended": period_date,
            "fiscal_period": next((x for x in group["__fiscal_period"] if x), ""),
            "result_announcement_datetime": result_dt,
            "announcement_date": result_dt.date() if result_dt else None,
            "announcement_time": result_dt.time().isoformat() if result_dt else None,
            "document_count": len(group),
            "has_financial_results": bool(group["__classified_type"].isin(["financial_results", "financial_results_correction"]).any()),
            "has_media_release": "media_release" in types,
            "has_earnings_call": "earnings_call" in types,
            "has_transcript": "earnings_call_transcript" in types,
            "first_document_datetime": first_document_datetime,
            "last_document_datetime": last_document_datetime,
            "quality_flag": quality_flag,
        })

        for row in group.to_dict("records"):
            document_rows.append({
                "document_id": row["__document_id"],
                "event_key": event_key,
                "symbol": symbol,
                "period_ended": period_date,
                "announcement_datetime": row["__announcement_datetime"],
                "document_type": row["__classified_type"],
                "document_subtype": row["__subtype"],
                "announcement_text": row["__announcement_text"],
                "filing_url": row["__filing_url"],
                "is_primary_result": bool(primary_row is not None and row["__document_id"] == primary_row["__document_id"]),
                "is_followup_document": bool(row["__followup"] or (primary_row is not None and row["__document_id"] != primary_row["__document_id"])),
                "source": row["__source"],
            })

        quality.append({
            "event_key": event_key,
            "symbol": symbol,
            "period_ended": period_date,
            "document_count": len(group),
            "first_document_datetime": group["__announcement_datetime"].min(),
            "result_announcement_datetime": result_dt,
            "last_document_datetime": group["__announcement_datetime"].max(),
            "days_from_period_end": (result_dt.date() - period_date).days if result_dt else None,
            "primary_result_count": len(primary_candidates),
            "classification_status": "OK",
            "quality_flag": quality_flag,
        })

    return pd.DataFrame(event_rows), pd.DataFrame(document_rows), pd.DataFrame(quality)


def persist_events(events: pd.DataFrame, documents: pd.DataFrame) -> tuple[int, int]:
    db = get_session()
    inserted_events = inserted_documents = 0
    try:
        stocks = {stock.symbol: stock for stock in db.scalars(select(Stock)).all()}
        for row in events.itertuples(index=False):
            stock = stocks.get(row.symbol)
            if stock is None:
                raise ValueError(f"Stock row missing for symbol {row.symbol}")
            existing = db.scalar(select(EarningsEvent).where(EarningsEvent.event_key == row.event_key))
            values = dict(
                stock_id=stock.id,
                event_key=row.event_key,
                period_ended=row.period_ended,
                fiscal_period=row.fiscal_period or f"period_ended:{row.period_ended}",
                event_date=row.period_ended,
                result_announcement_datetime=row.result_announcement_datetime,
                announced_at=row.result_announcement_datetime,
                announcement_date=row.announcement_date,
                announcement_time=row.announcement_time,
                event_status="result_announced" if pd.notna(row.result_announcement_datetime) else "identified",
                document_count=int(row.document_count),
                has_financial_results=bool(row.has_financial_results),
                has_media_release=bool(row.has_media_release),
                has_earnings_call=bool(row.has_earnings_call),
                has_transcript=bool(row.has_transcript),
                first_document_datetime=row.first_document_datetime,
                last_document_datetime=row.last_document_datetime,
                source="earnings_event_pipeline",
            )
            if existing is None:
                existing = EarningsEvent(**values)
                db.add(existing)
                db.flush()
                inserted_events += 1
            else:
                for key, value in values.items():
                    setattr(existing, key, value)

        db.commit()

        for row in documents.itertuples(index=False):
            event = db.scalar(select(EarningsEvent).where(EarningsEvent.event_key == row.event_key))
            stock = stocks.get(row.symbol)
            if event is None or stock is None:
                raise ValueError(f"Missing event/stock for document {row.document_id}")
            existing = db.scalar(select(EarningsDocument).where(EarningsDocument.document_id == row.document_id))
            values = dict(
                event_id=event.id,
                event_key=row.event_key,
                stock_id=stock.id,
                symbol=row.symbol,
                period_ended=row.period_ended,
                announcement_datetime=row.announcement_datetime,
                document_type=row.document_type,
                document_subtype=row.document_subtype,
                announcement_text=row.announcement_text,
                filing_url=row.filing_url,
                is_primary_result=bool(row.is_primary_result),
                is_followup_document=bool(row.is_followup_document),
                source=row.source,
            )
            if existing is None:
                db.add(EarningsDocument(document_id=row.document_id, **values))
                inserted_documents += 1
            else:
                for key, value in values.items():
                    setattr(existing, key, value)
        db.commit()
        return inserted_events, inserted_documents
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical EarningsOS events/documents from source filings.")
    parser.add_argument("--input", required=True, help="Source parquet/csv containing earnings filings/documents.")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--persist", action="store_true", help="Persist canonical events/documents into PostgreSQL.")
    args = parser.parse_args()

    path = Path(args.input)
    if path.suffix.lower() == ".parquet":
        source = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        source = pd.read_csv(path)
    else:
        raise ValueError("Input must be .parquet or .csv")

    events, documents, quality = build_event_documents(source)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output / "earnings_events.parquet", index=False)
    documents.to_parquet(output / "earnings_documents.parquet", index=False)
    quality.to_csv(output / "earnings_event_quality_report.csv", index=False)

    print(f"source rows: {len(source)}")
    print(f"documents after deduplication: {len(documents)}")
    print(f"canonical events: {len(events)}")
    print(f"missing primary results: {(events['result_announcement_datetime'].isna()).sum() if len(events) else 0}")
    print(f"quality flags: {quality['quality_flag'].value_counts().to_dict() if len(quality) else {}}")

    if args.persist:
        inserted_events, inserted_documents = persist_events(events, documents)
        print(f"persisted events: {inserted_events}")
        print(f"persisted documents: {inserted_documents}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
