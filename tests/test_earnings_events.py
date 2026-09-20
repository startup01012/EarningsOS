from datetime import datetime, timezone

import pandas as pd

from scripts.build_earnings_events import build_event_documents, classify_document


def test_document_classification_is_deterministic():
    assert classify_document("financial_results", "Financial Results for the quarter ended 31/03/2024", "").document_type == "financial_results"
    assert classify_document("transcript", "earnings call transcript", "").document_type == "earnings_call_transcript"
    assert classify_document("media release", "press release", "").document_type == "media_release"


def test_multiple_documents_make_one_event():
    source = pd.DataFrame(
        [
            {
                "symbol": "ADANIENT",
                "period_ended": "2024-03-31",
                "announcement_datetime": "2024-02-01T14:09:11+05:30",
                "document_type": "financial_result_update",
                "announcement_text": "Financial Results for the quarter ended 31/03/2024",
                "filing_url": "https://example/result-1",
            },
            {
                "symbol": "ADANIENT",
                "period_ended": "2024-03-31",
                "announcement_datetime": "2024-02-01T14:14:37+05:30",
                "document_type": "financial_results",
                "announcement_text": "Financial Results for the quarter ended 31/03/2024",
                "filing_url": "https://example/result-2",
            },
            {
                "symbol": "ADANIENT",
                "period_ended": "2024-03-31",
                "announcement_datetime": "2024-02-08T19:07:56+05:30",
                "document_type": "transcript",
                "announcement_text": "Earnings call transcript",
                "filing_url": "https://example/transcript",
            },
        ]
    )
    events, documents, quality = build_event_documents(source)
    assert len(events) == 1
    assert events.iloc[0].event_key == "ADANIENT_2024-03-31"
    assert len(documents) == 3
    assert events.iloc[0].result_announcement_datetime == datetime(2024, 2, 1, 8, 39, 11, tzinfo=timezone.utc)


def test_non_consolidated_is_not_consolidated_by_substring_matching():
    source = pd.DataFrame(
        [
            {
                "symbol": "RELIANCE",
                "period_ended": "2024-06-30",
                "announcement_datetime": "2024-07-20T10:00:00+05:30",
                "document_type": "financial_results",
                "consolidated_status": "Non-Consolidated",
                "announcement_text": "Financial Results for the quarter ended 30/06/2024",
                "filing_url": "https://example/non-consolidated",
            },
        ]
    )
    events, documents, _ = build_event_documents(source)
    assert len(events) == 1
    assert len(documents) == 1


def test_missing_period_is_flagged_not_invented():
    source = pd.DataFrame(
        [
            {
                "symbol": "INFY",
                "announcement_datetime": "2024-07-18T10:00:00+05:30",
                "document_type": "media_release",
                "announcement_text": "Investor media release",
                "filing_url": "https://example/media",
            },
        ]
    )
    events, documents, quality = build_event_documents(source)
    assert events.empty
    assert documents.empty
    assert quality.iloc[0].quality_flag == "MISSING_PERIOD_END"


def test_duplicate_ingestion_is_removed_but_distinct_urls_remain():
    row = {
        "symbol": "TCS",
        "period_ended": "2024-03-31",
        "announcement_datetime": "2024-04-12T10:00:00+05:30",
        "document_type": "financial_results",
        "announcement_text": "Financial Results for the quarter ended 31/03/2024",
        "filing_url": "https://example/results",
    }
    source = pd.DataFrame([row, row, {**row, "filing_url": "https://example/results-copy"}])
    events, documents, _ = build_event_documents(source)
    assert len(events) == 1
    assert len(documents) == 2
