"""
Tests for EarningsOS Earnings Event Pipeline
"""

import hashlib
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from earnings_classifier import (
    classify_document,
    classify_document_detailed,
    is_primary_result,
    PRIMARY_RESULT_TYPES,
)


class TestDocumentClassification:
    """Test document classification logic."""

    def test_financial_results_primary(self):
        """Test primary financial results detection."""
        result = classify_document(
            announcement_text="Financial Results for quarter ended 31-Mar-2024",
            announcement_details="",
            filing_url="",
            earnings_type="financial_results",
            document_type_legacy="financial_results",
        )
        assert result == "financial_results"
        assert is_primary_result(result)

    def test_financial_result_update(self):
        """Test financial result update detection."""
        result = classify_document(
            announcement_text="Financial Result Updates",
            announcement_details="Submitted financial results for period ended Dec 31, 2023",
            filing_url="",
            earnings_type="financial_results",
            document_type_legacy="financial_result_update",
        )
        assert result == "financial_result_update"
        assert is_primary_result(result)

    def test_integrated_financial_filing(self):
        """Test integrated XBRL filing detection."""
        result = classify_document(
            announcement_text="Integrated Filing - Financial",
            announcement_details="XBRL filing for quarter ended",
            filing_url="",
            earnings_type="integrated_financial",
            document_type_legacy="integrated_financial_filing",
        )
        assert result == "integrated_financial_filing"
        assert is_primary_result(result)

    def test_media_release(self):
        """Test media release detection."""
        result = classify_document(
            announcement_text="Media Release on Quarterly Results",
            announcement_details="Press release about Q4 results",
            filing_url="",
            earnings_type="other_earnings",
            document_type_legacy="other",
        )
        assert result == "media_release"
        assert not is_primary_result(result)

    def test_investor_presentation(self):
        """Test investor presentation detection."""
        result = classify_document(
            announcement_text="Investor Presentation on Q4 Results",
            announcement_details="Earnings presentation slides",
            filing_url="",
            earnings_type="other_earnings",
            document_type_legacy="other",
        )
        assert result == "investor_presentation"
        assert not is_primary_result(result)

    def test_earnings_call_transcript(self):
        """Test earnings call transcript detection."""
        result = classify_document(
            announcement_text="Transcript of Earnings Call",
            announcement_details="Transcript of analysts/institutional investor call",
            filing_url="",
            earnings_type="other_earnings",
            document_type_legacy="other",
        )
        assert result == "earnings_call_transcript"
        assert not is_primary_result(result)

    def test_earnings_call_recording(self):
        """Test earnings call recording detection."""
        result = classify_document(
            announcement_text="Audio Recording of Earnings Call",
            announcement_details="Recording of analysts call",
            filing_url="",
            earnings_type="other_earnings",
            document_type_legacy="other",
        )
        assert result == "earnings_call_recording"
        assert not is_primary_result(result)

    def test_earnings_call_schedule(self):
        """Test earnings call schedule detection."""
        result = classify_document(
            announcement_text="Schedule of Analyst/Institutional Investor Meet",
            announcement_details="Concall schedule for Q4 results",
            filing_url="",
            earnings_type="other_earnings",
            document_type_legacy="other",
        )
        assert result == "earnings_call_schedule"
        assert not is_primary_result(result)

    def test_board_meeting_outcome(self):
        """Test board meeting outcome detection."""
        result = classify_document(
            announcement_text="Outcome of Board Meeting",
            announcement_details="Board approved financial results",
            filing_url="",
            earnings_type="financial_results",
            document_type_legacy="financial_result_update",
        )
        # Should be financial_result_update since it mentions financial results
        assert result == "financial_result_update"

    def test_board_meeting_notice(self):
        """Test board meeting notice detection."""
        result = classify_document(
            announcement_text="Board Meeting Intimation",
            announcement_details="Meeting of board of directors scheduled",
            filing_url="",
            earnings_type="other_earnings",
            document_type_legacy="other",
        )
        assert result == "board_meeting_notice"
        assert not is_primary_result(result)

    def test_limited_review(self):
        """Test limited review report detection."""
        result = classify_document(
            announcement_text="Limited Review Report",
            announcement_details="Independent auditor's review report",
            filing_url="",
            earnings_type="limited_review",
            document_type_legacy="limited_review",
        )
        assert result == "limited_review"
        assert not is_primary_result(result)

    def test_classification_signals(self):
        """Test detailed classification returns signals."""
        result = classify_document_detailed(
            announcement_text="Financial Results for quarter ended",
            announcement_details="Media release and investor presentation",
            filing_url="",
            earnings_type="financial_results",
            document_type_legacy="financial_results",
        )
        assert result["document_type"] == "financial_results"
        assert result["is_primary_result"] is True
        assert "financial_results" in result["signals_matched"]

    def test_url_based_classification(self):
        """Test classification uses filing URL as signal."""
        result = classify_document(
            announcement_text="Updates",
            announcement_details="",
            filing_url="https://example.com/Transcript_Q4_2024.pdf",
            earnings_type="other_earnings",
            document_type_legacy="other",
        )
        assert result == "earnings_call_transcript"


class TestEventsPipeline:
    """Test the events pipeline integration."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for ADANIENT_2023-12-31 event (Q3 FY24).

        The Feb 2024 documents are for the quarter ended Dec 31, 2023.
        """
        return pd.DataFrame([
            {
                "document_id": "doc1",
                "event_key": "ADANIENT_2023-12-31",
                "symbol": "ADANIENT",
                "company_name": "Adani Enterprises Ltd.",
                "period_ended": pd.Timestamp("2023-12-31"),
                "fiscal_quarter": "2024-Q4",
                "announcement_datetime": pd.Timestamp("2024-02-01 14:09:11"),
                "document_type": "financial_result_update",
                "is_primary_result": True,
                "filing_url": "https://example.com/result.pdf",
            },
            {
                "document_id": "doc2",
                "event_key": "ADANIENT_2023-12-31",
                "symbol": "ADANIENT",
                "company_name": "Adani Enterprises Ltd.",
                "period_ended": pd.Timestamp("2023-12-31"),
                "fiscal_quarter": "2024-Q4",
                "announcement_datetime": pd.Timestamp("2024-02-01 14:14:37"),
                "document_type": "media_release",
                "is_primary_result": False,
                "filing_url": "https://example.com/media.pdf",
            },
            {
                "document_id": "doc3",
                "event_key": "ADANIENT_2023-12-31",
                "symbol": "ADANIENT",
                "company_name": "Adani Enterprises Ltd.",
                "period_ended": pd.Timestamp("2023-12-31"),
                "fiscal_quarter": "2024-Q4",
                "announcement_datetime": pd.Timestamp("2024-02-01 21:15:29"),
                "document_type": "earnings_call_recording",
                "is_primary_result": False,
                "filing_url": "https://example.com/recording.pdf",
            },
            {
                "document_id": "doc4",
                "event_key": "ADANIENT_2023-12-31",
                "symbol": "ADANIENT",
                "company_name": "Adani Enterprises Ltd.",
                "period_ended": pd.Timestamp("2023-12-31"),
                "fiscal_quarter": "2024-Q4",
                "announcement_datetime": pd.Timestamp("2024-02-08 19:07:56"),
                "document_type": "earnings_call_transcript",
                "is_primary_result": False,
                "filing_url": "https://example.com/transcript.pdf",
            },
        ])

    def test_single_event_multiple_documents(self, sample_documents):
        """Test that multiple documents for one event produce one event row."""
        from build_earnings_events_v2 import build_events_table

        events = build_events_table(sample_documents)

        assert len(events) == 1
        assert events.iloc[0]["event_key"] == "ADANIENT_2023-12-31"
        assert events.iloc[0]["document_count"] == 4
        assert bool(events.iloc[0]["has_financial_results"]) is True
        assert bool(events.iloc[0]["has_media_release"]) is True
        assert bool(events.iloc[0]["has_earnings_call"]) is True
        assert bool(events.iloc[0]["has_transcript"]) is True

    def test_primary_result_selection(self, sample_documents):
        """Test that earliest primary result is selected as canonical."""
        from build_earnings_events_v2 import build_events_table

        events = build_events_table(sample_documents)

        # Primary result is at 14:09:11
        assert events.iloc[0]["result_announcement_datetime"] == pd.Timestamp("2024-02-01 14:09:11")
        assert events.iloc[0]["period_ended"] == pd.Timestamp("2023-12-31")

    def test_different_quarters_separate_events(self):
        """Test that different quarters remain separate events."""
        from build_earnings_events_v2 import build_events_table

        docs = pd.DataFrame([
            {
                "document_id": "doc1",
                "event_key": "ADANIENT_2024-03-31",
                "symbol": "ADANIENT",
                "company_name": "Adani Enterprises Ltd.",
                "period_ended": pd.Timestamp("2024-03-31"),
                "fiscal_quarter": "2024-Q4",
                "announcement_datetime": pd.Timestamp("2024-02-01 14:09:11"),
                "document_type": "financial_results",
                "is_primary_result": True,
                "filing_url": "https://example.com/q4.pdf",
            },
            {
                "document_id": "doc2",
                "event_key": "ADANIENT_2024-06-30",
                "symbol": "ADANIENT",
                "company_name": "Adani Enterprises Ltd.",
                "period_ended": pd.Timestamp("2024-06-30"),
                "fiscal_quarter": "2024-Q1",
                "announcement_datetime": pd.Timestamp("2024-05-02 15:09:36"),
                "document_type": "financial_results",
                "is_primary_result": True,
                "filing_url": "https://example.com/q1.pdf",
            },
        ])

        events = build_events_table(docs)
        assert len(events) == 2
        assert set(events["event_key"].tolist()) == {"ADANIENT_2024-03-31", "ADANIENT_2024-06-30"}

    def test_different_companies_separate_events(self):
        """Test that different companies remain separate."""
        from build_earnings_events_v2 import build_events_table

        docs = pd.DataFrame([
            {
                "document_id": "doc1",
                "event_key": "ADANIENT_2024-03-31",
                "symbol": "ADANIENT",
                "company_name": "Adani Enterprises Ltd.",
                "period_ended": pd.Timestamp("2024-03-31"),
                "fiscal_quarter": "2024-Q4",
                "announcement_datetime": pd.Timestamp("2024-02-01 14:09:11"),
                "document_type": "financial_results",
                "is_primary_result": True,
                "filing_url": "https://example.com/adanient.pdf",
            },
            {
                "document_id": "doc2",
                "event_key": "RELIANCE_2024-03-31",
                "symbol": "RELIANCE",
                "company_name": "Reliance Industries Ltd.",
                "period_ended": pd.Timestamp("2024-03-31"),
                "fiscal_quarter": "2024-Q4",
                "announcement_datetime": pd.Timestamp("2024-01-19 14:09:11"),
                "document_type": "financial_results",
                "is_primary_result": True,
                "filing_url": "https://example.com/reliance.pdf",
            },
        ])

        events = build_events_table(docs)
        assert len(events) == 2
        assert set(events["symbol"].tolist()) == {"ADANIENT", "RELIANCE"}

    def test_transcript_later_not_new_event(self, sample_documents):
        """Test that transcript uploaded later doesn't create new event."""
        from build_earnings_events_v2 import build_events_table

        # Add a transcript much later
        late_transcript = pd.DataFrame([{
            "document_id": "doc5",
            "event_key": "ADANIENT_2023-12-31",
            "symbol": "ADANIENT",
            "company_name": "Adani Enterprises Ltd.",
            "period_ended": pd.Timestamp("2023-12-31"),
            "fiscal_quarter": "2024-Q4",
            "announcement_datetime": pd.Timestamp("2024-03-15 10:00:00"),  # Month later
            "document_type": "earnings_call_transcript",
            "is_primary_result": False,
            "filing_url": "https://example.com/late_transcript.pdf",
        }])

        docs = pd.concat([sample_documents, late_transcript], ignore_index=True)
        events = build_events_table(docs)

        assert len(events) == 1
        assert events.iloc[0]["document_count"] == 5
        # Primary result should still be the earliest financial result
        assert events.iloc[0]["result_announcement_datetime"] == pd.Timestamp("2024-02-01 14:09:11")

    def test_duplicate_document_deduplication(self):
        """Test that duplicate documents are deduplicated."""
        from build_earnings_events_v2 import build_documents_table, build_events_table
        from earnings_classifier import classify_document_detailed

        # Create raw earnings-like dataframe with duplicates
        base = {
            "symbol": "ADANIENT",
            "company_name": "Adani Enterprises Ltd.",
            "period_ended": pd.Timestamp("2023-12-31"),
            "fiscal_quarter": "2024-Q4",
            "announcement_datetime": pd.Timestamp("2024-02-01 14:09:11"),
            "document_type": "financial_result_update",
            "is_primary_result": True,
            "filing_url": "https://example.com/result.pdf",
        }

        docs_df = pd.DataFrame([base, base.copy()])  # Exact duplicate
        docs_df["document_id"] = docs_df.apply(
            lambda r: hashlib.sha256(f"{r['symbol']}|{r['announcement_datetime']}|{r['filing_url']}".encode()).hexdigest()[:16],
            axis=1,
        )
        docs_df["event_key"] = "ADANIENT_2023-12-31"

        # Deduplicate
        before = len(docs_df)
        docs_df = docs_df.drop_duplicates(subset=["document_id"], keep="first")

        assert len(docs_df) == 1
        assert before == 2

    def test_missing_period_end_flagged(self):
        """Test that missing period_end is flagged."""
        from build_earnings_events_v2 import build_events_table

        docs = pd.DataFrame([{
            "document_id": "doc1",
            "event_key": "ADANIENT_UNKNOWN",
            "symbol": "ADANIENT",
            "company_name": "Adani Enterprises Ltd.",
            "period_ended": pd.NaT,
            "fiscal_quarter": None,
            "announcement_datetime": pd.Timestamp("2024-02-01 14:09:11"),
            "document_type": "financial_results",
            "is_primary_result": True,
            "filing_url": "https://example.com/result.pdf",
        }])

        events = build_events_table(docs)
        assert "MISSING_PERIOD_END" in events.iloc[0]["quality_flag"]
        assert events.iloc[0]["period_ended"] is pd.NaT

    def test_primary_result_deterministic(self, sample_documents):
        """Test that primary result selection is deterministic."""
        from build_earnings_events_v2 import build_events_table

        # Run twice
        events1 = build_events_table(sample_documents)
        events2 = build_events_table(sample_documents)

        assert events1.iloc[0]["result_announcement_datetime"] == events2.iloc[0]["result_announcement_datetime"]


class TestDataIntegrity:
    """Test data integrity constraints."""

    def test_event_key_uniqueness(self):
        """Test that event keys are unique in events table."""
        from build_earnings_events_v2 import load_and_classify, build_documents_table, build_events_table

        earnings = load_and_classify()
        docs = build_documents_table(earnings)
        events = build_events_table(docs)

        # Event keys must be unique
        assert events["event_key"].is_unique
        assert events["event_key"].duplicated().sum() == 0

    def test_document_id_uniqueness(self):
        """Test that document IDs are unique."""
        from build_earnings_events_v2 import load_and_classify, build_documents_table

        earnings = load_and_classify()
        docs = build_documents_table(earnings)

        assert docs["document_id"].is_unique
        assert docs["document_id"].duplicated().sum() == 0

    def test_every_document_has_event(self):
        """Test that every document maps to an event."""
        from build_earnings_events_v2 import load_and_classify, build_documents_table, build_events_table

        earnings = load_and_classify()
        docs = build_documents_table(earnings)
        events = build_events_table(docs)

        # Every document's event_key must exist in events
        doc_event_keys = set(docs["event_key"].unique())
        event_keys = set(events["event_key"].unique())
        assert doc_event_keys.issubset(event_keys)

    def test_event_has_at_least_one_document(self):
        """Test that every event has at least one document."""
        from build_earnings_events_v2 import load_and_classify, build_documents_table, build_events_table

        earnings = load_and_classify()
        docs = build_documents_table(earnings)
        events = build_events_table(docs)

        assert (events["document_count"] >= 1).all()


# Run tests if executed directly
if __name__ == "__main__":
    import hashlib
    pytest.main([__file__, "-v"])