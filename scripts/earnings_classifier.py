"""
Earnings Document Classification Module

Provides deterministic classification of NSE corporate announcements
into structured document types for the EarningsOS pipeline.
"""

import re
from typing import Optional


# Primary result document types (contain actual financial results)
PRIMARY_RESULT_TYPES = {
    "financial_results",
    "financial_result_update",
    "integrated_financial_filing",
}

# Document type categories
DOCUMENT_CATEGORIES = {
    "financial_results": "Primary financial results filing",
    "financial_result_update": "Financial results update/outcome",
    "integrated_financial_filing": "Integrated XBRL financial filing",
    "media_release": "Media/press release about results",
    "investor_presentation": "Investor presentation/slides",
    "earnings_call_recording": "Earnings call audio/video recording",
    "earnings_call_transcript": "Earnings call transcript",
    "earnings_call_schedule": "Earnings call schedule/announcement",
    "board_meeting_notice": "Board meeting notice",
    "board_meeting_outcome": "Board meeting outcome",
    "limited_review": "Limited review report",
    "other_filing": "Other follow-up filing",
    "unclassified": "Could not classify",
}


def classify_document(
    announcement_text: str,
    announcement_details: str,
    filing_url: str,
    earnings_type: str,
    document_type_legacy: str,
) -> str:
    """
    Classify a document into a structured category.

    Uses multiple signals: announcement text, attachment details,
    filing URL filename, and legacy classification fields.

    Priority: Check announcement_text first for primary results,
    then combined text for follow-up types.
    """
    text = " ".join(
        filter(
            None,
            [
                str(announcement_text).lower(),
                str(announcement_details).lower(),
                str(filing_url).lower(),
            ],
        )
    )
    announcement_text_lower = str(announcement_text).lower()

    # Check for primary result types in announcement_text FIRST (highest priority)
    # Integrated Financial Filing (XBRL) - very specific
    if _is_integrated_financial(announcement_text_lower):
        return "integrated_financial_filing"

    # Financial Result Update / Outcome (primary)
    if _is_financial_result_update(announcement_text_lower):
        return "financial_result_update"

    # Financial Results (primary) - check announcement_text first
    if _is_financial_results(announcement_text_lower):
        return "financial_results"

    # Now check combined text for follow-up types

    # Trading Window - not a financial result
    if _is_trading_window(text):
        return "other_filing"

    # Earnings Call Transcript - very specific
    if _is_earnings_call_transcript(text):
        return "earnings_call_transcript"

    # Earnings Call Recording - very specific
    if _is_earnings_call_recording(text):
        return "earnings_call_recording"

    # Investor Presentation - specific
    if _is_investor_presentation(text):
        return "investor_presentation"

    # Media Release - specific
    if _is_media_release(text):
        return "media_release"

    # Earnings Call Schedule
    if _is_earnings_call_schedule(text):
        return "earnings_call_schedule"

    # Board Meeting Outcome
    if _is_board_meeting_outcome(text):
        return "board_meeting_outcome"

    # Board Meeting Notice
    if _is_board_meeting_notice(text):
        return "board_meeting_notice"

    # Limited Review Report
    if _is_limited_review(text):
        return "limited_review"

    # Option to submit financial results - this is a board meeting notice, not results
    if _is_option_to_submit(text):
        return "board_meeting_notice"

    # Follow-up result filings (machine readable copies, legible format submissions)
    # These are not primary results but follow-ups to already-filed results
    if _is_followup_result_filing(text):
        return "other_filing"

    # Fallback to legacy classification for backwards compatibility
    legacy_map = {
        "financial_results": "financial_results",
        "financial_result_update": "financial_result_update",
        "integrated_financial_filing": "integrated_financial_filing",
        "results_update": "financial_result_update",
        "quarterly_results": "financial_results",
        "annual_results": "financial_results",
        "limited_review": "limited_review",
        "other_earnings": "other_filing",
    }

    return legacy_map.get(document_type_legacy, "unclassified")


def _is_financial_results(text: str) -> bool:
    """Detect primary financial results filing."""
    # Must be the main results filing, not a follow-up
    # Exclude if it's clearly a media/press release about results
    if re.search(r"\b(media|press)\s+release\b", text, re.IGNORECASE):
        return False
    
    # Exclude follow-up filings (machine readable copies, legible formats, etc.)
    if re.search(r"\b(machine\s+readable|legible\s+copy|readable\s+format|xbrl\s+filing|filing\s+of.*financial\s+results)\b", text, re.IGNORECASE):
        return False
    
    # Exclude "option to submit" / "opted not to submit" - these are board meeting notices
    if re.search(r"option\s+to\s+submit.*financial\s+results?", text, re.IGNORECASE):
        return False
    if re.search(r"opted\s+not\s+to\s+submit.*financial\s+results?", text, re.IGNORECASE):
        return False
    if re.search(r"intimation.*option.*submit.*financial\s+results?", text, re.IGNORECASE):
        return False
    
    # Exclude clarifications and replies to clarifications on financial results
    if re.search(r"clarification.*financial\s+results?", text, re.IGNORECASE):
        return False
    if re.search(r"reply\s+to\s+clarification.*financial\s+results?", text, re.IGNORECASE):
        return False
    
    # Exclude newspaper publications/advertisements/extracts of financial results
    if re.search(r"(copy\s+of\s+newspaper|newspaper\s+(advertisement|publication)|extract\s+of\s+financial\s+results)", text, re.IGNORECASE):
        return False
    
    # Exclude "General updates" / "Updates" that are follow-ups
    if re.search(r"general\s+updates?", text, re.IGNORECASE):
        return False
    if re.search(r"^updates?$", text.strip(), re.IGNORECASE):
        return False

    patterns = [
        r"\bfinancial\s+results?\b(?!\s+(update|submission|recording|transcript|presentation|media))",
        r"^financial\s+results?$",
        r"quarterly\s+results?\b",
        r"audited\s+financial\s+results",
        r"unaudited\s+financial\s+results",
        r"consolidated\s+financial\s+results\b",
        r"standalone\s+financial\s+results\b",
        r"results?\s+for\s+the\s+(quarter|year)\b",
        r"results?\s+of\s+the\s+(quarter|year)\b",
        r"^outcome\s+of\s+board\s+meeting.*financial\s+results",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_financial_result_update(text: str) -> bool:
    """Detect financial result update/outcome filings (the actual results submission)."""
    # Exclude generic board meeting outcomes that aren't about financial results
    if re.search(r"outcome\s+of\s+board\s+meeting", text, re.IGNORECASE):
        if not re.search(r"financial\s+result", text, re.IGNORECASE):
            return False

    patterns = [
        r"financial\s+result\s+updates?",
        r"^results?\s+update$",
        r"^outcome\s+of\s+board\s+meeting$",
        r"^board\s+meeting.*outcome$",
        r"submission\s+of.*financial\s+results",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_integrated_financial(text: str) -> bool:
    """Detect integrated XBRL financial filing."""
    patterns = [
        r"integrated\s+filing",
        r"xbrl.*financial",
        r"financial.*xbrl",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_media_release(text: str) -> bool:
    """Detect media/press release."""
    patterns = [
        r"media\s+release",
        r"media\s+statement",
        r"press\s+release",
        r"press\s+note",
        r"submission\s+of.*media\s+release",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_investor_presentation(text: str) -> bool:
    """Detect investor presentation."""
    patterns = [
        r"investor\s+presentation",
        r"presentation\s+on.*results",
        r"earnings\s+presentation",
        r"results\s+presentation",
        r"analyst\s+presentation",
        r"submission\s+of.*investor\s+presentation",
        r"investor\s+presentation.*financial\s+results",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_earnings_call_transcript(text: str) -> bool:
    """Detect earnings call transcript."""
    patterns = [
        r"transcript",
        r"earnings\s+call\s+transcript",
        r"concall\s+transcript",
        r"conference\s+call\s+transcript",
        r"transcript\s+of.*earnings",
        r"transcript\s+of.*call",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_earnings_call_recording(text: str) -> bool:
    """Detect earnings call audio/video recording."""
    patterns = [
        r"audio\s+recording",
        r"recording\s+of.*call",
        r"earnings\s+call\s+recording",
        r"concall\s+recording",
        r"conference\s+call\s+recording",
        r"analysts?/institutional\s+investor.*call.*recording",
        r"recording.*analysts?.*investors?",
        r"audio.*earnings\s+call",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_earnings_call_schedule(text: str) -> bool:
    """Detect earnings call schedule/announcement."""
    patterns = [
        r"schedule\s+of.*call",
        r"schedule\s+of.*meet",
        r"earnings\s+call\s+schedule",
        r"concall\s+schedule",
        r"conference\s+call\s+schedule",
        r"intimation\s+for.*call",
        r"analysts?/institutional\s+investor.*meet",
        r"schedule.*analyst.*meet",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_board_meeting_outcome(text: str) -> bool:
    """Detect board meeting outcome (non-results)."""
    patterns = [
        r"outcome\s+of\s+board\s+meeting",
        r"board\s+meeting.*outcome",
        r"board\s+meeting.*held",
    ]
    # Exclude if it's about financial results (handled above)
    if any(re.search(p, text, re.IGNORECASE) for p in patterns):
        if not _is_financial_results(text) and not _is_financial_result_update(text):
            return True
    return False


def _is_board_meeting_notice(text: str) -> bool:
    """Detect board meeting notice/intimation."""
    patterns = [
        r"board\s+meeting\s+intimation",
        r"intimation.*board\s+meeting",
        r"notice.*board\s+meeting",
        r"board\s+meeting\s+on",
        r"meeting\s+of\s+board",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_limited_review(text: str) -> bool:
    """Detect limited review report."""
    patterns = [
        r"limited\s+review",
        r"review\s+report",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_followup_result_filing(text: str) -> bool:
    """Detect follow-up result filings (machine readable copies, legible format submissions).
    
    These are follow-up documents filed after the primary results, such as:
    - Machine readable form / XBRL filing of already-announced results
    - Legible copy of financial results
    - Readable format submissions
    - Additional information / updates on already-filed financial results
    - Newspaper publications / advertisements of financial results
    - Clarifications / replies to clarifications on financial results
    - Generic "Updates" that mention financial results (not "Financial Result Updates")
    """
    patterns = [
        r"machine\s+readable",
        r"legible\s+copy",
        r"readable\s+format",
        r"xbrl\s+filing",
        r"filing\s+of.*financial\s+results",
        r"additional\s+information\s+on.*financial\s+results?",
        r"update\s+on.*financial\s+results?",
        r"corrigendum.*financial\s+results?",
        r"revised.*financial\s+results?",
        r"copy\s+of\s+newspaper\s+publication",
        r"newspaper\s+advertisement",
        r"newspaper\s+publication",
        r"extract\s+of\s+financial\s+results",
        r"clarification.*financial\s+results?",
        r"reply\s+to\s+clarification.*financial\s+results?",
        r"general\s+updates?",
        r"^updates?$",
        r"updates?.*financial\s+results?",
    ]
    # Must also mention financial results to avoid false positives
    if not re.search(r"financial\s+result", text, re.IGNORECASE):
        return False
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_trading_window(text: str) -> bool:
    """Detect trading window closure/intimation announcements."""
    patterns = [
        r"trading\s+window",
        r"closure\s+of\s+trading\s+window",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_option_to_submit(text: str) -> bool:
    """Detect option to submit / opted not to submit financial results announcements."""
    patterns = [
        r"option\s+to\s+submit.*financial\s+results?",
        r"opted\s+not\s+to\s+submit.*financial\s+results?",
        r"intimation.*option.*submit.*financial\s+results?",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def is_primary_result(document_type: str) -> bool:
    """Check if document type represents a primary financial results announcement."""
    return document_type in PRIMARY_RESULT_TYPES


def get_document_category_description(document_type: str) -> str:
    """Get human-readable description of document type."""
    return DOCUMENT_CATEGORIES.get(document_type, "Unknown document type")


def classify_document_detailed(
    announcement_text: str,
    announcement_details: str,
    filing_url: str,
    earnings_type: str,
    document_type_legacy: str,
) -> dict:
    """
    Classify document with detailed metadata.

    Returns dict with:
    - document_type: primary classification
    - is_primary_result: bool
    - category_description: str
    - signals_matched: list of matched pattern categories
    """
    doc_type = classify_document(
        announcement_text,
        announcement_details,
        filing_url,
        earnings_type,
        document_type_legacy,
    )

    signals = []
    announcement_text_lower = str(announcement_text).lower()
    text = " ".join(
        filter(
            None,
            [
                announcement_text_lower,
                str(announcement_details).lower(),
                str(filing_url).lower(),
            ],
        )
    )

    # Check announcement_text first for primary results (matching classify_document logic)
    if _is_financial_results(announcement_text_lower):
        signals.append("financial_results")
    if _is_financial_result_update(announcement_text_lower):
        signals.append("financial_result_update")
    if _is_integrated_financial(announcement_text_lower):
        signals.append("integrated_financial")

    # Check combined text for follow-up types
    if _is_media_release(text):
        signals.append("media_release")
    if _is_investor_presentation(text):
        signals.append("investor_presentation")
    if _is_earnings_call_transcript(text):
        signals.append("earnings_call_transcript")
    if _is_earnings_call_recording(text):
        signals.append("earnings_call_recording")
    if _is_earnings_call_schedule(text):
        signals.append("earnings_call_schedule")
    if _is_board_meeting_outcome(text):
        signals.append("board_meeting_outcome")
    if _is_board_meeting_notice(text):
        signals.append("board_meeting_notice")
    if _is_limited_review(text):
        signals.append("limited_review")

    return {
        "document_type": doc_type,
        "is_primary_result": is_primary_result(doc_type),
        "category_description": get_document_category_description(doc_type),
        "signals_matched": signals,
    }