"""
Build Earnings Events and Documents Tables

Creates the canonical earnings events table (one row per symbol + period_ended)
and the earnings documents table (one row per filing/document).
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from earnings_classifier import (
    classify_document,
    classify_document_detailed,
    is_primary_result,
)

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("data/raw/earnings")
REFERENCE_FILE = Path("data/reference/nifty50_clean.csv")

EVENTS_OUTPUT = Path("data/processed/earnings_events.parquet")
DOCUMENTS_OUTPUT = Path("data/processed/earnings_documents.parquet")
QUALITY_REPORT_OUTPUT = Path("data/processed/earnings_event_quality_report.csv")

for p in [EVENTS_OUTPUT, DOCUMENTS_OUTPUT, QUALITY_REPORT_OUTPUT]:
    p.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_document_id(symbol: str, announcement_datetime: pd.Timestamp, filing_url: str) -> str:
    """Generate a stable, deterministic document ID."""
    key = f"{symbol}|{announcement_datetime.isoformat()}|{filing_url}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _reassign_followup_periods(earnings: pd.DataFrame) -> pd.DataFrame:
    """Reassign period_ended for follow-up documents AND primary results that used fallback.

    For documents classified as follow-up types (board_meeting_outcome, board_meeting_notice,
    earnings_call_recording, earnings_call_transcript, investor_presentation, media_release, other_filing)
    that don't have an explicit period_ended extracted, OR have a period_ended that is
    suspiciously far in the future (>90 days after announcement), find the nearest primary result
    document for the same symbol within a 14-day window and adopt its period_ended.

    Also reassign primary results (financial_results, financial_result_update, integrated_financial_filing)
    that used fallback, by finding other primary results with explicit periods on the same day.
    """
    # Follow-up document types that should be grouped with primary results
    followup_types = {
        "board_meeting_outcome",
        "board_meeting_notice",
        "earnings_call_recording",
        "earnings_call_transcript",
        "investor_presentation",
        "media_release",
        "other_filing",
    }

    # Primary result types
    primary_types = {
        "financial_results",
        "financial_result_update",
        "integrated_financial_filing",
    }

    earnings = earnings.copy()

    # For each symbol, build a lookup of primary result periods by date
    for symbol in earnings["symbol"].unique():
        symbol_mask = earnings["symbol"] == symbol
        symbol_data = earnings[symbol_mask].copy()

        # Primary results with explicit period_ended
        primary_mask = symbol_data["document_type"].isin(primary_types) & ~symbol_data["period_ended_from_fallback"]
        primary_docs = symbol_data[primary_mask][["announcement_datetime", "period_ended"]].copy()

        if primary_docs.empty:
            continue

        # Follow-up docs that used fallback OR have suspicious period (announcement > 90 days before period_end)
        followup_mask = (
            symbol_data["document_type"].isin(followup_types)
            & (
                symbol_data["period_ended_from_fallback"]
                | ((symbol_data["period_ended"] - symbol_data["announcement_datetime"]).dt.days > 90)
            )
        )
        followup_indices = symbol_data[followup_mask].index

        # Primary results that used fallback - reassign based on same-day primary results with explicit period
        primary_fallback_mask = (
            symbol_data["document_type"].isin(primary_types)
            & symbol_data["period_ended_from_fallback"]
        )
        primary_fallback_indices = symbol_data[primary_fallback_mask].index

        all_reassign_indices = followup_indices.union(primary_fallback_indices)

        for idx in all_reassign_indices:
            doc_time = earnings.loc[idx, "announcement_datetime"]
            if pd.isna(doc_time):
                continue

            # Find nearest primary result with explicit period within 14 days
            time_diffs = (primary_docs["announcement_datetime"] - doc_time).abs()
            nearest_idx = time_diffs.idxmin()
            min_diff = time_diffs.min()

            if min_diff <= pd.Timedelta(days=14):
                new_period = primary_docs.loc[nearest_idx, "period_ended"]
                earnings.loc[idx, "period_ended"] = new_period
                earnings.loc[idx, "period_ended_from_fallback"] = False
                logger.debug(f"Reassigned {symbol} doc {idx} to period {new_period.date()}")

    return earnings


def infer_quarter_end(date: pd.Timestamp) -> pd.Timestamp:
    """Infer financial quarter end from announcement date (fallback only)."""
    if pd.isna(date):
        return pd.NaT

    month = date.month
    year = date.year

    if month <= 3:
        return pd.Timestamp(year, 3, 31)
    if month <= 6:
        return pd.Timestamp(year, 6, 30)
    if month <= 9:
        return pd.Timestamp(year, 9, 30)
    return pd.Timestamp(year, 12, 31)


def quarter_label(date: pd.Timestamp) -> Optional[str]:
    """Generate fiscal quarter label (e.g., '2024-Q4')."""
    if pd.isna(date):
        return None

    month = date.month
    if month == 3:
        quarter = "Q4"
    elif month == 6:
        quarter = "Q1"
    elif month == 9:
        quarter = "Q2"
    elif month == 12:
        quarter = "Q3"
    else:
        return None

    return f"{date.year}-{quarter}"


def extract_period_from_text(text: str) -> Optional[pd.Timestamp]:
    """Extract period ended date from announcement text.

    Only matches dates explicitly associated with reporting period keywords.
    Handles multiple formats:
    - DD-MMM-YYYY (e.g., 30-Sep-2015)
    - DDth Month YYYY (e.g., 30th September 2024)
    - Month DD, YYYY (e.g., September 30, 2024)
    - MMM DD, YYYY (e.g., Jun 30, 2025)
    - Month YYYY (e.g., March 2022)
    - Quarter and half year / nine months ended
    - Q3 FY24 format
    - Dot-separated dates (30.06.2026)
    """
    # Preprocess text to fix common spacing issues from PDF extraction
    # Insert space between concatenated words like "quarterand" -> "quarter and"
    text = re.sub(r'(quarter|period|year)(and|or)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(and|or)(quarter|period|year|half|nine)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(for|the|of|in|on|to|as|per)(quarter|period|year|half|nine|financial)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(financial|unaudited|audited|consolidated|standalone)(results|result)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(results|result)(for|the|ended|period|quarter|year)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'(ended)(quarter|period|year|half|nine)', r'\1 \2', text, flags=re.IGNORECASE)
    
    # Month name patterns (full and abbreviated)
    month_full = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    month_abbr = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    month_any = f"(?:{month_full}|{month_abbr})"
    
    # Day pattern (with optional ordinal suffix)
    day_pat = r"\d{1,2}(?:st|nd|rd|th)?"
    
    # Year pattern
    year_pat = r"\d{4}"
    
    # Date patterns
    # DD-MMM-YYYY or DD/MM/YYYY
    date_dash = rf"\d{{1,2}}[-/]{month_abbr}[-/]\d{{4}}"
    # DDth Month YYYY or DD Month YYYY
    date_ordinal = rf"{day_pat}\s+{month_full}\s*,?\s*{year_pat}"
    # DDth MMM YYYY or DD MMM YYYY (day-first with abbreviated month)
    date_ordinal_abbr = rf"{day_pat}\s+{month_abbr}\s*,?\s*{year_pat}"
    # Month DD, YYYY
    date_month_first = rf"{month_full}\s+\d{{1,2}},?\s*{year_pat}"
    # MMM DD, YYYY (abbreviated month)
    date_abbr_first = rf"{month_abbr}\s+\d{{1,2}},?\s*{year_pat}"
    # Month YYYY (no day)
    date_month_year = rf"{month_full}\s+{year_pat}"
    # DD.MM.YYYY or DD.MM.YY
    date_dot = rf"\d{{1,2}}\.\d{{1,2}}\.\d{{2,4}}"

    # Patterns that explicitly reference the reporting period
    patterns = [
        # "period ended 30-Sep-2015", "quarter ended on 30-Jun-2008", "year ended 31-Mar-2008"
        rf"(?:period|quarter|year)\s+ended(?:\s+on)?\s*({date_dash})",
        
        # "period ended 30th September 2024", "quarter ended on 30th June, 2021"
        rf"(?:period|quarter|year)\s+ended(?:\s+on)?\s*({date_ordinal})",
        
        # "for the quarter ended 30th September 2024", "for the quarter and half year ended September 30, 2024"
        rf"for\s+the\s+(?:quarter|year|period)(?:\s+and\s+(?:\w+\s+)?(?:quarter|half\s+year|six\s+months|nine\s+months|year))?\s+ended\s+({date_ordinal})",
        
        # "for the quarter ended September 30, 2024", "for the period ended June 30, 2024"
        rf"for\s+the\s+(?:quarter|year|period)(?:\s+and\s+(?:\w+\s+)?(?:quarter|half\s+year|six\s+months|nine\s+months|year))?\s+ended\s+({date_month_first})",
        
        # "for the quarter ended Jun 30, 2025" (abbreviated month)
        rf"for\s+the\s+(?:quarter|year|period)(?:\s+and\s+(?:\w+\s+)?(?:quarter|half\s+year|six\s+months|nine\s+months|year))?\s+ended\s+({date_abbr_first})",
        
        # "results for the quarter ended 30th September 2024"
        rf"results?\s+for\s+the\s+(?:quarter|year|period)(?:\s+and\s+(?:\w+\s+)?(?:quarter|half\s+year|six\s+months|nine\s+months|year))?\s+ended\s+({date_ordinal})",
        
        # "results for the quarter ended September 30, 2024"
        rf"results?\s+for\s+the\s+(?:quarter|year|period)(?:\s+and\s+(?:\w+\s+)?(?:quarter|half\s+year|six\s+months|nine\s+months|year))?\s+ended\s+({date_month_first})",
        
        # "results for the quarter ended Jun 30, 2025" (abbreviated month)
        rf"results?\s+for\s+the\s+(?:quarter|year|period)(?:\s+and\s+(?:\w+\s+)?(?:quarter|half\s+year|six\s+months|nine\s+months|year))?\s+ended\s+({date_abbr_first})",
        
        # "period ended September 30, 2024" (without "for the" prefix)
        rf"(?:period|quarter|year)\s+ended(?:\s+on)?\s+({date_month_first})",
        
        # "period ended Jun 30, 2025" (abbreviated month, without "for the" prefix)
        rf"(?:period|quarter|year)\s+ended(?:\s+on)?\s+({date_abbr_first})",
        
        # "period ended March 2022" (no day)
        rf"(?:period|quarter|year)\s+ended(?:\s+on)?\s+({date_month_year})",
        
        # "quarter and half year ended September 30, 2024", "quarter and nine months ended December 31, 2023"
        # Also handle "3rd quarter and nine months ended", "1st quarter and half year ended", "quarter and six months ended"
        rf"(?:\d+(?:st|nd|rd|th)\s+)?(?:quarter|period)\s+and\s+(?:half\s+year|six\s+months|nine\s+months|year)\s+ended\s+({date_ordinal})",
        rf"(?:\d+(?:st|nd|rd|th)\s+)?(?:quarter|period)\s+and\s+(?:half\s+year|six\s+months|nine\s+months|year)\s+ended\s+({date_month_first})",
        rf"(?:\d+(?:st|nd|rd|th)\s+)?(?:quarter|period)\s+and\s+(?:half\s+year|six\s+months|nine\s+months|year)\s+ended\s+({date_abbr_first})",
        rf"(?:\d+(?:st|nd|rd|th)\s+)?(?:quarter|period)\s+and\s+(?:half\s+year|six\s+months|nine\s+months|year)\s+ended\s+({date_ordinal_abbr})",
        
        # "unaudited financial results for the quarter ended on 30.06.2026"
        rf"(?:period|quarter|year)\s+ended(?:\s+on)?\s*({date_dot})",
        
        # "financial results for the period ended 30-Jun-2015" (in longer text)
        rf"financial\s+results?\s+(?:for\s+the\s+)?(?:period|quarter|year)\s+ended\s+({date_dash})",
        rf"financial\s+results?\s+(?:for\s+the\s+)?(?:period|quarter|year)\s+ended\s+({date_ordinal})",
        rf"financial\s+results?\s+(?:for\s+the\s+)?(?:period|quarter|year)\s+ended\s+({date_month_first})",
        rf"financial\s+results?\s+(?:for\s+the\s+)?(?:period|quarter|year)\s+ended\s+({date_abbr_first})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                date_str = match.group(1)
                # Clean up ordinal suffixes (st, nd, rd, th)
                date_str = re.sub(r'(\d+)(?:st|nd|rd|th)', r'\1', date_str)
                # Fix missing space after comma (e.g., "March 31,2011" -> "March 31, 2011")
                date_str = re.sub(r',(\d{4})', r', \1', date_str)
                # Handle dot-separated dates (DD.MM.YYYY)
                if re.match(r'\d{1,2}\.\d{1,2}\.\d{4}', date_str):
                    return pd.to_datetime(date_str, format='%d.%m.%Y', errors="coerce")
                elif re.match(r'\d{1,2}\.\d{1,2}\.\d{2}', date_str):
                    return pd.to_datetime(date_str, format='%d.%m.%y', errors="coerce")
                
                # Check if date_str is "Month YYYY" format (no day)
                # If so, convert to quarter-end date (last day of quarter month)
                month_year_match = re.match(rf'^({month_full})\s+(\d{{4}})$', date_str.strip(), re.IGNORECASE)
                if month_year_match:
                    month_name = month_year_match.group(1)
                    year = int(month_year_match.group(2))
                    # Map month to quarter-end date
                    month_lower = month_name.lower()
                    if month_lower in ['march', 'mar']:
                        return pd.Timestamp(year, 3, 31)
                    elif month_lower in ['june', 'jun']:
                        return pd.Timestamp(year, 6, 30)
                    elif month_lower in ['september', 'sep', 'sept']:
                        return pd.Timestamp(year, 9, 30)
                    elif month_lower in ['december', 'dec']:
                        return pd.Timestamp(year, 12, 31)
                    # For non-quarter months, default to month end (fallback)
                    import calendar
                    last_day = calendar.monthrange(year, pd.to_datetime(month_name, format='%B').month)[1]
                    return pd.Timestamp(year, pd.to_datetime(month_name, format='%B').month, last_day)
                
                return pd.to_datetime(date_str, dayfirst=True, errors="coerce")
            except Exception:
                continue

    # Special handling for "Q3 FY 24" format
    qfy_match = re.search(r"\bQ([1-4])\s*FY\s*(\d{2})\b", text, re.IGNORECASE)
    if qfy_match:
        quarter = int(qfy_match.group(1))
        fy_year = int(qfy_match.group(2))
        # FY24 means financial year ending March 2024
        # Q1 = Jun 30, Q2 = Sep 30, Q3 = Dec 31, Q4 = Mar 31
        full_year = 2000 + fy_year
        if quarter == 1:
            return pd.Timestamp(full_year - 1, 6, 30)
        elif quarter == 2:
            return pd.Timestamp(full_year - 1, 9, 30)
        elif quarter == 3:
            return pd.Timestamp(full_year - 1, 12, 31)
        elif quarter == 4:
            return pd.Timestamp(full_year, 3, 31)

    return None


# ============================================================
# MAIN PIPELINE
# ============================================================

def load_and_classify() -> pd.DataFrame:
    """Load raw NSE data, classify documents, extract periods."""
    logger.info("Loading reference data...")
    reference = pd.read_csv(REFERENCE_FILE)
    reference["symbol"] = reference["symbol"].astype(str).str.strip()
    reference["company_name"] = reference["company_name"].astype(str).str.strip()
    symbol_to_company = dict(zip(reference["symbol"], reference["company_name"]))

    logger.info("Loading raw earnings files...")
    files = sorted(INPUT_DIR.glob("*.csv"))
    files = [f for f in files if f.name.lower() != "reliance_test.csv"]

    all_data = []
    for i, file in enumerate(files, 1):
        symbol = file.stem
        try:
            df = pd.read_csv(file)
            if df.empty:
                continue
            df["symbol"] = symbol
            all_data.append(df)
            if i % 10 == 0:
                logger.info(f"  Loaded {i}/{len(files)} files")
        except Exception as e:
            logger.warning(f"  Failed to load {symbol}: {e}")

    if not all_data:
        raise RuntimeError("No earnings files loaded")

    df = pd.concat(all_data, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]

    # Ensure required columns exist
    for col in ["desc", "attchmntText", "attchmntFile", "an_dt", "exchdisstime", "hasXbrl"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Build searchable text
    df["announcement_text"] = df["desc"].fillna("").astype(str).str.strip()
    df["announcement_details"] = df["attchmntText"].fillna("").astype(str).str.strip()
    df["search_text"] = (df["announcement_text"] + " " + df["announcement_details"]).str.lower()

    # Parse datetimes
    df["announcement_datetime"] = pd.to_datetime(df["an_dt"], dayfirst=True, errors="coerce")
    df["dissemination_datetime"] = pd.to_datetime(df["exchdisstime"], dayfirst=True, errors="coerce")

    # Filter to earnings candidates using broad pattern
    earnings_pattern = re.compile(
        r"""
        financial\s+results?|
        financial\s+result\s+updates?|
        integrated\s+filing|
        quarterly\s+results?|
        results?\s+for\s+the\s+(quarter|year)|
        results?\s+of\s+the\s+(quarter|year)|
        standalone\s+financial|
        consolidated\s+financial|
        unaudited\s+financial|
        audited\s+financial|
        limited\s+review\s+report|
        results?\s+update.*quarter|
        media\s+release|
        press\s+release|
        investor\s+presentation|
        earnings\s+call|
        concall|
        conference\s+call|
        transcript|
        recording.*call|
        board\s+meeting|
        outcome.*board
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    df["is_earnings_candidate"] = df["search_text"].str.contains(earnings_pattern, regex=True, na=False)
    earnings = df[df["is_earnings_candidate"]].copy()

    logger.info(f"Total raw rows: {len(df)}, Earnings candidates: {len(earnings)}")

    # Legacy classification for fallback
    def classify_earnings_type_legacy(text):
        text = str(text).lower()
        if "integrated filing" in text and "financial" in text:
            return "integrated_financial"
        if "financial result" in text:
            return "financial_results"
        if "quarterly result" in text:
            return "quarterly_results"
        if "results update" in text:
            return "results_update"
        if "limited review" in text:
            return "limited_review"
        if "annual" in text:
            return "annual_results"
        return "other_earnings"

    def classify_document_legacy(text):
        text = str(text).lower()
        if "integrated filing" in text:
            return "integrated_financial_filing"
        if "financial result updates" in text:
            return "financial_result_update"
        if "financial results" in text:
            return "financial_results"
        if "results update" in text:
            return "results_update"
        if "limited review" in text:
            return "limited_review"
        return "other"

    earnings["earnings_type"] = earnings["search_text"].apply(classify_earnings_type_legacy)
    earnings["document_type_legacy"] = earnings["search_text"].apply(classify_document_legacy)

    # Extract period_ended from text
    earnings["period_ended_extracted"] = earnings["search_text"].apply(extract_period_from_text)

    # Fallback to inferred quarter
    fallback_period = earnings["announcement_datetime"].apply(infer_quarter_end)
    earnings["period_ended"] = earnings["period_ended_extracted"].fillna(fallback_period)

    # Track which rows used fallback (no explicit period found)
    earnings["period_ended_from_fallback"] = earnings["period_ended_extracted"].isna()

    # Company info
    earnings["company_name"] = earnings["symbol"].map(symbol_to_company)

    # Fiscal quarter label
    earnings["fiscal_quarter"] = earnings["period_ended"].apply(quarter_label)

    # Reporting scope
    def detect_scope(text):
        text = str(text).lower()
        # Use word boundaries to avoid matching "non-consolidated" as "consolidated"
        import re
        has_consolidated = bool(re.search(r"\bconsolidated\b", text))
        has_standalone = bool(re.search(r"\bstandalone\b", text))
        has_non_consolidated = bool(re.search(r"\bnon[-\s]?consolidated\b", text))
        if has_non_consolidated:
            # If explicitly non-consolidated, treat as standalone scope
            has_consolidated = False
        if has_consolidated and has_standalone:
            return "both"
        if has_consolidated:
            return "consolidated"
        if has_standalone:
            return "standalone"
        return "unknown"

    earnings["reporting_scope"] = earnings["search_text"].apply(detect_scope)

    # XBRL flag
    earnings["has_xbrl"] = (
        earnings["hasXbrl"].astype(str).str.lower().isin(["true", "1", "yes"])
    )

    # Filing URL
    earnings["filing_url"] = earnings["attchmntFile"].fillna("").astype(str).str.strip()

    # Detailed classification
    classification_results = earnings.apply(
        lambda row: classify_document_detailed(
            row["announcement_text"],
            row["announcement_details"],
            row["filing_url"],
            row["earnings_type"],
            row["document_type_legacy"],
        ),
        axis=1,
    )

    earnings["document_type"] = classification_results.apply(lambda x: x["document_type"])
    earnings["is_primary_result"] = classification_results.apply(lambda x: x["is_primary_result"])
    earnings["classification_signals"] = classification_results.apply(lambda x: x["signals_matched"])

    # Post-process: For follow-up documents without explicit period_ended,
    # try to assign them to the nearest primary result event for the same symbol
    earnings = _reassign_followup_periods(earnings)

    # Document ID
    earnings["document_id"] = earnings.apply(
        lambda row: generate_document_id(row["symbol"], row["announcement_datetime"], row["filing_url"]),
        axis=1,
    )

    # Event key
    earnings["event_key"] = (
        earnings["symbol"].astype(str)
        + "_"
        + earnings["period_ended"].dt.strftime("%Y-%m-%d").fillna("UNKNOWN")
    )

    # Sort
    earnings = earnings.sort_values(
        ["symbol", "period_ended", "announcement_datetime"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return earnings


def select_canonical_primary(doc: pd.Series) -> int:
    """
    Score a primary result document for canonical selection.
    Higher score = more preferred as canonical primary result.
    
    Priority order:
    1. Consolidated specific filing (scope=consolidated, mentions quarter/year ended)
    2. Standalone specific filing (scope=standalone, mentions quarter/year ended)
    3. Both scope specific filing (scope=both, mentions quarter/year ended)
    4. Consolidated generic (scope=consolidated, "Financial Result Updates")
    5. Standalone generic (scope=standalone, "Financial Result Updates")
    6. Both scope generic (scope=both, "Financial Result Updates")
    7. Unknown scope specific (scope=unknown, mentions quarter/year ended)
    8. Unknown scope generic (scope=unknown, "Financial Result Updates" or board outcome)
    9. financial_result_update > financial_results (more specific to outcome)
    10. Earlier announcement (first filing) as tiebreaker
    """
    score = 0
    
    # Base score by document type
    if doc["document_type"] == "financial_result_update":
        score += 100
    elif doc["document_type"] == "financial_results":
        score += 50
    
    # Scope specificity
    scope = doc.get("reporting_scope", "unknown")
    text = str(doc.get("announcement_text", "")).lower()
    has_specific_period = "quarter ended" in text or "year ended" in text
    has_generic_fru = text.strip() == "financial result updates"
    has_board_outcome = "outcome of board meeting" in text
    
    if scope == "consolidated":
        if has_specific_period:
            score += 1000  # Highest: consolidated specific
        elif has_generic_fru:
            score += 500   # Consolidated generic
        else:
            score += 300
    elif scope == "standalone":
        if has_specific_period:
            score += 900   # Standalone specific
        elif has_generic_fru:
            score += 400   # Standalone generic
        else:
            score += 200
    elif scope == "both":
        if has_specific_period:
            score += 800   # Both specific
        elif has_generic_fru:
            score += 300   # Both generic
        else:
            score += 100
    else:  # unknown
        if has_specific_period:
            score += 200   # Unknown specific
        elif has_generic_fru:
            score += 100   # Unknown generic
        elif has_board_outcome:
            score += 50    # Board outcome (often first filing)
        else:
            score += 10
    
    # For Q4 (March periods), prefer annual over quarterly
    period_ended = doc.get("period_ended")
    if pd.notna(period_ended) and period_ended.month == 3:
        if "year ended" in text:
            score += 200
        elif "quarter ended" in text:
            score += 100
    
    # Tiebreaker: earlier announcement gets slightly higher score (first filing)
    # We'll subtract a tiny amount based on seconds since epoch
    ann_dt = doc.get("announcement_datetime")
    if pd.notna(ann_dt):
        score -= ann_dt.timestamp() / 1e12  # Very small penalty for later dates
    
    # Penalize announcements before period end (preliminary/early filings)
    # These are often board meeting approvals before quarter end or misassigned fallback periods
    period_ended = doc.get("period_ended")
    if pd.notna(ann_dt) and pd.notna(period_ended):
        if ann_dt < period_ended:
            days_before = (period_ended - ann_dt).days
            # Heavy penalty for announcements before period end
            score -= 10000 + days_before * 10
    
    return int(score)


def build_documents_table(earnings: pd.DataFrame) -> pd.DataFrame:
    """Build the earnings_documents table."""
    docs = earnings.copy()

    # Deduplicate exact duplicate documents
    before = len(docs)
    docs = docs.drop_duplicates(subset=["document_id"], keep="first")
    logger.info(f"Removed {before - len(docs)} duplicate documents")

    # Select canonical primary result per event
    human_primary_types = {"financial_results", "financial_result_update"}
    primary_mask = docs["document_type"].isin(human_primary_types)
    
    if primary_mask.any():
        primary_docs = docs[primary_mask].copy()
        primary_docs["canonical_score"] = primary_docs.apply(select_canonical_primary, axis=1)
        
        # Find best primary per event_key
        best_primary_idx = primary_docs.groupby("event_key")["canonical_score"].idxmax()
        canonical_primary_ids = set(best_primary_idx)
        
        docs["is_canonical_primary"] = docs.index.isin(canonical_primary_ids)
    else:
        docs["is_canonical_primary"] = False

    # Select and rename columns for documents table
    doc_columns = {
        "document_id": "document_id",
        "event_key": "event_key",
        "symbol": "symbol",
        "company_name": "company_name",
        "period_ended": "period_ended",
        "fiscal_quarter": "fiscal_quarter",
        "announcement_datetime": "announcement_datetime",
        "announcement_date": "announcement_datetime",  # will extract date
        "announcement_time": "announcement_datetime",  # will extract time
        "dissemination_datetime": "dissemination_datetime",
        "document_type": "document_type",
        "earnings_type": "earnings_type",
        "reporting_scope": "reporting_scope",
        "announcement_text": "announcement_text",
        "announcement_details": "announcement_details",
        "filing_url": "filing_url",
        "has_xbrl": "has_xbrl",
        "is_primary_result": "is_primary_result",
        "is_canonical_primary": "is_canonical_primary",
        "classification_signals": "classification_signals",
        "seq_id": "seq_id",
        "orgid": "orgid",
    }

    # Keep only existing columns
    doc_columns = {k: v for k, v in doc_columns.items() if k in docs.columns}

    docs_out = docs[list(doc_columns.keys())].copy()
    docs_out = docs_out.rename(columns=doc_columns)

    # Extract date/time
    docs_out["announcement_date"] = docs_out["announcement_datetime"].dt.date
    docs_out["announcement_time"] = docs_out["announcement_datetime"].dt.time

    # Add is_followup_document flag
    docs_out["is_followup_document"] = ~docs_out["is_primary_result"]

    # Source
    docs_out["source"] = "nse_corporate_announcements"

    return docs_out


def build_events_table(docs: pd.DataFrame) -> pd.DataFrame:
    """Build the earnings_events table (one row per event_key)."""
    # Pre-compute aggregations for efficiency
    grouped = docs.groupby("event_key")

    # First/last announcement per event
    first_ann = grouped["announcement_datetime"].min()
    last_ann = grouped["announcement_datetime"].max()

    # Symbol, company_name, period_ended, fiscal_quarter (should be same per event)
    event_meta = grouped.agg({
        "symbol": "first",
        "company_name": "first",
        "period_ended": "first",
        "fiscal_quarter": "first",
    })

    # Document count
    doc_count = grouped.size().rename("document_count")

    # Document types per event
    doc_types = grouped["document_type"].value_counts().unstack(fill_value=0).apply(
        lambda row: {k: int(v) for k, v in row.items() if v > 0}, axis=1
    ).rename("doc_types_dict")

    # Primary result info - only human-readable results (not XBRL filings) for dating
    human_primary_types = {"financial_results", "financial_result_update"}
    human_primary_docs = docs[docs["document_type"].isin(human_primary_types)].sort_values("announcement_datetime")
    human_primary_first = human_primary_docs.groupby("event_key")["announcement_datetime"].first()
    human_primary_count = human_primary_docs.groupby("event_key").size().rename("human_primary_result_count")

    # Canonical primary result (deterministically selected best primary)
    if "is_canonical_primary" in docs.columns:
        canonical_primary_docs = docs[docs["is_canonical_primary"] == True].sort_values("announcement_datetime")
    else:
        # Fallback: use first human primary result (backward compatibility)
        canonical_primary_docs = human_primary_docs
    canonical_primary_first = canonical_primary_docs.groupby("event_key")["announcement_datetime"].first()
    canonical_primary_count = canonical_primary_docs.groupby("event_key").size().rename("canonical_primary_count")

    # All primary results (including XBRL) for counting
    primary_docs = docs[docs["is_primary_result"]].sort_values("announcement_datetime")
    primary_first = primary_docs.groupby("event_key")["announcement_datetime"].first()
    primary_count = primary_docs.groupby("event_key").size().rename("primary_result_count")

    # Has flags
    has_financial_results = grouped["document_type"].apply(
        lambda x: any(is_primary_result(t) for t in x)
    ).rename("has_financial_results")
    has_media_release = grouped["document_type"].apply(
        lambda x: "media_release" in x.values
    ).rename("has_media_release")
    has_investor_presentation = grouped["document_type"].apply(
        lambda x: "investor_presentation" in x.values
    ).rename("has_investor_presentation")
    has_earnings_call = grouped["document_type"].apply(
        lambda x: any(t in x.values for t in ["earnings_call_recording", "earnings_call_schedule"])
    ).rename("has_earnings_call")
    has_transcript = grouped["document_type"].apply(
        lambda x: "earnings_call_transcript" in x.values
    ).rename("has_transcript")

    # Duplicate URLs
    dup_urls = grouped["filing_url"].apply(lambda x: x.duplicated().sum()).rename("dup_urls")

    # Unclassified count
    unclassified = grouped["document_type"].apply(
        lambda x: (x == "unclassified").sum()
    ).rename("unclassified")

    # Suspicious dates: HUMAN primary result announcement before period_end
    human_primary_suspicious = human_primary_docs.groupby("event_key").apply(
        lambda g: (g["announcement_datetime"] < g["period_ended"]).sum()
    ).rename("human_primary_suspicious_dates")

    # Also check all primary results (including XBRL) for reference
    primary_suspicious = primary_docs.groupby("event_key").apply(
        lambda g: (g["announcement_datetime"] < g["period_ended"]).sum()
    ).rename("primary_suspicious_dates")

    # Also check all documents for reference
    all_suspicious = grouped.apply(
        lambda g: (g["announcement_datetime"] < g["period_ended"]).sum()
    ).rename("all_suspicious_dates")

    # Missing period_end
    missing_period = grouped["period_ended"].apply(lambda x: pd.isna(x.iloc[0])).rename("missing_period")

    # Combine all
    events_df = pd.concat([
        event_meta,
        doc_count,
        first_ann.rename("first_document_datetime"),
        last_ann.rename("last_document_datetime"),
        canonical_primary_first.rename("result_announcement_datetime"),
        human_primary_count.rename("human_primary_result_count"),
        canonical_primary_count.rename("canonical_primary_count"),
        primary_first.rename("all_primary_first_datetime"),
        primary_count,
        has_financial_results,
        has_media_release,
        has_investor_presentation,
        has_earnings_call,
        has_transcript,
        doc_types.rename("doc_types_dict"),
        dup_urls,
        unclassified,
        primary_suspicious,
        human_primary_suspicious.rename("human_primary_suspicious_dates"),
        all_suspicious,
        missing_period,
    ], axis=1).reset_index()

    # Fill missing counts with 0
    events_df["primary_result_count"] = events_df["primary_result_count"].fillna(0).astype(int)
    events_df["human_primary_result_count"] = events_df["human_primary_result_count"].fillna(0).astype(int)
    events_df["canonical_primary_count"] = events_df["canonical_primary_count"].fillna(0).astype(int)

    # Compute result_announcement_date/time (based on canonical primary result)
    events_df["result_announcement_date"] = events_df["result_announcement_datetime"].dt.date
    events_df["result_announcement_time"] = events_df["result_announcement_datetime"].dt.time

    # Compute announcement date/time
    events_df["announcement_date"] = events_df["first_document_datetime"].dt.date
    events_df["announcement_time"] = events_df["first_document_datetime"].dt.time

    # Days from period end
    mask = events_df["period_ended"].notna() & events_df["result_announcement_datetime"].notna()
    events_df.loc[mask, "days_from_period_end"] = (
        events_df.loc[mask, "result_announcement_datetime"].dt.date -
        events_df.loc[mask, "period_ended"].dt.date
    ).apply(lambda x: x.days)
    events_df["days_from_period_end"] = events_df["days_from_period_end"].astype("Int64")

    # Quality flags
    def compute_quality_flag(row):
        flags = []
        if row["missing_period"]:
            flags.append("MISSING_PERIOD_END")
        if row["human_primary_result_count"] == 0:
            flags.append("MISSING_RESULT")
        elif row["human_primary_result_count"] > 1:
            flags.append("MULTIPLE_HUMAN_PRIMARY_RESULTS")
        if pd.notna(row["days_from_period_end"]):
            if row["days_from_period_end"] < 0:
                flags.append("ANNOUNCEMENT_BEFORE_PERIOD_END")
            elif row["days_from_period_end"] > 90:
                flags.append("VERY_LATE_FILING")
        if row["dup_urls"] > 0:
            flags.append("DUPLICATE_DOCUMENT")
        if row["unclassified"] > 0:
            flags.append("UNCLASSIFIED_DOCUMENT")
        # Only flag if HUMAN PRIMARY RESULT announcement is before period end
        if row.get("human_primary_suspicious_dates", 0) > 0:
            flags.append("HUMAN_PRIMARY_RESULT_BEFORE_PERIOD_END")
        # Also flag if many pre-announcement documents (informational)
        if row.get("all_suspicious_dates", 0) > 5:
            flags.append("MANY_PRE_ANNOUNCEMENTS")
        return "|".join(flags) if flags else "OK"

    events_df["quality_flag"] = events_df.apply(compute_quality_flag, axis=1)
    events_df["event_status"] = events_df["human_primary_result_count"].apply(
        lambda x: "confirmed" if x > 0 else "no_primary_result"
    )
    events_df["document_types"] = events_df["doc_types_dict"].apply(
        lambda d: "|".join(f"{k}:{v}" for k, v in sorted(d.items())) if isinstance(d, dict) else ""
    )

    # Select final columns (rename first_document_datetime to announcement_datetime for consistency)
    events_df = events_df.rename(columns={"first_document_datetime": "announcement_datetime"})

    final_cols = [
        "event_key", "symbol", "company_name", "period_ended", "fiscal_quarter",
        "announcement_datetime", "announcement_date", "announcement_time",
        "result_announcement_datetime", "result_announcement_date", "result_announcement_time",
        "event_status", "document_count", "has_financial_results", "has_media_release",
        "has_investor_presentation", "has_earnings_call", "has_transcript",
        "first_document_datetime", "last_document_datetime", "days_from_period_end",
        "primary_result_count", "human_primary_result_count", "canonical_primary_count", "all_primary_first_datetime",
        "quality_flag", "document_types",
    ]

    # Ensure first_document_datetime exists (it was renamed, so add it back)
    events_df["first_document_datetime"] = events_df["announcement_datetime"]

    events_df = events_df[final_cols].sort_values(["symbol", "period_ended"]).reset_index(drop=True)
    return events_df


def build_quality_report(events: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    """Build quality report for events."""
    return events[[
        "event_key", "symbol", "period_ended", "document_count",
        "first_document_datetime", "result_announcement_datetime",
        "last_document_datetime", "days_from_period_end",
        "human_primary_result_count", "quality_flag"
    ]].copy()


def main():
    logger.info("=" * 70)
    logger.info("BUILD EARNINGS EVENTS & DOCUMENTS")
    logger.info("=" * 70)

    # Load and classify
    earnings = load_and_classify()

    # Build documents table
    logger.info("Building documents table...")
    docs = build_documents_table(earnings)

    # Build events table
    logger.info("Building events table...")
    events = build_events_table(docs)

    # Build quality report
    logger.info("Building quality report...")
    quality_report = build_quality_report(events, docs)

    # Save
    logger.info(f"Saving events to {EVENTS_OUTPUT}")
    events.to_parquet(EVENTS_OUTPUT, index=False)

    logger.info(f"Saving documents to {DOCUMENTS_OUTPUT}")
    docs.to_parquet(DOCUMENTS_OUTPUT, index=False)

    logger.info(f"Saving quality report to {QUALITY_REPORT_OUTPUT}")
    quality_report.to_csv(QUALITY_REPORT_OUTPUT, index=False)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total documents: {len(docs)}")
    logger.info(f"Unique events: {len(events)}")
    logger.info(f"Companies: {events['symbol'].nunique()}")

    logger.info("\nDocument types:")
    logger.info(f"\n{docs['document_type'].value_counts().to_string()}")

    logger.info("\nQuality flags:")
    flag_counts = events["quality_flag"].str.split("|").explode().value_counts()
    logger.info(f"\n{flag_counts.to_string()}")

    logger.info("\nEvents with missing primary result:")
    missing = events[events["primary_result_count"] == 0]
    logger.info(f"  Count: {len(missing)}")
    if len(missing) > 0:
        logger.info(f"  Examples: {missing['event_key'].head(10).tolist()}")

    logger.info("\nEvents with multiple primary results:")
    multiple = events[events["primary_result_count"] > 1]
    logger.info(f"  Count: {len(multiple)}")
    if len(multiple) > 0:
        logger.info(f"  Examples: {multiple['event_key'].head(10).tolist()}")

    logger.info("\nDone!")


if __name__ == "__main__":
    import re
    main()