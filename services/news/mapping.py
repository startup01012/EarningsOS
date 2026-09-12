from __future__ import annotations

import re
from dataclasses import dataclass

from apps.api.db.models import Stock


@dataclass(frozen=True)
class SymbolMatch:
    symbol: str
    reason: str


def _token_pattern(value: str) -> re.Pattern[str]:
    escaped = re.escape(value.strip())
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


def match_stock(text: str, stocks: list[Stock]) -> SymbolMatch | None:
    """Conservatively map a news item to one stock.

    Exact ticker-token matches take priority. Company-name matches are accepted
    only when exactly one stock matches. Ambiguous or weak matches stay unmapped
    rather than guessing, which is safer for downstream market features.
    """
    if not text or not text.strip():
        return None

    for stock in stocks:
        if _token_pattern(stock.symbol).search(text):
            return SymbolMatch(stock.symbol, "symbol")

    matches: list[Stock] = []
    for stock in stocks:
        name = " ".join(stock.company_name.split()).strip()
        if len(name) < 5:
            continue
        if _token_pattern(name).search(text):
            matches.append(stock)

    if len(matches) == 1:
        return SymbolMatch(matches[0].symbol, "company_name")
    return None
