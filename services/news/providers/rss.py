from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser

from ..base import NewsProvider
from ..schemas import NewsItem


class RSSNewsProvider(NewsProvider):
    """Read public RSS/Atom feeds; source-specific logic stays behind the interface."""

    source = "rss"

    def __init__(self, feed_urls: list[str]):
        self.feed_urls = feed_urls

    @staticmethod
    def _published(entry) -> datetime | None:
        value = entry.get("published") or entry.get("updated")
        if not value:
            return None
        try:
            parsed = parsedate_to_datetime(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None

    def search(self, query: str, start: datetime | None = None, end: datetime | None = None) -> list[NewsItem]:
        terms = [part.lower() for part in query.split() if part.strip()]
        results: list[NewsItem] = []
        seen: set[str] = set()
        for feed_url in self.feed_urls:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title = str(entry.get("title", "")).strip()
                summary = str(entry.get("summary", "")).strip() or None
                url = str(entry.get("link", "")).strip()
                haystack = f"{title} {summary or ''}".lower()
                if not url or url in seen or any(term not in haystack for term in terms):
                    continue
                published = self._published(entry)
                if start and published and published < start:
                    continue
                if end and published and published >= end:
                    continue
                seen.add(url)
                results.append(NewsItem(title=title, url=url, source=feed_url, published_at=published, summary=summary))
        results.sort(key=lambda item: item.published_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return results
