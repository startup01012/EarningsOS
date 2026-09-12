from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256

from sqlalchemy import select

from apps.api.db.models import NewsArticle, Stock
from apps.api.db.session import SessionLocal

from .base import NewsProvider
from .mapping import match_stock


class NewsService:
    def __init__(self, provider: NewsProvider):
        self.provider = provider

    @staticmethod
    def content_hash(item) -> str:
        payload = f"{item.title}\n{item.summary or ''}".encode("utf-8")
        return sha256(payload).hexdigest()

    def ingest(
        self,
        query: str,
        start: datetime | None = None,
        end: datetime | None = None,
        symbol: str | None = None,
    ) -> tuple[int, int]:
        """Ingest news idempotently and conservatively map it to a stock.

        When ``symbol`` is supplied, all ingested items are explicitly attached
        to that stock. Without it, ticker/company-name matching is attempted and
        ambiguous items remain unmapped instead of being guessed.
        """
        items = self.provider.search(query, start=start, end=end)
        db = SessionLocal()
        inserted = skipped = 0
        try:
            stocks = list(db.scalars(select(Stock)).all()) if symbol is None else []
            target_stock = None
            if symbol is not None:
                target_stock = db.scalar(select(Stock).where(Stock.symbol == symbol.strip().upper()))
                if target_stock is None:
                    raise ValueError(f"Unknown stock symbol: {symbol}")

            for item in items:
                url = str(item.url)
                if db.execute(select(NewsArticle.id).where(NewsArticle.url == url)).scalar_one_or_none():
                    skipped += 1
                    continue

                published = item.published_at
                if published and published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
                if published is None:
                    # NewsArticle.published_at is required; do not fabricate a timestamp.
                    skipped += 1
                    continue

                matched_stock = target_stock
                if matched_stock is None:
                    match = match_stock(f"{item.title}\n{item.summary or ''}", stocks)
                    if match is not None:
                        matched_stock = next((s for s in stocks if s.symbol == match.symbol), None)

                db.add(
                    NewsArticle(
                        stock_id=matched_stock.id if matched_stock else None,
                        title=item.title,
                        url=url,
                        source=item.source,
                        published_at=published,
                        fetched_at=datetime.now(timezone.utc),
                        summary=item.summary,
                        content_hash=self.content_hash(item),
                    )
                )
                inserted += 1
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return inserted, skipped
