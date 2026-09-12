from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256

from sqlalchemy import select

from apps.api.db.models import NewsArticle
from apps.api.db.session import SessionLocal

from .base import NewsProvider


class NewsService:
    def __init__(self, provider: NewsProvider):
        self.provider = provider

    @staticmethod
    def content_hash(item) -> str:
        payload = f"{item.title}\n{item.summary or ''}".encode("utf-8")
        return sha256(payload).hexdigest()

    def ingest(self, query: str, start: datetime | None = None, end: datetime | None = None) -> tuple[int, int]:
        items = self.provider.search(query, start=start, end=end)
        db = SessionLocal()
        inserted = skipped = 0
        try:
            for item in items:
                url = str(item.url)
                if db.execute(select(NewsArticle.id).where(NewsArticle.url == url)).scalar_one_or_none():
                    skipped += 1
                    continue
                published = item.published_at
                if published and published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
                db.add(NewsArticle(
                    title=item.title,
                    url=url,
                    source=item.source,
                    published_at=published,
                    fetched_at=datetime.now(timezone.utc),
                    summary=item.summary,
                    content_hash=self.content_hash(item),
                ))
                inserted += 1
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return inserted, skipped
