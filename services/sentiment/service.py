from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from apps.api.db.models import NewsArticle, SentimentScore
from apps.api.db.session import SessionLocal

from .finbert import FinBERTSentiment


class SentimentService:
    """Persist pretrained sentiment scores for news articles idempotently."""

    def __init__(self, adapter: FinBERTSentiment | None = None):
        self.adapter = adapter or FinBERTSentiment()

    @staticmethod
    def _text(article: NewsArticle) -> str:
        return f"{article.title}\n{article.summary or ''}".strip()

    def score_pending(self, limit: int = 100) -> tuple[int, int]:
        """Score articles that do not yet have a result from this model."""
        if limit < 1:
            raise ValueError("limit must be >= 1")

        db = SessionLocal()
        scored = skipped = 0
        try:
            model_name = self.adapter.model_id
            stmt = (
                select(NewsArticle)
                .where(
                    ~NewsArticle.sentiment_scores.any(
                        SentimentScore.model_name == model_name
                    )
                )
                .order_by(NewsArticle.published_at.asc())
                .limit(limit)
            )
            articles = list(db.scalars(stmt).all())

            for article in articles:
                text = self._text(article)
                if not text:
                    skipped += 1
                    continue
                result = self.adapter.predict(text)
                db.add(
                    SentimentScore(
                        article_id=article.id,
                        model_name=model_name,
                        label=result.label,
                        score=result.score,
                        positive_probability=result.positive,
                        negative_probability=result.negative,
                        neutral_probability=result.neutral,
                        created_at=datetime.now(timezone.utc),
                    )
                )
                scored += 1

            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return scored, skipped
