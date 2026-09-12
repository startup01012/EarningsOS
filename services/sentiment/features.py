from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class DailySentimentFeature:
    symbol: str
    feature_date: date
    model_name: str
    article_count: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    mean_score: float
    sentiment_balance: float
    last_published_at: datetime | None


def build_daily_features(rows: list[dict]) -> list[DailySentimentFeature]:
    """Aggregate scored news by symbol/date/model.

    ``published_at`` is the information timestamp. Inference/ingestion time is
    deliberately not used to assign the feature date.
    """
    groups: dict[tuple[str, date, str], list[dict]] = defaultdict(list)
    for row in rows:
        published_at = row.get("published_at")
        symbol = row.get("symbol")
        model_name = row.get("model_name")
        if published_at is None or not symbol or not model_name:
            continue
        groups[(symbol, published_at.date(), model_name)].append(row)

    features: list[DailySentimentFeature] = []
    for (symbol, feature_date, model_name), items in sorted(groups.items()):
        count = len(items)
        positive = [float(x.get("positive_probability") or 0) for x in items]
        negative = [float(x.get("negative_probability") or 0) for x in items]
        neutral = [float(x.get("neutral_probability") or 0) for x in items]
        scores = [float(x.get("score") or 0) for x in items]
        positive_count = sum(p > n and p > z for p, n, z in zip(positive, negative, neutral))
        negative_count = sum(n > p and n > z for p, n, z in zip(positive, negative, neutral))
        neutral_count = count - positive_count - negative_count
        features.append(
            DailySentimentFeature(
                symbol=symbol,
                feature_date=feature_date,
                model_name=model_name,
                article_count=count,
                positive_ratio=positive_count / count,
                negative_ratio=negative_count / count,
                neutral_ratio=neutral_count / count,
                mean_score=sum(scores) / count,
                sentiment_balance=sum(p - n for p, n in zip(positive, negative)) / count,
                last_published_at=max(x["published_at"] for x in items),
            )
        )
    return features
