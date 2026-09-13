from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo


FEATURE_TIMEZONE = ZoneInfo("Asia/Kolkata")


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


@dataclass(frozen=True)
class RollingSentimentFeature:
    symbol: str
    feature_date: date
    model_name: str

    article_count_3d: int
    article_count_7d: int
    article_count_14d: int
    article_count_30d: int

    sentiment_balance_3d: float
    sentiment_balance_7d: float
    sentiment_balance_14d: float
    sentiment_balance_30d: float

    sentiment_momentum_3d_vs_7d: float


def _feature_date(published_at: datetime) -> date:
    """Convert an information timestamp to the Indian calendar date."""
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    return published_at.astimezone(FEATURE_TIMEZONE).date()


def build_daily_features(rows: list[dict]) -> list[DailySentimentFeature]:
    """Aggregate scored news by symbol/date/model.

    ``published_at`` is the information timestamp. Inference/ingestion time is
    deliberately not used to assign the feature date. Dates are normalized to
    India Standard Time because EarningsOS targets NSE/BSE securities.
    """
    groups: dict[tuple[str, date, str], list[dict]] = defaultdict(list)

    for row in rows:
        published_at = row.get("published_at")
        symbol = row.get("symbol")
        model_name = row.get("model_name")

        if published_at is None or not symbol or not model_name:
            continue

        groups[(symbol, _feature_date(published_at), model_name)].append(row)

    features: list[DailySentimentFeature] = []

    for (symbol, feature_date, model_name), items in sorted(groups.items()):
        count = len(items)

        positive = [
            float(x.get("positive_probability") or 0)
            for x in items
        ]
        negative = [
            float(x.get("negative_probability") or 0)
            for x in items
        ]
        neutral = [
            float(x.get("neutral_probability") or 0)
            for x in items
        ]
        scores = [
            float(x.get("score") or 0)
            for x in items
        ]

        positive_count = sum(
            p > n and p > z
            for p, n, z in zip(positive, negative, neutral)
        )
        negative_count = sum(
            n > p and n > z
            for p, n, z in zip(positive, negative, neutral)
        )
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
                sentiment_balance=sum(
                    p - n for p, n in zip(positive, negative)
                ) / count,
                last_published_at=max(
                    x["published_at"] for x in items
                ),
            )
        )

    return features


def build_rolling_features(
    daily_features: list[DailySentimentFeature],
) -> list[RollingSentimentFeature]:
    """Build causal rolling sentiment features.

    For feature date D, only daily observations with feature_date <= D
    are included. No future information is used.
    """
    grouped: dict[tuple[str, str], list[DailySentimentFeature]] = defaultdict(list)

    for feature in daily_features:
        grouped[(feature.symbol, feature.model_name)].append(feature)

    results: list[RollingSentimentFeature] = []

    for (symbol, model_name), items in grouped.items():
        items.sort(key=lambda x: x.feature_date)

        for current in items:
            current_date = current.feature_date

            def window(days: int) -> list[DailySentimentFeature]:
                start_date = current_date - timedelta(days=days - 1)
                return [
                    item
                    for item in items
                    if start_date <= item.feature_date <= current_date
                ]

            def total_articles(values: list[DailySentimentFeature]) -> int:
                return sum(item.article_count for item in values)

            def weighted_balance(
                values: list[DailySentimentFeature],
            ) -> float:
                articles = total_articles(values)
                if articles == 0:
                    return 0.0

                return sum(
                    item.sentiment_balance * item.article_count
                    for item in values
                ) / articles

            values_3d = window(3)
            values_7d = window(7)
            values_14d = window(14)
            values_30d = window(30)

            balance_3d = weighted_balance(values_3d)
            balance_7d = weighted_balance(values_7d)

            results.append(
                RollingSentimentFeature(
                    symbol=symbol,
                    feature_date=current_date,
                    model_name=model_name,
                    article_count_3d=total_articles(values_3d),
                    article_count_7d=total_articles(values_7d),
                    article_count_14d=total_articles(values_14d),
                    article_count_30d=total_articles(values_30d),
                    sentiment_balance_3d=balance_3d,
                    sentiment_balance_7d=balance_7d,
                    sentiment_balance_14d=weighted_balance(values_14d),
                    sentiment_balance_30d=weighted_balance(values_30d),
                    sentiment_momentum_3d_vs_7d=balance_3d - balance_7d,
                )
            )

    return sorted(
        results,
        key=lambda x: (x.symbol, x.model_name, x.feature_date),
    )
