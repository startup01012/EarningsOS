from datetime import datetime, timezone

from apps.api.intelligence import _baseline


def test_baseline_is_unavailable_without_scored_news():
    class EmptyDB:
        def execute(self, _statement):
            class Result:
                def all(self):
                    return []
            return Result()

    result = _baseline(
        EmptyDB(),
        "RELIANCE",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert result["available"] is False
    assert result["article_count"] == 0
    assert result["model"] == "ProsusAI/finbert"


def test_baseline_score_is_bounded():
    rows = []

    class Score:
        positive_probability = 0.9
        negative_probability = 0.05
        neutral_probability = 0.05

    class Article:
        published_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for _ in range(3):
        rows.append((Article(), Score()))

    class Result:
        def all(self):
            return rows

    class DB:
        def execute(self, _statement):
            return Result()

    result = _baseline(
        DB(),
        "RELIANCE",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert 0 <= result["score"] <= 100
    assert result["label"] == "positive"
