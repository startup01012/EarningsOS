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


def test_baseline_requires_explicit_result_boundary():
    class EmptyDB:
        pass

    result = _baseline(EmptyDB(), "RELIANCE", None)
    assert result["available"] is False
    assert result["boundary"] is None
    assert "result_announcement_datetime" in result["method"]


def test_baseline_uses_strict_pre_result_boundary():
    captured = []

    class Result:
        def all(self):
            return []

    class DB:
        def execute(self, statement):
            captured.append(str(statement))
            return Result()

    boundary = datetime(2026, 1, 2, tzinfo=timezone.utc)
    _baseline(DB(), "RELIANCE", boundary)
    assert captured
    assert "published_at <" in captured[0]
