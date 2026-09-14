from datetime import datetime, timezone

import pytest

from services.forecasting.backtest import BacktestCase, REGIME_BULL
from services.forecasting.paired import compare_paired, dm_style_hac


def make_cases() -> list[BacktestCase]:
    return [
        BacktestCase(datetime(2026, 1, 1, tzinfo=timezone.utc), [110, 120], [0, 0], 100, [90, 100], REGIME_BULL),
        BacktestCase(datetime(2026, 1, 2, tzinfo=timezone.utc), [90, 80], [0, 0], 100, [110, 100], REGIME_BULL),
        BacktestCase(datetime(2026, 1, 3, tzinfo=timezone.utc), [105, 95], [0, 0], 100, [95, 100], REGIME_BULL),
    ]


def test_compare_paired_preserves_same_cutoff_pairing():
    cases = make_cases()
    model = [[101, 121], [99, 81], [101, 94]]
    baseline = [[102, 118], [98, 82], [103, 93]]

    summary, rows = compare_paired(
        cases,
        model,
        baseline,
        model="Model",
        baseline="last_close",
        horizon=1,
        symbol="TEST",
    )

    assert summary.cases == 3
    assert len(rows) == 3
    assert [row.symbol for row in rows] == ["TEST"] * 3
    assert summary.mean_loss_diff == pytest.approx(sum(row.loss_diff for row in rows) / 3)
    assert summary.model_win_rate + summary.tie_rate + summary.baseline_win_rate == pytest.approx(1.0)


def test_compare_paired_bootstrap_is_about_paired_loss_difference():
    cases = make_cases()
    model = [[100, 120], [100, 80], [100, 95]]
    baseline = [[110, 120], [90, 80], [105, 95]]

    summary, rows = compare_paired(
        cases,
        model,
        baseline,
        model="Model",
        baseline="baseline",
        horizon=1,
    )

    expected = [row.model_loss - row.baseline_loss for row in rows]
    assert summary.mean_loss_diff == pytest.approx(sum(expected) / len(expected))
    assert summary.ci_low is not None
    assert summary.ci_high is not None
    assert summary.ci_low <= summary.mean_loss_diff <= summary.ci_high


def test_dm_style_hac_has_no_result_for_too_few_cases():
    assert dm_style_hac([0.1, -0.1], 1) == (None, None)


def test_compare_paired_rejects_mismatched_predictions():
    with pytest.raises(ValueError, match="one row per case"):
        compare_paired(
            make_cases(),
            [[100, 100]],
            [[100, 100], [100, 100], [100, 100]],
            model="Model",
            baseline="baseline",
            horizon=1,
        )
