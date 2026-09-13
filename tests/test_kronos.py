from datetime import datetime, timezone

import pandas as pd
import pytest

from services.forecasting.base import ForecastRequest
from services.forecasting.kronos import KronosSmallAdapter


def _request() -> ForecastRequest:
    timestamps = [datetime(2026, 1, day, tzinfo=timezone.utc) for day in range(1, 4)]
    return ForecastRequest(
        symbol="RELIANCE",
        values=[100.0, 101.0, 102.0],
        timestamps=timestamps,
        horizon=2,
        opens=[99.0, 100.0, 101.0],
        highs=[101.0, 102.0, 103.0],
        lows=[98.0, 99.0, 100.0],
        volumes=[1000.0, 1100.0, 1200.0],
        amounts=[100000.0, 111100.0, 122400.0],
    )


def test_kronos_adapter_requires_complete_ohlcv():
    adapter = KronosSmallAdapter()
    request = ForecastRequest(
        symbol="RELIANCE",
        values=[100.0, 101.0, 102.0],
        timestamps=[datetime(2026, 1, day, tzinfo=timezone.utc) for day in range(1, 4)],
        horizon=2,
    )
    with pytest.raises(ValueError, match="requires OHLCV context"):
        adapter.forecast(request)


def test_kronos_request_accepts_complete_ohlcv():
    request = _request()
    assert request.opens == [99.0, 100.0, 101.0]
    assert request.volumes == [1000.0, 1100.0, 1200.0]


def test_kronos_adapter_validates_max_context():
    with pytest.raises(ValueError, match="max_context must be >= 2"):
        KronosSmallAdapter(max_context=1)


def test_kronos_adapter_validates_seed():
    with pytest.raises(ValueError, match="seed must be >= 0"):
        KronosSmallAdapter(seed=-1)


def test_kronos_passes_timestamps_as_series(monkeypatch):
    captured = {}

    class FakePredictor:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return pd.DataFrame({"close": [103.0, 104.0]})

    adapter = KronosSmallAdapter()
    adapter._predictor = FakePredictor()
    adapter.forecast(_request())

    assert isinstance(captured["x_timestamp"], pd.Series)
    assert isinstance(captured["y_timestamp"], pd.Series)
    assert captured["x_timestamp"].name == "timestamps"
    assert captured["y_timestamp"].name == "timestamps"
    assert len(captured["y_timestamp"]) == 2


def test_kronos_seed_is_applied_before_prediction(monkeypatch):
    calls = []

    class FakeTorch:
        @staticmethod
        def manual_seed(seed):
            calls.append(("manual_seed", seed))

        class cuda:
            @staticmethod
            def is_available():
                return False

    class FakePredictor:
        def predict(self, **kwargs):
            calls.append(("predict", kwargs["pred_len"]))
            return pd.DataFrame({"close": [103.0, 104.0]})

    adapter = KronosSmallAdapter(seed=12345)
    adapter._predictor = FakePredictor()
    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch)

    adapter.forecast(_request())

    assert calls == [("manual_seed", 12345), ("predict", 2)]
