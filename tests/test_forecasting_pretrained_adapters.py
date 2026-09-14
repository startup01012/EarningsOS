from __future__ import annotations

import pytest

from services.forecasting.base import ForecastRequest
from services.forecasting.fincast import FinCastAdapter
from services.forecasting.timesfm import TimesFM25Adapter


def request(n: int = 64, horizon: int = 10) -> ForecastRequest:
    from datetime import datetime, timedelta, timezone

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return ForecastRequest(
        symbol="RELIANCE",
        values=[100.0 + i for i in range(n)],
        timestamps=[start + timedelta(days=i) for i in range(n)],
        horizon=horizon,
    )


def test_timesfm_metadata_and_limits() -> None:
    adapter = TimesFM25Adapter()
    assert adapter.model_id == "google/timesfm-2.5-200m-pytorch"
    assert adapter.max_context == 16_384


def test_timesfm_rejects_non_finite_without_loading_model() -> None:
    adapter = TimesFM25Adapter()
    bad = request()
    bad = ForecastRequest(
        symbol=bad.symbol,
        values=[float("nan")] + bad.values[1:],
        timestamps=bad.timestamps,
        horizon=bad.horizon,
    )
    with pytest.raises(ValueError, match="non-finite"):
        adapter.forecast(bad)


def test_fincast_requires_explicit_model_configuration() -> None:
    adapter = FinCastAdapter(model_path=None, fincast_root=None)
    with pytest.raises(RuntimeError, match="FINCAST_MODEL_PATH"):
        adapter.forecast(request())


def test_fincast_metadata() -> None:
    adapter = FinCastAdapter(model_path="/tmp/model.pth", fincast_root="/tmp/fincast")
    assert adapter.model_id == "Vincent05R/FinCast:v1"
    assert adapter.max_context == 1024
    assert adapter.max_horizon == 256
