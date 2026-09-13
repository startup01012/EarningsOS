from datetime import datetime, timezone

import pytest

from services.forecasting.base import ForecastRequest, ForecastResult
from services.forecasting.chronos import Chronos2Adapter


def test_forecast_request_validates_lengths_and_horizon():
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    request = ForecastRequest(
        symbol="RELIANCE",
        values=[100.0, 101.0],
        timestamps=[timestamp, timestamp],
        horizon=3,
    )

    assert request.symbol == "RELIANCE"
    assert request.horizon == 3

    with pytest.raises(ValueError):
        ForecastRequest(
            symbol="RELIANCE",
            values=[100.0],
            timestamps=[timestamp, timestamp],
            horizon=1,
        )


def test_forecast_result_requires_horizon_length():
    generated = datetime(2026, 1, 1, tzinfo=timezone.utc)

    result = ForecastResult(
        model_id="test-model",
        symbol="RELIANCE",
        generated_at=generated,
        horizon=2,
        median=[101.0, 102.0],
        lower=[99.0, 100.0],
        upper=[103.0, 104.0],
    )

    assert result.median == [101.0, 102.0]

    with pytest.raises(ValueError):
        ForecastResult(
            model_id="test-model",
            symbol="RELIANCE",
            generated_at=generated,
            horizon=2,
            median=[101.0],
        )


def test_chronos_adapter_is_lazy():
    adapter = Chronos2Adapter(device_map="cpu")
    assert adapter.model_id == "amazon/chronos-2"
    assert adapter._pipeline is None
