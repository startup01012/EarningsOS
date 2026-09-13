from datetime import datetime, timezone

import pytest

from services.forecasting.price_series import PriceObservation, build_forecast_request


def test_build_forecast_request_sorts_observations_without_future_data():
    observations = [
        PriceObservation(datetime(2026, 1, 3, tzinfo=timezone.utc), 103.0),
        PriceObservation(datetime(2026, 1, 1, tzinfo=timezone.utc), 101.0),
        PriceObservation(datetime(2026, 1, 2, tzinfo=timezone.utc), 102.0),
    ]

    request = build_forecast_request("RELIANCE", observations, horizon=2)

    assert request.symbol == "RELIANCE"
    assert request.values == [101.0, 102.0, 103.0]
    assert request.timestamps == [
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 2, tzinfo=timezone.utc),
        datetime(2026, 1, 3, tzinfo=timezone.utc),
    ]
    assert request.horizon == 2


def test_build_forecast_request_rejects_duplicate_timestamps():
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    observations = [
        PriceObservation(timestamp, 100.0),
        PriceObservation(timestamp, 101.0),
    ]

    with pytest.raises(ValueError, match="timestamps must be unique"):
        build_forecast_request("RELIANCE", observations, horizon=1)


def test_build_forecast_request_requires_enough_observations():
    observation = PriceObservation(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        100.0,
    )

    with pytest.raises(ValueError, match="at least two observations"):
        build_forecast_request("RELIANCE", [observation], horizon=1)
