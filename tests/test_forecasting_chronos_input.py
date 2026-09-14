from datetime import datetime, timezone

import pandas as pd

from services.forecasting.base import ForecastRequest
from services.forecasting.chronos import Chronos2Adapter


class FakeChronosPipeline:
    def __init__(self):
        self.context = None
        self.kwargs = None

    def predict_df(self, context_df, **kwargs):
        self.context = context_df.copy()
        self.kwargs = kwargs
        return pd.DataFrame({
            "item_id": ["TEST"] * 2,
            "timestamp": pd.date_range("2026-01-05", periods=2, freq="B"),
            "0.1": [99.0, 100.0],
            "0.5": [101.0, 102.0],
            "0.9": [103.0, 104.0],
        })


def test_chronos_receives_required_univariate_dataframe_and_business_frequency():
    adapter = Chronos2Adapter(device_map="cpu")
    fake = FakeChronosPipeline()
    adapter._pipeline = fake
    timestamps = [
        datetime(2026, 1, 1, 18, 30, tzinfo=timezone.utc),
        datetime(2026, 1, 2, 18, 30, tzinfo=timezone.utc),
        datetime(2026, 1, 5, 18, 30, tzinfo=timezone.utc),
    ]
    request = ForecastRequest("TEST", [100.0, 101.0, 102.0], timestamps, 2)

    result = adapter.forecast(request)

    assert list(fake.context.columns) == ["item_id", "timestamp", "target"]
    assert fake.context["item_id"].tolist() == ["TEST", "TEST", "TEST"]
    assert str(fake.context["timestamp"].dtype).startswith("datetime64[")
    assert fake.context["timestamp"].dt.tz is None
    assert fake.context["target"].tolist() == [100.0, 101.0, 102.0]
    assert fake.context["timestamp"].is_monotonic_increasing
    assert fake.context["timestamp"].is_unique
    assert fake.kwargs["prediction_length"] == 2
    assert fake.kwargs["id_column"] == "item_id"
    assert fake.kwargs["timestamp_column"] == "timestamp"
    assert fake.kwargs["target"] == "target"
    assert fake.kwargs["freq"] == "B"
    assert fake.kwargs["quantile_levels"] == [0.1, 0.5, 0.9]
    assert result.median == [101.0, 102.0]
