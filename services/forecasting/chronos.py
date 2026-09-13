from __future__ import annotations

from datetime import datetime, timezone

from .base import ForecastAdapter, ForecastRequest, ForecastResult


class Chronos2Adapter(ForecastAdapter):
    """Lazy-loaded adapter for Amazon's pretrained Chronos-2 model.

    Inference uses the public ``predict_df`` API. No training or fine-tuning is
    performed. The optional dependency is imported only when inference is
    requested so importing this package does not download model weights.
    """

    model_id = "amazon/chronos-2"

    def __init__(self, device_map: str = "cpu") -> None:
        self.device_map = device_map
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            try:
                from chronos import BaseChronosPipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Chronos is not installed. Install chronos-forecasting "
                    "before running Chronos inference."
                ) from exc
            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device_map,
            )
        return self._pipeline

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        import pandas as pd

        pipeline = self._load()

        # Chronos-2's dataframe normalizer currently calls ``to_numpy().view("int64")``
        # on timestamps. Passing a timezone-aware pandas extension array can become
        # object dtype with the installed pandas/Chronos combination, which causes:
        # ``TypeError: Cannot change data-type for array of references.``
        # Keep the timestamps in UTC, then remove the timezone while preserving the
        # instant. This gives Chronos a concrete datetime64[ns] ndarray that its
        # normalizer can safely reinterpret as int64 nanoseconds.
        timestamps = pd.to_datetime(request.timestamps, utc=True).tz_localize(None)
        context_df = pd.DataFrame(
            {
                "item_id": [request.symbol] * len(request.values),
                "timestamp": timestamps,
                "target": request.values,
            }
        )

        # NSE equity daily observations occur on trading days, so the timestamp
        # sequence is intentionally not calendar-daily (weekends and exchange
        # holidays are absent). Chronos-2 requires a regular/inferable frequency
        # when ``freq`` is omitted. Explicitly use pandas BusinessDay frequency to
        # describe the trading-day cadence instead of asking Chronos to infer a
        # calendar frequency from the irregular calendar dates.
        result = pipeline.predict_df(
            context_df,
            prediction_length=request.horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            freq="B",
        )

        if result.empty:
            raise RuntimeError("Chronos-2 returned an empty forecast")

        result = result.sort_values("timestamp")
        expected_columns = {"0.1", "0.5", "0.9"}
        missing = expected_columns.difference(result.columns)
        if missing:
            raise RuntimeError(
                f"Chronos-2 forecast is missing quantile columns: {sorted(missing)}"
            )

        if len(result) != request.horizon:
            raise RuntimeError(
                f"Chronos-2 returned {len(result)} rows for horizon "
                f"{request.horizon}"
            )

        return ForecastResult(
            model_id=self.model_id,
            symbol=request.symbol.strip().upper(),
            generated_at=datetime.now(timezone.utc),
            horizon=request.horizon,
            median=result["0.5"].astype(float).tolist(),
            lower=result["0.1"].astype(float).tolist(),
            upper=result["0.9"].astype(float).tolist(),
        )
