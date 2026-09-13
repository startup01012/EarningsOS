from __future__ import annotations

from datetime import datetime, timezone

from .base import ForecastAdapter, ForecastRequest, ForecastResult


class Chronos2Adapter(ForecastAdapter):
    """Lazy-loaded adapter for Amazon's pretrained Chronos-2 model.

    No training or fine-tuning is performed. The optional dependency is imported
    only when inference is requested so the API/tests do not download model
    weights just by importing the package.
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
        import torch

        if len(request.values) != len(request.timestamps):
            raise ValueError("values and timestamps must have the same length")

        pipeline = self._load()
        context = torch.tensor(request.values, dtype=torch.float32)
        forecast = pipeline.predict(context, request.horizon)

        samples = forecast[0].detach().cpu().numpy()
        # Chronos returns [series, samples, horizon]. Keep probabilistic output
        # rather than reducing the model to a fabricated point probability.
        import numpy as np

        lower, median, upper = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)

        return ForecastResult(
            model_id=self.model_id,
            symbol=request.symbol.strip().upper(),
            generated_at=datetime.now(timezone.utc),
            horizon=request.horizon,
            median=median.tolist(),
            lower=lower.tolist(),
            upper=upper.tolist(),
        )
