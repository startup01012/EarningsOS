from __future__ import annotations

from datetime import datetime, timezone

from .base import ForecastAdapter, ForecastRequest, ForecastResult


class TimesFM25Adapter(ForecastAdapter):
    """Lazy-loaded adapter for Google's pretrained TimesFM 2.5."""

    model_id = "google/timesfm-2.5-200m-pytorch"
    max_context = 16_384

    def __init__(self, device: str = "cpu", per_core_batch_size: int = 1) -> None:
        self.device = device
        self.per_core_batch_size = per_core_batch_size
        self._model = None

    def _load(self, context_length: int, horizon: int):
        if self._model is None:
            try:
                import torch
                import timesfm
            except ImportError as exc:
                raise RuntimeError(
                    "TimesFM 2.5 is not installed. Install the Google Research "
                    "TimesFM source package with its PyTorch extra before inference."
                ) from exc
            if context_length < 32:
                raise ValueError("TimesFM 2.5 requires at least 32 context observations")
            if context_length > self.max_context:
                raise ValueError(f"TimesFM 2.5 context {context_length} exceeds {self.max_context}")
            if horizon < 1 or horizon > 256:
                raise ValueError("TimesFM 2.5 adapter supports horizons from 1 to 256")

            torch.set_float32_matmul_precision("high")
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
            if self.device != "cpu" and hasattr(model, "to"):
                model = model.to(self.device)
            model.compile(
                timesfm.ForecastConfig(
                    max_context=context_length,
                    max_horizon=max(256, horizon),
                    normalize_inputs=True,
                    per_core_batch_size=self.per_core_batch_size,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            self._model = model
        return self._model

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        import numpy as np

        if any(not np.isfinite(value) for value in request.values):
            raise ValueError("TimesFM input contains non-finite values")
        model = self._load(len(request.values), request.horizon)
        point, quantiles = model.forecast(
            horizon=request.horizon,
            inputs=[np.asarray(request.values, dtype=np.float32)],
        )
        median = np.asarray(point)[0, : request.horizon].astype(float).tolist()
        q = np.asarray(quantiles)
        if q.ndim != 3 or q.shape[2] < 10:
            raise RuntimeError(f"TimesFM 2.5 returned unexpected quantile shape: {q.shape}")
        lower = q[0, : request.horizon, 1].astype(float).tolist()
        upper = q[0, : request.horizon, 9].astype(float).tolist()
        return ForecastResult(
            model_id=self.model_id,
            symbol=request.symbol.strip().upper(),
            generated_at=datetime.now(timezone.utc),
            horizon=request.horizon,
            median=median,
            lower=lower,
            upper=upper,
        )
