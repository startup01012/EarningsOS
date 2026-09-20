from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .base import ForecastAdapter, ForecastRequest, ForecastResult


class FinCastAdapter(ForecastAdapter):
    """Adapter for the pretrained FinCast v1 financial foundation model.

    FinCast is loaded lazily from an external checkout because its official
    implementation is a research repository rather than a small pip package.
    No training or fine-tuning is performed by this adapter.
    """

    model_id = "Vincent05R/FinCast:v1"
    max_context = 1024
    max_horizon = 256

    def __init__(
        self,
        model_path: str | None = None,
        fincast_root: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path or os.getenv("FINCAST_MODEL_PATH")
        self.fincast_root = fincast_root or os.getenv("FINCAST_ROOT")
        self.device = device
        self._api = None
        self._loaded_shape: tuple[int, int] | None = None

    def _load(self, context_length: int, horizon: int):
        if self._api is not None and self._loaded_shape == (context_length, horizon):
            return self._api
        if not self.model_path:
            raise RuntimeError(
                "FINCAST_MODEL_PATH is not set. Point it to the pretrained FinCast "
                "v1 checkpoint before running FinCast inference."
            )
        if not self.fincast_root:
            raise RuntimeError(
                "FINCAST_ROOT is not set. Point it to a checkout of the official "
                "vincent05r/FinCast-fts repository."
            )
        if context_length < 32 or context_length > self.max_context:
            raise ValueError(
                f"FinCast context must be between 32 and {self.max_context}"
            )
        if context_length % 32 != 0:
            raise ValueError(
                "FinCast context_length must be a multiple of 32 according to the "
                "official inference configuration"
            )
        if horizon < 1 or horizon > self.max_horizon:
            raise ValueError(
                f"FinCast horizon must be between 1 and {self.max_horizon}"
            )
        if not Path(self.model_path).is_file():
            raise RuntimeError(f"FinCast model checkpoint not found: {self.model_path}")

        root = Path(self.fincast_root).resolve()
        src = root / "src"
        if not src.is_dir():
            raise RuntimeError(f"FinCast source directory not found: {src}")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        try:
            from types import SimpleNamespace
            from tools.inference_utils import get_model_api
        except ImportError as exc:
            raise RuntimeError(
                "FinCast dependencies/source are not importable. Follow the official "
                "FinCast-fts environment setup before inference."
            ) from exc

        config = SimpleNamespace(
            backend="gpu" if self.device not in {"cpu", "none"} else "cpu",
            model_path=self.model_path,
            model_version="v1",
            horizon_len=horizon,
            context_len=context_length,
            num_experts=4,
            gating_top_n=2,
            load_from_compile=True,
            forecast_mode="mean",
        )
        self._api = get_model_api(config)
        self._loaded_shape = (context_length, horizon)
        return self._api

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        import numpy as np

        values = np.asarray(request.values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("FinCast input contains non-finite values")

        api = self._load(len(values), request.horizon)
        # FinCast's official frequency mapper treats 0 as the fast-frequency
        # bucket used for daily/business-day data.
        outputs = api.forecast([values], [0])
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise RuntimeError("FinCast returned an unexpected forecast structure")

        point, full = outputs
        point = np.asarray(point)
        full = None if full is None else np.asarray(full)

        if point.ndim != 2 or point.shape[0] < 1 or point.shape[1] < request.horizon:
            raise RuntimeError(
                f"FinCast returned an unexpected point forecast shape: {point.shape}"
            )

        median = point[0, -request.horizon :].astype(float).tolist()
        lower = upper = None
        if full is not None:
            if full.ndim != 3 or full.shape[0] < 1 or full.shape[1] < request.horizon:
                raise RuntimeError(
                    f"FinCast returned an unexpected distribution shape: {full.shape}"
                )
            if full.shape[2] >= 10:
                lower = full[0, -request.horizon :, 1].astype(float).tolist()
                upper = full[0, -request.horizon :, 9].astype(float).tolist()
                median = full[0, -request.horizon :, 5].astype(float).tolist()

        return ForecastResult(
            model_id=self.model_id,
            symbol=request.symbol.strip().upper(),
            generated_at=datetime.now(timezone.utc),
            horizon=request.horizon,
            median=median,
            lower=lower,
            upper=upper,
        )
