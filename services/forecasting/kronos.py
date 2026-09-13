from __future__ import annotations

from datetime import datetime, timezone

from .base import ForecastAdapter, ForecastRequest, ForecastResult


class KronosSmallAdapter(ForecastAdapter):
    """Lazy-loaded, pretrained Kronos-small financial K-line forecaster.

    Kronos consumes OHLCV/amount candles rather than a close-only series. This
    adapter therefore requires the corresponding fields on ForecastRequest.
    No training or fine-tuning is performed.

    Kronos uses torch.multinomial during autoregressive token sampling. When a
    seed is supplied, the adapter resets the PyTorch RNG immediately before
    inference so repeated forecasts with identical inputs are reproducible.
    """

    model_id = "NeoQuasar/Kronos-small"
    tokenizer_id = "NeoQuasar/Kronos-Tokenizer-base"

    def __init__(
        self,
        device: str | None = None,
        max_context: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
        seed: int | None = None,
    ) -> None:
        if max_context < 2:
            raise ValueError("max_context must be >= 2")
        if sample_count < 1:
            raise ValueError("sample_count must be >= 1")
        if seed is not None and seed < 0:
            raise ValueError("seed must be >= 0")
        self.device = device
        self.max_context = max_context
        self.temperature = temperature
        self.top_p = top_p
        self.sample_count = sample_count
        self.seed = seed
        self._predictor = None

    def _load(self):
        if self._predictor is None:
            try:
                from model import Kronos, KronosPredictor, KronosTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Kronos is not installed. Install kronos-model-arch==0.1.0 "
                    "before running Kronos inference."
                ) from exc

            tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_id)
            model = Kronos.from_pretrained(self.model_id)
            self._predictor = KronosPredictor(
                model,
                tokenizer,
                device=self.device,
                max_context=self.max_context,
            )
        return self._predictor

    def _seed_rng(self) -> None:
        if self.seed is None:
            return

        import torch

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        import pandas as pd

        required = {
            "opens": request.opens,
            "highs": request.highs,
            "lows": request.lows,
            "volumes": request.volumes,
        }
        missing = [name for name, values in required.items() if values is None]
        if missing:
            raise ValueError(
                "Kronos requires OHLCV context; missing: " + ", ".join(missing)
            )

        amounts = request.amounts
        if amounts is None:
            amounts = [0.0] * len(request.values)

        context = pd.DataFrame(
            {
                "open": request.opens,
                "high": request.highs,
                "low": request.lows,
                "close": request.values,
                "volume": request.volumes,
                "amount": amounts,
            }
        )
        x_timestamp = pd.Series(
            pd.to_datetime(request.timestamps, utc=True).tz_localize(None),
            name="timestamps",
        )
        y_timestamp = pd.Series(
            pd.bdate_range(
                start=x_timestamp.iloc[-1] + pd.Timedelta(days=1),
                periods=request.horizon,
            ),
            name="timestamps",
        )

        predictor = self._load()
        self._seed_rng()
        result = predictor.predict(
            df=context,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=request.horizon,
            T=self.temperature,
            top_p=self.top_p,
            sample_count=self.sample_count,
            verbose=False,
        )
        if result.empty or "close" not in result.columns:
            raise RuntimeError("Kronos returned an invalid forecast")
        if len(result) != request.horizon:
            raise RuntimeError(
                f"Kronos returned {len(result)} rows for horizon {request.horizon}"
            )

        return ForecastResult(
            model_id=self.model_id,
            symbol=request.symbol.strip().upper(),
            generated_at=datetime.now(timezone.utc),
            horizon=request.horizon,
            median=result["close"].astype(float).tolist(),
        )
