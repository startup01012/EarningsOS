# Pretrained forecasting models

EarningsOS benchmarks forecasting models zero-shot. The benchmark does not train or fine-tune any of these models.

## TimesFM 2.5

Model: `google/timesfm-2.5-200m-pytorch`

- 200M parameters
- up to 16,384 context points
- point forecast plus q10/q90 quantiles
- Apache-2.0 model weights
- adapter: `services/forecasting/timesfm.py`

The official TimesFM 2.5 API requires the Google Research source package rather than the old PyPI 1.x API. Install the current TimesFM source with its PyTorch extra according to the official repository instructions.

The adapter supplies only the causal close history in `ForecastRequest.values`. It does not pass future observations or future covariates.

## FinCast v1

Model: `Vincent05R/FinCast` checkpoint `v1.pth`

- financial time-series foundation model
- pretrained zero-shot inference
- context supported by the official inference configuration: 32–1024
- horizon supported by the official inference configuration: 1–256
- native probabilistic output when the checkpoint returns the full distribution
- adapter: `services/forecasting/fincast.py`

The released `v1.pth` checkpoint is approximately 3.97 GB, so it is intentionally not committed to EarningsOS. Download it from the official Hugging Face model repository and set:

```bash
export FINCAST_ROOT=/path/to/FinCast-fts
export FINCAST_MODEL_PATH=/path/to/FinCast/v1.pth
```

The official FinCast repository currently documents a Python 3.11 environment. If the EarningsOS Codespace Python version cannot import the official FinCast stack, keep FinCast in a compatible environment rather than modifying the benchmark methodology or checkpoint.

## Common paired benchmark

The corrected benchmark is causal rolling-origin and compares each pretrained model against the same baselines:

- last close
- previous return
- drift
- moving average 20

It evaluates 1-day, 5-day and 10-day horizons using the same context, stride, case selection, paired loss differences, case-level bootstrap confidence intervals, and strongest-baseline production gate used for the Chronos-2 evaluation.

Run both models after their dependencies are installed:

```bash
PYTHONPATH=. python scripts/backtest_nifty50_pretrained.py \
  --models timesfm,fincast \
  --max-cases 60
```

For TimesFM only:

```bash
PYTHONPATH=. python scripts/backtest_nifty50_pretrained.py \
  --models timesfm \
  --max-cases 60
```

For FinCast only:

```bash
PYTHONPATH=. python scripts/backtest_nifty50_pretrained.py \
  --models fincast \
  --fincast-root "$FINCAST_ROOT" \
  --fincast-model-path "$FINCAST_MODEL_PATH" \
  --max-cases 60
```

Do not declare a production winner from model-vs-model performance alone. The corrected gate requires a model to beat the strongest practical baseline at every evaluated horizon.
