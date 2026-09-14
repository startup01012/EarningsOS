# Pretrained forecasting model screening

EarningsOS uses a staged evaluation funnel so expensive full NIFTY 50 benchmarks are reserved for models that survive a cheaper screen.

## Stage 1: fast screen

Default command:

```bash
PYTHONPATH=. python scripts/screen_pretrained_models.py
```

Defaults:

- ADANIENT, AXISBANK, HDFCBANK, INFY, RELIANCE
- 15 causal rolling-origin cases per stock/model
- 1d, 5d and 10d horizons from one 10-step forecast
- 256 observations of context
- yfinance source
- no training or fine-tuning

The screen currently exercises the four pretrained adapters already integrated in EarningsOS:

- Chronos-2
- TimesFM-2.5
- Kronos-small
- FinCast-v1

Models that cannot initialize because their optional dependency/checkpoint is unavailable are reported as `SKIPPED` rather than causing the other models to fail.

## Stage 2: shortlist

Compare models primarily against the strongest simple baseline (`last_close`) and inspect direction, confidence intervals, runtime, and consistency across the five representative stocks. A model that only beats weak baselines should not automatically become a production model.

## Stage 3: full benchmark

Only shortlisted models should be run through the full 50-stock paired benchmark. The full benchmark remains the final production gate.

## Important interpretation rule

The model is not required to beat every baseline to remain useful. `previous_return` and moving-average baselines can be weak. The production decision should use the strongest baseline and should consider direction, calibration, regime behavior, runtime, and eventual earnings-event performance in addition to raw price error.

## Finance-specific versus general TSFM

Kronos is evaluated through its OHLCV interface. Chronos-2, TimesFM-2.5 and FinCast are evaluated through their close-series interfaces in the current benchmark harness. This distinction must remain explicit; an OHLCV model should not silently be reduced to a close-only interface when the model requires its full input schema.
