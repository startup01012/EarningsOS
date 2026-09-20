# EarningsOS ML worker architecture

## Deployment boundary

The web/API deployment must remain lightweight.

- `apps/web`: Next.js frontend on Vercel.
- `apps/api`: FastAPI + SQLAlchemy + Psycopg only. It reads validated results from Neon and does not load PyTorch or forecasting models.
- `requirements/ml.txt`: heavyweight pretrained inference dependencies for offline/batch execution.
- `services/ml_worker/`: batch inference and persistence layer.
- `scripts/run_forecasts.py`: command-line entry point for scheduled/manual batches.
- `.github/workflows/forecast-batch.yml`: scheduled/manual GitHub Actions worker.

## Data flow

```
pretrained model
      |
      v
ML worker -> validation -> Neon Forecast/ModelRun tables
                                      |
                                      v
                               FastAPI API
                                      |
                                      v
                                  Next.js
```

The API never executes heavy inference during an HTTP request.

## Models

The worker supports:

- Chronos-2
- TimesFM 2.5
- Kronos-small
- FinCast v1 when `FINCAST_MODEL_PATH` and `FINCAST_ROOT` are configured

All models are used pretrained-only. No training or fine-tuning is performed by the worker.

## Validation

Every successful run:

1. loads only historical `yfinance` price bars;
2. builds a causal `ForecastRequest`;
3. runs the selected pretrained adapter;
4. rejects non-finite/non-positive predictions;
5. validates prediction intervals when supplied;
6. records a `ModelRun` with status `validated`;
7. stores each forecast step in `Forecast`.

Failed model executions are recorded as `ModelRun.status = failed` with the error in `metadata_json`.

## Database secret

GitHub Actions requires a repository secret named `DATABASE_URL`. Never commit the Neon connection string to source control.

## Manual execution

```bash
python scripts/run_forecasts.py --models chronos --symbols RELIANCE,INFY --horizons 1,5,10
```

TimesFM/Kronos/FinCast can be selected with `--models` when their corresponding pretrained runtime dependencies are available.

The GitHub Actions workflow defaults to Chronos because it is the currently validated lightweight starting point. Models that have not passed the project's backtest production gate must not be treated as production-approved merely because inference succeeds.

## Vercel fix

`apps/api/requirements.txt` intentionally contains no PyTorch, Transformers, Chronos, Kronos, or other heavy inference dependencies. This prevents the API function from exceeding Vercel's serverless function bundle limit.
