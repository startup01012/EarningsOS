# EarningsOS runtime architecture

EarningsOS is split into a lightweight browser surface and a data/model API.

## Web

Vercel project:
- Repository: `startup01012/EarningsOS`
- Root directory: `apps/web`
- Framework: Next.js
- Production domain: `https://earnings-os.vercel.app`

Set:

```text
NEXT_PUBLIC_API_URL=https://<your-api-deployment>
```

Without this variable the UI deliberately remains in explicit fallback mode. It does not present fallback numbers as live market data.

## API

The FastAPI service lives under `apps/api`.

Endpoints:

- `GET /health`
- `GET /health/db`
- `GET /api/v1/models`
- `GET /api/v1/market/{symbol}`
- `GET /api/v1/watchlist`
- `GET /api/v1/intelligence/{symbol}`
- `GET /api/v1/forecast/{symbol}`

The API reads Neon PostgreSQL and serves stored forecasts. Heavy pretrained-model inference is intentionally not triggered by a browser request.

For a separate Vercel deployment, create another Vercel project against the same repository and set its Root Directory to `apps/api`. Configure:

```text
DATABASE_URL=postgresql+psycopg://...
FRONTEND_ORIGINS=https://earnings-os.vercel.app
```

If the Python runtime cannot resolve the repository package imports in the selected Vercel configuration, deploy the same `apps/api` service to a Python-capable service instead; the web contract remains unchanged.

## Forecasting policy

The forecasting adapters are pretrained-only:

- Amazon Chronos-2
- Google TimesFM 2.5
- NeoQuasar Kronos-small
- FinCast v1

The web API only reads validated forecasts stored in PostgreSQL. Forecast generation belongs to a worker/benchmark environment with model weights installed.

This prevents a large model checkpoint from being downloaded during a user request and avoids unpredictable serverless latency.

## Leakage boundary

For baseline sentiment and other pre-result features:

`published_at <= data_boundary`

is mandatory. Historical evaluation must pass the actual result-announcement timestamp as the boundary. Transcripts, calls, media releases, corrections, or other post-result documents must not be used to construct pre-result features.

## Forecast evaluation

The pretrained screening harness compares every model against:

- last close
- previous return
- drift
- 20-session moving average

Paired comparisons use identical causal cutoffs. HAC dependence lag is derived from forecast horizon and backtest stride so overlapping forecast windows are not treated as independent.
