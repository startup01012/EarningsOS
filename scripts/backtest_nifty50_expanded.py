from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from apps.api.db.session import SessionLocal
from services.forecasting.backtest import BacktestCase, REGIME_BEAR, REGIME_BULL, REGIME_SIDEWAYS_HIGH_VOLATILITY, classify_regime, evaluate_cases, run_backtest
from services.forecasting.benchmarks import BASELINE_DRIFT, BASELINE_LAST_CLOSE, BASELINE_MOVING_AVERAGE_20, BASELINE_PREVIOUS_RETURN, baseline_predictions, bootstrap_mean_ci, evaluate_return_predictions
from services.forecasting.base import ForecastRequest
from services.forecasting.chronos import Chronos2Adapter
from services.forecasting.kronos import KronosSmallAdapter
from services.forecasting.kronos_loader import load_ohlcv_observations
from services.forecasting.price_loader import load_close_observations

HORIZONS = (1, 5, 10)
REGIMES = ("all", REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS_HIGH_VOLATILITY)
BASELINES = (BASELINE_LAST_CLOSE, BASELINE_PREVIOUS_RETURN, BASELINE_DRIFT, BASELINE_MOVING_AVERAGE_20)

@dataclass(frozen=True)
class Row:
    symbol: str
    model: str
    horizon: int
    regime: str
    cases: int
    price_mae: float
    price_rmse: float
    return_mae: float
    return_rmse: float
    directional_accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    information_coefficient: float | None


def rows_for(symbol: str, model: str, cases: list[BacktestCase], predictions: list[list[float]]) -> list[Row]:
    output: list[Row] = []
    for horizon in HORIZONS:
        for regime in REGIMES:
            idx = [i for i, case in enumerate(cases) if regime == "all" or case.regime == regime]
            if not idx:
                continue
            selected_cases = [cases[i] for i in idx]
            selected_predictions = [predictions[i] for i in idx]
            ret = evaluate_return_predictions(selected_cases, selected_predictions, horizon=horizon)
            price_cases = [BacktestCase(c.cutoff_timestamp, c.actual[:horizon], selected_predictions[j][:horizon], c.cutoff_close, c.context_closes, c.regime) for j, c in enumerate(selected_cases)]
            price = evaluate_cases(price_cases)
            output.append(Row(symbol, model, horizon, regime, len(idx), price.mae, price.rmse, ret.return_mae, ret.return_rmse, ret.directional_accuracy, ret.precision, ret.recall, ret.f1, ret.information_coefficient))
    return output


def symbols_from_csv(path: str) -> list[str]:
    frame = pd.read_csv(path)
    if "Symbol" not in frame.columns:
        raise ValueError(f"{path} must contain Symbol")
    symbols = list(dict.fromkeys(str(v).strip().upper() for v in frame["Symbol"].dropna() if str(v).strip()))
    if len(symbols) != 50:
        raise ValueError(f"expected 50 NIFTY 50 symbols, found {len(symbols)}")
    return symbols


def chronos_cases(symbol: str, args, adapter, db):
    observations = load_close_observations(db, symbol, interval=args.interval, source=args.source, limit=args.history_limit)
    return run_backtest(adapter, symbol, observations, context_length=args.context_length, horizon=10, stride=args.stride, start_case=args.start_case, max_cases=args.max_cases)[0]


def kronos_cases(symbol: str, args, adapter, db):
    observations = load_ohlcv_observations(db, symbol, interval=args.interval, source=args.source, limit=args.history_limit)
    need = args.context_length + 10
    if len(observations) < need:
        raise ValueError(f"{symbol}: need {need} OHLCV rows, got {len(observations)}")
    cases: list[BacktestCase] = []
    cutoff = args.context_length + args.start_case * args.stride
    while cutoff + 10 <= len(observations) and len(cases) < args.max_cases:
        context = observations[cutoff - args.context_length:cutoff]
        future = observations[cutoff:cutoff + 10]
        closes = [x.close for x in context]
        request = ForecastRequest(symbol=symbol, values=closes, timestamps=[x.timestamp for x in context], horizon=10, opens=[x.open for x in context], highs=[x.high for x in context], lows=[x.low for x in context], volumes=[x.volume for x in context], amounts=[x.amount for x in context])
        forecast = adapter.forecast(request)
        cases.append(BacktestCase(context[-1].timestamp, [x.close for x in future], forecast.median, context[-1].close, closes, classify_regime(closes)))
        cutoff += args.stride
    return cases


def write_csv(rows: list[Row], path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(Row.__dataclass_fields__)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def summarize(rows: list[Row]):
    print("\n=== RETURN MAE: MEAN / STD / 95% BOOTSTRAP CI ===")
    models = sorted({r.model for r in rows})
    for model in models:
        for horizon in HORIZONS:
            for regime in REGIMES:
                values = [r.return_mae for r in rows if r.model == model and r.horizon == horizon and r.regime == regime]
                if not values:
                    continue
                s = bootstrap_mean_ci(values, seed=20260913 + horizon)
                print(f"{model:22s} h={horizon:2d} {regime:24s} n={s.n:2d} mean={s.mean*100:.4f}% std={s.std*100:.4f}% CI=[{s.ci_low*100:.4f}%, {s.ci_high*100:.4f}%]")

    print("\n=== DIRECTIONAL ACCURACY: MEAN / STD / 95% BOOTSTRAP CI ===")
    for model in models:
        for horizon in HORIZONS:
            for regime in REGIMES:
                values = [r.directional_accuracy for r in rows if r.model == model and r.horizon == horizon and r.regime == regime and r.directional_accuracy is not None]
                if not values:
                    continue
                s = bootstrap_mean_ci([float(v) for v in values], seed=303000 + horizon)
                print(f"{model:22s} h={horizon:2d} {regime:24s} n={s.n:2d} mean={s.mean*100:.2f}% std={s.std*100:.2f}% CI=[{s.ci_low*100:.2f}%, {s.ci_high*100:.2f}%]")

    print("\n=== MODEL VS BEST STRONGER BASELINE ===")
    for horizon in HORIZONS:
        baseline_means = {}
        for baseline in BASELINES:
            values = [r.return_mae for r in rows if r.model == baseline and r.horizon == horizon and r.regime == "all"]
            if values:
                baseline_means[baseline] = sum(values) / len(values)
        if not baseline_means:
            continue
        best_name, best = min(baseline_means.items(), key=lambda item: item[1])
        print(f"\nHorizon {horizon}d: best baseline={best_name} MAE={best*100:.4f}%")
        for model in models:
            if model in BASELINES:
                continue
            values = [r.return_mae for r in rows if r.model == model and r.horizon == horizon and r.regime == "all"]
            if not values:
                continue
            s = bootstrap_mean_ci(values, seed=400000 + horizon)
            print(f"  {model}: MAE={s.mean*100:.4f}% delta_vs_best={((s.mean-best)*100):+.4f}pp CI=[{s.ci_low*100:.4f}%, {s.ci_high*100:.4f}%]")

    print("\n=== PRODUCTION DECISION GATE ===")
    print("Pass requires lower return MAE than the best baseline at 1d, 5d and 10d, with the model's 95% bootstrap CI entirely below that baseline mean.")
    candidates = []
    for model in models:
        if model in BASELINES:
            continue
        passed = True
        for horizon in HORIZONS:
            baseline_values = [r.return_mae for r in rows if r.model in BASELINES and r.horizon == horizon and r.regime == "all"]
            model_values = [r.return_mae for r in rows if r.model == model and r.horizon == horizon and r.regime == "all"]
            if not baseline_values or not model_values:
                passed = False
                break
            best = min(baseline_values)
            ci = bootstrap_mean_ci(model_values, seed=500000 + horizon)
            if ci.ci_low is None or ci.ci_low >= best:
                passed = False
        if passed:
            candidates.append(model)
    if candidates:
        print(f"Candidate models: {', '.join(candidates)}")
    else:
        print("No pretrained model passes the production gate.")
        print("Decision: keep Chronos-2/Kronos-small as research candidates; do not promote either to production yet.")


def main():
    parser = argparse.ArgumentParser(description="Expanded 1d/5d/10d causal NIFTY 50 benchmark")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--nifty50-csv", default="data/reference/nifty50.csv")
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--history-limit", type=int, default=1000)
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--kronos-seed", type=int, default=101)
    parser.add_argument("--models", default="chronos,kronos")
    parser.add_argument("--output", default="data/processed/forecast_benchmark_v2.csv")
    args = parser.parse_args()
    models = tuple(x.strip().lower() for x in args.models.split(",") if x.strip())
    if not models or any(x not in ("chronos", "kronos") for x in models):
        raise ValueError("--models must contain chronos and/or kronos")
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()] if args.symbols else symbols_from_csv(args.nifty50_csv)
    if not symbols:
        raise ValueError("no symbols")
    print("=== NIFTY 50 EXPANDED CAUSAL FORECAST BENCHMARK ===")
    print(f"Stocks: {len(symbols)} | Horizons: 1d, 5d, 10d | Context: {args.context_length} | Stride: {args.stride} | Cases/stock: {args.max_cases}")
    print(f"Models: {', '.join(models)} | Baselines: {', '.join(BASELINES)}")
    print("Regime is determined from each cutoff context only: bull, bear, sideways/high-volatility.")

    db = SessionLocal()
    rows: list[Row] = []
    chronos = Chronos2Adapter(device_map=args.device_map) if "chronos" in models else None
    kronos = KronosSmallAdapter(device=args.device_map, seed=args.kronos_seed) if "kronos" in models else None
    try:
        for number, symbol in enumerate(symbols, 1):
            print(f"[{number:02d}/{len(symbols)}] {symbol}")
            shared_cases = None
            if chronos is not None:
                try:
                    cases = chronos_cases(symbol, args, chronos, db)
                    rows.extend(rows_for(symbol, "Chronos-2", cases, [c.predicted for c in cases]))
                    shared_cases = cases
                    print(f"  Chronos-2: {len(cases)} cases")
                except Exception as exc:
                    print(f"  Chronos-2 ERROR: {exc}")
            if kronos is not None:
                try:
                    cases = kronos_cases(symbol, args, kronos, db)
                    rows.extend(rows_for(symbol, "Kronos-small", cases, [c.predicted for c in cases]))
                    if shared_cases is None:
                        shared_cases = cases
                    print(f"  Kronos-small: {len(cases)} cases")
                except Exception as exc:
                    print(f"  Kronos-small ERROR: {exc}")
            if shared_cases is not None:
                for name, predictions in baseline_predictions(shared_cases).items():
                    rows.extend(rows_for(symbol, name, shared_cases, predictions))
    finally:
        db.close()
    if not rows:
        raise RuntimeError("No benchmark rows were produced")
    write_csv(rows, args.output)
    print(f"\nWrote {len(rows)} rows to {args.output}")
    summarize(rows)


if __name__ == "__main__":
    main()
