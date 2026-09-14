from __future__ import annotations

import argparse

from apps.api.db.session import SessionLocal
from scripts.backtest_nifty50_expanded_v2 import (
    BASELINES,
    HORIZONS,
    REGIMES,
    rows_for,
    symbols_from_csv,
    write_csv,
)
from scripts.backtest_nifty50_paired import pooled_summary, print_gate
from services.forecasting.backtest import BacktestCase, run_backtest
from services.forecasting.benchmarks import baseline_predictions
from services.forecasting.chronos import Chronos2Adapter
from services.forecasting.fincast import FinCastAdapter
from services.forecasting.price_loader import load_close_observations
from services.forecasting.paired import PairedCaseResult, PairedComparison, compare_paired, dedupe_paired_cases
from services.forecasting.timesfm import TimesFM25Adapter


def standard_cases(symbol: str, args, adapter, db) -> list[BacktestCase]:
    observations = load_close_observations(
        db,
        symbol,
        interval=args.interval,
        source=args.source,
        limit=args.history_limit,
    )
    return run_backtest(
        adapter,
        symbol,
        observations,
        context_length=args.context_length,
        horizon=10,
        stride=args.stride,
        start_case=args.start_case,
        max_cases=args.max_cases,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained-only paired NIFTY 50 causal benchmark")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--nifty50-csv", default="data/reference/nifty50.csv")
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--history-limit", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fincast-root", default=None)
    parser.add_argument("--fincast-model-path", default=None)
    parser.add_argument("--models", default="timesfm,fincast")
    parser.add_argument("--output", default="data/processed/forecast_benchmark_v5.csv")
    parser.add_argument("--case-output", default="data/processed/forecast_benchmark_cases_v5.csv")
    parser.add_argument("--paired-output", default="data/processed/forecast_benchmark_paired_v5.csv")
    args = parser.parse_args()

    models = tuple(x.strip().lower() for x in args.models.split(",") if x.strip())
    allowed = {"chronos", "kronos", "timesfm", "fincast"}
    if not models or any(model not in allowed for model in models):
        raise ValueError(f"--models must contain only: {', '.join(sorted(allowed))}")
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()] if args.symbols else symbols_from_csv(args.nifty50_csv)
    expected = len(symbols) * args.max_cases

    print("=== NIFTY 50 PRETRAINED-ONLY CORRECTED PAIRED BENCHMARK ===")
    print(f"Stocks: {len(symbols)} | Horizons: {HORIZONS} | Context: {args.context_length} | Stride: {args.stride} | Cases/stock: {args.max_cases}")
    print(f"Models: {models} | Baselines: {BASELINES}")
    print(f"Expected canonical paired cases per horizon/model/baseline: {expected}")

    db = SessionLocal()
    aggregate_rows = []
    paired_rows: list[PairedCaseResult] = []
    per_stock_rows: list[PairedComparison] = []
    adapters = {
        "Chronos-2": Chronos2Adapter(device_map=args.device) if "chronos" in models else None,
        "TimesFM-2.5": TimesFM25Adapter(device=args.device) if "timesfm" in models else None,
        "FinCast-v1": FinCastAdapter(
            model_path=args.fincast_model_path,
            fincast_root=args.fincast_root,
            device=args.device,
        ) if "fincast" in models else None,
    }

    try:
        for number, symbol in enumerate(symbols, 1):
            print(f"[{number:02d}/{len(symbols)}] {symbol}")
            model_results: dict[str, tuple[list[BacktestCase], list[list[float]]]] = {}
            for model_name, adapter in adapters.items():
                if adapter is None:
                    continue
                try:
                    cases = standard_cases(symbol, args, adapter, db)
                    predictions = [case.predicted for case in cases]
                    model_results[model_name] = (cases, predictions)
                    aggregate_rows.extend(rows_for(symbol, model_name, cases, predictions))
                    print(f"  {model_name}: {len(cases)} cases")
                except Exception as exc:
                    print(f"  {model_name} ERROR: {exc}")

            for model_name, (cases, predictions) in model_results.items():
                for baseline_name, baseline_prediction in baseline_predictions(cases).items():
                    for horizon in HORIZONS:
                        for regime in REGIMES:
                            comparison, details = compare_paired(
                                cases,
                                predictions,
                                baseline_prediction,
                                model=model_name,
                                baseline=baseline_name,
                                horizon=horizon,
                                regime=regime,
                                symbol=symbol,
                                dependence_lag=max(0, (horizon - 1) // max(1, args.stride)),
                            )
                            per_stock_rows.append(comparison)
                            paired_rows.extend(details)
    finally:
        db.close()

    paired_rows = dedupe_paired_cases(paired_rows, include_regime=True)
    if not aggregate_rows or not paired_rows:
        raise RuntimeError("No benchmark rows were produced. Check model installation and input paths.")

    write_csv(aggregate_rows, args.output)
    write_csv(paired_rows, args.case_output)
    write_csv(per_stock_rows, args.paired_output)
    print(f"\nWrote {len(aggregate_rows)} aggregate rows to {args.output}")
    print(f"Wrote {len(paired_rows)} unique case/regime rows to {args.case_output}")
    print(f"Wrote {len(per_stock_rows)} per-stock paired summaries to {args.paired_output}")

    pooled: list[PairedComparison] = []
    for model in sorted({row.model for row in paired_rows}):
        model_summary = pooled_summary(paired_rows, model)
        pooled.extend(model_summary)
        print(f"\n=== {model} POOLED RESULTS ===")
        for item in model_summary:
            if item.regime != "all":
                continue
            print(
                f"{item.model:20s} vs {item.baseline:20s} h={item.horizon:2d} n={item.cases:4d} "
                f"model={item.model_loss_mean*100:.4f}% baseline={item.baseline_loss_mean*100:.4f}% "
                f"diff={item.mean_loss_diff*100:+.4f}pp CI=[{item.ci_low*100:+.4f},{item.ci_high*100:+.4f}]pp "
                f"wins={item.model_win_rate*100:.2f}%"
            )

    for model in sorted({row.model for row in paired_rows}):
        for horizon in HORIZONS:
            counts = [
                item.cases
                for item in pooled
                if item.model == model and item.horizon == horizon and item.regime == "all"
            ]
            if not counts or any(count != expected for count in counts):
                raise RuntimeError(
                    f"case-count invariant failed for {model} h={horizon}: expected {expected}, got {counts}"
                )
    print(f"\nCASE-COUNT INVARIANT: PASS ({expected} unique paired cases per horizon/model/baseline)")
    print_gate(pooled)


if __name__ == "__main__":
    main()
