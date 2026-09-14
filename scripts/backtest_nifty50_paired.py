from __future__ import annotations

import argparse

from apps.api.db.session import SessionLocal
from scripts.backtest_nifty50_expanded_v2 import BASELINES, HORIZONS, REGIMES, chronos_cases, kronos_cases, rows_for, symbols_from_csv, write_csv
from services.forecasting.benchmarks import baseline_predictions, bootstrap_mean_ci
from services.forecasting.chronos import Chronos2Adapter
from services.forecasting.kronos import KronosSmallAdapter
from services.forecasting.paired import PairedCaseResult, PairedComparison, compare_paired, dedupe_paired_cases, strongest_baseline


def pooled_summary(case_rows: list[PairedCaseResult], model: str) -> list[PairedComparison]:
    summaries: list[PairedComparison] = []
    for horizon in HORIZONS:
        for baseline in BASELINES:
            for regime in REGIMES:
                selected = [
                    row for row in case_rows
                    if row.model == model and row.baseline == baseline and row.horizon == horizon
                    and (regime == "all" or row.regime == regime)
                ]
                selected = dedupe_paired_cases(selected, include_regime=(regime != "all"))
                if not selected:
                    continue
                diffs = [row.loss_diff for row in selected]
                ci = bootstrap_mean_ci(diffs, seed=610000 + horizon + len(baseline))
                model_hits = [row.model_direction_correct for row in selected if row.model_direction_correct is not None]
                baseline_hits = [row.baseline_direction_correct for row in selected if row.baseline_direction_correct is not None]
                model_da = sum(model_hits) / len(model_hits) if model_hits else None
                baseline_da = sum(baseline_hits) / len(baseline_hits) if baseline_hits else None
                ordered = sorted(diffs)
                mid = len(ordered) // 2
                median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
                summaries.append(PairedComparison(
                    model, baseline, horizon, regime, len(selected),
                    sum(row.model_loss for row in selected) / len(selected),
                    sum(row.baseline_loss for row in selected) / len(selected),
                    sum(diffs) / len(selected), median, ci.std, ci.ci_low, ci.ci_high,
                    sum(diff < 0 for diff in diffs) / len(selected),
                    sum(diff == 0 for diff in diffs) / len(selected),
                    sum(diff > 0 for diff in diffs) / len(selected),
                    None, None, model_da, baseline_da,
                    model_da - baseline_da if model_da is not None and baseline_da is not None else None,
                ))
    return summaries


def print_gate(summaries: list[PairedComparison]) -> None:
    print("\n=== CORRECTED PRODUCTION GATE ===")
    print("The gate uses the strongest baseline (lowest return MAE), not whichever baseline the model happens to beat.")
    print("Pass requires mean paired difference < 0, CI upper < 0, and model win rate > 50% at 1d, 5d and 10d.")
    for model in sorted({item.model for item in summaries}):
        passed = True
        for horizon in HORIZONS:
            candidates = [item for item in summaries if item.model == model and item.horizon == horizon and item.regime == "all"]
            best = strongest_baseline(candidates, horizon=horizon)
            if best is None:
                passed = False
                print(f"{model} h={horizon}: MISSING")
                continue
            print(f"{model} h={horizon}: strongest={best.baseline} baseline={best.baseline_loss_mean*100:.4f}% diff={best.mean_loss_diff*100:+.4f}pp CI=[{best.ci_low*100:+.4f},{best.ci_high*100:+.4f}]pp wins={best.model_win_rate*100:.2f}%")
            if best.mean_loss_diff >= 0 or best.ci_high is None or best.ci_high >= 0 or best.model_win_rate <= 0.50:
                passed = False
        print(f"{model}: {'PASS' if passed else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected paired NIFTY 50 causal forecast benchmark")
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
    parser.add_argument("--models", default="chronos")
    parser.add_argument("--output", default="data/processed/forecast_benchmark_v4.csv")
    parser.add_argument("--case-output", default="data/processed/forecast_benchmark_cases_v4.csv")
    parser.add_argument("--paired-output", default="data/processed/forecast_benchmark_paired_v4.csv")
    args = parser.parse_args()
    models = tuple(x.strip().lower() for x in args.models.split(",") if x.strip())
    if not models or any(model not in {"chronos", "kronos"} for model in models):
        raise ValueError("--models must contain chronos and/or kronos")
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()] if args.symbols else symbols_from_csv(args.nifty50_csv)
    expected = len(symbols) * args.max_cases

    print("=== NIFTY 50 CORRECTED PAIRED CAUSAL BENCHMARK ===")
    print(f"Stocks: {len(symbols)} | Horizons: {HORIZONS} | Context: {args.context_length} | Stride: {args.stride} | Cases/stock: {args.max_cases}")
    print(f"Models: {models} | Baselines: {BASELINES}")
    print(f"Expected canonical paired cases per horizon/model/baseline: {expected}")

    db = SessionLocal()
    aggregate_rows = []
    paired_rows: list[PairedCaseResult] = []
    per_stock_rows: list[PairedComparison] = []
    adapters = {
        "Chronos-2": Chronos2Adapter(device_map=args.device_map) if "chronos" in models else None,
        "Kronos-small": KronosSmallAdapter(device=args.device_map, seed=args.kronos_seed) if "kronos" in models else None,
    }
    try:
        for number, symbol in enumerate(symbols, 1):
            print(f"[{number:02d}/{len(symbols)}] {symbol}")
            model_results = {}
            if adapters["Chronos-2"] is not None:
                cases = chronos_cases(symbol, args, adapters["Chronos-2"], db)
                predictions = [case.predicted for case in cases]
                model_results["Chronos-2"] = (cases, predictions)
                aggregate_rows.extend(rows_for(symbol, "Chronos-2", cases, predictions))
                print(f"  Chronos-2: {len(cases)} cases")
            if adapters["Kronos-small"] is not None:
                cases = kronos_cases(symbol, args, adapters["Kronos-small"], db)
                predictions = [case.predicted for case in cases]
                model_results["Kronos-small"] = (cases, predictions)
                aggregate_rows.extend(rows_for(symbol, "Kronos-small", cases, predictions))
                print(f"  Kronos-small: {len(cases)} cases")

            for model_name, (cases, predictions) in model_results.items():
                for baseline_name, baseline_prediction in baseline_predictions(cases).items():
                    for horizon in HORIZONS:
                        for regime in REGIMES:
                            comparison, details = compare_paired(
                                cases, predictions, baseline_prediction,
                                model=model_name, baseline=baseline_name,
                                horizon=horizon, regime=regime, symbol=symbol,
                                dependence_lag=max(0, (horizon - 1) // max(1, args.stride)),
                            )
                            per_stock_rows.append(comparison)
                            paired_rows.extend(details)
    finally:
        db.close()

    paired_rows = dedupe_paired_cases(paired_rows, include_regime=True)
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
            print(f"{item.model:20s} vs {item.baseline:20s} h={item.horizon:2d} n={item.cases:4d} model={item.model_loss_mean*100:.4f}% baseline={item.baseline_loss_mean*100:.4f}% diff={item.mean_loss_diff*100:+.4f}pp CI=[{item.ci_low*100:+.4f},{item.ci_high*100:+.4f}]pp wins={item.model_win_rate*100:.2f}%")

    for model in sorted({row.model for row in paired_rows}):
        for horizon in HORIZONS:
            counts = [item.cases for item in pooled if item.model == model and item.horizon == horizon and item.regime == "all"]
            if not counts or any(count != expected for count in counts):
                raise RuntimeError(f"case-count invariant failed for {model} h={horizon}: expected {expected}, got {counts}")
    print(f"\nCASE-COUNT INVARIANT: PASS ({expected} unique paired cases per horizon/model/baseline)")
    print_gate(pooled)


if __name__ == "__main__":
    main()
