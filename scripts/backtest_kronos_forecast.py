from __future__ import annotations

import argparse
from statistics import mean, median, pstdev

from apps.api.db.session import SessionLocal
from services.forecasting.backtest import BacktestCase, evaluate_cases
from services.forecasting.benchmarks import evaluate_predictions, naive_last_close_predictions
from services.forecasting.base import ForecastRequest
from services.forecasting.kronos import KronosSmallAdapter
from services.forecasting.kronos_loader import load_ohlcv_observations


def build_request(symbol: str, context, horizon: int) -> ForecastRequest:
    return ForecastRequest(
        symbol=symbol,
        values=[row.close for row in context],
        timestamps=[row.timestamp for row in context],
        horizon=horizon,
        opens=[row.open for row in context],
        highs=[row.high for row in context],
        lows=[row.low for row in context],
        volumes=[row.volume for row in context],
        amounts=[row.amount for row in context],
    )


def build_cases(observations, symbol: str, context_length: int, horizon: int, stride: int,
                start_case: int, max_cases: int) -> list[tuple[BacktestCase, object]]:
    cases: list[tuple[BacktestCase, object]] = []
    cutoff_end = context_length + start_case * stride

    if cutoff_end + horizon > len(observations):
        raise ValueError(
            "start_case points beyond the available observations: "
            f"cutoff requires {cutoff_end + horizon} observations, got {len(observations)}"
        )

    while cutoff_end + horizon <= len(observations):
        context = observations[cutoff_end - context_length : cutoff_end]
        future = observations[cutoff_end : cutoff_end + horizon]
        cases.append(
            (
                BacktestCase(
                    cutoff_timestamp=context[-1].timestamp,
                    actual=[row.close for row in future],
                    predicted=[],
                    cutoff_close=context[-1].close,
                ),
                context,
            )
        )
        if len(cases) >= max_cases:
            break
        cutoff_end += stride

    return cases


def metric_summary(values: list[float]) -> tuple[float, float, float]:
    return mean(values), median(values), pstdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a strictly causal multi-seed rolling backtest with pretrained Kronos-small"
    )
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--start-case", type=int, default=21)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--seeds",
        default="101,202,303,404,505",
        help="Comma-separated PyTorch seeds used for independent stochastic runs",
    )
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    if any(seed < 0 for seed in seeds):
        raise ValueError("all seeds must be >= 0")
    if args.context_length < 2:
        raise ValueError("--context-length must be >= 2")
    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.start_case < 0:
        raise ValueError("--start-case must be >= 0")
    if args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1")
    if args.context_length > 512:
        raise ValueError("Kronos-small supports max_context <= 512")

    db = SessionLocal()
    try:
        observations = load_ohlcv_observations(
            db,
            args.symbol,
            interval=args.interval,
            source=args.source,
            limit=args.history_limit,
        )
    finally:
        db.close()

    minimum = args.context_length + args.horizon
    if len(observations) < minimum:
        raise ValueError(
            f"at least {minimum} observations are required, got {len(observations)}"
        )

    case_contexts = build_cases(
        observations,
        args.symbol,
        args.context_length,
        args.horizon,
        args.stride,
        args.start_case,
        args.max_cases,
    )

    naive_cases = [case for case, _ in case_contexts]
    naive_predictions = naive_last_close_predictions(naive_cases)
    naive_metrics = evaluate_predictions(naive_cases, naive_predictions)

    adapter = KronosSmallAdapter(device=args.device, max_context=512)
    seed_metrics = []

    for seed in seeds:
        adapter.seed = seed
        cases: list[BacktestCase] = []
        for template, context in case_contexts:
            request = build_request(args.symbol, context, args.horizon)
            result = adapter.forecast(request)
            cases.append(
                BacktestCase(
                    cutoff_timestamp=template.cutoff_timestamp,
                    actual=template.actual,
                    predicted=result.median,
                    cutoff_close=template.cutoff_close,
                )
            )

        metrics = evaluate_cases(cases)
        seed_metrics.append(metrics)

        print(f"\n=== Kronos-small seed {seed} ===")
        print(f"MAE: {metrics.mae:.6f}")
        print(f"RMSE: {metrics.rmse:.6f}")
        print(f"MAPE: {metrics.mape:.6f}%" if metrics.mape is not None else "MAPE: N/A")
        print(f"sMAPE: {metrics.smape:.6f}%" if metrics.smape is not None else "sMAPE: N/A")
        print(
            f"Directional accuracy: {metrics.directional_accuracy:.2%}"
            if metrics.directional_accuracy is not None
            else "Directional accuracy: N/A"
        )

    print(f"Model: {adapter.model_id}")
    print(f"Symbol: {args.symbol.strip().upper()}")
    print(f"Available observations: {len(observations)}")
    print(f"Start case: {args.start_case}")
    print(f"Cases per seed: {len(case_contexts)}")
    print(f"Horizon: {args.horizon}")
    print(f"Context length: {args.context_length}")
    print(f"Seeds: {seeds}")

    mae = metric_summary([m.mae for m in seed_metrics])
    rmse = metric_summary([m.rmse for m in seed_metrics])
    mape_values = [m.mape for m in seed_metrics if m.mape is not None]
    smape_values = [m.smape for m in seed_metrics if m.smape is not None]
    direction_values = [m.directional_accuracy for m in seed_metrics if m.directional_accuracy is not None]

    print("\n=== Kronos-small multi-seed summary ===")
    print(f"MAE mean/median/std: {mae[0]:.6f} / {mae[1]:.6f} / {mae[2]:.6f}")
    print(f"RMSE mean/median/std: {rmse[0]:.6f} / {rmse[1]:.6f} / {rmse[2]:.6f}")
    if mape_values:
        summary = metric_summary(mape_values)
        print(f"MAPE mean/median/std: {summary[0]:.6f}% / {summary[1]:.6f}% / {summary[2]:.6f}%")
    else:
        print("MAPE mean/median/std: N/A")
    if smape_values:
        summary = metric_summary(smape_values)
        print(f"sMAPE mean/median/std: {summary[0]:.6f}% / {summary[1]:.6f}% / {summary[2]:.6f}%")
    else:
        print("sMAPE mean/median/std: N/A")
    if direction_values:
        summary = metric_summary(direction_values)
        print(f"Directional accuracy mean/median/std: {summary[0]:.2%} / {summary[1]:.2%} / {summary[2]:.2%}")
    else:
        print("Directional accuracy mean/median/std: N/A")

    print("\n=== Naive last-close baseline ===")
    print(f"MAE: {naive_metrics.mae:.6f}")
    print(f"RMSE: {naive_metrics.rmse:.6f}")
    print(f"MAPE: {naive_metrics.mape:.6f}%" if naive_metrics.mape is not None else "MAPE: N/A")
    print(f"sMAPE: {naive_metrics.smape:.6f}%" if naive_metrics.smape is not None else "sMAPE: N/A")
    print("Directional accuracy: N/A (constant forecast has no directional signal)")

    print("\n=== Kronos mean improvement vs naive ===")
    print(f"MAE change: {mae[0] - naive_metrics.mae:+.6f} (negative is better)")
    print(f"RMSE change: {rmse[0] - naive_metrics.rmse:+.6f} (negative is better)")
    if mape_values and naive_metrics.mape is not None:
        print(f"MAPE change: {mean(mape_values) - naive_metrics.mape:+.6f} percentage points (negative is better)")
    if smape_values and naive_metrics.smape is not None:
        print(f"sMAPE change: {mean(smape_values) - naive_metrics.smape:+.6f} percentage points (negative is better)")


if __name__ == "__main__":
    main()
