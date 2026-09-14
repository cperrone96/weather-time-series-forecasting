#!/usr/bin/env python3
"""Run model selection, pre-test backtesting, and one final holdout evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from weather_forecasting import (
    ModelSpec,
    diagnostic_pairs,
    error_metrics,
    fit_model,
    forecast_model,
    load_temperature_series,
    seasonal_naive,
    split_series,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "jfk_daily_temperature_2017.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"


def rounded(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 4) for key, value in metrics.items()}


def rolling_origin(series: pd.Series, spec: ModelSpec) -> dict[str, object]:
    origins = [240, 254, 268, 282, 296]
    rows: list[dict[str, float | int | str]] = []
    for origin in origins:
        history = series.iloc[:origin]
        actual = series.iloc[origin : origin + 7]
        fitted = fit_model(history, spec)
        forecast, _, _ = forecast_model(fitted, actual.index, spec.harmonics)
        baseline = seasonal_naive(history, len(actual))
        for horizon, (observed, model_value, baseline_value) in enumerate(
            zip(actual, forecast, baseline), start=1
        ):
            rows.append(
                {
                    "origin": history.index[-1].date().isoformat(),
                    "horizon_day": horizon,
                    "actual_f": float(observed),
                    "model_forecast_f": float(model_value),
                    "seasonal_naive_f": float(baseline_value),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "rolling_origin_predictions.csv", index=False, float_format="%.4f")
    by_horizon = []
    for horizon, group in frame.groupby("horizon_day"):
        by_horizon.append(
            {
                "horizon_day": int(horizon),
                "model_rmse_f": round(float(np.sqrt(np.mean((group.actual_f - group.model_forecast_f) ** 2))), 4),
                "seasonal_naive_rmse_f": round(float(np.sqrt(np.mean((group.actual_f - group.seasonal_naive_f) ** 2))), 4),
            }
        )
    return {"origins": len(origins), "horizon_days": 7, "rmse_by_horizon": by_horizon}


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    series = load_temperature_series(DATA_PATH)
    train, validation, test = split_series(series)

    candidates = [
        ModelSpec("Fourier-1 + SARIMA(1,0,1)x(1,0,0,7)", harmonics=1),
        ModelSpec("Fourier-2 + SARIMA(1,0,1)x(1,0,0,7)", harmonics=2),
    ]
    validation_results: list[dict[str, object]] = []
    for spec in candidates:
        fitted = fit_model(train, spec)
        forecast, _, _ = forecast_model(fitted, validation.index, spec.harmonics)
        validation_results.append({"name": spec.name, **rounded(error_metrics(validation, forecast))})
    selected = candidates[int(np.argmin([row["rmse_f"] for row in validation_results]))]
    validation_baseline = seasonal_naive(train, len(validation))

    pretest = pd.concat([train, validation])
    backtest = rolling_origin(pretest, selected)
    final_fit = fit_model(pretest, selected)
    test_forecast, lower, upper = forecast_model(final_fit, test.index, selected.harmonics)
    test_baseline = seasonal_naive(pretest, len(test))
    selected_metrics = rounded(error_metrics(test, test_forecast))
    selected_metrics["interval_95_coverage"] = round(
        float(np.mean((test.to_numpy() >= lower) & (test.to_numpy() <= upper))), 4
    )
    selected_metrics["mean_interval_width_f"] = round(float(np.mean(upper - lower)), 4)

    predictions = pd.DataFrame(
        {
            "date": test.index.strftime("%Y-%m-%d"),
            "actual_f": test.to_numpy(),
            "model_forecast_f": test_forecast,
            "lower_95_f": lower,
            "upper_95_f": upper,
            "seasonal_naive_f": test_baseline,
        }
    )
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, float_format="%.4f")

    # Exclude diffuse-initialization residuals before diagnostic checks. Those
    # early state estimates are not comparable with steady-state innovations.
    diagnostic_burn = max(int(final_fit.loglikelihood_burn), 14)
    diagnostic_data = diagnostic_pairs(
        pd.Series(final_fit.fittedvalues), pd.Series(final_fit.resid), diagnostic_burn
    )
    residuals = diagnostic_data["residual"]
    ljung_lag = 14
    ljung = acorr_ljungbox(residuals, lags=[ljung_lag], return_df=True)
    metrics = {
        "data": {
            "rows": len(series),
            "start": series.index.min().date().isoformat(),
            "end": series.index.max().date().isoformat(),
            "observation_unit": "calendar-day mean of valid hourly dry-bulb temperature observations",
        },
        "design": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "test_used_for_model_selection": False,
            "selection_metric": "validation RMSE",
            "selected_model": selected.name,
        },
        "validation": {
            "candidates": validation_results,
            "seasonal_naive": rounded(error_metrics(validation, validation_baseline)),
        },
        "rolling_origin_pretest": backtest,
        "test": {
            "selected_model": selected_metrics,
            "seasonal_naive": rounded(error_metrics(test, test_baseline)),
            "horizon_errors": [
                {
                    "horizon_day": day,
                    "absolute_error_f": round(float(abs(test.iloc[day - 1] - test_forecast[day - 1])), 4),
                }
                for day in (1, 3, 7, 14, 28, len(test))
            ],
        },
        "diagnostics": {
            "initial_residuals_excluded": diagnostic_burn,
            "ljung_box_lag": ljung_lag,
            "ljung_box_pvalue": round(float(ljung.iloc[0]["lb_pvalue"]), 6),
            "residual_mean_f": round(float(residuals.mean()), 4),
            "residual_std_f": round(float(residuals.std()), 4),
        },
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test, color="#1f2937", label="Observed", linewidth=1.8)
    ax.plot(test.index, test_forecast, color="#2563eb", label="Selected model")
    ax.plot(test.index, test_baseline, color="#d97706", label="Seasonal naive", alpha=0.8)
    ax.fill_between(test.index, lower, upper, color="#93c5fd", alpha=0.35, label="95% model interval")
    ax.set(title="Untouched test-period forecasts", ylabel="Daily mean temperature (°F)")
    ax.legend(frameon=False, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "test_forecast.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(diagnostic_data["fitted"], residuals, s=13, alpha=0.65)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Fitted temperature (°F)", ylabel="Residual (°F)", title="Residuals vs fitted")
    from statsmodels.graphics.gofplots import qqplot

    qqplot(residuals, line="45", ax=axes[1], fit=True)
    axes[1].set_title("Normal Q-Q diagnostic")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "residual_diagnostics.png", dpi=160)
    plt.close(fig)

    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
