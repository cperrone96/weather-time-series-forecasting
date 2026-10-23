#!/usr/bin/env python3
"""Select a forecaster on 2018, then evaluate it once on 2019 with rolling 1-7 day forecasts.

Writes ``outputs/metrics.json``, the row-level predictions in ``outputs/`` and
the figures in ``figures/``.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from weather_forecasting import (
    RollingForecast,
    SarimaxSpec,
    actuals,
    climatology,
    diebold_mariano,
    evaluation_origins,
    fit_climatology,
    fit_sarimax,
    holm_adjust,
    load_temperature_series,
    persistence,
    rmse_by_horizon,
    sarimax_rolling,
    seasonal_naive,
    split_by_year,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "jfk_daily_temperature_2013_2019.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
HORIZON = 7
VALIDATION_YEAR, TEST_YEAR = 2018, 2019
CLIMATOLOGY_HARMONICS = (1, 2, 3)
SARIMAX_GRID = [
    SarimaxSpec(k, order) for k, order in product((1, 2, 3), ((1, 0, 0), (2, 0, 0), (3, 0, 0), (1, 0, 1), (2, 0, 1)))
]
LABELS = {
    "sarimax": "SARIMAX (selected)",
    "persistence": "Persistence",
    "seasonal_naive": "Seasonal naive (same day last year)",
    "climatology": "Climatology",
}
COLORS = {"sarimax": "#2a6fb0", "persistence": "#d1495b", "seasonal_naive": "#7a9e3a", "climatology": "#e09f3e"}


def r4(values) -> list[float]:
    return [round(float(v), 4) for v in values]


def sig(value: float) -> float:
    """Four significant digits, so small p-values are not rounded to zero."""
    return float(f"{value:.4g}")


def long_table(series: pd.Series, forecasts: dict[str, RollingForecast]) -> pd.DataFrame:
    """One row per (method, origin, horizon) with the observed value."""
    first = next(iter(forecasts.values()))
    observed = actuals(series, first.origins, HORIZON)
    rows = []
    for method, fc in forecasts.items():
        for i, origin in enumerate(fc.origins):
            for h in range(HORIZON):
                row = {
                    "method": method,
                    "origin": origin.date().isoformat(),
                    "horizon_day": h + 1,
                    "target_date": (origin + pd.Timedelta(days=h + 1)).date().isoformat(),
                    "actual_f": observed[i, h],
                    "forecast_f": fc.mean[i, h],
                }
                if fc.lower is not None:
                    row["lower_95_f"], row["upper_95_f"] = fc.lower[i, h], fc.upper[i, h]
                rows.append(row)
    return pd.DataFrame(rows)


def baselines(
    series: pd.Series, fit_window: pd.Series, origins: pd.DatetimeIndex, harmonics: int
) -> dict[str, RollingForecast]:
    clim_fit = fit_climatology(fit_window, harmonics)
    return {
        "persistence": persistence(series, origins, HORIZON),
        "seasonal_naive": seasonal_naive(series, origins, HORIZON),
        "climatology": climatology(clim_fit, harmonics, origins, HORIZON),
    }


def plot_results(
    series: pd.Series, test_scores: dict[str, list[float]], test_fc: dict[str, RollingForecast], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_h, ax_t) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    days = np.arange(1, HORIZON + 1)
    for method in ("persistence", "seasonal_naive", "climatology", "sarimax"):
        ax_h.plot(days, test_scores[method], "o-", color=COLORS[method], lw=2, label=LABELS[method])
    ax_h.set(
        xlabel="Forecast horizon (days ahead)",
        ylabel="RMSE (°F)",
        title=f"Test-year error by horizon ({TEST_YEAR})",
        xticks=days,
    )
    ax_h.set_ylim(bottom=0)
    ax_h.legend(frameon=False, fontsize=9, loc="lower right")

    sar = test_fc["sarimax"]
    window = (sar.origins >= f"{TEST_YEAR}-01-01") & (sar.origins < f"{TEST_YEAR}-04-01")
    target = sar.origins[window] + pd.Timedelta(days=1)
    ax_t.plot(series.loc[target].index, series.loc[target], color="#222222", lw=1.6, label="Observed daily mean")
    ax_t.fill_between(
        target,
        sar.lower[window, 0],
        sar.upper[window, 0],
        color=COLORS["sarimax"],
        alpha=0.2,
        lw=0,
        label="95% prediction interval",
    )
    ax_t.plot(target, sar.mean[window, 0], color=COLORS["sarimax"], lw=1.6, label="SARIMAX, 1 day ahead")
    ax_t.plot(
        target,
        test_fc["climatology"].mean[window, 0],
        color=COLORS["climatology"],
        lw=1.4,
        ls="--",
        label="Climatology",
    )
    ax_t.set(ylabel="Daily mean temperature (°F)", title=f"One-day-ahead forecasts, Jan to Mar {TEST_YEAR}")
    ax_t.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    for ax in (ax_h, ax_t):
        ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(residuals: pd.Series, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.graphics.gofplots import qqplot
    from statsmodels.graphics.tsaplots import plot_acf

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_acf(residuals, lags=30, ax=axes[0], zero=False)
    axes[0].set(title="Residual autocorrelation, final fit (2013-2018)", xlabel="Lag (days)", ylim=(-0.25, 0.25))
    qqplot(residuals, line="45", fit=True, ax=axes[1], markersize=3, alpha=0.5)
    axes[1].set_title("Normal Q-Q plot of residuals")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    series = load_temperature_series(DATA_PATH)
    train, validation, test = split_by_year(series, VALIDATION_YEAR, TEST_YEAR)

    # 1. Selection on the validation year (parameters estimated on training years only).
    val_origins = evaluation_origins(series, validation, HORIZON)
    val_actual = actuals(series, val_origins, HORIZON)
    clim_scores = {
        k: float(
            rmse_by_horizon(val_actual, climatology(fit_climatology(train, k), k, val_origins, HORIZON).mean).mean()
        )
        for k in CLIMATOLOGY_HARMONICS
    }
    clim_k = min(clim_scores, key=clim_scores.get)
    candidates, val_forecasts = [], {}
    for spec in SARIMAX_GRID:
        result, converged = fit_sarimax(train, spec)
        row = {"model": spec.name, "harmonics": spec.harmonics, "order": list(spec.order), "converged": converged}
        if converged:
            fc = sarimax_rolling(result, spec, series, val_origins, HORIZON)
            row["rmse_by_horizon_f"] = r4(rmse_by_horizon(val_actual, fc.mean))
            row["mean_rmse_f"] = round(float(np.mean(row["rmse_by_horizon_f"])), 4)
            val_forecasts[spec] = fc
        candidates.append(row)
    selected = min(val_forecasts, key=lambda s: rmse_by_horizon(val_actual, val_forecasts[s].mean).mean())
    val_all = {"sarimax": val_forecasts[selected], **baselines(series, train, val_origins, clim_k)}
    long_table(series, val_all).to_csv(OUTPUT_DIR / "rolling_origin_predictions.csv", index=False, float_format="%.4f")

    # 2. Refit the selected specification on 2013-2018 and score 2019 once.
    pretest = pd.concat([train, validation])
    final, final_converged = fit_sarimax(pretest, selected)
    test_origins = evaluation_origins(series, test, HORIZON)
    test_actual = actuals(series, test_origins, HORIZON)
    test_fc = {
        "sarimax": sarimax_rolling(final, selected, series, test_origins, HORIZON),
        **baselines(series, pretest, test_origins, clim_k),
    }
    long_table(series, test_fc).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, float_format="%.4f")

    test_rmse = {m: rmse_by_horizon(test_actual, fc.mean) for m, fc in test_fc.items()}
    test_mae = {m: np.mean(np.abs(test_actual - fc.mean), axis=0) for m, fc in test_fc.items()}
    sar = test_fc["sarimax"]
    inside = (test_actual >= sar.lower) & (test_actual <= sar.upper)
    seasons = {"Dec-Feb": (12, 1, 2), "Mar-May": (3, 4, 5), "Jun-Aug": (6, 7, 8), "Sep-Nov": (9, 10, 11)}
    target_months = np.array([[(o + pd.Timedelta(days=h + 1)).month for h in range(HORIZON)] for o in test_origins])

    tests = []
    for baseline in ("persistence", "climatology"):
        for h in range(HORIZON):
            dm = diebold_mariano(
                test_actual[:, h] - sar.mean[:, h], test_actual[:, h] - test_fc[baseline].mean[:, h], h + 1
            )
            tests.append({"versus": baseline, "horizon_day": h + 1, **{k: sig(v) for k, v in dm.items()}})
    for row, adjusted in zip(tests, holm_adjust([t["p_value"] for t in tests])):
        row["holm_adjusted_p_value"] = sig(adjusted)

    burn = max(int(final.loglikelihood_burn), 30)
    residuals = pd.Series(final.resid).iloc[burn:]
    ljung = acorr_ljungbox(residuals, lags=[7, 14], return_df=True)

    metrics = {
        "data": {
            "days": len(series),
            "start": series.index.min().date().isoformat(),
            "end": series.index.max().date().isoformat(),
            "unit": "calendar-day mean of routine hourly dry-bulb temperature, °F",
        },
        "design": {
            "train_years": [int(train.index.year.min()), int(train.index.year.max())],
            "validation_year": VALIDATION_YEAR,
            "test_year": TEST_YEAR,
            "horizon_days": HORIZON,
            "validation_origins": len(val_origins),
            "test_origins": len(test_origins),
            "parameters": "estimated once per stage (training years for validation, 2013-2018 for test), then fixed while the origin rolls daily",
            "selection_metric": "validation RMSE averaged over horizons 1-7, among converged fits",
        },
        "validation": {
            "sarimax_candidates": candidates,
            "climatology_harmonics_rmse_f": clim_scores,
            "baselines_rmse_by_horizon_f": {
                m: r4(rmse_by_horizon(val_actual, fc.mean)) for m, fc in val_all.items() if m != "sarimax"
            },
        },
        "selected": {
            "model": selected.name,
            "harmonics": selected.harmonics,
            "order": list(selected.order),
            "climatology_harmonics": clim_k,
            "final_fit_converged": final_converged,
        },
        "test": {
            "rmse_by_horizon_f": {m: r4(v) for m, v in test_rmse.items()},
            "mae_by_horizon_f": {m: r4(v) for m, v in test_mae.items()},
            "skill_vs_climatology_by_horizon": r4(1 - test_rmse["sarimax"] / test_rmse["climatology"]),
            "skill_vs_persistence_by_horizon": r4(1 - test_rmse["sarimax"] / test_rmse["persistence"]),
            "sarimax_interval_95_coverage_by_horizon": r4(inside.mean(axis=0)),
            "sarimax_interval_95_mean_width_by_horizon_f": r4((sar.upper - sar.lower).mean(axis=0)),
            "sarimax_interval_95_coverage_by_season": {
                name: round(float(inside[np.isin(target_months, months)].mean()), 4) for name, months in seasons.items()
            },
            "diebold_mariano": tests,
        },
        "diagnostics": {
            "residuals_excluded_for_burn_in": burn,
            "ljung_box_pvalue_lag7": round(float(ljung.loc[7, "lb_pvalue"]), 6),
            "ljung_box_pvalue_lag14": round(float(ljung.loc[14, "lb_pvalue"]), 6),
            "residual_std_f": round(float(residuals.std()), 4),
        },
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_results(series, metrics["test"]["rmse_by_horizon_f"], test_fc, FIGURE_DIR / "test_forecast.png")
    plot_residuals(residuals, FIGURE_DIR / "residual_diagnostics.png")
    return metrics


if __name__ == "__main__":
    out = main()
    print(json.dumps({"selected": out["selected"], "test_rmse": out["test"]["rmse_by_horizon_f"]}, indent=2))
