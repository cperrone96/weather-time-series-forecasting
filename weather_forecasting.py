"""Leakage-safe daily temperature forecasting utilities.

The forecasting model is a dynamic regression with annual Fourier terms and
SARIMA errors.  All split and baseline helpers are deliberately small so their
time-ordering assumptions can be tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


DATA_COLUMNS = ["date", "mean_dry_bulb_temp_f", "observation_count"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    harmonics: int
    order: tuple[int, int, int] = (1, 0, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 0, 7)


def load_temperature_series(path: str | Path) -> pd.Series:
    frame = pd.read_csv(path)
    if list(frame.columns) != DATA_COLUMNS:
        raise ValueError(f"Expected columns {DATA_COLUMNS}; got {list(frame.columns)}")
    if frame["date"].duplicated().any():
        raise ValueError("Dates must be unique")
    if frame["mean_dry_bulb_temp_f"].isna().any():
        raise ValueError("Temperature target contains missing values")
    index = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    series = pd.Series(
        frame["mean_dry_bulb_temp_f"].to_numpy(dtype=float),
        index=index,
        name="mean_dry_bulb_temp_f",
    ).sort_index()
    expected = pd.date_range(series.index.min(), series.index.max(), freq="D")
    if not series.index.equals(expected):
        raise ValueError("Daily time index must be complete and consecutive")
    series.index.freq = "D"
    return series


def split_series(
    series: pd.Series, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    train_end = int(np.floor(len(series) * train_fraction))
    validation_end = train_end + int(np.floor(len(series) * validation_fraction))
    if train_end < 30 or validation_end >= len(series):
        raise ValueError("Series is too short for requested train/validation/test split")
    return series.iloc[:train_end], series.iloc[train_end:validation_end], series.iloc[validation_end:]


def build_lagged_features(series: pd.Series) -> pd.DataFrame:
    """Create diagnostics/features using prior observations only."""
    prior = series.shift(1)
    return pd.DataFrame(
        {
            "lag_1": prior,
            "lag_7": series.shift(7),
            "rolling_mean_7": prior.rolling(7).mean(),
        }
    ).dropna()


def seasonal_naive(history: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    if horizon < 1 or len(history) < season:
        raise ValueError("Need a positive horizon and at least one full season")
    last_season = history.iloc[-season:].to_numpy(dtype=float)
    return np.resize(last_season, horizon)


def fourier_terms(index: pd.DatetimeIndex, harmonics: int, period: float = 365.2425) -> pd.DataFrame:
    day = (index - pd.Timestamp("2017-01-01")).days.to_numpy(dtype=float)
    values: dict[str, np.ndarray] = {}
    for k in range(1, harmonics + 1):
        angle = 2 * np.pi * k * day / period
        values[f"sin_{k}"] = np.sin(angle)
        values[f"cos_{k}"] = np.cos(angle)
    return pd.DataFrame(values, index=index)


def fit_model(series: pd.Series, spec: ModelSpec):
    exog = fourier_terms(series.index, spec.harmonics)
    model = SARIMAX(
        series,
        exog=exog,
        order=spec.order,
        seasonal_order=spec.seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=300)


def forecast_model(fitted, index: pd.DatetimeIndex, harmonics: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = fitted.get_forecast(steps=len(index), exog=fourier_terms(index, harmonics))
    mean = np.asarray(prediction.predicted_mean, dtype=float)
    interval = np.asarray(prediction.conf_int(alpha=0.05), dtype=float)
    return mean, interval[:, 0], interval[:, 1]


def error_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    error = actual_array - predicted_array
    return {
        "rmse_f": float(np.sqrt(np.mean(np.square(error)))),
        "mae_f": float(np.mean(np.abs(error))),
    }


def diagnostic_pairs(
    fitted: pd.Series, residuals: pd.Series, burn: int
) -> pd.DataFrame:
    """Return aligned steady-state fitted values and residuals."""
    if burn < 0:
        raise ValueError("Burn-in must be nonnegative")
    return pd.DataFrame(
        {
            "fitted": np.asarray(fitted, dtype=float)[burn:],
            "residual": np.asarray(residuals, dtype=float)[burn:],
        },
        index=fitted.index[burn:],
    ).dropna()
