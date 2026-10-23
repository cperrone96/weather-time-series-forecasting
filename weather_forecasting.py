"""Daily temperature forecasting at JFK: baselines, SARIMAX and rolling-origin evaluation.

Every method produces, for each forecast origin (the last observed day), a
forecast for the next 1 to ``H`` days. Forecasts at origin ``t`` use only
observations up to and including ``t``; model parameters are estimated once on
a training window and then held fixed while the origin rolls forward.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_COLUMNS = ["date", "mean_dry_bulb_temp_f", "observation_count"]
FOURIER_EPOCH = pd.Timestamp("2013-01-01")
YEAR_DAYS = 365.2425


@dataclass(frozen=True)
class SarimaxSpec:
    """Dynamic regression: annual Fourier terms plus ARMA(p, q) errors."""

    harmonics: int
    order: tuple[int, int, int]

    @property
    def name(self) -> str:
        p, d, q = self.order
        return f"Fourier K={self.harmonics} + ARMA({p},{q}) errors"


@dataclass
class RollingForecast:
    """Forecasts for horizons 1..H from consecutive origins (rows) as arrays of shape (n_origins, H)."""

    origins: pd.DatetimeIndex
    mean: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


# --------------------------------------------------------------------------- data


def load_temperature_series(path: str | Path) -> pd.Series:
    """Load the daily series and enforce a complete, unique, ordered daily index."""
    frame = pd.read_csv(path)
    if list(frame.columns) != DATA_COLUMNS:
        raise ValueError(f"Expected columns {DATA_COLUMNS}; got {list(frame.columns)}")
    if frame["date"].duplicated().any():
        raise ValueError("Dates must be unique")
    if frame["mean_dry_bulb_temp_f"].isna().any():
        raise ValueError("Temperature target contains missing values")
    index = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    series = pd.Series(frame["mean_dry_bulb_temp_f"].to_numpy(dtype=float), index=index, name="mean_dry_bulb_temp_f").sort_index()
    if not series.index.equals(pd.date_range(series.index.min(), series.index.max(), freq="D")):
        raise ValueError("Daily time index must be complete and consecutive")
    return series.asfreq("D")


def split_by_year(series: pd.Series, validation_year: int, test_year: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Train on every year before ``validation_year``; validate and test on whole calendar years."""
    if not series.index.year.min() < validation_year < test_year <= series.index.year.max():
        raise ValueError("Need training years before the validation year and a later test year")
    train = series.loc[series.index.year < validation_year]
    validation = series.loc[series.index.year == validation_year]
    test = series.loc[series.index.year == test_year]
    return train, validation, test


def fourier_terms(index: pd.DatetimeIndex, harmonics: int) -> pd.DataFrame:
    """Annual sine/cosine pairs, so the seasonal cycle is smooth and needs 2K parameters."""
    t = (index - FOURIER_EPOCH).days.to_numpy(dtype=float) / YEAR_DAYS
    columns = {}
    for k in range(1, harmonics + 1):
        columns[f"sin_{k}"] = np.sin(2 * np.pi * k * t)
        columns[f"cos_{k}"] = np.cos(2 * np.pi * k * t)
    return pd.DataFrame(columns, index=index)


# ------------------------------------------------------------------------ methods


def _origin_positions(series: pd.Series, origins: pd.DatetimeIndex) -> np.ndarray:
    return series.index.get_indexer(origins)


def persistence(series: pd.Series, origins: pd.DatetimeIndex, horizon: int) -> RollingForecast:
    """Tomorrow (and every later day) equals today."""
    values = series.to_numpy()[_origin_positions(series, origins)]
    return RollingForecast(origins, np.repeat(values[:, None], horizon, axis=1))


def seasonal_naive(series: pd.Series, origins: pd.DatetimeIndex, horizon: int, season: int = 365) -> RollingForecast:
    """Each target day equals the same day one year (365 days) earlier."""
    if horizon > season:
        raise ValueError("Horizon cannot exceed the season length")
    pos = _origin_positions(series, origins)
    if pos.min() + 1 - season < 0:
        raise ValueError("Need a full season of history before the first origin")
    values = series.to_numpy()
    return RollingForecast(origins, np.array([[values[i + h - season] for h in range(1, horizon + 1)] for i in pos]))


def fit_climatology(train: pd.Series, harmonics: int):
    """OLS of temperature on an intercept and annual Fourier terms (the long-run seasonal mean)."""
    return sm.OLS(train.to_numpy(), sm.add_constant(fourier_terms(train.index, harmonics)).to_numpy()).fit()


def climatology(fit, harmonics: int, origins: pd.DatetimeIndex, horizon: int) -> RollingForecast:
    """Forecast each target day with its climatological mean, ignoring recent weather."""
    target_days = pd.DatetimeIndex([o + pd.Timedelta(days=h) for o in origins for h in range(1, horizon + 1)])
    mean = fit.predict(sm.add_constant(fourier_terms(target_days, harmonics), has_constant="add").to_numpy())
    return RollingForecast(origins, mean.reshape(len(origins), horizon))


def fit_sarimax(train: pd.Series, spec: SarimaxSpec):
    """Maximum-likelihood fit. Returns the results and whether the optimizer converged cleanly."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = SARIMAX(train, exog=fourier_terms(train.index, spec.harmonics), order=spec.order, trend="c").fit(disp=False, maxiter=500)
    converged = bool(result.mle_retvals.get("converged", False)) and not any("converge" in str(w.message).lower() for w in caught)
    return result, converged


def sarimax_rolling(result, spec: SarimaxSpec, series: pd.Series, origins: pd.DatetimeIndex, horizon: int, alpha: float = 0.05) -> RollingForecast:
    """Rolling 1..H-day forecasts with parameters fixed at their training estimates.

    ``apply`` re-runs the Kalman filter over ``series`` with the fitted
    parameters; a dynamic prediction starting the day after each origin then
    conditions only on data up to that origin.
    """
    last_needed = series.index.get_loc(origins[-1]) + horizon
    data = series.iloc[: last_needed + 1]
    filtered = result.apply(data, exog=fourier_terms(data.index, spec.harmonics))
    mean = np.empty((len(origins), horizon))
    lower, upper = np.empty_like(mean), np.empty_like(mean)
    for row, pos in enumerate(_origin_positions(series, origins)):
        pred = filtered.get_prediction(start=pos + 1, end=pos + horizon, dynamic=True)
        interval = np.asarray(pred.conf_int(alpha=alpha))
        mean[row], lower[row], upper[row] = np.asarray(pred.predicted_mean), interval[:, 0], interval[:, 1]
    return RollingForecast(origins, mean, lower, upper)


# ------------------------------------------------------------------------ scoring


def actuals(series: pd.Series, origins: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    """Observed values for every (origin, horizon) cell."""
    pos = _origin_positions(series, origins)
    values = series.to_numpy()
    return np.array([[values[i + h] for h in range(1, horizon + 1)] for i in pos])


def evaluation_origins(series: pd.Series, period: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Origins from the day before ``period`` starts such that every target falls inside ``period``."""
    first = series.index.get_loc(period.index[0]) - 1
    last = series.index.get_loc(period.index[-1]) - horizon
    return series.index[first : last + 1]


def rmse_by_horizon(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((actual - forecast) ** 2, axis=0))


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray, horizon: int) -> dict[str, float]:
    """Diebold-Mariano test of equal squared-error loss with the Harvey-Leybourne-Newbold correction.

    ``h``-step errors from consecutive origins overlap, so the loss-difference
    variance uses autocovariances up to lag ``h - 1``. A negative statistic
    means model A has lower loss.
    """
    d = errors_a**2 - errors_b**2
    n = len(d)
    centered = d - d.mean()
    gamma = [np.dot(centered[k:], centered[: n - k]) / n for k in range(horizon)]
    variance = (gamma[0] + 2 * sum(gamma[1:])) / n
    if variance <= 0:
        variance = gamma[0] / n
    statistic = d.mean() / np.sqrt(variance)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic *= correction
    p_value = 2 * stats.t.sf(abs(statistic), df=n - 1)
    return {"statistic": float(statistic), "p_value": float(p_value), "mean_loss_difference": float(d.mean())}


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control)."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()
