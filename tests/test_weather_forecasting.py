from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from weather_forecasting import (
    DATA_COLUMNS,
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
    sarimax_rolling,
    seasonal_naive,
    split_by_year,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "jfk_daily_temperature_2013_2019.csv"


@pytest.fixture(scope="module")
def series() -> pd.Series:
    return load_temperature_series(DATA_PATH)


def _synthetic(days: int = 3 * 365, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2013-01-01", periods=days, freq="D")
    noise = np.zeros(days)
    for t in range(1, days):
        noise[t] = 0.6 * noise[t - 1] + rng.normal(0, 3)
    seasonal = 55 - 20 * np.cos(2 * np.pi * (index - index[0]).days / 365.2425)
    return pd.Series(seasonal + noise, index=index)


def test_checked_in_dataset_contract(series):
    frame = pd.read_csv(DATA_PATH)
    assert list(frame.columns) == DATA_COLUMNS
    assert frame["observation_count"].min() >= 20
    assert series.index.freqstr == "D" and len(series) == 2556
    assert series.index[0] == pd.Timestamp("2013-01-01") and series.index[-1] == pd.Timestamp("2019-12-31")


def test_year_split_is_ordered_and_complete(series):
    train, validation, test = split_by_year(series, 2018, 2019)
    assert train.index.max() < validation.index.min() and validation.index.max() < test.index.min()
    assert len(train) + len(validation) + len(test) == len(series)
    assert set(test.index.year) == {2019}


def test_origins_keep_every_target_inside_the_period(series):
    _, validation, _ = split_by_year(series, 2018, 2019)
    origins = evaluation_origins(series, validation, 7)
    assert origins[0] == pd.Timestamp("2017-12-31")
    assert origins[-1] + pd.Timedelta(days=7) == pd.Timestamp("2018-12-31")


def test_baselines_use_only_past_values():
    s = pd.Series(np.arange(800.0), index=pd.date_range("2013-01-01", periods=800, freq="D"))
    origins = s.index[[400, 500]]
    assert persistence(s, origins, 3).mean.tolist() == [[400.0] * 3, [500.0] * 3]
    assert seasonal_naive(s, origins, 3).mean.tolist() == [[36.0, 37.0, 38.0], [136.0, 137.0, 138.0]]
    assert actuals(s, origins, 2).tolist() == [[401.0, 402.0], [501.0, 502.0]]


def test_sarimax_rolling_forecasts_never_see_the_future():
    s = _synthetic()
    spec = SarimaxSpec(1, (1, 0, 0))
    train = s.iloc[: 2 * 365]
    result, converged = fit_sarimax(train, spec)
    assert converged
    origins = s.index[800:810]
    clean = sarimax_rolling(result, spec, s, origins, 7)
    tampered = s.copy()
    tampered.iloc[811:] += 50  # change only values after the first origin's targets begin
    changed = sarimax_rolling(result, spec, tampered, origins, 7)
    assert np.allclose(clean.mean[0], changed.mean[0])  # origin day 800: all targets <= 807 are forecasts, inputs <= 800
    assert np.all(clean.lower < clean.mean) and np.all(clean.mean < clean.upper)


def test_sarimax_beats_climatology_one_day_ahead_on_autocorrelated_data():
    s = _synthetic(days=4 * 365, seed=1)
    train = s.iloc[: 3 * 365]
    spec = SarimaxSpec(1, (1, 0, 0))
    result, _ = fit_sarimax(train, spec)
    origins = s.index[3 * 365 : 4 * 365 - 7]
    observed = actuals(s, origins, 7)
    sar = sarimax_rolling(result, spec, s, origins, 7)
    clim = climatology(fit_climatology(train, 1), 1, origins, 7)
    sar_rmse = np.sqrt(((observed - sar.mean) ** 2).mean(axis=0))
    clim_rmse = np.sqrt(((observed - clim.mean) ** 2).mean(axis=0))
    assert sar_rmse[0] < 0.9 * clim_rmse[0]
    assert abs(sar_rmse[6] - clim_rmse[6]) < 0.1 * clim_rmse[6]  # AR(0.6) memory is gone after a week


def test_diebold_mariano_detects_a_clearly_better_forecast():
    rng = np.random.default_rng(2)
    small, large = rng.normal(0, 1, 300), rng.normal(0, 2, 300)
    assert diebold_mariano(small, large, 1)["p_value"] < 0.001
    a, b = rng.normal(0, 1, 300), rng.normal(0, 1, 300)
    assert diebold_mariano(a, b, 1)["p_value"] > 0.05  # equal accuracy: no rejection for this seed


def test_holm_adjustment():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_committed_results_follow_the_selection_protocol():
    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    converged = [c for c in metrics["validation"]["sarimax_candidates"] if c["converged"]]
    best = min(converged, key=lambda c: c["mean_rmse_f"])
    assert metrics["selected"]["model"] == best["model"]
    assert metrics["design"]["test_year"] > metrics["design"]["validation_year"]
