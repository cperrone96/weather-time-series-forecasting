from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from weather_forecasting import (
    DATA_COLUMNS,
    build_lagged_features,
    diagnostic_pairs,
    load_temperature_series,
    seasonal_naive,
    split_series,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "jfk_daily_temperature_2017.csv"


def test_checked_in_dataset_has_expected_contract() -> None:
    frame = pd.read_csv(DATA_PATH)
    assert list(frame.columns) == DATA_COLUMNS
    assert len(frame) == 365
    assert frame["date"].is_unique
    assert frame["mean_dry_bulb_temp_f"].notna().all()
    assert frame["observation_count"].min() >= 20

    series = load_temperature_series(DATA_PATH)
    assert series.index.is_monotonic_increasing
    assert series.index.freqstr == "D"
    assert np.isfinite(series).all()


def test_chronological_split_is_disjoint_and_ordered() -> None:
    series = load_temperature_series(DATA_PATH)
    train, validation, test = split_series(series)

    assert len(train) + len(validation) + len(test) == len(series)
    assert train.index.max() < validation.index.min() < test.index.min()
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)


def test_lagged_features_exclude_contemporaneous_target() -> None:
    series = pd.Series(
        np.arange(20.0), index=pd.date_range("2024-01-01", periods=20, freq="D")
    )
    features = build_lagged_features(series)

    target_day = features.index[-1]
    assert features.loc[target_day, "lag_1"] == series.shift(1).loc[target_day]
    assert features.loc[target_day, "rolling_mean_7"] == series.shift(1).rolling(7).mean().loc[target_day]
    assert features.loc[target_day, "rolling_mean_7"] != series.rolling(7).mean().loc[target_day]


def test_seasonal_naive_repeats_only_observed_history() -> None:
    history = pd.Series(np.arange(14.0))
    forecast = seasonal_naive(history, horizon=10, season=7)
    assert forecast.tolist() == [7, 8, 9, 10, 11, 12, 13, 7, 8, 9]


def test_diagnostic_pairs_exclude_burn_in_and_remain_aligned() -> None:
    fitted = pd.Series(np.arange(20.0), index=pd.date_range("2024-01-01", periods=20))
    residuals = pd.Series(np.arange(100.0, 120.0), index=fitted.index)
    pairs = diagnostic_pairs(fitted, residuals, burn=5)
    assert len(pairs) == 15
    assert pairs.iloc[0].to_dict() == {"fitted": 5.0, "residual": 105.0}


def test_checked_in_metrics_capture_holdout_and_uncertainty() -> None:
    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text())

    assert metrics["data"]["rows"] == 365
    assert metrics["design"]["test_used_for_model_selection"] is False
    assert metrics["test"]["selected_model"]["rmse_f"] > 0
    assert metrics["test"]["seasonal_naive"]["rmse_f"] > 0
    assert metrics["test"]["selected_model"]["rmse_f"] == pytest.approx(11.3797, abs=0.05)
    assert metrics["test"]["selected_model"]["rmse_f"] < metrics["test"]["seasonal_naive"]["rmse_f"]
    assert 0 <= metrics["test"]["selected_model"]["interval_95_coverage"] <= 1
    assert metrics["diagnostics"]["ljung_box_lag"] >= 7


def test_source_metadata_records_exact_lineage() -> None:
    metadata = json.loads((ROOT / "data" / "source_metadata.json").read_text())
    assert metadata["station_id"] == "74486094789"
    assert metadata["raw_sha256"] == "8caa2df61d05802b4cd63195dc882369b5ba012592989217b525bf70265f98e8"
    assert metadata["clean_rows"] == 365
