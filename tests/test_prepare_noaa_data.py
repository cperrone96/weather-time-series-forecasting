from __future__ import annotations

import pandas as pd
import pytest

from scripts.prepare_noaa_data import MIN_DAILY_OBSERVATIONS, prepare_daily_data


def _raw_year(observations_per_day: int = 20) -> pd.DataFrame:
    timestamps = []
    for day in pd.date_range("2017-01-01", "2017-12-31", freq="D"):
        timestamps.extend(day + pd.to_timedelta(range(observations_per_day), unit="h"))
    return pd.DataFrame(
        {
            "STATION": "74486094789",
            "NAME": "JFK INTERNATIONAL AIRPORT, NY US",
            "DATE": timestamps,
            "HourlyDryBulbTemperature": 50.0,
        }
    )


def test_preparation_rejects_day_below_documented_observation_minimum() -> None:
    raw = _raw_year(MIN_DAILY_OBSERVATIONS)
    raw = raw.drop(raw.index[0])

    with pytest.raises(ValueError, match="at least 20 valid hourly observations"):
        prepare_daily_data(raw)


def test_preparation_accepts_complete_year_meeting_minimum() -> None:
    cleaned = prepare_daily_data(_raw_year(MIN_DAILY_OBSERVATIONS))

    assert len(cleaned) == 365
    assert cleaned["observation_count"].min() == MIN_DAILY_OBSERVATIONS
