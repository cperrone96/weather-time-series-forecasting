from __future__ import annotations

import pandas as pd
import pytest

from scripts.prepare_noaa_data import MIN_DAILY_OBSERVATIONS, prepare_daily_data, routine_hourly


def _raw(days: int = 3, per_day: int = 24) -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2017-01-01", periods=days, freq="D"):
        for hour in range(per_day):
            rows.append({"STATION": "74486094789", "DATE": day + pd.Timedelta(hours=hour, minutes=51), "REPORT_TYPE": "FM-15", "HourlyDryBulbTemperature": "50"})
    return pd.DataFrame(rows)


def test_routine_hourly_drops_specials_synoptic_duplicates_and_flagged_values():
    raw = _raw(days=1)
    extra = pd.DataFrame(
        [
            {"STATION": "74486094789", "DATE": pd.Timestamp("2017-01-01 03:00"), "REPORT_TYPE": "FM-12", "HourlyDryBulbTemperature": "90"},
            {"STATION": "74486094789", "DATE": pd.Timestamp("2017-01-01 05:20"), "REPORT_TYPE": "FM-16", "HourlyDryBulbTemperature": "90"},
            {"STATION": "74486094789", "DATE": pd.Timestamp("2017-01-01 06:51"), "REPORT_TYPE": "SY-MT", "HourlyDryBulbTemperature": "45s"},
        ]
    )
    hourly = routine_hourly(pd.concat([raw, extra], ignore_index=True))
    assert len(hourly) == 24 and (hourly == 50).all()


def test_daily_contract_rejects_thin_days_and_accepts_complete_ones():
    hourly = routine_hourly(_raw(days=3))
    cleaned = prepare_daily_data(hourly, "2017-01-01", "2017-01-03")
    assert len(cleaned) == 3 and cleaned["observation_count"].min() == 24
    thin = hourly.drop(hourly.index[: 24 - MIN_DAILY_OBSERVATIONS + 1])
    with pytest.raises(ValueError, match="at least 20 valid hourly observations"):
        prepare_daily_data(thin, "2017-01-01", "2017-01-03")
