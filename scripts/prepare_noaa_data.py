#!/usr/bin/env python3
"""Download NOAA LCD hourly files for JFK (2013-2019) and build the daily series.

Each yearly source file is verified against a recorded SHA-256 before use.
The daily value is the mean of the routine hourly observations (report types
FM-15 and SY-MT, one per clock hour). Special reports (FM-16), synoptic
duplicates (FM-12) and daily/monthly summary rows are excluded so that hours
with extra reports, which cluster in bad weather, are not over-weighted.

Usage:
    python scripts/prepare_noaa_data.py                    # download to data/raw/
    python scripts/prepare_noaa_data.py --raw-dir <folder> # use files already downloaded
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

STATION_ID = "74486094789"
URL_TEMPLATE = "https://www.ncei.noaa.gov/data/local-climatological-data/access/{year}/74486094789.csv"
EXPECTED_SHA256 = {
    2013: "99a7a60432a97df672971e442e69e2ff8b1f7288494299c844efa58dcd66f91b",
    2014: "14cbae8e10eac00f5ae1b3032e32f1c70f6471a326239a10354f6eb7fcab96fc",
    2015: "f80ea2eb204c9b1a459d4f4b80dc1acb57a03fc8039d343759bc61c3c1e3820b",
    2016: "f2dd990d35ee90917c019c31e1f22d5ef14139b4e855baa7724c1e18a23882bd",
    2017: "8caa2df61d05802b4cd63195dc882369b5ba012592989217b525bf70265f98e8",
    2018: "568670875fb666939bf6fe3b70bc830bf1b4c00b8421ff9e33336682407b7929",
    2019: "84d448f0509ed2e239edcebc0fe1be9ac6d1ed098dc2834d89cb54323ea36cc4",
}
ROUTINE_REPORT_TYPES = {"FM-15", "SY-MT"}
MIN_DAILY_OBSERVATIONS = 20
CHECKSUMS_RECORDED = "2026-10-23"  # date the source files above were downloaded and hashed
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def routine_hourly(raw: pd.DataFrame) -> pd.Series:
    """One valid dry-bulb temperature per clock hour from routine reports."""
    required = {"STATION", "DATE", "REPORT_TYPE", "HourlyDryBulbTemperature"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required NOAA columns: {sorted(missing)}")
    if set(raw["STATION"].astype(str)) != {STATION_ID}:
        raise ValueError("Unexpected station identifier")
    frame = raw.assign(
        DATE=pd.to_datetime(raw["DATE"], errors="raise"),
        temperature_f=pd.to_numeric(
            raw["HourlyDryBulbTemperature"], errors="coerce"
        ),  # flagged values such as "45s" become NaN
    )
    frame = frame.loc[
        frame["REPORT_TYPE"].astype(str).str.strip().isin(ROUTINE_REPORT_TYPES) & frame["temperature_f"].notna()
    ]
    return frame.groupby(frame["DATE"].dt.floor("h"))["temperature_f"].first()


def prepare_daily_data(hourly: pd.Series, start: str, end: str) -> pd.DataFrame:
    """Aggregate hourly temperatures to daily means and enforce the coverage contract."""
    daily = hourly.resample("D").agg(["mean", "count"])
    expected_days = pd.date_range(start, end, freq="D")
    if not daily.index.equals(expected_days) or daily["mean"].isna().any():
        raise ValueError(f"Expected a complete daily series from {start} to {end}")
    below_minimum = daily.index[daily["count"] < MIN_DAILY_OBSERVATIONS]
    if len(below_minimum):
        dates = ", ".join(day.date().isoformat() for day in below_minimum[:5])
        raise ValueError(
            f"Every day must have at least {MIN_DAILY_OBSERVATIONS} valid hourly observations; below-threshold dates include: {dates}"
        )
    cleaned = daily.rename(columns={"mean": "mean_dry_bulb_temp_f", "count": "observation_count"}).reset_index(
        names="date"
    )
    cleaned["date"] = cleaned["date"].dt.strftime("%Y-%m-%d")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw", help="folder for the yearly NOAA CSVs")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "jfk_daily_temperature_2013_2019.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data" / "source_metadata.json")
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    hourly_parts, sources, raw_rows = [], [], 0
    for year, expected in EXPECTED_SHA256.items():
        path = args.raw_dir / f"{STATION_ID}_{year}.csv"
        url = URL_TEMPLATE.format(year=year)
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"{year}: checksum mismatch (expected {expected}, got {observed})")
        raw = pd.read_csv(
            path, usecols=["STATION", "DATE", "REPORT_TYPE", "HourlyDryBulbTemperature"], low_memory=False
        )
        raw_rows += len(raw)
        hourly_parts.append(routine_hourly(raw))
        sources.append({"year": year, "url": url, "sha256": observed, "rows": int(len(raw))})

    hourly = pd.concat(hourly_parts).sort_index()
    first, last = min(EXPECTED_SHA256), max(EXPECTED_SHA256)
    cleaned = prepare_daily_data(hourly, f"{first}-01-01", f"{last}-12-31")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False, float_format="%.6f")

    metadata = {
        "dataset": f"NOAA Local Climatological Data (LCD), {first}-{last}",
        "station": "JFK INTERNATIONAL AIRPORT, NY US",
        "station_id": STATION_ID,
        "checksums_recorded_utc": CHECKSUMS_RECORDED,
        "sources": sources,
        "raw_rows": raw_rows,
        "routine_hourly_observations": int(len(hourly)),
        "clean_rows": int(len(cleaned)),
        "minimum_valid_hourly_observations_per_day": MIN_DAILY_OBSERVATIONS,
        "transformation": "Routine hourly reports (FM-15, SY-MT), one per clock hour, with numeric dry-bulb temperature; calendar-day arithmetic mean; no daily values imputed.",
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{len(cleaned)} days written to {args.output}")


if __name__ == "__main__":
    main()
