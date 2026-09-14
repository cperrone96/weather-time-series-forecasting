#!/usr/bin/env python3
"""Download and prepare the versioned 2017 NOAA LCD JFK source file."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd


SOURCE_URL = "https://www.ncei.noaa.gov/data/local-climatological-data/access/2017/74486094789.csv"
EXPECTED_SHA256 = "8caa2df61d05802b4cd63195dc882369b5ba012592989217b525bf70265f98e8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", type=Path, help="Use an already-downloaded NOAA CSV")
    parser.add_argument("--output", type=Path, default=Path("data/jfk_daily_temperature_2017.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/source_metadata.json"))
    args = parser.parse_args()

    raw_path = args.raw_file or Path("data/raw/74486094789_2017.csv")
    if args.raw_file is None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, raw_path)

    observed_sha = sha256(raw_path)
    if observed_sha != EXPECTED_SHA256:
        raise ValueError(f"Source checksum mismatch: expected {EXPECTED_SHA256}, got {observed_sha}")

    raw = pd.read_csv(
        raw_path,
        usecols=["STATION", "NAME", "DATE", "HourlyDryBulbTemperature"],
        parse_dates=["DATE"],
    )
    if set(raw["STATION"].astype(str)) != {"74486094789"}:
        raise ValueError("Unexpected station identifier")
    raw["temperature_f"] = pd.to_numeric(raw["HourlyDryBulbTemperature"], errors="coerce")
    valid = raw.dropna(subset=["temperature_f"]).set_index("DATE")["temperature_f"]
    daily = valid.resample("D").agg(["mean", "count"])
    if len(daily) != 365 or daily["mean"].isna().any():
        raise ValueError("Expected a complete 365-day daily series")
    cleaned = daily.rename(
        columns={"mean": "mean_dry_bulb_temp_f", "count": "observation_count"}
    ).reset_index(names="date")
    cleaned["date"] = cleaned["date"].dt.strftime("%Y-%m-%d")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False, float_format="%.6f")

    metadata = {
        "dataset": "NOAA Local Climatological Data (LCD), 2017",
        "station": "JFK INTERNATIONAL AIRPORT, NY US",
        "station_id": "74486094789",
        "source_url": SOURCE_URL,
        "retrieved_utc": "2026-09-14",
        "raw_sha256": observed_sha,
        "raw_rows": int(len(raw)),
        "valid_temperature_observations": int(len(valid)),
        "clean_rows": int(len(cleaned)),
        "transformation": "Numeric hourly dry-bulb observations aggregated to calendar-day arithmetic means; missing/non-numeric hourly values excluded; no daily values imputed.",
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
