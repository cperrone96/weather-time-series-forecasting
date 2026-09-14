# JFK Weather Forecasting: Time-Ordered Evaluation

An independently remediated time-series portfolio project that forecasts the
**daily mean dry-bulb temperature at JFK Airport**. The analysis demonstrates
data provenance, leakage-safe model selection, expanding-window evaluation,
an untouched test period, forecast intervals, and residual diagnostics.

![Observed temperatures, forecasts, and model intervals](figures/test_forecast.png)

## Question and decision context

How accurately can a small dynamic-regression model forecast future daily mean
temperature, relative to a transparent seven-day seasonal-naive benchmark?
This is an educational demonstration of forecast evaluation. It is not an
operational weather-warning system and should not be used for safety decisions.

## Data contract and provenance

- **Source:** [NOAA/NCEI Local Climatological Data](https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data), station `74486094789` (JFK International Airport), calendar year 2017.
- **Versioned file:** `https://www.ncei.noaa.gov/data/local-climatological-data/access/2017/74486094789.csv`
- **Retrieved:** 2026-09-14 UTC.
- **Raw SHA-256:** `8caa2df61d05802b4cd63195dc882369b5ba012592989217b525bf70265f98e8`.
- **Observation unit:** one calendar-day arithmetic mean of valid hourly dry-bulb temperature observations.
- **Transformation:** nonnumeric/missing hourly target values are excluded; valid readings are aggregated by day. No daily target is imputed. The checked-in table has 365 complete, unique, ordered days and at least 20 valid observations per day.
- **Use:** U.S. government public data. See NOAA's site for dataset documentation and attribution guidance.

Exact lineage and row counts are recorded in
[`data/source_metadata.json`](data/source_metadata.json). Rebuild the checked-in
table from the archived NOAA file with:

```bash
python scripts/prepare_noaa_data.py
```

The script downloads the fixed 2017 source, verifies its checksum, and refuses
to continue if the source bytes differ.

## Design

The 365-day series is split strictly by time: first 255 days for training, next
54 for validation, and final 56 for a one-time test evaluation. The test segment
is not consulted for model selection.

Two prespecified candidates use annual Fourier terms with weekly
`SARIMA(1,0,1)x(1,0,0,7)` errors. Calling this SARIMA/SARIMAX is accurate: the
model includes an explicit seasonal autoregressive term with period seven and
exogenous Fourier regressors. Validation RMSE selects the number of annual
harmonics. A seven-day seasonal-naive forecast is reported throughout. Five
expanding-window origins in the pre-test period estimate errors for horizons
one through seven. Any lagged/rolling features are shifted before rolling, so
they cannot contain the contemporaneous target.

## Results

The selected one-harmonic model produced:

| Evaluation | Selected model RMSE | Seasonal-naive RMSE |
|---|---:|---:|
| Validation (54 days) | 6.03 °F | 7.72 °F |
| Untouched test (56 days) | 11.38 °F | 20.10 °F |

The nominal 95% model interval covered only **75%** of test observations, with
mean width 24.65 °F. That undercoverage is a material limitation, not evidence
of a calibrated operational interval. Pre-test rolling-origin errors by horizon
are in [`outputs/metrics.json`](outputs/metrics.json), and row-level test
predictions are in [`outputs/test_predictions.csv`](outputs/test_predictions.csv).
Residuals from the final pre-test fit did not show detectable autocorrelation at
lag 14 in this sample (Ljung-Box p=0.850), but a single nonsignificant diagnostic
does not prove model adequacy.

## Limitations and nonclaims

- One station-year is too narrow for operational validation or claims about
  other stations, years, extremes, or climate trends.
- Aggregating to daily means discards intraday structure and extreme values.
- The 56-day test is a single contiguous autumn/winter period; errors are not
  independent replications.
- The test advantage over seasonal naive is descriptive for this split, not a
  guarantee of future performance.
- Interval undercoverage indicates unresolved uncertainty misspecification.
- No causal or meteorological mechanism is inferred.

## Reproduce

Python 3.11 is used in CI.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_analysis.py
python -m pytest -q
jupyter nbconvert --to notebook --execute weather_forecasting_sarimax.ipynb \
  --output /tmp/weather_forecasting_sarimax.executed.ipynb \
  --ExecutePreprocessor.timeout=300
```

`run_analysis.py` writes deterministic JSON/CSV artifacts and figures. The
notebook is a concise, executable walkthrough over the same tested pipeline.

## Repository map

- `weather_forecasting.py` — tested data, split, feature, model, and metric utilities
- `scripts/prepare_noaa_data.py` — acquisition, checksum, and cleaning path
- `run_analysis.py` — selection, backtesting, final evaluation, diagnostics, outputs
- `tests/` — data contract, split integrity, leakage, baseline, and metric checks
- `outputs/` — machine-readable metrics and predictions
- `figures/` — forecast and residual-diagnostic visuals

This repository is the canonical, corrected version of earlier IBM Watson
Studio coursework retained separately in `DAX-Weather-Project` as an archive
pointer.
