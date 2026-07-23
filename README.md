# Weather Time-Series Forecasting (SARIMAX)

Forecasting **hourly dry-bulb temperature** at JFK Airport using a seasonal
ARIMA (SARIMAX) model. The notebook walks through the full time-series
workflow: framing the target, engineering lag/rolling features, splitting the
series chronologically into train / validation / test sets, fitting the model,
and evaluating forecast error with RMSE.

## Notebook

`weather_forecasting_sarimax.ipynb`

- Loads and indexes an hourly weather series (`DATE` parsed as the time index)
- Builds a baseline and shifts/rolls the target to create supervised features
- Chronological `split_data()` into train / validation / test (no leakage)
- Fits `statsmodels` **SARIMAX**, forecasts, and scores with `mean_squared_error`
- Plots predictions against actuals

## Skills demonstrated

- Time-series framing (stationarity, seasonality, leakage-safe splitting)
- Feature engineering with shifting and rolling averages
- `statsmodels` SARIMAX modeling and forecast evaluation (RMSE)

## Data

Built on the **NOAA hourly weather data for JFK Airport** (Local
Climatological Data). A sample of this dataset is publicly available through
the NOAA LCD portal. Saved cell outputs are included so the analysis renders on
GitHub as-is; to re-run, point the data-loading cell at a local
`jfk_weather.csv`.

## Run it

```bash
pip install -r requirements.txt
jupyter lab
```
