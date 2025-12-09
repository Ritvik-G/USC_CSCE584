# code/time_series_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import pandas as pd

from price_forecast import load_price_data

try:
    # Stronger time-series model: SARIMA (seasonal ARIMA)
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


@dataclass
class TimeSeriesModels:
    """Container for time-series models for price and net load."""
    use_sarima: bool
    price_result: Any
    load_result: Any
    train_length: int
    seasonal_period: int = 24  # 24-hour daily seasonality


def _prepare_series(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Flatten Day/Hour data into two time series indexed by hourly time.

    Returns:
      price_series: Series indexed 0..N-1 with Electricity Price [$/MWh]
      load_series:  Series indexed 0..N-1 with Net Load [MW]
    """
    required_cols = {"Day", "Hour", "Electricity Price [$/MWh]", "Net Load [MW]"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in price/net-load data: {missing}")

    df_sorted = df.sort_values(["Day", "Hour"]).reset_index(drop=True)

    price_series = df_sorted["Electricity Price [$/MWh]"].copy()
    price_series.index = np.arange(len(price_series))

    load_series = df_sorted["Net Load [MW]"].copy()
    load_series.index = np.arange(len(load_series))

    return price_series, load_series


def train_time_series_models(maxiter: int = 50, seasonal_period: int = 24) -> TimeSeriesModels:
    """Train SARIMA models for price and net load, if possible.

    - If statsmodels is available: fit SARIMA(1,0,1)x(1,1,1,24) on the hourly series.
    - If not: fall back to using simple hourly-average baselines (no model).

    Returns a TimeSeriesModels object.
    """
    df = load_price_data()
    price_series, load_series = _prepare_series(df)
    train_length = len(price_series)

    if not _HAS_STATSMODELS:
        print("[TimeSeriesModels] statsmodels not available; "
              "using hourly-average fallback (no SARIMA).")
        return TimeSeriesModels(
            use_sarima=False,
            price_result=None,
            load_result=None,
            train_length=train_length,
            seasonal_period=seasonal_period,
        )

    # SARIMA parameters: light but expressive, with daily seasonality
    order = (1, 0, 1)
    seasonal_order = (1, 1, 1, seasonal_period)

    print("[TimeSeriesModels] Training SARIMA for price...")
    price_model = SARIMAX(
        price_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    price_result = price_model.fit(maxiter=maxiter, disp=False)

    print("[TimeSeriesModels] Training SARIMA for net load...")
    load_model = SARIMAX(
        load_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    load_result = load_model.fit(maxiter=maxiter, disp=False)

    print(f"[TimeSeriesModels] Training complete on {train_length} hourly points.")

    return TimeSeriesModels(
        use_sarima=True,
        price_result=price_result,
        load_result=load_result,
        train_length=train_length,
        seasonal_period=seasonal_period,
    )


def forecast_future(
    models: TimeSeriesModels,
    horizon_days: int,
) -> Tuple[pd.DataFrame, float, float]:
    """Forecast future price and net load for a given number of days.

    Args:
      models: trained TimeSeriesModels
      horizon_days: how many days into the future (integer)
                    each day has 24 hourly steps

    Returns:
      df_forecast: DataFrame with columns:
        - 'HorizonHour'      : 0..(H-1)
        - 'FutureDayIndex'   : floor(HorizonHour / 24)
        - 'HourOfDay'        : 0..23
        - 'PriceForecast'    : $/MWh
        - 'NetLoadForecast'  : MW
      total_energy_MWh: approximate total energy over horizon (sum load * 1h)
      avg_price: average forecasted price over horizon
    """
    steps = int(horizon_days * 24)
    if steps <= 0:
        raise ValueError("horizon_days must be positive.")

    df_hist = load_price_data()
    _, _ = _prepare_series(df_hist)  # sanity check

    if models.use_sarima and models.price_result is not None:
        price_forecast = models.price_result.get_forecast(steps=steps).predicted_mean
        load_forecast = models.load_result.get_forecast(steps=steps).predicted_mean
    else:
        # Fallback: hourly-average heuristic
        print("[forecast_future] Using hourly-average fallback (no SARIMA models).")
        grouped = df_hist.groupby("Hour")
        hourly_price = grouped["Electricity Price [$/MWh]"].mean()
        hourly_load = grouped["Net Load [MW]"].mean()

        horizon_hours = np.arange(steps)
        hours_of_day = horizon_hours % 24
        price_forecast = [hourly_price[h + 1] for h in hours_of_day]  # Hour is 1..24
        load_forecast = [hourly_load[h + 1] for h in hours_of_day]

    horizon_hours = np.arange(steps)
    future_day_index = horizon_hours // 24
    hour_of_day = horizon_hours % 24

    df_forecast = pd.DataFrame(
        {
            "HorizonHour": horizon_hours,
            "FutureDayIndex": future_day_index,
            "HourOfDay": hour_of_day,
            "PriceForecast": np.asarray(price_forecast, dtype=float),
            "NetLoadForecast": np.asarray(load_forecast, dtype=float),
        }
    )

    total_energy_MWh = float(df_forecast["NetLoadForecast"].sum())
    avg_price = float(df_forecast["PriceForecast"].mean())

    return df_forecast, total_energy_MWh, avg_price


if __name__ == "__main__":
    models = train_time_series_models(maxiter=50, seasonal_period=24)
    df_forecast, total_E, avg_P = forecast_future(models, horizon_days=7)
    print(df_forecast.head())
    print(f"\nTotal energy over 7 days: {total_E:.2f} MWh")
    print(f"Average forecasted price: {avg_P:.2f} $/MWh")
