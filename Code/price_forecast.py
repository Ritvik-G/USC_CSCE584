# code/price_forecast.py
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PRICE_FILE = DATA_DIR / "Historical_Net_Load_and_Electricity_Price.csv"


def load_price_data():
    """Load historical price data as a pandas DataFrame."""
    df = pd.read_csv(PRICE_FILE)
    # Basic sanity check
    expected_cols = {"Day", "Hour", "Net Load [MW]", "Electricity Price [$/MWh]"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in price file: {missing}")
    return df


def compute_hourly_price_profile(df: pd.DataFrame) -> pd.Series:
    """Return average price for each hour of day (1-24) as a pandas Series.

    This is a very simple but strong baseline: forecast = typical price per hour.
    """
    hourly_profile = (
        df.groupby("Hour")["Electricity Price [$/MWh]"]
        .mean()
        .sort_index()
    )
    return hourly_profile


def forecast_next_day_prices(df: pd.DataFrame) -> pd.Series:
    """Forecast prices for the next 24 hours using the hourly average profile.

    Returns a Series indexed 0..23 representing hours of the next day (0-based).
    """
    hourly_profile = compute_hourly_price_profile(df)
    # Map 0..23 -> 1..24 hour labels
    next_day_hours = {h: (h + 1) for h in range(24)}
    forecast_values = [hourly_profile[next_day_hours[h]] for h in range(24)]
    return pd.Series(forecast_values, index=range(24), name="forecast_price")


if __name__ == "__main__":
    df_price = load_price_data()
    forecast = forecast_next_day_prices(df_price)
    print("Next-day hourly price forecast ($/MWh):")
    print(forecast.to_string())
