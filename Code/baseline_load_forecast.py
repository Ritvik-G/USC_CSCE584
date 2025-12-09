# code/baseline_load_forecast.py
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SMART_FILE = DATA_DIR / "smart_home_energy_consumption_large.csv"

# You can tweak this set depending on what you consider "essential".
ESSENTIAL_APPLIANCES = {"Fridge", "Heater", "Lights"}


def load_smart_home_data():
    df = pd.read_csv(SMART_FILE)
    expected_cols = {
        "Home ID",
        "Appliance Type",
        "Energy Consumption (kWh)",
        "Time",
        "Date",
        "Outdoor Temperature (°C)",
        "Season",
        "Household Size",
    }
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in smart home file: {missing}")
    # Parse datetime for convenience
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df["hour"] = df["datetime"].dt.hour
    return df


def compute_baseline_profile(df: pd.DataFrame, home_id=None) -> pd.DataFrame:
    """Compute typical baseline (essential) load per hour & season.

    If home_id is provided, restrict to that home.

    Returns a DataFrame indexed by ['Season', 'hour'] with a column 'baseline_kwh'.
    """
    df_use = df.copy()
    if home_id is not None:
        df_use = df_use[df_use["Home ID"] == home_id]

    # Keep only essential appliances
    df_ess = df_use[df_use["Appliance Type"].isin(ESSENTIAL_APPLIANCES)].copy()
    if df_ess.empty:
        raise ValueError("No essential appliance data found with current settings.")

    grouped = (
        df_ess.groupby(["Season", "hour"])["Energy Consumption (kWh)"]
        .mean()
        .rename("baseline_kwh")
        .to_frame()
    )
    return grouped


def forecast_baseline_for_day(baseline_profile: pd.DataFrame, season: str) -> pd.Series:
    """Given a baseline profile and a season, return 24h baseline forecast.

    baseline_profile: DataFrame from compute_baseline_profile
    season: one of the values present in the 'Season' column
    """
    # Select given season
    season_profile = baseline_profile.loc[season]
    # Ensure all 24 hours are present; if not, fill missing with overall mean
    all_hours = pd.Index(range(24), name="hour")
    season_profile = season_profile.reindex(all_hours)
    if season_profile["baseline_kwh"].isna().any():
        overall_mean = baseline_profile["baseline_kwh"].mean()
        season_profile["baseline_kwh"].fillna(overall_mean, inplace=True)
    return season_profile["baseline_kwh"].rename("baseline_kwh")


if __name__ == "__main__":
    df_smart = load_smart_home_data()
    baseline_profile_all = compute_baseline_profile(df_smart, home_id=None)
    # Example: forecast for "Winter"
    baseline_winter = forecast_baseline_for_day(baseline_profile_all, season="Winter")
    print("Example 24h baseline forecast for Winter (kWh):")
    print(baseline_winter.to_string())
