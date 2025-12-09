# code/new_house_planner.py
"""
New house planner:

- Use time-series data to:
  * estimate total / forecasted neighborhood energy usage (net load)
  * forecast price for a "typical" day (hour-of-day profile)

- Assume a new house joins with the same baseline requirements as the
  average existing home (computed from the smart-home dataset).

- For the new house's NON-ESSENTIAL appliances:
  * use user-provided average run time + average energy consumption
  * suggest the best start time window (schedule) to minimize their cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

from price_forecast import load_price_data
from baseline_load_forecast import (
    load_smart_home_data,
    compute_baseline_profile,
    forecast_baseline_for_day,
)
from scheduler import Job, schedule_jobs, compute_total_load, compute_cost


@dataclass
class ApplianceSpec:
    """Description of a non-essential appliance in the NEW house.

    avg_duration_h: average run time per use (hours)
    avg_energy_kwh: total energy consumption per use (kWh)
    earliest_start: earliest hour (0-23) the user is willing to start
    latest_end: latest hour (1-24) the user wants it finished by (exclusive)
    """
    name: str
    avg_duration_h: int
    avg_energy_kwh: float
    earliest_start: int
    latest_end: int

    def to_job(self) -> Job:
        # Power (kW) = Energy (kWh) / Duration (h)
        power_kw = self.avg_energy_kwh / float(self.avg_duration_h)
        return Job(
            name=self.name,
            power_kw=power_kw,
            duration_h=self.avg_duration_h,
            earliest_start=self.earliest_start,
            latest_end=self.latest_end,
        )


def forecast_price_and_net_load(df_price: pd.DataFrame) -> (pd.Series, pd.Series, float):
    """
    Compute simple hour-of-day average forecasts for:
    - price ($/MWh)
    - net load (MW)

    Returns:
      price_forecast: 24-length Series indexed 0..23
      net_load_forecast: 24-length Series indexed 0..23
      total_daily_energy_MWh: scalar, approximate total daily energy
    """
    hourly_group = df_price.groupby("Hour")

    hourly_price = hourly_group["Electricity Price [$/MWh]"].mean()
    hourly_net_load = hourly_group["Net Load [MW]"].mean()

    # Map 0..23 -> Hour 1..24
    hours_0_23 = range(24)
    price_vals = [hourly_price[h + 1] for h in hours_0_23]
    net_load_vals = [hourly_net_load[h + 1] for h in hours_0_23]

    price_forecast = pd.Series(price_vals, index=hours_0_23, name="price_forecast")
    net_load_forecast = pd.Series(net_load_vals, index=hours_0_23, name="net_load_forecast_MW")

    # Approximate daily energy (MWh) as sum over 24h of load[MW] * 1h
    total_daily_energy_MWh = float(net_load_forecast.sum())

    return price_forecast, net_load_forecast, total_daily_energy_MWh


def compute_new_house_baseline(df_smart: pd.DataFrame, season: str = "Winter") -> pd.Series:
    """
    Compute a 24h baseline forecast (kWh) for the NEW house,
    assuming it has the same essential load pattern as the average existing home.

    Uses:
      - compute_baseline_profile with home_id=None (all homes)
      - forecast_baseline_for_day for the chosen season
    """
    baseline_profile_all = compute_baseline_profile(df_smart, home_id=None)
    # If the requested season is not present, fall back to the first available
    seasons_available = baseline_profile_all.index.get_level_values("Season").unique()
    if season not in seasons_available:
        season = seasons_available[0]
    baseline_24h = forecast_baseline_for_day(baseline_profile_all, season=season)
    return baseline_24h.rename("new_house_baseline_kWh")


def build_jobs_from_appliance_specs(specs: List[ApplianceSpec]) -> List[Job]:
    """Convert high-level appliance specs into Job objects for scheduling."""
    return [spec.to_job() for spec in specs]


def main():
    # 1) Load time-series price + net load (neighborhood/system-level)
    df_price = load_price_data()
    price_forecast, net_load_forecast, daily_energy_MWh = forecast_price_and_net_load(df_price)

    print("\n=== Neighborhood/System-Level Forecast (from time-series) ===")
    print("Hourly price forecast ($/MWh):")
    print(price_forecast.to_string())
    print("\nHourly net load forecast (MW):")
    print(net_load_forecast.to_string())
    print(f"\nApproximate total daily energy (MWh) for the system: {daily_energy_MWh:.2f}")

    # 2) Compute baseline load for the NEW house (same base as average existing home)
    df_smart = load_smart_home_data()
    new_house_baseline = compute_new_house_baseline(df_smart, season="Winter")

    print("\n=== New House Baseline Forecast ===")
    print("Baseline essential load (kWh, approx kW over each hour):")
    print(new_house_baseline.to_string())

    # 3) Collect non-essential appliance info for the NEW house.
    #    In a real system, this would come from user input / UI or file.
    #    Here we hardcode an example list.
    new_house_appliances = [
        ApplianceSpec(
            name="Dishwasher",
            avg_duration_h=2,
            avg_energy_kwh=1.8,   # kWh per run
            earliest_start=18,    # 6pm
            latest_end=24,        # must finish by midnight
        ),
        ApplianceSpec(
            name="WashingMachine",
            avg_duration_h=1,
            avg_energy_kwh=0.7,
            earliest_start=7,     # 7am earliest
            latest_end=22,        # 10pm latest
        ),
        ApplianceSpec(
            name="Dryer",
            avg_duration_h=1,
            avg_energy_kwh=1.0,
            earliest_start=9,
            latest_end=22,
        ),
        ApplianceSpec(
            name="EV_Charging",
            avg_duration_h=3,
            avg_energy_kwh=10.5,  # e.g. 3.5 kW * 3h
            earliest_start=0,     # anytime after midnight
            latest_end=7,         # done by 7am
        ),
    ]

    jobs = build_jobs_from_appliance_specs(new_house_appliances)

    # 4) Schedule these non-essential appliances to minimize cost for the NEW house
    schedules = schedule_jobs(jobs, price_forecast)

    # 5) Compute the NEW house's load profile and cost
    total_load_new_house = compute_total_load(new_house_baseline, jobs, schedules)
    total_cost_new_house = compute_cost(price_forecast, total_load_new_house)

    print("\n=== Suggested Schedule for New House (Non-Essential Appliances) ===")
    for job in jobs:
        sched_series = pd.Series(schedules[job.name])
        print(f"\n{job.name} (power ~ {job.power_kw:.2f} kW, duration {job.duration_h}h):")
        print("Recommended ON hours (1 = ON):")
        print(sched_series.to_string())

    print("\n=== New House 24h Load Profile (approx kW) ===")
    print(total_load_new_house.to_string())

    print(f"\nEstimated total daily cost for the new house: ${total_cost_new_house:.2f}")
    print("\nInterpretation:")
    print("  • The schedule above tells the new homeowner WHEN to run each non-essential appliance")
    print("    to minimize their electricity bill, given the typical price and load patterns.")
    print("  • Baseline essential usage is assumed similar to the neighborhood average.")
    print("  • This keeps the whole system time-series based, but extremely simple in compute.")


if __name__ == "__main__":
    main()
