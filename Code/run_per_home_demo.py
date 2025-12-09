# code/run_per_home_demo.py
"""
Per-home orchestration demo:

- Train a simple model on historical price time series
  to mimic the optimal scheduler's start-time decisions.
- For a few homes:
  * build a baseline forecast
  * schedule non-essential devices using the trained model
  * compute total load & cost.
"""

import pandas as pd

from price_forecast import load_price_data
from baseline_load_forecast import (
    load_smart_home_data,
    compute_baseline_profile,
    forecast_baseline_for_day,
)
from scheduler import Job, compute_total_load, compute_cost
from per_home_orchestrator import PerHomeOrchestrator


def main():
    # 1) Load price data (time series) and smart-home data
    df_price = load_price_data()
    df_smart = load_smart_home_data()

    # 2) Define job templates (non-essential devices)
    jobs = [
        Job(
            name="Dishwasher",
            power_kw=1.2,
            duration_h=2,
            earliest_start=18,
            latest_end=24,
        ),
        Job(
            name="WashingMachine",
            power_kw=1.0,
            duration_h=1,
            earliest_start=7,
            latest_end=22,
        ),
        Job(
            name="EV_Charging",
            power_kw=3.5,
            duration_h=3,
            earliest_start=0,
            latest_end=7,
        ),
    ]

    # 3) Train the per-home orchestrator on historical price time series
    orchestrator = PerHomeOrchestrator(jobs=jobs)
    orchestrator.fit(df_price, num_samples=200, random_state=0)

    # 4) Choose a test day (last day in price dataset) and build its price curve
    test_day = int(df_price["Day"].max())
    test_day_prices = df_price[df_price["Day"] == test_day].sort_values("Hour")
    prices_24h = test_day_prices["Electricity Price [$/MWh]"].reset_index(drop=True)
    prices_24h.index = range(24)
    prices_24h.name = "price"

    print(f"\nUsing Day={test_day} as test day.")
    print("Test-day hourly prices ($/MWh):")
    print(prices_24h.to_string())

    # 5) Get a few homes to demonstrate per-home orchestration
    all_home_ids = df_smart["Home ID"].unique()
    demo_home_ids = all_home_ids[:3]  # first 3 homes

    print(f"\nRunning orchestration for homes: {demo_home_ids.tolist()}")

    for home_id in demo_home_ids:
        print("\n" + "=" * 60)
        print(f"Home ID: {home_id}")

        # 5a) Compute per-home baseline profile and 24h forecast
        try:
            baseline_profile_home = compute_baseline_profile(df_smart, home_id=home_id)
        except ValueError:
            print("  Skipping: no essential appliance data for this home.")
            continue

        # For simplicity, pick a season; in a real system you'd map test_day to season
        # Here we use "Winter" if present, else first available season for this home.
        seasons_for_home = baseline_profile_home.index.get_level_values("Season").unique()
        if "Winter" in seasons_for_home:
            season = "Winter"
        else:
            season = seasons_for_home[0]

        baseline_24h = forecast_baseline_for_day(baseline_profile_home, season=season)

        # 5b) Schedule jobs for this home using the trained orchestrator
        schedules = orchestrator.schedule_jobs_for_home(home_id=home_id, prices=prices_24h)

        # 5c) Compute total load and cost
        total_load = compute_total_load(baseline_24h, jobs, schedules)
        total_cost = compute_cost(prices_24h, total_load)

        print(f"  Season used for baseline: {season}")
        print("  Baseline load (kWh ~ kW over 1h):")
        print("  " + baseline_24h.to_string().replace("\n", "\n  "))

        print("\n  Job schedules (1 = ON):")
        for job in jobs:
            sched_series = pd.Series(schedules[job.name])
            print(f"  {job.name}:")
            print("  " + sched_series.to_string().replace("\n", "\n  "))

        print("\n  Total load (approx kW):")
        print("  " + total_load.to_string().replace("\n", "\n  "))

        print(f"\n  Total cost over 24h for home {home_id}: ${total_cost:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
