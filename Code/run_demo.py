# code/run_demo.py
"""End-to-end example:
- load historical price data and compute a simple 24h price forecast
- load smart-home data and compute a 24h baseline forecast
- define a few example non-essential jobs
- schedule them to minimize cost
"""

import pandas as pd

from price_forecast import load_price_data, forecast_next_day_prices
from baseline_load_forecast import (
    load_smart_home_data,
    compute_baseline_profile,
    forecast_baseline_for_day,
)
from scheduler import Job, schedule_jobs, compute_total_load, compute_cost


def main():
    # 1) Price forecast
    price_df = load_price_data()
    prices_forecast = forecast_next_day_prices(price_df)  # length 24

    # 2) Baseline forecast (using all homes, Winter as an example)
    smart_df = load_smart_home_data()
    baseline_profile = compute_baseline_profile(smart_df, home_id=None)
    # You can change "Winter" to "Summer", "Spring", "Fall"
    baseline_24h = forecast_baseline_for_day(baseline_profile, season="Winter")

    # 3) Define some example non-essential jobs
    # Times are in 0-23 (0 = midnight, 23 = 11pm)
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

    # 4) Schedule jobs independently to minimize cost
    schedules = schedule_jobs(jobs, prices_forecast)

    # 5) Compute resulting total load and cost
    total_load = compute_total_load(baseline_24h, jobs, schedules)
    total_cost = compute_cost(prices_forecast, total_load)

    print("Hourly price forecast ($/MWh):")
    print(prices_forecast.to_string())
    print("\nBaseline load forecast (kWh, approximated as kW over 1h):")
    print(baseline_24h.to_string())

    print("\nJob schedules (1 = ON):")
    for job in jobs:
        sched = schedules[job.name]
        print(f"\n{job.name}:")
        print(pd.Series(sched).to_string())

    print("\nTotal load (approx kW):")
    print(total_load.to_string())
    print(f"\nTotal cost over 24h: ${total_cost:.2f}")


if __name__ == "__main__":
    main()
