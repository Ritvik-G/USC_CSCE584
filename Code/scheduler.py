# code/scheduler.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd


@dataclass
class Job:
    name: str
    power_kw: float
    duration_h: int
    earliest_start: int  # inclusive, 0-23
    latest_end: int      # exclusive, 1-24


def schedule_single_job(job: Job, prices: pd.Series) -> Dict[int, int]:
    """Greedy-optimal schedule for a single job.

    Because cost is linear and there is no maximum power constraint, each job
    can be scheduled independently in the cheapest contiguous block.

    Returns a dict {hour: 1/0} for hours when the job runs.
    """
    T = len(prices)
    assert T == 24, "This simple scheduler assumes a 24h horizon."

    # Valid start times: start + duration <= latest_end
    latest_start = job.latest_end - job.duration_h
    starts = range(job.earliest_start, latest_start + 1)

    best_start = None
    best_cost = float("inf")
    for s in starts:
        window = range(s, s + job.duration_h)
        # Cost ~ sum(price[h] * power_kw) over the job duration
        cost = sum(prices[h] * job.power_kw for h in window)
        if cost < best_cost:
            best_cost = cost
            best_start = s

    schedule = {h: 0 for h in range(T)}
    for h in range(best_start, best_start + job.duration_h):
        schedule[h] = 1
    return schedule


def schedule_jobs(jobs: List[Job], prices: pd.Series) -> Dict[str, Dict[int, int]]:
    """Schedule a list of jobs independently.

    Returns a dict mapping job.name -> {hour: 0/1}.
    """
    schedules: Dict[str, Dict[int, int]] = {}
    for job in jobs:
        schedules[job.name] = schedule_single_job(job, prices)
    return schedules


def compute_total_load(
    baseline: pd.Series,
    jobs: List[Job],
    schedules: Dict[str, Dict[int, int]],
) -> pd.Series:
    """Compute total load (kW) per hour given baseline and scheduled jobs."""
    T = len(baseline)
    total_load = baseline.astype(float).copy()
    if not total_load.index.equals(pd.Index(range(T))):
        total_load.index = range(T)
    for job in jobs:
        sched = schedules[job.name]
        for h in range(T):
            total_load[h] += job.power_kw * sched[h]
    return total_load.rename("total_load_kw")


def compute_cost(prices: pd.Series, load_kw: pd.Series) -> float:
    """Compute total cost over horizon (assuming 1h time steps)."""
    if not prices.index.equals(load_kw.index):
        prices = prices.copy()
        prices.index = load_kw.index
    # price is $/MWh, load is kW -> convert kW to MW
    load_mw = load_kw / 1000.0
    cost = float((prices * load_mw).sum())
    return cost


if __name__ == "__main__":
    # Small self-test with dummy data
    import numpy as np

    prices = pd.Series(np.linspace(50, 100, 24), index=range(24), name="price")
    baseline = pd.Series(1.0, index=range(24), name="baseline_kw")

    jobs = [
        Job(name="Dishwasher", power_kw=1.2, duration_h=2, earliest_start=18, latest_end=24),
        Job(name="WashingMachine", power_kw=1.0, duration_h=1, earliest_start=8, latest_end=20),
    ]

    schedules = schedule_jobs(jobs, prices)
    total_load = compute_total_load(baseline, jobs, schedules)
    cost = compute_cost(prices, total_load)

    print("Schedules:")
    for name, sched in schedules.items():
        print(name, sched)
    print("\nTotal load (kW):")
    print(total_load.to_string())
    print(f"\nTotal cost over 24h: ${cost:.2f}")
