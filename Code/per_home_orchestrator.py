# code/per_home_orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from scheduler import Job, schedule_single_job
from price_forecast import load_price_data
from baseline_load_forecast import load_smart_home_data, compute_baseline_profile

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    RandomForestRegressor = None


@dataclass
class PerHomeOrchestrator:
    jobs: List[Job]
    model: Optional["RandomForestRegressor"] = None

    def _build_features_for_job(self, prices: pd.Series, job: Job) -> np.ndarray:
        """Build a simple feature vector:
        [price_0, ..., price_23, earliest_start, latest_end, duration, power_kw].
        """
        prices = prices.values.astype(float)
        meta = np.array(
            [job.earliest_start, job.latest_end, job.duration_h, job.power_kw],
            dtype=float,
        )
        feats = np.concatenate([prices, meta])
        return feats

    def _get_day_price_curve(self, df_price: pd.DataFrame, day: int) -> pd.Series:
        """Return a 24-length Series of prices for a given day, index 0..23."""
        sub = df_price[df_price["Day"] == day].sort_values("Hour")
        if len(sub) != 24:
            raise ValueError(f"Day {day} does not have exactly 24 rows in price data.")
        prices = sub["Electricity Price [$/MWh]"].reset_index(drop=True)
        prices.index = range(24)
        prices.name = "price"
        return prices

    def fit(self, df_price: pd.DataFrame, num_samples: int = 200, random_state: int = 0):
        """Train a small model to mimic the optimal scheduler's start time decisions.

        - We sample `num_samples` days from the historical price data.
        - For each day and each job, we:
          * compute the optimal schedule using schedule_single_job()
          * extract the start hour as the label
          * build a feature vector from the price curve + job metadata

        If sklearn is not available, this becomes a no-op and the orchestrator
        will simply fall back to using the scheduler at inference time.
        """
        if RandomForestRegressor is None:
            print(
                "[PerHomeOrchestrator] sklearn not available; "
                "will fall back to exact scheduler (no training)."
            )
            self.model = None
            return

        rng = np.random.default_rng(seed=random_state)
        days = df_price["Day"].unique()
        if len(days) == 0:
            raise ValueError("No days found in price data.")

        feature_rows = []
        targets = []

        for _ in range(num_samples):
            day = int(rng.choice(days))
            prices = self._get_day_price_curve(df_price, day)
            for job in self.jobs:
                sched = schedule_single_job(job, prices)
                start_hour = min(h for h, v in sched.items() if v == 1)
                feats = self._build_features_for_job(prices, job)
                feature_rows.append(feats)
                targets.append(start_hour)

        X = np.vstack(feature_rows)
        y = np.array(targets, dtype=float)

        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X, y)
        self.model = model
        print("[PerHomeOrchestrator] Training complete. "
              f"Trained on {len(y)} examples.")

    def predict_start_hour(self, job: Job, prices: pd.Series) -> int:
        """Predict a start hour for a single job.

        If model is not trained or sklearn is missing, fall back to exact scheduler.
        """
        # Fallback: exact optimal schedule
        if self.model is None:
            sched = schedule_single_job(job, prices)
            return min(h for h, v in sched.items() if v == 1)

        feats = self._build_features_for_job(prices, job).reshape(1, -1)
        pred = float(self.model.predict(feats)[0])

        # Round to nearest hour, clamp to valid window
        start_hour = int(round(pred))
        # enforce window and feasibility
        earliest = job.earliest_start
        latest_start = job.latest_end - job.duration_h
        start_hour = max(earliest, min(start_hour, latest_start))

        return start_hour

    def schedule_jobs_for_home(
        self,
        home_id: int,
        prices: pd.Series,
    ) -> Dict[str, Dict[int, int]]:
        """Schedule all jobs for a given home.

        Currently, the ML model does not explicitly depend on home_id,
        but we keep the interface so that extending to per-home features
        later is easy.

        Returns: {job_name: {hour: 0/1}}
        """
        T = len(prices)
        if T != 24:
            raise ValueError("This orchestrator assumes a 24h horizon.")

        schedules: Dict[str, Dict[int, int]] = {}
        for job in self.jobs:
            start_hour = self.predict_start_hour(job, prices)
            sched = {h: 0 for h in range(T)}
            for h in range(start_hour, start_hour + job.duration_h):
                sched[h] = 1
            schedules[job.name] = sched
        return schedules


def example_build_baseline_per_home():
    """Utility showing how to compute per-home baseline profiles.

    Not used directly in training, but useful if you want per-home baselines.
    """
    df_smart = load_smart_home_data()
    home_ids = df_smart["Home ID"].unique()
    print(f"Found {len(home_ids)} homes in smart-home dataset.")

    baseline_profiles = {}
    for h in home_ids:
        try:
            prof = compute_baseline_profile(df_smart, home_id=h)
            baseline_profiles[h] = prof
        except ValueError:
            # Skip homes with no essential data
            continue

    print(f"Computed baseline profiles for {len(baseline_profiles)} homes.")
    return baseline_profiles


if __name__ == "__main__":
    # Small self-test training
    df_price = load_price_data()

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

    orchestrator = PerHomeOrchestrator(jobs=jobs)
    orchestrator.fit(df_price, num_samples=200)

    # Test on the last day in the dataset
    last_day = int(df_price["Day"].max())
    prices_last_day = orchestrator._get_day_price_curve(df_price, last_day)

    schedules_example = orchestrator.schedule_jobs_for_home(home_id=1, prices=prices_last_day)
    print("\nExample schedules for home 1 on the last day:")
    for name, sched in schedules_example.items():
        print(name, pd.Series(sched).to_string())
