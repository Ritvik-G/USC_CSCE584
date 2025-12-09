# code/app.py
from __future__ import annotations

from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from baseline_load_forecast import (
    load_smart_home_data,
    compute_baseline_profile,
    forecast_baseline_for_day,
)
from price_forecast import load_price_data
from scheduler import Job, schedule_jobs, compute_total_load, compute_cost
from new_house_planner import ApplianceSpec  # reuse the dataclass
from time_series_models import train_time_series_models, forecast_future


# ---------- CACHING HELPERS ----------

@st.cache_data
def cached_load_price_data():
    return load_price_data()


@st.cache_data
def cached_load_smart_home_data():
    return load_smart_home_data()


@st.cache_resource
def cached_train_models():
    # You can tweak maxiter for SARIMA training here
    return train_time_series_models(maxiter=60, seasonal_period=24)


@st.cache_data
def cached_baseline_profile_all():
    df_smart = cached_load_smart_home_data()
    return compute_baseline_profile(df_smart, home_id=None)


def compute_new_house_baseline(season: str) -> pd.Series:
    baseline_profile_all = cached_baseline_profile_all()
    seasons_available = baseline_profile_all.index.get_level_values("Season").unique()
    if season not in seasons_available:
        season = seasons_available[0]
    baseline_24h = forecast_baseline_for_day(baseline_profile_all, season=season)
    return baseline_24h.rename("Baseline_kWh")


# ---------- APPLIANCE CONFIG UI ----------

def build_appliance_specs_ui() -> List[ApplianceSpec]:
    """UI for selecting common appliances from dataset + adding custom ones."""
    df_smart = cached_load_smart_home_data()
    appliance_types = sorted(df_smart["Appliance Type"].unique())

    st.sidebar.subheader("New House: Appliances")

    selected_types = st.sidebar.multiselect(
        "Common appliances to include in the new house",
        options=appliance_types,
        help="Choose from appliances observed in the dataset.",
    )

    add_other = st.sidebar.checkbox(
        "Add other / custom appliances",
        value=False,
        help="Enable to define appliances not found in the dataset list.",
    )

    specs: List[ApplianceSpec] = []

    st.markdown("### 3. Configure non-essential appliances for the new house")

    if selected_types:
        st.markdown("#### Common appliances from dataset")
    for name in selected_types:
        with st.expander(f"{name}", expanded=False):
            # Rough default based on dataset mean energy per record (user can adjust)
            mask = df_smart["Appliance Type"] == name
            mean_energy = float(df_smart.loc[mask, "Energy Consumption (kWh)"].mean())
            if np.isnan(mean_energy) or mean_energy <= 0:
                mean_energy = 1.0

            duration = st.number_input(
                f"{name} - run duration (hours)",
                min_value=1, max_value=8, value=1, step=1,
                key=f"dur_{name}",
            )
            energy = st.number_input(
                f"{name} - energy per run (kWh)",
                min_value=0.1, max_value=50.0, value=round(mean_energy, 2), step=0.1,
                key=f"energy_{name}",
            )

            # Simple defaults: evening/night for dishwashers, overnight for EV, etc.
            default_earliest = 18 if "Dish" in name else 7
            default_latest = 24 if "Dish" in name else 22
            if "EV" in name or "Car" in name:
                default_earliest, default_latest = 0, 7

            earliest = st.number_input(
                f"{name} - earliest start hour (0-23)",
                min_value=0, max_value=23, value=default_earliest, step=1,
                key=f"earliest_{name}",
            )
            latest = st.number_input(
                f"{name} - latest end hour (1-24)",
                min_value=1, max_value=24, value=default_latest, step=1,
                key=f"latest_{name}",
            )

            specs.append(
                ApplianceSpec(
                    name=name,
                    avg_duration_h=int(duration),
                    avg_energy_kwh=float(energy),
                    earliest_start=int(earliest),
                    latest_end=int(latest),
                )
            )

    if add_other:
        st.markdown("#### Other / custom appliances")
        num_other = st.number_input(
            "Number of custom appliances",
            min_value=1, max_value=10, value=1, step=1,
        )
        for i in range(int(num_other)):
            with st.expander(f"Custom appliance {i+1}", expanded=False):
                name = st.text_input(
                    f"Name for custom appliance {i+1}",
                    value=f"Custom_{i+1}",
                    key=f"other_name_{i}",
                )
                duration = st.number_input(
                    f"{name} - run duration (hours)",
                    min_value=1, max_value=8, value=1, step=1,
                    key=f"other_dur_{i}",
                )
                energy = st.number_input(
                    f"{name} - energy per run (kWh)",
                    min_value=0.1, max_value=50.0, value=1.0, step=0.1,
                    key=f"other_energy_{i}",
                )
                earliest = st.number_input(
                    f"{name} - earliest start hour (0-23)",
                    min_value=0, max_value=23, value=8, step=1,
                    key=f"other_earliest_{i}",
                )
                latest = st.number_input(
                    f"{name} - latest end hour (1-24)",
                    min_value=1, max_value=24, value=22, step=1,
                    key=f"other_latest_{i}",
                )

                specs.append(
                    ApplianceSpec(
                        name=name,
                        avg_duration_h=int(duration),
                        avg_energy_kwh=float(energy),
                        earliest_start=int(earliest),
                        latest_end=int(latest),
                    )
                )

    return specs


# ---------- FORECAST VISUALIZATION ----------

def build_hist_and_forecast_frames(df_hist: pd.DataFrame, df_forecast: pd.DataFrame):
    """Prepare long-form data for price and load, with clear historical vs forecast labels."""
    df_hist_sorted = df_hist.sort_values(["Day", "Hour"]).reset_index(drop=True)
    hist_hours = len(df_hist_sorted)
    df_hist_sorted["TimeIndex"] = np.arange(hist_hours)

    # Historical frames
    hist_price = df_hist_sorted[["TimeIndex", "Electricity Price [$/MWh]"]].copy()
    hist_price.rename(columns={"Electricity Price [$/MWh]": "Value"}, inplace=True)
    hist_price["Type"] = "Historical"
    hist_price["Series"] = "Price"

    hist_load = df_hist_sorted[["TimeIndex", "Net Load [MW]"]].copy()
    hist_load.rename(columns={"Net Load [MW]": "Value"}, inplace=True)
    hist_load["Type"] = "Historical"
    hist_load["Series"] = "Load"

    # Forecast frames (shifted in time after historical)
    df_forecast = df_forecast.copy()
    df_forecast["TimeIndex"] = hist_hours + df_forecast["HorizonHour"]

    f_price = df_forecast[["TimeIndex", "PriceForecast"]].copy()
    f_price.rename(columns={"PriceForecast": "Value"}, inplace=True)
    f_price["Type"] = "Forecast"
    f_price["Series"] = "Price"

    f_load = df_forecast[["TimeIndex", "NetLoadForecast"]].copy()
    f_load.rename(columns={"NetLoadForecast": "Value"}, inplace=True)
    f_load["Type"] = "Forecast"
    f_load["Series"] = "Load"

    df_price_long = pd.concat([hist_price, f_price], ignore_index=True)
    df_load_long = pd.concat([hist_load, f_load], ignore_index=True)

    return df_price_long, df_load_long


def plot_forecast(df_hist: pd.DataFrame, df_forecast: pd.DataFrame):
    st.subheader("Forecasted Price and Load (time series)")

    df_price_long, df_load_long = build_hist_and_forecast_frames(df_hist, df_forecast)

    # Price chart
    price_chart = (
        alt.Chart(df_price_long)
        .mark_line()
        .encode(
            x=alt.X("TimeIndex:Q", title="Time (hours from start of history)"),
            y=alt.Y("Value:Q", title="Price ($/MWh)"),
            color=alt.Color("Type:N", title="Segment"),
        )
        .properties(height=250)
    )

    # Load chart
    load_chart = (
        alt.Chart(df_load_long)
        .mark_line()
        .encode(
            x=alt.X("TimeIndex:Q", title="Time (hours from start of history)"),
            y=alt.Y("Value:Q", title="Net Load (MW)"),
            color=alt.Color("Type:N", title="Segment"),
        )
        .properties(height=250)
    )

    st.altair_chart(price_chart, use_container_width=True)
    st.altair_chart(load_chart, use_container_width=True)


# ---------- SCHEDULING VISUALS ----------

def build_daily_price_curve(df_forecast: pd.DataFrame, day_index: int) -> pd.Series:
    """Extract a 24h price curve for a selected future day index."""
    subset = df_forecast[df_forecast["FutureDayIndex"] == day_index].copy()
    if subset.empty:
        raise ValueError("Selected future day index not in forecast horizon.")
    subset = subset.sort_values("HourOfDay")
    prices = subset["PriceForecast"].reset_index(drop=True)
    prices.index = range(24)
    prices.name = "PriceForecast_Day"
    return prices


def plot_new_house_schedule(
    baseline_24h: pd.Series,
    jobs: List[Job],
    schedules: Dict[str, Dict[int, int]],
    prices_24h: pd.Series,
):
    st.subheader("New House: Appliance Scheduling for Selected Future Day")

    # Tabular schedule
    schedule_df = pd.DataFrame(
        {job.name: schedules[job.name] for job in jobs},
        index=pd.Index(range(24), name="Hour"),
    )
    st.write("**Appliance ON/OFF schedule (1 = ON)**")
    st.dataframe(schedule_df)

    # Individual appliance load (kW) per hour
    appliance_loads = {}
    for job in jobs:
        sched_series = pd.Series(schedules[job.name])
        appliance_loads[job.name] = sched_series * job.power_kw

    df_app_loads = pd.DataFrame(appliance_loads, index=range(24))
    df_app_loads.index.name = "Hour"

    total_load = compute_total_load(baseline_24h, jobs, schedules)
    total_cost = compute_cost(prices_24h, total_load)

    # Line chart: baseline vs total
    st.write("**Baseline vs total load (kW approx)**")
    df_plot = pd.DataFrame(
        {
            "Baseline": baseline_24h.values,
            "TotalLoad": total_load.values,
        },
        index=pd.Index(range(24), name="Hour"),
    )
    st.line_chart(df_plot)

    # Stacked bar: baseline + each appliance
    st.write("**Type-based hourly usage (appliances stacked on top of baseline)**")
    df_stacked = df_app_loads.copy()
    df_stacked["Baseline"] = baseline_24h.values
    st.bar_chart(df_stacked)

    # Daily energy per appliance
    st.write("**Daily energy by appliance (kWh)**")
    daily_energy = df_app_loads.sum(axis=0)
    st.dataframe(daily_energy.rename("Daily_kWh").to_frame())

    st.markdown(
        f"**Estimated daily cost for this new house (selected day):** "
        f"`{total_cost:.2f} $`"
    )


# ---------- MAIN STREAMLIT APP ----------

def main():
    st.set_page_config(
        page_title="Smart Appliance Scheduler & Energy Forecaster",
        layout="wide",
    )
    st.title("⚡ Smart Appliance Scheduler & Energy Forecaster")
    st.caption(
        "One-stop tool for forecasting electricity prices & net load, "
        "and scheduling household appliances in a new home for minimum cost."
    )

    # Sidebar: forecast configuration
    st.sidebar.header("Forecast Configuration")

    months_ahead = st.sidebar.number_input(
        "Forecast horizon (months ahead)",
        min_value=1, max_value=60, value=6, step=1,
        help="How many months into the future you want to forecast. "
             "Assumes seasonal patterns from historical data.",
    )
    horizon_days = int(months_ahead * 30)

    st.sidebar.header("Baseline / Season")
    season_choice = st.sidebar.selectbox(
        "Season for new house baseline",
        options=["Winter", "Spring", "Summer", "Fall"],
        index=0,
    )

    selected_future_day = st.sidebar.number_input(
        "Day index ahead for scheduling this new house",
        min_value=0, max_value=horizon_days - 1, value=0, step=1,
        help="0 = the first forecast day after the historical period.",
    )

    # Appliances UI
    appliance_specs = build_appliance_specs_ui()

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Run Forecast & Scheduling")

    if not run_button:
        st.info(
            "Configure the forecast horizon, baseline season, and appliances in the sidebar, "
            "then click **Run Forecast & Scheduling**."
        )
        return

    # --- Train models & forecast ---
    st.markdown("### 1. Train models and build future forecasts (price & load)")
    with st.spinner("Training time-series models (SARIMA / fallback)..."):
        models = cached_train_models()

    df_hist = cached_load_price_data()

    with st.spinner("Forecasting future price and net load..."):
        df_forecast, total_E, avg_P = forecast_future(models, horizon_days=horizon_days)

    st.write(
        f"**Forecast horizon:** {horizon_days} days "
        f"(~{months_ahead} months)  \n"
        f"- Approximate total system energy over horizon: `{total_E:.2f} MWh`  \n"
        f"- Average forecasted price: `{avg_P:.2f} $/MWh`"
    )

    plot_forecast(df_hist, df_forecast)

    # --- New house baseline ---
    st.markdown("### 2. New house baseline (essential usage)")
    baseline_24h = compute_new_house_baseline(season_choice)
    st.write(f"Using season: **{season_choice}** for new house baseline.")
    st.line_chart(
        pd.DataFrame(
            {"Baseline_kWh": baseline_24h.values},
            index=pd.Index(range(24), name="Hour"),
        )
    )

    # --- Show appliance specs table ---
    st.markdown("### 3. New house non-essential appliances (summary)")
    if not appliance_specs:
        st.warning(
            "No appliances configured for the new house. "
            "Go to the sidebar and select common appliances or add custom ones."
        )
        return

    app_table = pd.DataFrame(
        {
            "Name": [a.name for a in appliance_specs],
            "AvgDuration_h": [a.avg_duration_h for a in appliance_specs],
            "AvgEnergy_kWh": [a.avg_energy_kwh for a in appliance_specs],
            "EarliestStart_h": [a.earliest_start for a in appliance_specs],
            "LatestEnd_h": [a.latest_end for a in appliance_specs],
        }
    )
    st.dataframe(app_table)

    # --- Scheduling for selected future day ---
    st.markdown("### 4. Schedule new house appliances on a specific future day")

    try:
        prices_24h = build_daily_price_curve(df_forecast, day_index=int(selected_future_day))
    except ValueError as e:
        st.error(str(e))
        return

    st.write(
        f"Price forecast for **future day index {int(selected_future_day)}** "
        "(0 = first forecast day after the historical period):"
    )
    price_df = pd.DataFrame(
        {"PriceForecast": prices_24h.values},
        index=pd.Index(range(24), name="Hour"),
    )
    st.bar_chart(price_df)

    # Build Job objects and schedule
    jobs: List[Job] = [
        a_spec.to_job() for a_spec in appliance_specs
    ]

    schedules = schedule_jobs(jobs, prices_24h)

    plot_new_house_schedule(
        baseline_24h=baseline_24h,
        jobs=jobs,
        schedules=schedules,
        prices_24h=prices_24h,
    )

    st.markdown("---")
    st.markdown(
        "This app treats forecasting as **two coupled time series**: price and load.  \n"
        "- SARIMA (or a simple fallback) learns patterns from historical data.  \n"
        "- The new house baseline is anchored to typical essential usage from the dataset.  \n"
        "- Non-essential appliances are scheduled in a cost-minimizing way based on the "
        "forecasted hourly price for a chosen future day."
    )


if __name__ == "__main__":
    main()
