"""
data_fetcher_combined.py
Fetches Formula 1 race and qualifying data (2022 → latest race)
and ensures 'Constructor_ID' feature is present in raw_data.csv
"""

import os
import time
import pandas as pd
from tqdm import tqdm
import fastf1
from fastf1 import plotting

# Enable cache
fastf1.Cache.enable_cache('E:/Projects/F1_predictor_samp/data/cache')

# ============ CONFIG ============
START_YEAR = 2022
END_YEAR = 2025
OUTPUT_PATH = "E:/Projects/F1_predictor_samp/data/raw_data.csv"
MAX_RETRIES = 3
# ================================


def get_session_data(year: int, gp_name: str, session_type: str):
    """Helper to safely load a session with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            session = fastf1.get_session(year, gp_name, session_type)
            session.load()
            return session
        except Exception as e:
            print(f"   ❌ {gp_name} failed to load ({session_type}) (attempt {attempt}): {e}")
            time.sleep(2)
    return None


def collect_season_data(year: int):
    """Fetch all races and qualifying sessions for a given year."""
    results = []
    schedule = fastf1.get_event_schedule(year, include_testing=False)

    for _, event in tqdm(schedule.iterrows(), total=len(schedule), desc=f"{year} Season Progress"):
        gp_name = event["EventName"]

        race_session = get_session_data(year, gp_name, "R")
        qual_session = get_session_data(year, gp_name, "Q")

        if race_session is None or qual_session is None:
            print(f"   🔁 Skipping {gp_name} after {MAX_RETRIES} retries.")
            continue

        # --------------------- Qualifying Data ---------------------
        qual_results = qual_session.results
        if qual_results is not None:
            qual_cols = ["DriverNumber", "Abbreviation", "TeamName", "Position", "Q1", "Q2", "Q3"]
            if "TeamId" in qual_results.columns:
                qual_cols.append("TeamId")

            qual_df = qual_results[qual_cols].copy()
            qual_df.rename(
                columns={
                    "DriverNumber": "Driver_ID",
                    "Abbreviation": "Driver",
                    "TeamName": "Constructor",
                    "TeamId": "Constructor_ID",
                    "Position": "Grid_Position",
                },
                inplace=True,
            )
            # Fastest Q time
            qual_df["Qualifying_Time"] = qual_df[["Q1", "Q2", "Q3"]].min(axis=1, skipna=True)
        else:
            print(f"   ⚠️ No qualifying results found for {gp_name}")
            continue

        # --------------------- Race Data ---------------------
        race_results = race_session.results
        if race_results is None:
            print(f"   ⚠️ No race results found for {gp_name}")
            continue

        race_cols = ["DriverNumber", "Abbreviation", "TeamName", "Position", "Status"]
        if "TeamId" in race_results.columns:
            race_cols.append("TeamId")

        race_df = race_results[race_cols].copy()
        race_df.rename(
            columns={
                "DriverNumber": "Driver_ID",
                "Abbreviation": "Driver",
                "TeamName": "Constructor",
                "TeamId": "Constructor_ID",
                "Position": "Finish_Position",
            },
            inplace=True,
        )

        # Fastest lap time per driver
        try:
            laps = race_session.laps
            fastest_laps = (
                laps.groupby("Driver")["LapTime"]
                .min()
                .reset_index()
                .rename(columns={"LapTime": "Fastest_Lap_Time"})
            )
        except Exception:
            fastest_laps = pd.DataFrame(columns=["Driver", "Fastest_Lap_Time"])

        # ✅ PIT STOP SECTION
        try:
            pitstops = race_session.laps[
                race_session.laps["PitInTime"].notna() & race_session.laps["PitOutTime"].notna()
            ]
            if not pitstops.empty:
                pit_times = (
                    pitstops.groupby("Driver")
                    .apply(lambda df: (df["PitOutTime"] - df["PitInTime"]).dt.total_seconds().min())
                    .reset_index()
                    .rename(columns={0: "Pit_Stop_Duration"})
                )
            else:
                pit_times = pd.DataFrame(columns=["Driver", "Pit_Stop_Duration"])
        except Exception:
            pit_times = pd.DataFrame(columns=["Driver", "Pit_Stop_Duration"])

        # --------------------- Merge Data ---------------------
        merged = (
            race_df.merge(
                qual_df[["Driver", "Grid_Position", "Qualifying_Time", "Constructor_ID"]],
                on="Driver",
                how="left",
            )
            .merge(fastest_laps, on="Driver", how="left")
            .merge(pit_times, on="Driver", how="left")
        )

        # Fallback: if Constructor_ID missing, assign numeric mapping
        if "Constructor_ID" not in merged.columns or merged["Constructor_ID"].isna().all():
            merged["Constructor_ID"] = merged["Constructor"].factorize()[0] + 1

        merged["Season"] = year
        merged["Circuit_Name"] = gp_name
        merged["Race_ID"] = f"{year}_{event['RoundNumber']}"

        results.append(merged)

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


def ensure_constructor_id(csv_path: str):
    """Ensures the dataset has a proper 'Constructor_ID' column."""
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if "Constructor" not in df.columns:
        raise ValueError("❌ 'Constructor' column not found in the dataset.")

    if "Constructor_ID" not in df.columns:
        df["Constructor_ID"] = df["Constructor"].astype("category").cat.codes + 1
        df.to_csv(csv_path, index=False)
        print(f"✅ Added 'Constructor_ID' column successfully and updated {csv_path}")
    else:
        print("ℹ️ 'Constructor_ID' already exists — no changes made.")


def main():
    all_data = []
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n========== Fetching {year} Season ==========")
        season_df = collect_season_data(year)
        if not season_df.empty:
            all_data.append(season_df)

    if not all_data:
        print("❌ No data fetched.")
        return

    final_df = pd.concat(all_data, ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Data collection complete! Saved to {OUTPUT_PATH}")

    # Ensure Constructor_ID column exists
    ensure_constructor_id(OUTPUT_PATH)


if __name__ == "__main__":
    main()
