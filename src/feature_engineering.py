"""
feature_engineering.py
Generates engineered features from raw_data.csv
Output -> data/processed_data.csv
"""

import os
import pandas as pd
import numpy as np

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw_data.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed_data.csv")
# =========================================


def compute_driver_features(df):
    df = df.sort_values(["Driver", "Season", "Race_ID"])
    grouped = df.groupby("Driver", group_keys=False)

    # Average Finish Position (Last 5)
    df["Avg_Finish_Position_L5"] = (
        grouped["Finish_Position"]
        .apply(lambda x: x.rolling(5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # Recent DNF Count (Last 5)
    df["Recent_DNF_Count_L5"] = (
        grouped["Status"]
        .apply(lambda x: x.ne("Finished").rolling(5, min_periods=1).sum())
        .reset_index(level=0, drop=True)
    )

    # Racecraft Score = Grid - Finish (positive = gained places)
    df["Racecraft_Score"] = df["Grid_Position"] - df["Finish_Position"]
    df["Avg_Racecraft_Score_L22"] = (
        grouped["Racecraft_Score"]
        .apply(lambda x: x.rolling(22, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # Track Specialization Index (mean deviation on track)
    df["Track_Specialization_Index_L22"] = (
        grouped.apply(
            lambda g: g.set_index("Circuit_Name")
            .groupby("Circuit_Name")["Finish_Position"]
            .transform(lambda x: (x - x.mean()).rolling(22, min_periods=1).mean())
        )
        .reset_index(level=0, drop=True)
    )

    return df


def compute_team_features(df):
    df = df.sort_values(["Constructor_ID", "Season", "Race_ID"])
    team_group = df.groupby("Constructor_ID", group_keys=False)

    # Compute team average lap time per race
    df["Team_Avg_Lap"] = df.groupby(["Race_ID", "Constructor_ID"])["Fastest_Lap_Time"].transform("mean")

    # Recent Car Pace Delta (Last 5 races)
    df["Car_Pace_Delta"] = (df["Fastest_Lap_Time"] - df["Team_Avg_Lap"]).dt.total_seconds()
    df["Recent_Car_Pace_Delta_L5"] = (
        team_group["Car_Pace_Delta"]
        .apply(lambda x: x.rolling(5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # Team Avg Pace Delta (Last 22)
    df["Team_Avg_Pace_Delta_L22"] = (
        team_group["Car_Pace_Delta"]
        .apply(lambda x: x.rolling(22, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # Reliability Rate (Last 22)
    df["Reliability_Binary"] = np.where(df["Status"] == "Finished", 1, 0)
    df["Overall_Reliability_Rate_L22"] = (
        team_group["Reliability_Binary"]
        .apply(lambda x: x.rolling(22, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    return df


def compute_race_context(df):
    # Qualifying gap to pole
    df["Qualifying_Gap_to_Pole"] = (
        df.groupby("Race_ID")["Qualifying_Time"]
        .transform(lambda x: (x - x.min()).dt.total_seconds())
    )
    return df


def main():
    print("📂 Loading raw data...")
    df = pd.read_csv(RAW_PATH)

    # Convert times properly
    for col in ["Fastest_Lap_Time", "Qualifying_Time"]:
        df[col] = pd.to_timedelta(df[col], errors="coerce")

    # Compute each feature group
    print("⚙️ Computing driver-level features...")
    df = compute_driver_features(df)

    print("⚙️ Computing team-level features...")
    df = compute_team_features(df)

    print("⚙️ Computing race context features...")
    df = compute_race_context(df)

    # Cleanup temporary columns
    df.drop(columns=["Racecraft_Score", "Team_Avg_Lap", "Car_Pace_Delta", "Reliability_Binary"], inplace=True, errors="ignore")

    # Save processed data
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Processed features saved to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
