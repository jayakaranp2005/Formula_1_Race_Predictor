import fastf1
import pandas as pd
import os
from datetime import datetime

def fetch_upcoming_qualifying(season, grand_prix_name, race_round, output_path):
    """
    Fetches qualifying results from FastF1 and saves to CSV with:
    Driver_ID, Grid_Position, Qualifying_Time, Race_ID, Circuit_Name
    """

    # ✅ Ensure cache directory exists (inside data/)
    os.makedirs('data/cache', exist_ok=True)

    # ✅ Enable FastF1 cache in that directory
    fastf1.Cache.enable_cache('data/cache')
    print("✅ FastF1 cache enabled at: data/cache")

    # Load qualifying session
    session = fastf1.get_session(season, race_round, 'Q')
    session.load()

    # Prepare dataframe from results
    results = []
    for row in session.results.itertuples():
        try:
            driver_id = getattr(row, 'DriverNumber')
            grid_pos = getattr(row, 'Position')

            # Get best qualifying time in seconds
            qual_time = row.Q3
            if pd.isna(qual_time):
                qual_time = row.Q2
            if pd.isna(qual_time):
                qual_time = row.Q1
            qual_time_sec = qual_time.total_seconds() if not pd.isna(qual_time) else None

            race_id = f"{season}_{race_round}"
            circuit_name = session.event['EventName']

            results.append({
                'Driver_ID': driver_id,
                'Grid_Position': grid_pos,
                'Qualifying_Time': qual_time_sec,
                'Race_ID': race_id,
                'Circuit_Name': circuit_name
            })
        except Exception as e:
            print(f"⚠️ Skipped driver due to error: {e}")

    # Convert to DataFrame and sort
    df = pd.DataFrame(results)
    df.sort_values(by='Grid_Position', inplace=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Upcoming qualifying data saved to {output_path}")
    print(df.head(10))


if __name__ == "__main__":
    fetch_upcoming_qualifying(
        season=2025,
        grand_prix_name='São Paulo Grand Prix',
        race_round=21,
        output_path='data/upcoming_qualifying.csv'
    )
