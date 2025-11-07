import pandas as pd
import numpy as np

def generate_prediction_input(processed_data_path, qualifying_data_path, x_train_path, upcoming_race_id, upcoming_circuit_name, output_path):
    # Load processed historical data
    processed_df = pd.read_csv(processed_data_path)

    # Load qualifying data
    qual_df = pd.read_csv(qualifying_data_path)
    qual_df.columns = qual_df.columns.str.strip()

    # Load training data to get one-hot encoded circuit columns
    x_train = pd.read_csv(x_train_path)
    circuit_cols = [col for col in x_train.columns if col.startswith('Circuit_Name_')]
    print(f"Circuits from training data: {[col.replace('Circuit_Name_', '') for col in circuit_cols]}")

    # Get unique drivers from qualifying
    drivers = qual_df['Driver_ID'].unique()

    feature_rows = []

    for driver_id in drivers:
        # Filter historical data for the driver for races before upcoming race
        driver_hist = processed_df[(processed_df['Driver_ID'] == driver_id) & (processed_df['Race_ID'] < upcoming_race_id)]

        def safe_mean(series):
            return series.mean() if len(series) > 0 else np.nan
        def safe_sum(series):
            return series.sum() if len(series) > 0 else 0

        avg_finish_l5 = safe_mean(driver_hist.tail(5)['Finish_Position'])
        recent_dnf_l5 = safe_sum(driver_hist.tail(5)['Status'].apply(lambda x: 1 if x == 'DNF' else 0))
        avg_racecraft_l22 = safe_mean(driver_hist.tail(22)['Avg_Racecraft_Score_L22'])
        track_spec_l22 = safe_mean(driver_hist.tail(22)['Track_Specialization_Index_L22'])
        recent_car_pace_delta_l5 = safe_mean(driver_hist.tail(5)['Recent_Car_Pace_Delta_L5'])
        team_avg_pace_delta_l22 = safe_mean(driver_hist.tail(22)['Team_Avg_Pace_Delta_L22'])
        overall_reliability_l22 = safe_mean(driver_hist.tail(22)['Overall_Reliability_Rate_L22'])

        pole_qual_time = qual_df['Qualifying_Time'].min() if 'Qualifying_Time' in qual_df.columns else np.nan
        driver_qual_time = qual_df[qual_df['Driver_ID'] == driver_id]['Qualifying_Time'].values
        qualifying_gap_to_pole = (driver_qual_time[0] - pole_qual_time) if len(driver_qual_time) > 0 and not np.isnan(pole_qual_time) else np.nan

        grid_position = qual_df[qual_df['Driver_ID'] == driver_id]['Grid_Position'].values[0]

        # Create circuit one-hot columns for this prediction input
        circuit_one_hot = {col: 0 for col in circuit_cols}
        circuit_col_name = f'Circuit_Name_{upcoming_circuit_name}'
        if circuit_col_name in circuit_one_hot:
            circuit_one_hot[circuit_col_name] = 1
        else:
            print(f"Warning: Circuit '{upcoming_circuit_name}' not found in training data circuits list. All set to 0.")

        feature_dict = {
            'Driver_ID': driver_id,  # <-- Added Driver_ID here
            'Avg_Finish_Position_L5': avg_finish_l5,
            'Recent_DNF_Count_L5': recent_dnf_l5,
            'Avg_Racecraft_Score_L22': avg_racecraft_l22,
            'Track_Specialization_Index_L22': track_spec_l22,
            'Recent_Car_Pace_Delta_L5': recent_car_pace_delta_l5,
            'Team_Avg_Pace_Delta_L22': team_avg_pace_delta_l22,
            'Overall_Reliability_Rate_L22': overall_reliability_l22,
            'Qualifying_Gap_to_Pole': qualifying_gap_to_pole,
            'Grid_Position': grid_position,
        }
        feature_dict.update(circuit_one_hot)

        feature_rows.append(feature_dict)

    features_df = pd.DataFrame(feature_rows)

    # Fill missing values with median of each column
    features_df.fillna(features_df.median(numeric_only=True), inplace=True)

    features_df.to_csv(output_path, index=False)
    print(f"✅ New prediction input CSV generated at: {output_path}")


if __name__ == "__main__":
    generate_prediction_input(
        processed_data_path='data/processed_data.csv',
        qualifying_data_path='data/upcoming_qualifying.csv',
        x_train_path='data/x_train.csv',
        upcoming_race_id='2025_20',
        upcoming_circuit_name='Mexico City Grand Prix',
        output_path='data/new_data.csv'
    )
