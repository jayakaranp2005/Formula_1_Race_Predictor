import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

# ============================
# Paths
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data.csv")

# ============================
# 1. Load processed data
# ============================
data = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {data.shape[0]} rows from processed_data.csv")

# ============================
# 2. Extract Round from Race_ID
# ============================
if "Race_ID" in data.columns and "Round" not in data.columns:
    data["Round"] = data["Race_ID"].apply(lambda x: int(str(x).split("_")[1]))

# ============================
# 3. Create Target Column: Is_Podium
# ============================
data["Is_Podium"] = data["Finish_Position"].apply(lambda x: 1 if x in [1, 2, 3] else 0)

# ============================
# 4. Sort chronologically by Season and Round
# ============================
if "Season" in data.columns and "Round" in data.columns:
    data = data.sort_values(by=["Season", "Round"]).reset_index(drop=True)
else:
    raise KeyError("Columns 'Season' and 'Round' are required for chronological split.")

# ============================
# 5. Define Feature Columns
# ============================
feature_cols = [
    "Avg_Finish_Position_L5",
    "Recent_DNF_Count_L5",
    "Avg_Racecraft_Score_L22",
    "Track_Specialization_Index_L22",
    "Recent_Car_Pace_Delta_L5",
    "Team_Avg_Pace_Delta_L22",
    "Overall_Reliability_Rate_L22",
    "Qualifying_Gap_to_Pole",
    "Grid_Position",
    "Circuit_Name"
]

X = data[feature_cols].copy()
y = data["Is_Podium"]

# ============================
# 6. Split based on chronology (≈1758 rows)
# ============================
train_end = 1200
val_end = 1550

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print(f"📘 Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ============================
# 7. One-Hot Encode Circuit_Name (fit only on train)
# ============================
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(X_train[["Circuit_Name"]])

def encode_circuit(df):
    encoded = pd.DataFrame(
        ohe.transform(df[["Circuit_Name"]]),
        columns=ohe.get_feature_names_out(["Circuit_Name"]),
        index=df.index
    )
    df = pd.concat([df.drop(columns=["Circuit_Name"]), encoded], axis=1)
    return df

X_train = encode_circuit(X_train)
X_val = encode_circuit(X_val)
X_test = encode_circuit(X_test)

# ============================
# 8. Save all splits to /data/
# ============================
save_dir = os.path.join(BASE_DIR, "data")

X_train.to_csv(os.path.join(save_dir, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(save_dir, "y_train.csv"), index=False)
X_val.to_csv(os.path.join(save_dir, "X_val.csv"), index=False)
y_val.to_csv(os.path.join(save_dir, "y_val.csv"), index=False)
X_test.to_csv(os.path.join(save_dir, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(save_dir, "y_test.csv"), index=False)

print("✅ All files saved to /data/: X_train, y_train, X_val, y_val, X_test, y_test")
