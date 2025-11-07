import os
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, classification_report

# ==============================
# 📂 Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_xgb_model.pkl")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed_data.csv")

# ==============================
# 1️⃣ Load Model and Data
# ==============================
print("✅ Loading model and test data...")

model = joblib.load(MODEL_PATH)

X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))

processed = pd.read_csv(PROCESSED_PATH)

# Determine test split start index
total_rows = processed.shape[0]
train_end, val_end = 1200, 1550
test_start, test_end = val_end, total_rows

# Columns we need for metadata (for ranking)
meta_cols = ["Driver", "Constructor", "Race_ID", "Finish_Position"]
meta_test = processed.iloc[test_start:test_end][meta_cols].reset_index(drop=True)

print(f"Test data shape: {X_test.shape}")

# ==============================
# 2️⃣ Ensure Feature Alignment
# ==============================
expected_features = model.get_booster().feature_names

# Drop extra columns not used during training
X_test = X_test[[c for c in X_test.columns if c in expected_features]]

# Add missing columns as zeros
for col in expected_features:
    if col not in X_test.columns:
        X_test[col] = 0

# Reorder columns to match training
X_test = X_test[expected_features]

print(f"✅ Aligned features: {X_test.shape[1]} columns now match model training set")

# ==============================
# 3️⃣ Predict and Evaluate
# ==============================
print("\n🚀 Making predictions...")
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"\n🎯 Test ROC AUC: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# 4️⃣ Podium Ranking Evaluation
# ==============================
print("\n🏁 Evaluating Podium Ranking Accuracy...")

meta_test["Predicted_Podium_Prob"] = y_pred_prob
meta_test["Actual_Is_Podium"] = (meta_test["Finish_Position"] <= 3).astype(int)

# Group by Race_ID and rank by predicted probability
podium_correct = 0
total_podiums = 0

for race_id, group in meta_test.groupby("Race_ID"):
    ranked = group.sort_values("Predicted_Podium_Prob", ascending=False).reset_index(drop=True)
    top3_pred = ranked.head(3)
    actual_podium = group[group["Actual_Is_Podium"] == 1]
    
    correct = len(set(top3_pred["Driver"]) & set(actual_podium["Driver"]))
    podium_correct += correct
    total_podiums += len(actual_podium)

podium_accuracy = podium_correct / total_podiums if total_podiums > 0 else 0

print(f"🏆 Podium Ranking Accuracy (Top-3 correct drivers): {podium_accuracy:.2%}")

# ==============================
# 5️⃣ Optional: Save Output
# ==============================
output_path = os.path.join(DATA_DIR, "test_predictions.csv")
meta_test.to_csv(output_path, index=False)
print(f"\n📄 Saved detailed predictions to: {output_path}")
