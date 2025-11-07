import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

# ============================
# 1. Paths
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_PATH, exist_ok=True)

# ============================
# 2. Load the train/val/test splits
# ============================
X_train = pd.read_csv(os.path.join(DATA_PATH, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv"))
X_val = pd.read_csv(os.path.join(DATA_PATH, "X_val.csv"))
y_val = pd.read_csv(os.path.join(DATA_PATH, "y_val.csv"))
X_test = pd.read_csv(os.path.join(DATA_PATH, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv"))

print(f"✅ Loaded training/validation/test data successfully.")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ============================
# 3. Combine Train + Validation sets
# ============================
X_final_train = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_final_train = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

print(f"📘 Final training set size: {X_final_train.shape[0]} rows")

# ============================
# 4. Initialize Final XGBoost Model
# ============================
best_params = {
    'subsample': 0.7,
    'reg_lambda': 1,
    'reg_alpha': 0,
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.01,
    'gamma': 0,
    'colsample_bytree': 0.7,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

final_model = XGBClassifier(**best_params)

# ============================
# 5. Train on Combined Data
# ============================
print("🚀 Training final model...")
final_model.fit(X_final_train, y_final_train)

print("✅ Final model training completed.")

# ============================
# 6. Evaluate on Test Set
# ============================
y_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_pred = final_model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n🎯 Test ROC AUC: {roc_auc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ============================
# 7. Save Final Model
# ============================
save_dir = os.path.join(BASE_DIR, "models")
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "final_xgb_model.pkl")
joblib.dump(final_model, model_path)

print(f"✅ Final model saved successfully at: {model_path}")
