import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report

# Load data with Windows-style paths
X_train = pd.read_csv(r"data\X_train.csv")
y_train = pd.read_csv(r"data\y_train.csv").squeeze()

X_val = pd.read_csv(r"data\X_val.csv")
y_val = pd.read_csv(r"data\y_val.csv").squeeze()

# Combine train + val for hyperparameter tuning with time series CV
X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

# Calculate scale_pos_weight for imbalance handling
neg = (y_full == 0).sum()
pos = (y_full == 1).sum()
scale_pos_weight = neg / pos
print(f"Scale_pos_weight calculated as: {scale_pos_weight:.2f}")

# Define XGBoost classifier with imbalance parameter
xgb_clf = xgb.XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

# Hyperparameter grid
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2],
}

# Setup TimeSeriesSplit cross-validator
tscv = TimeSeriesSplit(n_splits=5)

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=50,
    scoring='roc_auc',
    cv=tscv,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Run hyperparameter search
random_search.fit(X_full, y_full)

print("\nBest hyperparameters found:")
print(random_search.best_params_)

# Load test set
X_test = pd.read_csv(r"data\X_test.csv")
y_test = pd.read_csv(r"data\y_test.csv").squeeze()

# Evaluate best model on test set
best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nTest ROC AUC: {test_auc:.4f}")

# Optional: classification report at 0.5 threshold
y_pred = (y_pred_proba >= 0.5).astype(int)
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))
