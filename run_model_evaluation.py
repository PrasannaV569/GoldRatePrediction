"""
Script to run model evaluation and generate results report
"""
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from datetime import datetime

print("\n" + "="*60)
print("GOLD RATE PREDICTION MODEL - EVALUATION REPORT")
print("="*60)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load dataset
print("Loading and preprocessing data...")
df = pd.read_csv("GoldRate - History_Data.csv")

# --- CLEANING STEP ---
df["24K Rate"] = (
    df["24K Rate"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

df = df.dropna(subset=["24K Rate"])

# Create daily % changes
df["change"] = df["24K Rate"].pct_change() * 100
df = df.dropna()

# Feature engineering
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["dayofweek"] = df["Date"].dt.dayofweek
df["dayofyear"] = df["Date"].dt.dayofyear
df["month"] = df["Date"].dt.month
df["year"] = df["Date"].dt.year

# Lags
for lag in range(1, 8):
    df[f"lag_change_{lag}"] = df["change"].shift(lag)

# Rolling statistics
df["rolling_mean_change_7"] = df["change"].rolling(7).mean()
df["rolling_std_change_7"] = df["change"].rolling(7).std()

df = df.dropna().reset_index(drop=True)

print(f"✓ Dataset loaded: {len(df)} records")

# Features and target
feature_cols = [
    "dayofweek", "dayofyear", "month", "year",
    "lag_change_1", "lag_change_2", "lag_change_3",
    "lag_change_4", "lag_change_5", "lag_change_6", "lag_change_7",
    "rolling_mean_change_7", "rolling_std_change_7"
]
X = df[feature_cols]
y = df["change"]

# Ensure all features are numeric
for col in X.columns:
    if X[col].dtype == object:
        X[col] = X[col].astype(str).str.replace(",", "", regex=False)
        X[col] = pd.to_numeric(X[col], errors="coerce")
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

print(f"✓ Features engineered: {len(feature_cols)} features")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"✓ Data split: {len(X_train)} training, {len(X_test)} testing\n")

# Train model
print("Training Random Forest Regressor (200 estimators)...")
model = RandomForestRegressor(random_state=42, n_estimators=200)
model.fit(X_train, y_train)
print("✓ Model training complete\n")

# --- Model Evaluation ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics for training set
train_r2 = r2_score(y_train, y_pred_train)
train_rmse = math.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)

# Calculate metrics for test set
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = math.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

# Print evaluation metrics
print("="*60)
print("MODEL EVALUATION METRICS - RESULTS")
print("="*60)

print("\n📊 TRAINING SET PERFORMANCE:")
print(f"   ├─ R² Score (Accuracy):        {train_r2:.4f}")
print(f"   ├─ RMSE (Root Mean Sq Error):  {train_rmse:.4f}%")
print(f"   └─ MAE (Mean Absolute Error):  {train_mae:.4f}%")

print("\n📊 TEST SET PERFORMANCE:")
print(f"   ├─ R² Score (Accuracy):        {test_r2:.4f}")
print(f"   ├─ RMSE (Root Mean Sq Error):  {test_rmse:.4f}%")
print(f"   └─ MAE (Mean Absolute Error):  {test_mae:.4f}%")

print("\n" + "="*60)
print("INTERPRETATION:")
print("="*60)
print(f"• R² Score range: 0 to 1 (higher is better)")
print(f"  → Model explains {test_r2*100:.2f}% of variance in test data")
print(f"\n• RMSE: Average prediction error in percentage points")
print(f"  → Test predictions off by ±{test_rmse:.4f}% on average")
print(f"\n• MAE: Mean absolute deviation")
print(f"  → Test predictions off by ±{test_mae:.4f}% on average")

# Performance evaluation
print("\n" + "="*60)
print("MODEL PERFORMANCE ASSESSMENT:")
print("="*60)

if test_r2 > 0.7:
    performance = "✓ EXCELLENT"
elif test_r2 > 0.5:
    performance = "✓ GOOD"
elif test_r2 > 0.3:
    performance = "◐ FAIR"
else:
    performance = "✗ POOR"

print(f"\nOverall Performance: {performance}")
print(f"Overfitting Check: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")

if abs(train_r2 - test_r2) > 0.15:
    print("⚠️  WARNING: Possible overfitting detected")
else:
    print("✓ No significant overfitting detected")

print("\n" + "="*60 + "\n")

# Save the trained model
joblib.dump(model, "gold_model.pkl")

# Save metrics to a file for reference
metrics = {
    "train_r2": train_r2,
    "train_rmse": train_rmse,
    "train_mae": train_mae,
    "test_r2": test_r2,
    "test_rmse": test_rmse,
    "test_mae": test_mae,
    "timestamp": datetime.now().isoformat(),
}
joblib.dump(metrics, "model_metrics.pkl")

print("✓ Model saved to: gold_model.pkl")
print("✓ Metrics saved to: model_metrics.pkl")
