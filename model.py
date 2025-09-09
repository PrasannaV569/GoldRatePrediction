# model.py
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("GoldRate - History_Data.csv")

# --- CLEANING STEP ---
# Ensure the 24K Rate column is numeric (remove commas and convert)
df["24K Rate"] = (
    df["24K Rate"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Drop rows with missing values (if any)
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

# Lags (last 7 days of changes)
for lag in range(1, 8):
    df[f"lag_change_{lag}"] = df["change"].shift(lag)

# Rolling statistics
df["rolling_mean_change_7"] = df["change"].rolling(7).mean()
df["rolling_std_change_7"] = df["change"].rolling(7).std()

df = df.dropna().reset_index(drop=True)

# Only use engineered features for training (match prediction)
feature_cols = [
    "dayofweek", "dayofyear", "month", "year",
    "lag_change_1", "lag_change_2", "lag_change_3",
    "lag_change_4", "lag_change_5", "lag_change_6", "lag_change_7",
    "rolling_mean_change_7", "rolling_std_change_7"
]
X = df[feature_cols]
y = df["change"]

# Ensure all features are numeric (remove commas and convert if needed)
for col in X.columns:
    if X[col].dtype == object:
        X[col] = X[col].astype(str).str.replace(",", "", regex=False)
        X[col] = pd.to_numeric(X[col], errors="coerce")
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)  # or use dropna() if you prefer

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestRegressor(random_state=42, n_estimators=200)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "gold_model.pkl")

# --- Prediction function ---
def predict_gold_price(history):
    """
    history: list of recent gold prices (floats)
    """
    # Load the model
    model = joblib.load("gold_model.pkl")
    # Ensure history is numeric
    history_series = pd.Series(history).astype(float)

    # % changes from history
    if len(history_series) > 1:
        changes = history_series.pct_change().dropna() * 100
    else:
        changes = pd.Series(dtype=float)

    # Build last row features
    features = {
        "dayofweek": pd.Timestamp.now().dayofweek,
        "dayofyear": pd.Timestamp.now().dayofyear,
        "month": pd.Timestamp.now().month,
        "year": pd.Timestamp.now().year,
    }

    # Add lags (pad with 0 if not enough history)
    for lag in range(1, 8):
        if len(changes) >= lag:
            features[f"lag_change_{lag}"] = changes.iloc[-lag]
        else:
            features[f"lag_change_{lag}"] = 0.0

    # Rolling stats (use 0.0 if not enough data)
    features["rolling_mean_change_7"] = changes.tail(7).mean() if len(changes) > 0 else 0.0
    features["rolling_std_change_7"] = changes.tail(7).std() if len(changes) > 1 else 0.0

    X_new = pd.DataFrame([features])

    # Predict % change
    prediction = model.predict(X_new)[0]

    # Apply change to last known price
    last_price = history_series.iloc[-1]
    predicted_price = last_price * (1 + prediction / 100)

    return predicted_price

