from flask import Flask, render_template
from datetime import datetime
from model import predict_gold_price
import pandas as pd
import joblib
import requests
import os

app = Flask(__name__)
"""Flask app serving live and predicted gold prices.

Integrations:
- Live symbols: https://api.gold-api.com/symbols (no key required)
- Live price (XAU/INR): goldapi.io using API key from env GOLD_API_KEY
Fallback:
- Latest price from CSV if API fails.
"""

# Load the trained model once
model = joblib.load("gold_model.pkl")

def fetch_supported_symbols():
    """Fetch supported symbols from api.gold-api.com for validation."""
    try:
        symbols_resp = requests.get("https://api.gold-api.com/symbols", timeout=10)
        if symbols_resp.status_code == 200:
            return symbols_resp.json()
        return []
    except Exception:
        return []


def get_live_gold_price_in_inr():
    """Fetch live XAU price in INR using goldapi.io. Returns dict with gram and oz, or None."""
    api_key = os.getenv("GOLD_API_KEY") or "goldapi-4x8ixsmfc7b8tt-io"
    url = "https://www.goldapi.io/api/XAU/INR"
    headers = {
        "x-access-token": api_key or "",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            price_per_oz = float(data.get("price"))
            price_per_gram = price_per_oz / 31.1034768
            return {"gram": round(price_per_gram, 2), "oz": round(price_per_oz, 2)}
        return None
    except Exception:
        return None

@app.route("/")
def home():
    try:
        # Validate that XAU is a supported symbol (best-effort)
        _ = fetch_supported_symbols()

        # Try to get live prices first
        live_prices = get_live_gold_price_in_inr()
        
        # Get historical data from CSV
        df = pd.read_csv("GoldRate - History_Data.csv")
        
        if live_prices is not None:
            data_source = "GOLD API"
            latest_price_gram = live_prices['gram']
            latest_price_oz = live_prices['oz']
        else:
            data_source = "CSV"
            # Fallback to CSV if API fails
            latest_24k_rate = df["24K Rate"].iloc[-1]
            latest_price_gram = float(str(latest_24k_rate).replace(",", ""))
            latest_price_oz = latest_price_gram * 31.1034768
        
        # Get last 10 days history for prediction
        history = (
            df["24K Rate"]
            .tail(10)
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
            .tolist()
        )
        
        # Add current price to history for prediction
        history.append(latest_price_gram)
        
        # Predict next day's price
        predicted_price = round(predict_gold_price(history), 2)

        # Build 30-day trend for chart (from CSV for consistent history)
        df_clean = df.copy()
        df_clean["24K Rate"] = (
            df_clean["24K Rate"].astype(str).str.replace(",", "", regex=False).astype(float)
        )
        last30 = df_clean.tail(30)
        chart_labels = last30["Date"].tolist() if "Date" in last30.columns else list(range(len(last30)))
        chart_data = last30["24K Rate"].tolist()

        # Day-over-day change badge
        if len(df_clean) >= 2:
            prev_price = float(df_clean["24K Rate"].iloc[-2])
            price_change = latest_price_gram - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price else 0.0
        else:
            price_change = 0.0
            price_change_pct = 0.0

        return render_template(
            "index.html",
            gram="{:,.2f}".format(latest_price_gram),
            oz="{:,.2f}".format(latest_price_oz),
            predicted="{:,.2f}".format(predicted_price),
            date=datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            data_source=data_source,
            price_change=price_change,
            price_change_pct=price_change_pct,
            chart_labels=chart_labels,
            chart_data=chart_data,
        )
    except Exception as e:
        print(f"Error in home route: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        # Return a more user-friendly error message
        return render_template(
            "index.html",
            gram="Data Unavailable",
            oz="Data Unavailable",
            predicted="Data Unavailable",
            date=datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template(
            "index.html",
            oz="Error",
            gram="Error",
            predicted="Error",
            date=datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        )

if __name__ == "__main__":
    app.run(debug=True)
