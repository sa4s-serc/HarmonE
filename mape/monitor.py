import pandas as pd
import numpy as np
import threading
import time
from scipy.stats import entropy, wasserstein_distance
import json
import os
from sklearn.metrics import r2_score

mape_info_file = "knowledge/mape_info.json"
thresholds_file = "knowledge/thresholds.json"
model_file = "knowledge/model.csv"

def load_mape_info():
    # """Load stored MAPE info including model-specific EMA scores."""
    # if not os.path.exists(mape_info_file):
    #     return {
    #         "last_line": 0,
    #         "energy_debt": 0,
    #         "lr_model_version": 1,
    #         "lstm_model_version": 1,
    #         "svm_model_version": 1,
    #         "ema_scores": {"lstm": 0.5, "linear": 0.5, "svm": 0.5}
    #     }
    with open(mape_info_file, "r") as f:
        return json.load(f)

def save_mape_info(data):
    """Save updated MAPE info including model-specific EMA scores."""
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)

def get_current_model():
    """Fetch the currently active model from knowledge."""
    try:
        with open(model_file, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None  # Default behavior if file doesn't exist


def monitor_mape():
    """Monitor R² Score and Normalized Energy, and Compute Score."""
    info = load_mape_info()
    last_line = info["last_line"]
    current_model = get_current_model()
    if current_model is None:
        #print("⚠️ No model currently in use.")
        return None

    try:
        df = pd.read_csv("knowledge/predictions.csv", skiprows=range(1, last_line + 1))
        df.columns = df.columns.str.strip()
        if df.empty:
            #print("📉 No new data to process in predictions.csv")
            return None
    except FileNotFoundError:
        #print("⚠️ No predictions.csv file found.")
        return None

    #print(f"🆕 Processing {len(df)} new rows from predictions.csv for {current_model.upper()}")

    r2 = r2_score(df["true_value"], df["predicted_value"])

    # Compute Normalized Energy
    with open(thresholds_file, "r") as f:
        thresholds = json.load(f)
    energy_min, energy_max = thresholds["E_m"], thresholds["E_M"]
    #print(df["energy"].mean() ,energy_min,energy_max)
    energy_normalized = (df["energy"].mean() - energy_min)/(energy_max - energy_min)

    beta = thresholds.get("beta", 0.5)
    model_score = beta * r2 + (1 - beta) * (1 - energy_normalized)

    # Compute Exponential Moving Average (EMA)
    gamma = thresholds.get("gamma", 0.8)
    prev_score = info["ema_scores"][current_model]
    final_score = gamma * model_score + (1 - gamma) * prev_score

    # Log computed values
    info["ema_scores"][current_model] = final_score
    info["last_line"] += len(df)

    # Log computed values
    #print(f"🔹 R² Score: {r2:.4f}")
    #print(f"🔹 Normalized Energy: {energy_normalized:.4f}")
    #print(f"🔹 Model Score for {current_model.upper()}: {model_score:.4f}")
    #print(f"🔹 Updated EMA Score for {current_model.upper()}: {final_score:.4f}")

    save_mape_info(info)

    return {
        "r2_score": r2,
        "normalized_energy": energy_normalized,
        "score": final_score
    }



def monitor_drift():
    """Monitor data drift without enforcing immediate retraining."""
    try:
        df = pd.read_csv("knowledge/predictions.csv")
        df.columns = df.columns.str.strip()
        if df.empty:
            #print("Drift Monitor: No predictions yet.")
            return None

        window_size = 1200
        if len(df) >= window_size * 2:
            reference_window = df['true_value'].iloc[-2*window_size:-window_size]
            current_window = df['true_value'].iloc[-window_size:]
            kl_div = entropy(
                np.histogram(reference_window, bins=50, density=True)[0] + 1e-10,
                np.histogram(current_window, bins=50, density=True)[0] + 1e-10
            )
            #? energy_dist = wasserstein_distance(reference_window, current_window)
            #print(f"🌊 Drift: KL={kl_div:.4f}, Energy Distance={energy_dist:.4f}")
            return {"kl_div": kl_div}#?, "energy_distance": energy_dist}
        else:
            #print(f"Not enough data for drift detection. Have {len(df)} samples, need {window_size * 2}")
            return None
    
    except FileNotFoundError:
        #print("Drift Monitor: No predictions found.")
        return None
