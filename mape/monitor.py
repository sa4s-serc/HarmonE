import pandas as pd
import numpy as np
import threading
import time
from scipy.stats import entropy, wasserstein_distance
import json
import os

import pandas as pd
import numpy as np
import json
import os

mape_info_file = "knowledge/mape_info.json"
thresholds_file = "knowledge/thresholds.json"
debt_file = "knowledge/debt.json"

def load_mape_info():
    """Load last processed line from knowledge file."""
    if not os.path.exists(mape_info_file):
        return {"last_line": 0}
    with open(mape_info_file, "r") as f:
        return json.load(f)

def save_mape_info(data):
    """Save last processed line."""
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)

def load_debt():
    """Load current system debt from file."""
    if not os.path.exists(debt_file):
        return {"debt": 0}
    with open(debt_file, "r") as f:
        return json.load(f)

def save_debt(debt_data):
    """Save updated system debt."""
    with open(debt_file, "w") as f:
        json.dump(debt_data, f, indent=4)

def monitor_mape():
    """Monitor accuracy and accumulate debt if thresholds are exceeded."""
    info = load_mape_info()
    last_line = info["last_line"]
    
    try:
        df = pd.read_csv("knowledge/predictions.csv", skiprows=range(1, last_line + 1))
        df.columns = df.columns.str.strip()
        if df.empty:
            return None
    except FileNotFoundError:
        return None

    # Compute Accuracy Ai = 1 - MAPE
    df["mape"] = abs(df["true_value"] - df["predicted_value"]) / (df["true_value"] + 1e-5)
    df["accuracy"] = 1 - df["mape"]
    avg_accuracy = df["accuracy"].mean()

    # Load adaptation thresholds
    try:
        with open(thresholds_file, "r") as f:
            thresholds = json.load(f)
        min_acc = thresholds.get("min_accuracy", 0.8)
    except FileNotFoundError:
        min_acc = 0.8  # Default

    # Load system debt
    debt_data = load_debt()
    debt = debt_data["debt"]

    # If accuracy is above threshold, reduce debt
    if avg_accuracy >= min_acc:
        debt = max(0, debt - 0.05)  # Payback debt
    else:
        debt += 0.1  # Increase debt for using high-resource models

    save_debt({"debt": debt})

    # Store updated info
    info["last_line"] = last_line + len(df)
    save_mape_info(info)

    return {"accuracy": avg_accuracy, "debt": debt}

# Monitor Drift every 20 sec
def monitor_drift():
    try:
        df = pd.read_csv("knowledge/predictions.csv")
        df.columns = df.columns.str.strip()
        if df.empty:
            print("Drift Monitor: No predictions yet.")
            return None

        window_size = 500
        if len(df) >= window_size * 2:
            reference_window = df['true_value'].iloc[-2*window_size:-window_size]
            current_window = df['true_value'].iloc[-window_size:]
            # Compute KL Divergence between windows
            kl_div = entropy(np.histogram(reference_window, bins=50, density=True)[0] + 1e-10,
            np.histogram(current_window, bins=50, density=True)[0] + 1e-10)
            # Compute Energy Distance between windows
            energy_dist = wasserstein_distance(reference_window, current_window)
            print(f"🌊 Drift: KL={kl_div:.4f}, Energy Distance={energy_dist:.4f}")
            return {"kl_div": kl_div, "energy_distance": energy_dist}
        else:
            print(f"Not enough data for drift detection. Have {len(df)} samples, need {window_size * 2}")
            return None
        
    except FileNotFoundError:
        print("Drift Monitor: No predictions found.")
        return None

