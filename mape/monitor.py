import pandas as pd
import numpy as np
import threading
import time
from scipy.stats import entropy, wasserstein_distance
import json
import os
from sklearn.metrics import r2_score

mape_info_file = "knowledge/mape_info.json"
debt_file = "knowledge/debt.json"
thresholds_file = "knowledge/thresholds.json"
system_metrics_file = "knowledge/system_metrics.csv"

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

def log_system_metrics(accuracy, energy, debt, model_name):
    """Log accuracy, energy, debt, and model name to system metrics file."""
    log_entry = pd.DataFrame(
        [{"accuracy": accuracy, "energy": energy, "debt": debt, "model_used": model_name, "timestamp": time.time()}]
    )
    if not os.path.exists(system_metrics_file):
        log_entry.to_csv(system_metrics_file, index=False)
    else:
        log_entry.to_csv(system_metrics_file, mode='a', header=False, index=False)

def monitor_mape():
    """Monitor energy consumption and update debt accordingly."""
    info = load_mape_info()
    last_line = info["last_line"]
    try:
        df = pd.read_csv("knowledge/predictions.csv", skiprows=range(1, last_line + 1))
        df.columns = df.columns.str.strip()
        if df.empty:
            return None
    except FileNotFoundError:
        return None

    avg_energy = df["energy"].mean()
    max_energy = df["energy"].max()
    normalized_energy = avg_energy / max_energy if max_energy > 0 else 0  # Normalize energy
    model_used = df["model_used"].mode()[0]  # Most frequently used model

    accuracy = r2_score(df["true_value"], df["predicted_value"])  # Compute R^2 score as accuracy
    
    try:
        with open(thresholds_file, "r") as f:
            thresholds = json.load(f)
        B_max = thresholds.get("B_max", 1.0)  # Upper threshold for system performance
        B_min = thresholds.get("B_min", 0.5)  # Lower threshold for system performance
        beta = thresholds.get("beta", 0.5)  # Weight for accuracy vs energy
        debt_threshold = thresholds.get("debt_threshold", 1.5)  # Debt threshold for switching models
    except FileNotFoundError:
        B_max, B_min, beta, debt_threshold = 1.0, 0.5, 0.5, 1.5

    debt_data = load_debt()
    debt = debt_data["debt"]
    
    # Compute payback debt adjustment
    payback_debt = debt if debt > debt_threshold else 0
    
    # Compute system performance score S_i
    S_i = beta * accuracy + (1 - beta) * (1 - normalized_energy) - payback_debt
    
    # Update debt based on performance score
    if S_i > B_max:
        debt += (S_i - B_max)
    elif S_i < B_min:
        debt = max(0, debt - (B_min - S_i))

    save_debt({"debt": debt})
    log_system_metrics(accuracy, normalized_energy, debt, model_used)
    
    info["last_line"] = last_line + len(df)
    save_mape_info(info)

    return {"accuracy": accuracy, "energy": normalized_energy, "debt": debt, "model_used": model_used}

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
            kl_div = entropy(
                np.histogram(reference_window, bins=50, density=True)[0] + 1e-10,
                np.histogram(current_window, bins=50, density=True)[0] + 1e-10
            )
            energy_dist = wasserstein_distance(reference_window, current_window)
            print(f"🌊 Drift: KL={kl_div:.4f}, Energy Distance={energy_dist:.4f}")
            return {"kl_div": kl_div, "energy_distance": energy_dist}
        else:
            print(f"Not enough data for drift detection. Have {len(df)} samples, need {window_size * 2}")
            return None
    
    except FileNotFoundError:
        print("Drift Monitor: No predictions found.")
        return None
