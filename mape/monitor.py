import pandas as pd
import numpy as np
import threading
import time
from scipy.stats import entropy, wasserstein_distance

# Monitor MAPE every 5 sec
def monitor_mape():
    try:
        df = pd.read_csv("knowledge/predictions.csv")
        df.columns = df.columns.str.strip()
        if df.empty:
            print("MAPE Monitor: No predictions yet.")
            return None

        df["mape"] = abs(df["true_value"] - df["predicted_value"]) / df["true_value"]
        avg_mape = df["mape"].mean() * 100
        avg_time = df["inference_time"].mean()
        avg_energy = df["energy"].mean()
        print(f"📊 MAPE: {avg_mape:.2f}%, Time: {avg_time:.4f}s, Energy: {avg_energy:.2f} W")

        return {"mape": avg_mape, "avg_time": avg_time, "avg_energy": avg_energy}
    
    except FileNotFoundError:
        print("MAPE Monitor: No predictions found.")
        return None

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

