import pandas as pd
import numpy as np
import threading
import time
from scipy.stats import entropy, wasserstein_distance

# Monitor MAPE every 5 sec
def monitor_mape():
    while True:
        try:
            df = pd.read_csv("knowledge/predictions.csv")
            df.columns = df.columns.str.strip()

            if df.empty:
                print("MAPE Monitor: No predictions yet.")
            else:
                df["mape"] = abs(df["true_value"] - df["predicted_value"]) / df["true_value"]
                avg_mape = df["mape"].mean() * 100
                avg_time = df["inference_time"].mean()
                avg_energy = df["energy"].mean()
                print(f"📊 MAPE: {avg_mape:.2f}%, Time: {avg_time:.4f}s, Energy: {avg_energy:.2f} W")

                # Store monitoring data
                return {"mape": avg_mape, "avg_time": avg_time, "avg_energy": avg_energy}
        except FileNotFoundError:
            print("MAPE Monitor: No predictions found.")

        time.sleep(5)

# Monitor Drift every 20 sec
def monitor_drift():
    while True:
        try:
            df = pd.read_csv("knowledge/predictions.csv")
            df.columns = df.columns.str.strip()

            if df.empty:
                print("Drift Monitor: No predictions yet.")
            else:
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

                    # Define drift thresholds
                    drift_detected = kl_div > 0.5 or energy_dist > 0.1

                    if drift_detected:
                        print("🚨 Drift detected! Storing drift data...")
                        df.tail(500).to_csv("knowledge/drift.csv", index=False)

                    return {"kl_div": kl_div, "energy_distance": energy_dist}
                else:
                    print(f"Not enough data for drift detection. Have {len(df)} samples, need {window_size * 2}")
        except FileNotFoundError:
            print("Drift Monitor: No predictions found.")

        time.sleep(20)

# Run both monitors in parallel
t1 = threading.Thread(target=monitor_mape)
t2 = threading.Thread(target=monitor_drift)

t1.start()
t2.start()
