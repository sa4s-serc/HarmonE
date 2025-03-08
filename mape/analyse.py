from monitor import monitor_mape, monitor_drift
import pandas as pd
import json

thresholds_file = "knowledge/thresholds.json"

import json
from monitor import monitor_mape

thresholds_file = "knowledge/thresholds.json"
mape_info_file = "knowledge/mape_info.json"

def load_mape_info():
    """Load stored MAPE info including energy debt."""
    try:
        with open(mape_info_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"energy_debt": 0}
        
def save_mape_info(data):
    """Save updated MAPE info including energy debt."""
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)

def analyse_mape():
    """Analyze MAPE-based performance and decide if switching is needed."""
    mape_data = monitor_mape()
    if not mape_data:
        print("⚠️ No MAPE data available for analysis.")
        return None

    # Load thresholds
    try:
        with open(thresholds_file, "r") as f:
            thresholds = json.load(f)
    except FileNotFoundError:
        thresholds = {"min_accuracy": 0.8, "max_energy": 15, "min_score": 0.5, "max_debt": 10}

    min_acc = thresholds["min_accuracy"]
    max_energy = thresholds["max_energy"]
    min_score = thresholds["min_score"]
    max_debt = thresholds["max_debt"]

    # Load current energy debt
    mape_info = load_mape_info()
    energy_debt = mape_info["energy_debt"]

    # Compute energy excess or savings
    energy_excess = mape_data["normalized_energy"] - max_energy
    if energy_excess > 0:
        energy_debt += energy_excess  # Accumulate debt if above max_energy
    else:
        energy_debt = max(0, energy_debt + energy_excess)  # Reduce debt when below threshold

    # Check if model switch is needed
    switch_needed = False
    threshold_violated = None

    if mape_data["r2_score"] < min_acc:
        print("⚠️ R² score below threshold! Model switch required.")
        switch_needed = True
        threshold_violated = "accuracy"

    if mape_data["score"] < min_score:
        print("⚠️ Model score too low! Model switch required.")
        switch_needed = True
        threshold_violated = "score"

    if energy_debt > max_debt:
        print(f"⚠️ Energy Debt ({energy_debt:.2f}) exceeded max allowed ({max_debt}). Switching models!")
        switch_needed = True
        threshold_violated = "energy"

    # Save updated energy debt
    mape_info["energy_debt"] = energy_debt
    save_mape_info(mape_info)

    print(f"📊 Current Energy Debt: {energy_debt:.2f}, Max Allowed: {max_debt}")

    return {
        "switch_needed": switch_needed,
        "score": mape_data["score"],
        "threshold_violated": threshold_violated
    }




def analyse_drift():
    """Analyze drift & decide if retraining is needed."""
    drift_data = monitor_drift()
    if not drift_data:
        return None

    kl_div = drift_data["kl_div"]
    energy_dist = drift_data["energy_distance"]

    drift_detected = kl_div > 0.5 or energy_dist > 0.1

    if drift_detected:
        print("Drift detected! Storing drift data...")
        try:
            df = pd.read_csv("knowledge/predictions.csv")
            df.columns = df.columns.str.strip()
            df.tail(500).to_csv("knowledge/drift.csv", index=False)
        except FileNotFoundError:
            print("No predictions file found to store drift data.")
    
    return {"drift_detected": drift_detected}
