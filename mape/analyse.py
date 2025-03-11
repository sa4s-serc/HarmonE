from monitor import monitor_mape, monitor_drift
import pandas as pd
import json

import json
from monitor import monitor_mape

thresholds_file = "knowledge/thresholds.json"
mape_info_file = "knowledge/mape_info.json"

def load_mape_info():
    """Load stored MAPE info including energy debt and recovery cycles."""
    try:
        with open(mape_info_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "energy_debt": 0,
            "recovery_cycles": 0
        }

def save_mape_info(data):
    """Save updated MAPE info including energy debt and recovery cycles."""
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)

def analyse_mape():
    """Analyze performance and decide if switching is needed, with smarter debt management."""
    mape_data = monitor_mape()  
    if not mape_data:
        #print("⚠️ No MAPE data available for analysis.")
        return None

    # Load thresholds
    with open(thresholds_file, "r") as f:
        thresholds = json.load(f)

    min_score = thresholds["min_score"]
    max_energy = thresholds["max_energy"]
    max_debt = thresholds["max_debt"]
    recovery_time = thresholds["recovery_time"]
    debt_decay = thresholds["debt_decay"]

    # Load current energy debt and recovery cycle count
    mape_info = load_mape_info()
    energy_debt = mape_info["energy_debt"]
    recovery_cycles = mape_info["recovery_cycles"]

    # Compute energy excess or savings
    energy_excess = mape_data["normalized_energy"] - max_energy
    if energy_excess > 0:
        energy_debt += energy_excess  # Accumulate debt if above max_energy

    # Apply debt decay independent of performance
    energy_debt *= debt_decay  # Gradual decay every cycle

    # Ensure energy debt doesn’t go negative
    energy_debt = max(0, energy_debt)

    # Check if model switch is needed
    switch_needed = False
    threshold_violated = None

    # Block switching if in recovery mode
    if recovery_cycles > 0:
        recovery_cycles -= 1
        #print(f"⏳ Recovery mode active: {recovery_cycles} cycles remaining. No switching allowed.")
    else:
        if mape_data["score"] < min_score:
            #print("⚠️ Model score too low! Model switch required.")
            switch_needed = True
            threshold_violated = "score"

        if energy_debt > max_debt:
            #print(f"⚠️ Energy Debt ({energy_debt:.2f}) exceeded max allowed ({max_debt}). Switching models!")
            switch_needed = True
            threshold_violated = "energy"
            recovery_cycles = recovery_time  # Start recovery period

    # Save updated energy debt & recovery cycles
    mape_info["energy_debt"] = energy_debt
    mape_info["recovery_cycles"] = recovery_cycles
    save_mape_info(mape_info)

    #print(f"📊 Current Energy Debt: {energy_debt:.2f}, Max Allowed: {max_debt}, Recovery: {recovery_cycles}")

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
    #? energy distance no longer considered
    # energy_dist = drift_data["energy_distance"]
    drift_detected = kl_div > 0.75 #? or energy_dist > 0.1

    if drift_detected:
        #print("Drift detected! Storing drift data...")
        try:
            df = pd.read_csv("knowledge/predictions.csv")
            df.columns = df.columns.str.strip()
            df.tail(1200).to_csv("knowledge/drift.csv", index=False)
        except FileNotFoundError:
            print("No predictions file found to store drift data.")
    
    return {"drift_detected": drift_detected}
