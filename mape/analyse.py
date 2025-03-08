from monitor import monitor_mape, monitor_drift
import pandas as pd
import json

thresholds_file = "knowledge/thresholds.json"

def analyse_mape():
    """Analyze MAPE-based performance and decide if switching is needed."""
    mape_data = monitor_mape()
    if not mape_data:
        return None

    # Load thresholds
    try:
        with open(thresholds_file, "r") as f:
            thresholds = json.load(f)
    except FileNotFoundError:
        thresholds = {"min_accuracy": 0.8, "max_energy": 15, "min_score": 0.5}

    min_acc = thresholds["min_accuracy"]
    max_energy = thresholds["max_energy"]
    min_score = thresholds["min_score"]

    # Check thresholds
    switch_needed = (mape_data["accuracy"] < min_acc or 
                     mape_data["energy"] > max_energy or 
                     mape_data["score"] < min_score)

    return {"switch_needed": switch_needed, "score": mape_data["score"]}


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
