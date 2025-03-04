from monitor import monitor_mape, monitor_drift
import pandas as pd
import json

thresholds_file = "knowledge/thresholds.json"

def analyse_mape():
    """Analyze accuracy and debt-based performance to decide if debt needs repayment."""
    mape_data = monitor_mape()
    if not mape_data:
        return None

    accuracy = mape_data["accuracy"]
    debt = mape_data["debt"]

    # Load thresholds
    try:
        with open(thresholds_file, "r") as f:
            thresholds = json.load(f)
        min_acc = thresholds.get("min_accuracy", 0.8)
        debt_threshold = thresholds.get("debt_threshold", 1.5)
    except FileNotFoundError:
        min_acc, debt_threshold = 0.8, 1.5

    # Determine if action is required
    if debt > debt_threshold:
        return {"action_needed": "repay_debt", "accuracy": accuracy, "debt": debt}
    
    return {"action_needed": None, "accuracy": accuracy, "debt": debt}


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
