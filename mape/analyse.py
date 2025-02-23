from monitor import monitor_mape, monitor_drift
import pandas as pd

def analyse_mape():
    """Analyze MAPE & decide if a model switch is needed."""
    mape_data = monitor_mape()
    if not mape_data:
        return None

    return {
        "performance_issue": mape_data["mape"] > 10,
        "slow_model": mape_data["avg_time"] > 0.01,
        "high_energy": mape_data["avg_energy"] > 15
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
