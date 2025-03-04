import pandas as pd
import json
import os

system_metrics_file = "knowledge/system_metrics.csv"
thresholds_file = "knowledge/thresholds.json"


def analyze_metrics():
    """Analyze stored system metrics to determine optimal thresholds."""
    if not os.path.exists(system_metrics_file):
        print("❌ No system metrics file found. Collect more data first.")
        return
    
    df = pd.read_csv(system_metrics_file)
    
    if df.empty:
        print("❌ No data found in system metrics.")
        return
    
    # Compute statistical values
    min_acc = df["accuracy"].quantile(0.05)  # 5th percentile
    max_acc = df["accuracy"].quantile(0.95)  # 95th percentile
    min_energy = df["energy"].quantile(0.05)
    max_energy = df["energy"].quantile(0.95)
    min_debt = df["debt"].quantile(0.05)
    max_debt = df["debt"].quantile(0.95)
    
    # Define adaptation thresholds
    suggested_thresholds = {
        "min_accuracy": round(min_acc, 3),
        "max_accuracy": round(max_acc, 3),
        "min_energy": round(min_energy, 3),
        "max_energy": round(max_energy, 3),
        "debt_threshold": round(max_debt * 0.8, 3)  # Set at 80% of observed max
    }
    
    # Save suggested thresholds
    with open(thresholds_file, "w") as f:
        json.dump(suggested_thresholds, f, indent=4)
    
    print("✅ Suggested thresholds updated:")
    print(json.dumps(suggested_thresholds, indent=4))


if __name__ == "__main__":
    analyze_metrics()
