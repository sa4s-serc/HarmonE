import os
import json
import pandas as pd
from monitor import monitor_mape, monitor_drift

thresholds_file = "knowledge/thresholds.json"
mape_info_file = "knowledge/mape_info.json"

base_version_dir = "versionedMR"
current_model_file = "knowledge/model.csv"
drift_kl_file = "knowledge/drift_kl.json"
drift_data_file = "knowledge/drift.csv"

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


def get_model_versions(model_name):
    """Returns available versions for a given model."""
    model_dir = os.path.join(base_version_dir, model_name)
    if not os.path.exists(model_dir):
        return []
    return sorted([d for d in os.listdir(model_dir) if d.startswith("version_")], key=lambda x: int(x.split("_")[-1]))

def get_best_version(model_name):
    """Finds the version with the lowest KL divergence from past data."""
    versions = get_model_versions(model_name)
    if not versions:
        return None  # No previous versions exist

    if not os.path.exists(drift_data_file):
        print("⚠️ No drift.csv found. Cannot compare versions.")
        return None

    # Load current drift data
    try:
        drift_data = pd.read_csv(drift_data_file)["true_value"].values
        drift_hist, _ = np.histogram(drift_data, bins=50, density=True)
    except Exception as e:
        print(f"❌ Error reading drift.csv: {e}")
        return None

    min_kl_div = float("inf")
    best_version = None

    for version in versions:
        version_data_path = os.path.join(base_version_dir, model_name, version, "data.csv")
        if not os.path.exists(version_data_path):
            continue

        # Load versioned model's training data
        try:
            version_data = pd.read_csv(version_data_path)["train_data"].values
            version_hist, _ = np.histogram(version_data, bins=50, density=True)
        except Exception as e:
            print(f"❌ Error reading {version_data_path}: {e}")
            continue

        # Compute KL divergence
        kl_div = entropy(drift_hist, version_hist)
        print(f"🔎 KL divergence for {version}: {kl_div:.4f}")

        if kl_div < min_kl_div:
            min_kl_div = kl_div
            best_version = version_data_path  # Return the best version data path

    # Store KL divergences for debugging
    with open(drift_kl_file, "w") as f:
        json.dump({"best_version": best_version, "min_kl_div": min_kl_div}, f, indent=4)

    return best_version if min_kl_div < 0.75 else None  # Use version if KL is below threshold

def get_model_versions(model_name):
    """Returns available versions for a given model."""
    model_dir = os.path.join(base_version_dir, model_name)
    if not os.path.exists(model_dir):
        return []
    return sorted([d for d in os.listdir(model_dir) if d.startswith("version_")], key=lambda x: int(x.split("_")[-1]))

def get_best_version(model_name):
    """Finds the version with the lowest KL divergence from past data."""
    versions = get_model_versions(model_name)
    if not versions:
        return None  # No previous versions exist

    if not os.path.exists(drift_data_file):
        print("⚠️ No drift.csv found. Cannot compare versions.")
        return None

    # Load current drift data
    try:
        drift_data = pd.read_csv(drift_data_file)["true_value"].values
        drift_hist, _ = np.histogram(drift_data, bins=50, density=True)
    except Exception as e:
        print(f"❌ Error reading drift.csv: {e}")
        return None

    min_kl_div = float("inf")
    best_version = None

    for version in versions:
        version_data_path = os.path.join(base_version_dir, model_name, version, "data.csv")
        if not os.path.exists(version_data_path):
            continue

        # Load versioned model's training data
        try:
            version_data = pd.read_csv(version_data_path)["train_data"].values
            version_hist, _ = np.histogram(version_data, bins=50, density=True)
        except Exception as e:
            print(f"❌ Error reading {version_data_path}: {e}")
            continue

        # Compute KL divergence
        kl_div = entropy(drift_hist, version_hist)
        print(f"🔎 KL divergence for {version}: {kl_div:.4f}")

        if kl_div < min_kl_div:
            min_kl_div = kl_div
            best_version = version_data_path  # Return the best version data path

    # Store KL divergences for debugging
    with open(drift_kl_file, "w") as f:
        json.dump({"best_version": best_version, "min_kl_div": min_kl_div}, f, indent=4)

    return best_version if min_kl_div < 0.75 else None  # Use version if KL is below threshold

def analyse_drift():
    """Analyze drift & decide if retraining is needed or if an existing version can be used."""
    drift_data = monitor_drift()
    if not drift_data:
        return None

    kl_div = drift_data["kl_div"]
    drift_detected = kl_div > 0.75  #? Threshold for drift detection

    if drift_detected:
        print(f"🚨 Drift detected! KL divergence = {kl_div:.4f}")
        try:
            df = pd.read_csv("knowledge/predictions.csv")
            df.columns = df.columns.str.strip()
            df.tail(1200).to_csv(drift_data_file, index=False)
        except FileNotFoundError:
            print("No predictions file found to store drift data.")

        # Get the currently used model
        with open(current_model_file, "r") as f:
            current_model = f.read().strip()

        best_version = get_best_version(current_model)

        if best_version:
            print(f"✔ Best version found with lower KL divergence: {best_version}")
            return {"drift_detected": True, "best_version": best_version}

        # No suitable previous version found → Retrain needed
        return {"drift_detected": True, "best_version": None}

    return {"drift_detected": False}