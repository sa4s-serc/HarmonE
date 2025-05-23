import os
import json
import numpy as np
import pandas as pd
from scipy.stats import entropy
from monitor import monitor_mape, monitor_drift

thresholds_file = "knowledge/thresholds.json"
mape_info_file = "knowledge/mape_info.json"

base_version_dir = "versionedMR"
current_model_file = "knowledge/model.csv"
drift_kl_file = "knowledge/drift_kl.json"
drift_data_file = "knowledge/drift.csv"

def load_mape_info():
    """Load stored MAPE info including energy debt and recovery cycles."""
    with open(mape_info_file, "r") as f:
        return json.load(f)

def save_mape_info(data):
    """Save updated MAPE info including energy debt and recovery cycles."""
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)

def analyse_mape():
    """Analyze performance and decide if switching is needed, using dynamic energy thresholds."""
    mape_data = monitor_mape()
    if not mape_data:
        print("⚠️ No MAPE data available for analysis.")
        return None

    # Load thresholds
    with open(thresholds_file, "r") as f:
        thresholds = json.load(f)

    min_score = thresholds["min_score"]
    original_energy_threshold = thresholds["max_energy"]

    # Load current MAPE info
    mape_info = load_mape_info()
    current_energy_threshold = mape_info.get("current_energy_threshold", original_energy_threshold)
    recovery_cycles = mape_info["recovery_cycles"]

    # Update energy threshold dynamically
    used_energy = mape_data["normalized_energy"]
    new_energy_threshold = current_energy_threshold + 0.95 * (original_energy_threshold - used_energy)
    mape_info["current_energy_threshold"] = new_energy_threshold

    # Check if switching is needed
    switch_needed = False
    threshold_violated = None

    if recovery_cycles > 0:
        recovery_cycles -= 1
        print(f"⏳ Recovery mode active: {recovery_cycles} cycles remaining. No switching allowed.")
    else:
        if mape_data["score"] < min_score:
            print("⚠️ Model score too low! Model switch required.")
            switch_needed = True
            threshold_violated = "score"

        if used_energy > current_energy_threshold:
            print(f"⚠️ Energy threshold exceeded! Used: {used_energy:.4f}, Limit: {current_energy_threshold:.4f}")
            switch_needed = True
            threshold_violated = "energy"
            recovery_cycles = 3

    # Save updated info
    mape_info["recovery_cycles"] = recovery_cycles
    save_mape_info(mape_info)

    print(f"📊 Updated Energy Threshold: {new_energy_threshold:.4f}")

    return {
        "switch_needed": switch_needed,
        "score": mape_data["score"],
        "threshold_violated": threshold_violated
    }


def get_model_versions(model_name):
    """Returns available versions for a given model."""
    model_dir = os.path.join(base_version_dir, model_name)
    if not os.path.exists(model_dir):
        print("debug: Hey! this path is incorrect")
        return []
    return sorted([d for d in os.listdir(model_dir) if d.startswith("version_")], key=lambda x: int(x.split("_")[-1]))

def get_best_version(model_name):
    """Finds the version with the lowest KL divergence from past data."""
    versions = get_model_versions(model_name)
    if len(versions) <= 1:
        return None  # No previous versions exist

    if not os.path.exists(drift_data_file):
        print("⚠️ No drift.csv found. Cannot compare versions.")
        return None

    # Load current drift data
    try:
        drift_data = pd.read_csv(drift_data_file)["true_value"].values
        drift_hist, _ = np.histogram(drift_data, bins=50, density=True)
        drift_hist += 1e-10  
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
            version_hist += 1e-10  # ✅ Prevent zero probabilities
        except Exception as e:
            print(f"❌ Error reading {version_data_path}: {e}")
            continue

        # Compute KL divergence
        kl_div = entropy(drift_hist, version_hist)
        kl_div = np.clip(kl_div, 0, 10) 
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