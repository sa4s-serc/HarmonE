import json
import random
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from analyse import analyse_mape, analyse_drift


import json
import random
from analyse import analyse_mape

thresholds_file = "knowledge/thresholds.json"
model_file = "knowledge/model.csv"
mape_info_file = "knowledge/mape_info.json"

MODEL_ENERGY_EFFICIENCY = {
    "lstm": 0.7,    
    "linear": 0.3,  
    "svm": 0.5      
}

def load_mape_info():
    """Load stored MAPE info including model-specific EMA scores."""
    try:
        with open(mape_info_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "ema_scores": {"lstm": 0.5, "linear": 0.5, "svm": 0.5}
        }

def plan_mape():
    # Load exploration probability (alpha
    with open(thresholds_file, "r") as f:
        thresholds = json.load(f)
    alpha = thresholds.get("alpha", 0.1)
    if random.random() < alpha:
        chosen_model = random.choice(["lstm", "linear", "svm"])
        print(f"🎲 Exploratory switching active! Randomly selecting {chosen_model.upper()}.")
        return chosen_model
    
    analysis = analyse_mape()
    if not analysis or not analysis["switch_needed"]:
        print("✅ No model switch needed (Thresholds not violated).")
        return None

    threshold_violated = analysis["threshold_violated"]

    # Load model-specific EMA scores
    mape_info = load_mape_info()
    ema_scores = mape_info["ema_scores"]

    # Get currently used model
    try:
        with open(model_file, "r") as f:
            current_model = f.read().strip()
    except FileNotFoundError:
        current_model = None   
        
    # If energy threshold was exceeded, choose highest-scoring model not currently in use
    if threshold_violated == "energy":
        best_alternative = sorted(ema_scores.items(), key=lambda x: x[1], reverse=True)
        chosen_model = next((m for m, _ in best_alternative if m != current_model), None)
        if chosen_model:
            print(f"⚡ Energy threshold violated. Switching to best available model: {chosen_model.upper()}")
        else:
            print(f"⚠️ No alternative models available. Staying on {current_model.upper()}.")
            return None

    # Otherwise, choose the highest-scoring model
    else:
        chosen_model = max(ema_scores, key=ema_scores.get)
        print(f"🏆 Choosing best model based on EMA scores: {chosen_model.upper()}")

    # Check if already using the chosen model
    if chosen_model == current_model:
        print(f"🔄 Model {chosen_model.upper()} is already in use. No switch needed.")
        return None

    return chosen_model



def plan_drift():
    """Decide if retraining is needed or if an older version can be used."""
    drift = analyse_drift()
    if not drift or not drift["drift_detected"]:
        print("✅ No drift detected. No action required.")
        return None

    if drift["best_version"]:
        print(f"🔄 Switching to lower KL divergence model: {drift['best_version']}")
        return {"action": "replace", "version": drift["best_version"]}
    
    print("🔧 Drift detected! No previous version available. Retraining required.")
    return {"action": "retrain"}

