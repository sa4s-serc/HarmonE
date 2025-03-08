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
    """Select the best model based on its EMA score, prioritizing energy efficiency when needed."""
    analysis = analyse_mape()
    if not analysis or not analysis["switch_needed"]:
        print("✅ No model switch needed (Thresholds not violated).")
        return None

    threshold_violated = analysis["threshold_violated"]

    # Load model-specific EMA scores
    mape_info = load_mape_info()
    ema_scores = mape_info["ema_scores"]

    # If energy is the issue, pick the most energy-efficient model
    if threshold_violated == "energy":
        chosen_model = min(MODEL_ENERGY_EFFICIENCY, key=MODEL_ENERGY_EFFICIENCY.get)
        print(f"⚡ Energy threshold violated. Switching to the most efficient model: {chosen_model.upper()}")

    else:
        print(ema_scores)
        chosen_model = max(ema_scores, key=ema_scores.get)  # Select model with highest EMA score
        print(f"🏆 Choosing best model based on EMA scores: {chosen_model.upper()}")

    # Check if the chosen model is already in use
    try:
        with open(model_file, "r") as f:
            current_model = f.read().strip()
    except FileNotFoundError:
        current_model = None  

    if chosen_model == current_model:
        print(f"🔄 Model {chosen_model.upper()} is already in use. No switch needed.")
        return None

    return chosen_model


def plan_drift():
    """Decide if retraining is needed based on drift analysis."""
    drift = analyse_drift()
    if drift and drift["drift_detected"]:
        print("🔧 Drift detected! Retraining required.")
        return "retrain"
    return None
