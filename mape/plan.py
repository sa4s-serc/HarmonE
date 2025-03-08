import json
import random
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from analyse import analyse_mape, analyse_drift


thresholds_file = "knowledge/thresholds.json"
model_file = "knowledge/model.csv"

# Predefined model energy efficiency (lower = better)
MODEL_ENERGY_EFFICIENCY = {
    "lstm": 0.7,    # Medium energy consumption
    "linear": 0.3,  # Most energy-efficient
    "svm": 0.5      # Moderate energy consumption
}

def plan_mape():
    """Select the best model based on score, prioritizing energy efficiency when needed."""
    analysis = analyse_mape()
    if not analysis or not analysis["switch_needed"]:
        print("✅ No model switch needed (Thresholds not violated).")
        return None

    threshold_violated = analysis["threshold_violated"]

    # Load exploration probability (alpha)
    try:
        with open(thresholds_file, "r") as f:
            thresholds = json.load(f)
        alpha = thresholds.get("alpha", 0.1)
    except FileNotFoundError:
        alpha = 0.1

    # Load historical model scores
    scores_file = "knowledge/model_scores.json"
    try:
        with open(scores_file, "r") as f:
            model_scores = json.load(f)
    except FileNotFoundError:
        model_scores = {"lstm": 0.5, "linear": 0.5, "svm": 0.5}

    # If energy is the issue, pick the most energy-efficient model
    if threshold_violated == "energy":
        chosen_model = min(MODEL_ENERGY_EFFICIENCY, key=MODEL_ENERGY_EFFICIENCY.get)
        print(f"⚡ Energy threshold violated. Switching to the most efficient model: {chosen_model.upper()}")

    # Otherwise, use score-based selection with exploration
    else:
        if random.random() < alpha:
            chosen_model = random.choice(["lstm", "linear", "svm"])
            print(f"🎲 Random Exploration! Choosing {chosen_model.upper()}")
        else:
            chosen_model = max(model_scores, key=model_scores.get)
            print(f"🏆 Choosing best model: {chosen_model.upper()}")

    # Check if the chosen model is already in use
    try:
        with open(model_file, "r") as f:
            current_model = f.read().strip()
    except FileNotFoundError:
        current_model = None  # Default to switching if no model file exists

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
