import json
import random
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from analyse import analyse_mape, analyse_drift

thresholds_file = "knowledge/thresholds.json"
model_file = "knowledge/model.csv"
debt_file = "knowledge/debt.json"

def load_debt():
    """Load current system debt from file."""
    try:
        with open(debt_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"debt": 0}

def save_debt(debt_data):
    """Save updated system debt."""
    with open(debt_file, "w") as f:
        json.dump(debt_data, f, indent=4)

def compute_model_scores():
    """
    Compute performance scores for each model using historical data from predictions.csv.
    For each model m:
       - Compute accuracy using r2_score for all predictions made by model m.
       - Compute average energy consumption.
       - Normalize energy across models.
       - Score: S_m = beta * accuracy + (1 - beta) * (1 - normalized_energy)
    If predictions.csv is not available or insufficient, fallback to static scores.
    """
    try:
        df = pd.read_csv("knowledge/predictions.csv")
        df.columns = df.columns.str.strip()
        if df.empty:
            raise Exception("Empty predictions.csv")
        with open(thresholds_file, "r") as f:
            thresholds = json.load(f)
        beta = thresholds.get("beta", 0.5)
        scores = {}
        energy_means = {}
        for model, group in df.groupby("model_used"):
            if len(group) >= 2:
                accuracy = r2_score(group["true_value"], group["predicted_value"])
            else:
                # Fallback: if only one prediction, use a simple equality check
                accuracy = 1.0 if group["true_value"].iloc[0] == group["predicted_value"].iloc[0] else 0.0
            avg_energy = group["energy"].mean()
            scores[model] = {"accuracy": accuracy, "avg_energy": avg_energy}
            energy_means[model] = avg_energy
        max_energy = max(energy_means.values()) if energy_means else 1
        model_scores = {}
        for model, metrics in scores.items():
            normalized_energy = metrics["avg_energy"] / max_energy if max_energy > 0 else 0
            score = beta * metrics["accuracy"] + (1 - beta) * (1 - normalized_energy)
            model_scores[model] = score
        return model_scores
    except Exception as e:
        print("Could not compute model scores from predictions.csv, falling back to static scores.", e)
        try:
            with open(thresholds_file, "r") as f:
                thresholds = json.load(f)
            # Fallback static scores
            return thresholds.get("historical_model_scores", {"lstm": 0.5, "linear": 0.5, "svm": 0.5})
        except:
            return {"lstm": 0.5, "linear": 0.5, "svm": 0.5}

def plan_mape():
    """Plan strategy for model selection and debt management."""
    analysis = analyse_mape()
    if not analysis:
        print("✅ No action needed (No performance data).")
        return None

    accuracy = analysis["accuracy"]
    debt = analysis["debt"]

    # Load thresholds
    try:
        with open(thresholds_file, "r") as f:
            thresholds = json.load(f)
        min_acc = thresholds.get("min_accuracy", 0.8)
        debt_threshold = thresholds.get("debt_threshold", 1.5)
        exploration_prob = thresholds.get("exploration_prob", 0.1)
        debt_reduction_rate = thresholds.get("debt_reduction_on_switch", 0.2)
    except FileNotFoundError:
        min_acc, debt_threshold, exploration_prob, debt_reduction_rate = 0.8, 1.5, 0.1, 0.2

    # Compute updated model scores from historical data
    model_scores = compute_model_scores()
    print(f"Computed model scores: {model_scores}")

    # Debt Repayment Strategy: if debt is high, force a switch to a model that minimizes energy.
    if debt >= debt_threshold:
        print(f"⚠️ High debt detected ({debt:.2f}). Applying debt repayment strategy.")
        try:
            df = pd.read_csv("knowledge/predictions.csv")
            df.columns = df.columns.str.strip()
            avg_energy_by_model = df.groupby("model_used")["energy"].mean().to_dict()
            if avg_energy_by_model:
                chosen_model = min(avg_energy_by_model, key=avg_energy_by_model.get)
            else:
                chosen_model = min(model_scores, key=model_scores.get)
        except Exception as e:
            print("Error computing energy consumption, falling back to model scores.", e)
            chosen_model = min(model_scores, key=model_scores.get)
        print(f"🔄 Switching to a debt-efficient model: {chosen_model.upper()}")
        new_debt = max(0, debt - debt_reduction_rate)
        save_debt({"debt": new_debt})
        print(f"💰 Debt reduced from {debt:.2f} to {new_debt:.2f}")
        return chosen_model

    # Exploration vs Exploitation: With a probability, explore randomly; otherwise choose the best.
    if random.random() < exploration_prob:
        chosen_model = random.choice(list(model_scores.keys()))
        print(f"🎲 Random Exploration! Choosing {chosen_model.upper()}")
    else:
        chosen_model = max(model_scores, key=model_scores.get)
        print(f"🏆 Exploitation: Choosing model with highest score: {chosen_model.upper()}")

    # Check if the chosen model is already in use.
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
