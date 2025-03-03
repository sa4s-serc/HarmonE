import json
import random
from analyse import analyse_mape, analyse_drift

thresholds_file = "knowledge/thresholds.json"
model_file = "knowledge/model.csv"
debt_file = "knowledge/debt.json"
model_scores_file = "knowledge/model_scores.json"

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

def load_model_scores():
    """Load historical model scores from thresholds.json."""
    try:
        with open(thresholds_file, "r") as f:
            thresholds = json.load(f)
        
        # Access historical model scores from thresholds
        model_scores = thresholds.get("historical_model_scores", {})
        
        if not model_scores:
            print("❌ No historical model scores found. Using default scores.")
            model_scores = {"lstm": 0.5, "linear": 0.5, "svm": 0.5}
        
        return model_scores

    except FileNotFoundError:
        print("❌ Thresholds file not found. Using default model scores.")
        return {"lstm": 0.5, "linear": 0.5, "svm": 0.5}
    except json.JSONDecodeError:
        print("❌ Error reading thresholds file. Using default model scores.")
        return {"lstm": 0.5, "linear": 0.5, "svm": 0.5}


def plan_mape():
    """Select the best model while managing debt through exploration and switching."""
    analysis = analyse_mape()
    if not analysis:
        print("✅ No model switch needed (No performance data).")
        return None

    accuracy = analysis["accuracy"]
    debt = analysis["debt"]

    # Load thresholds
    try:
        with open(thresholds_file, "r") as f:
            thresholds = json.load(f)
        min_acc = thresholds.get("min_accuracy", 0.8)
        debt_threshold = thresholds.get("debt_threshold", 1.5)
        alpha = thresholds.get("alpha", 0.1)  # Exploration probability
    except FileNotFoundError:
        min_acc, debt_threshold, alpha = 0.8, 1.5, 0.1

    # Load model historical performance (Debt Propensity)
    try:
        model_scores = load_model_scores()
    except FileNotFoundError:
        model_scores = {"lstm": 0.5, "linear": 0.5, "svm": 0.5}

    # Choose model based on exploration or best debt repayment
    if random.random() < alpha:
        chosen_model = random.choice(["lstm", "linear", "svm"])
        print(f"🎲 Random Exploration! Choosing {chosen_model.upper()}")
    else:
        if debt >= debt_threshold:
            print(f"⚠️ High debt detected ({debt:.2f}), switching to a debt-efficient model.")
            chosen_model = min(model_scores, key=model_scores.get)  # Least resource-consuming model
        else:
            chosen_model = max(model_scores, key=model_scores.get)  # Most accurate model

    print(f"🏆 Choosing model: {chosen_model.upper()}")

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
