from analyse import analyse_mape, analyse_drift

def plan_mape():
    """Decide whether to switch models based on MAPE analysis."""
    issues = analyse_mape()
    if not issues:
        return None

    with open("knowledge/model.csv", "r") as f:
        current_model = f.read().strip()

    if issues["performance_issue"]:
        return "lstm" if current_model != "lstm" else current_model
    if issues["slow_model"]:
        return "linear"
    if issues["high_energy"]:
        return "svm" if current_model != "svm" else "linear"

    return current_model

def plan_drift():
    """Decide if retraining is needed based on drift analysis."""
    drift = analyse_drift()
    if drift and drift["drift_detected"]:
        return "retrain"
    return None
