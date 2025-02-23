from analyse import analyse

def plan():
    issues = analyse()
    if not issues:
        return None

    with open("knowledge/model.csv", "r") as f:
        current_model = f.read().strip()

    if issues["drift_detected"]:
        print("🔄 Drift detected! Model needs retraining.")
        return "retrain"

    if issues["performance_issue"]:
        return "lstm" if current_model != "lstm" else current_model

    if issues["slow_model"]:
        return "linear"

    if issues["high_energy"]:
        return "svm" if current_model != "svm" else "linear"

    return current_model
