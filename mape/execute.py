import os
import json
import time
import shutil
from plan import plan_mape, plan_drift

debt_file = "knowledge/debt.json"
model_file = "knowledge/model.csv"

def execute_mape():
    """Switch to the best model based on MAPE analysis."""
    decision = plan_mape()
    if not decision:
        print("MAPE: No action needed.")
        return

    print(f"⚡ Switching model to {decision.upper()}")
    with open("knowledge/model.csv", "w") as f:
        f.write(decision)

        # Reduce debt only if the planner selected a more efficient model
        try:
            with open(debt_file, "r") as f:
                debt_data = json.load(f)
            previous_debt = debt_data["debt"]
            new_debt = max(0, previous_debt - 0.2)  # Gradually reduce debt
            debt_data["debt"] = new_debt

            with open(debt_file, "w") as f:
                json.dump(debt_data, f, indent=4)
            print(f"💰 Debt reduced from {previous_debt:.2f} to {new_debt:.2f}")

        except FileNotFoundError:
            print("⚠️ No debt file found. Skipping debt reduction.")

def execute_drift():
    """Replaces model with best version or retrains if necessary."""
    decision = plan_drift()

    if not decision:
        print("Drift: No action needed.")
        return

    if decision["action"] == "replace":
        best_version_path = decision["version"]
        model_name = os.path.basename(best_version_path).split(".")[0]  # Extract model name

        # Determine correct file extension
        model_extension = ".pkl" if model_name in ["linear", "svm"] else ".pth"

        # Replace the model with the best version
        model_target_path = os.path.join("models", f"{model_name}{model_extension}")
        shutil.copy(best_version_path, model_target_path)

        with open(model_file, "w") as f:
            f.write(model_name)

        print(f"✔ Switched to lower KL divergence model: {best_version_path}")

    elif decision["action"] == "retrain":
        print("🚀 Triggering retraining...")
        os.system("python retrain.py")

    time.sleep(400)
