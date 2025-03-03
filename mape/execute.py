import os
import json
from plan import plan_mape

debt_file = "knowledge/debt.json"

def execute_mape():
    """Switch to the best model based on Debt-Based Logic."""
    decision = plan_mape()
    if not decision:
        print("MAPE: No action needed.")
        return

    print(f"⚡ Switching model to {decision.upper()}")
    with open("knowledge/model.csv", "w") as f:
        f.write(decision)

    # Reduce Debt if switching to a lower-energy model
    try:
        with open(debt_file, "r") as f:
            debt_data = json.load(f)
        debt_data["debt"] = max(0, debt_data["debt"] - 0.2)  # Gradually pay off debt
        with open(debt_file, "w") as f:
            json.dump(debt_data, f, indent=4)
    except FileNotFoundError:
        pass


def execute_drift():
    """Trigger retraining if drift is detected."""
    decision = plan_drift()
    if decision == "retrain":
        print("🔧 Retraining model due to drift...")
        os.system("python retrain.py")
