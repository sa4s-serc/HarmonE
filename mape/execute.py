import os
import json
from plan import plan_mape, plan_drift

debt_file = "knowledge/debt.json"
model_file = "knowledge/model.csv"

def execute_mape():
    """Switch to the best model based on MAPE analysis."""
    decision = plan_mape()
    if not decision:
        #print("MAPE: No action needed.")
        return

    #print(f"⚡ Switching model to {decision.upper()}")
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
            #print(f"💰 Debt reduced from {previous_debt:.2f} to {new_debt:.2f}")

        except FileNotFoundError:
            print("⚠️ No debt file found. Skipping debt reduction.")

def execute_drift():
    """Triggers model retraining if drift is detected."""
    decision = plan_drift()
    if decision == "retrain":
        #print("🔧 Retraining model due to drift...")
        os.system("python retrain.py")
