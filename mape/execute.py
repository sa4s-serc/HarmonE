import os
import json
from plan import plan_mape, plan_drift

def execute_mape():
    """Switch to the best model based on MAPE analysis."""
    decision = plan_mape()
    if not decision:
        print("MAPE: No action needed.")
        return

    print(f"⚡ Switching model to {decision.upper()}")
    with open("knowledge/model.csv", "w") as f:
        f.write(decision)

def execute_drift():
    """Trigger retraining if drift is detected."""
    decision = plan_drift()
    if decision == "retrain":
        print("🔧 Retraining model due to drift...")
        os.system("python retrain.py")
