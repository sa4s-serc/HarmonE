import os
from plan import plan

def execute():
    decision = plan()
    if not decision:
        print("No action needed.")
        return

    if decision == "retrain":
        print("🔧 Retraining model...")
        os.system("python retrain.py")
    else:
        print(f"⚡ Switching model to {decision.upper()}")
        with open("knowledge/model.csv", "w") as f:
            f.write(decision)
