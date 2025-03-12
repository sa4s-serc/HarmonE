import threading
import time
import pyRAPL
import csv
import os
import pandas as pd
from execute import execute_mape, execute_drift

pyRAPL.setup()
log_file = "knowledge/mape_log.csv"
predictions_file = "knowledge/predictions.csv"
drift_file = "knowledge/drift.csv"

# Ensure log file exists with header
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["function", "energy_joules"])

def log_energy(function_name, energy_joules):
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([function_name, energy_joules])

def run_execute_mape():
    while True:
        time.sleep(40)
        meter = pyRAPL.Measurement("execute_mape")
        meter.begin()
        execute_mape()
        meter.end()
        log_energy("execute_mape", meter.result.pkg[0]) 

def run_execute_drift():
    time.sleep(400)
    while True:
        time.sleep(3)
        meter = pyRAPL.Measurement("execute_drift")
        meter.begin()
        # execute_drift()
        meter.end()
        log_energy("execute_drift", meter.result.pkg[0])

def run_periodic_retrain():
    while True:
        time.sleep(500)
        # Ensure `drift.csv` has data by storing last 1200 rows from `predictions.csv`
        try:
            df = pd.read_csv(predictions_file)
            df.columns = df.columns.str.strip()
            if not df.empty:
                df.tail(1500).to_csv(drift_file, index=False)
                print("✔ Updated drift.csv with the last 1500 rows from predictions.csv")
            else:
                print("⚠️ Predictions file is empty. No data available for retraining.")
        except FileNotFoundError:
            print("❌ Error: predictions.csv not found. Cannot update drift.csv.")

        # Run retraining
        meter = pyRAPL.Measurement("periodic_retrain")
        meter.begin()
        os.system("python retrain.py")
        meter.end()
        log_energy("periodic_retrain", meter.result.pkg[0])

# Start all monitoring threads
t1 = threading.Thread(target=run_execute_mape, daemon=True)
t2 = threading.Thread(target=run_execute_drift, daemon=True)
# t3 = threading.Thread(target=run_periodic_retrain, daemon=True)

t1.start()
t2.start()
# t3.start()

# Prevent script from exiting
exit_event = threading.Event()
exit_event.wait()
