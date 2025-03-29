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
config_file = "approach.conf"

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
        start_time = time.perf_counter()
        execute_mape()
        end_time = time.perf_counter()
        meter.end()
        inference_time = end_time - start_time
        print(f"Execution time (overhead) of execute_mape: {inference_time:.4f} seconds")
        print(f"Energy consumption (pkg[0]): {meter.result.pkg[0]} uJ")
        log_energy("execute_mape", meter.result.pkg[0]) 

def run_execute_drift():
    time.sleep(400)
    while True:
        time.sleep(3)
        meter = pyRAPL.Measurement("execute_drift")
        meter.begin()
        execute_drift()
        meter.end()
        log_energy("execute_drift", meter.result.pkg[0])

def run_periodic_retrain():
    while True:
        time.sleep(500)
        # Ensure `drift.csv` has data by storing the last 1500 rows from `predictions.csv`
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

def get_approach_config():
    if not os.path.exists(config_file):
        print(f"Configuration file '{config_file}' not found. Defaulting to 'harmone'.")
        return "harmone"
    with open(config_file, 'r') as f:
        approach = f.read().strip().lower()
    return approach

approach = get_approach_config()
print(f"Running configuration: {approach}")

threads = []

if approach in ["harmone", "switch", "switch+retrain"]:
    # Always run t1 for these approaches
    t1 = threading.Thread(target=run_execute_mape, daemon=True)
    threads.append(t1)
    
    if approach == "harmone":
        # For HarmonE, also run t2 (drift detection)
        t2 = threading.Thread(target=run_execute_drift, daemon=True)
        threads.append(t2)
    elif approach == "switch+retrain":
        # For Switch + Retrain, run periodic retraining (t3)
        t3 = threading.Thread(target=run_periodic_retrain, daemon=True)
        threads.append(t3)
elif approach in ["single", "single+retrain"]:
    print("Single model approach selected: No dynamic model switching will be executed.")
    if approach == "single+retrain":
        # For single+retrain, run only periodic retraining (t3)
        t3 = threading.Thread(target=run_periodic_retrain, daemon=True)
        threads.append(t3)
else:
    print("Unknown approach configuration. No management threads will be started.")

for t in threads:
    t.start()

# Keep the script running indefinitely
exit_event = threading.Event()
exit_event.wait()
