import threading
import time
import pyRAPL
import csv
from execute import execute_mape, execute_drift

pyRAPL.setup()
log_file = "knowledge/mape_log.csv"

# Ensure the CSV file has a header
with open(log_file, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["function", "energy_joules"])

def log_energy(function_name, energy_joules):
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([function_name, energy_joules])

def run_execute_mape():
    while True:
        time.sleep(20)  # Runs every 20 seconds
        meter = pyRAPL.Measurement("execute_mape")
        meter.begin()
        execute_mape()
        meter.end()
        log_energy("execute_mape", meter.result.pkg[0]) 

def run_execute_drift():
    while True:
        time.sleep(120)  # Runs every 120 seconds
        meter = pyRAPL.Measurement("execute_drift")
        meter.begin()
        execute_drift()
        meter.end()
        log_energy("execute_drift", meter.result.pkg[0])

# Start both monitoring threads
t1 = threading.Thread(target=run_execute_mape, daemon=True)
t2 = threading.Thread(target=run_execute_drift, daemon=True)

t1.start()
t2.start()

while True:
    time.sleep(1)
