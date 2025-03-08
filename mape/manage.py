import threading
import time
from execute import execute_mape, execute_drift

def run_execute_mape():
    while True:
        time.sleep(30)  # Runs every 20 seconds
        execute_mape()

def run_execute_drift():
    while True:
        time.sleep(120)  # Runs every 120 seconds
        execute_drift()

# Start both monitoring threads
t1 = threading.Thread(target=run_execute_mape, daemon=True)
t2 = threading.Thread(target=run_execute_drift, daemon=True)

t1.start()
t2.start()

while True:
    time.sleep(1)
