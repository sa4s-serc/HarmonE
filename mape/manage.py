import time
from execute import execute

while True:
    print("\n--- Running MAPE-K Loop ---")
    execute()
    print("--- Cycle Complete ---")
    time.sleep(5)  # Run every 5 seconds
