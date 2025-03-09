import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/pems/flow_data_test.csv")
df.columns = df.columns.str.strip()  # Ensure column names are clean

# Define drift regions (start, end)
drift_regions = [(3000, 6000), (10000, 14000)]

# Apply sudden drift
for start, end in drift_regions:
    df.loc[start:end, 'flow'] *= 1.7
    df.loc[start:end, 'flow'] -= 70
# Save modified dataset
df.to_csv("data/pems/flow_data_drifted.csv", index=False)

# Plot original vs drifted data
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['flow'], label="Drifted Flow Data", color="red")
plt.axvspan(6000, 10000, color="gray", alpha=0.3, label="Drift Region")
plt.axvspan(12500, 15000, color="gray", alpha=0.3)
plt.xlabel("Index")
plt.ylabel("Flow Value")
plt.title("Traffic Flow with Sudden Drift Injected")
plt.legend()
plt.grid(True)
plt.savefig('output/data_drifted')
