import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("output/runs/predictions_4.csv")
# df = df[:500]
# Compute cumulative sum of energy
df["cumulative_sum_energy"] = df["energy"].cumsum()

# Generate threshold line (40000 * x)
df["threshold"] = 40000 * np.arange(1, len(df) + 1)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["cumulative_sum_energy"], label="Cumulative Sum Energy", color="blue")
plt.plot(df.index, df["threshold"], linestyle="dotted", color="red", label="Threshold (40,000 * x)")

plt.xlabel("Entry Index")
plt.ylabel("Cumulative Energy")
plt.title("Cumulative Energy Usage Over Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Save figure
plt.savefig("hello.png")

