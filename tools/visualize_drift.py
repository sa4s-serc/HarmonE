import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance

# Load dataset
df = pd.read_csv("data/pems/flow_data_drifted.csv")
df.columns = df.columns.str.strip()  # Strip any whitespace from column names

# Parameters
window_size = 750
step_size = 1  # Step size for sliding window
bins = 50  # Number of bins for KL divergence

# Store results
kl_values = []
energy_values = []
indices = []

# Function to compute KL Divergence and Energy Distance
def compute_drift(reference, current):
    ref_hist = np.histogram(reference, bins=bins, density=True)[0] + 1e-10
    curr_hist = np.histogram(current, bins=bins, density=True)[0] + 1e-10
    
    kl_div = entropy(ref_hist, curr_hist)
    energy_dist = wasserstein_distance(reference, current)
    
    return kl_div, energy_dist

# Sliding window approach
for start in range(0, len(df) - 2 * window_size, step_size):
    ref_window = df['flow'].iloc[start : start + window_size]
    curr_window = df['flow'].iloc[start + window_size : start + 2 * window_size]

    kl_div, energy_dist = compute_drift(ref_window, curr_window)
    
    kl_values.append(kl_div)
    energy_values.append(energy_dist)
    indices.append(start + 2*window_size)  # Middle point of window

# Plot results
plt.figure(figsize=(12, 5))

# KL Divergence Plot
plt.subplot(1, 2, 1)
plt.plot(indices, kl_values, label="KL Divergence", color="blue")
plt.xlabel("Data Index")
plt.ylabel("KL Divergence")
plt.title("KL Divergence Over Time")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.locator_params(axis="x", nbins=25)


# Energy Distance Plot
plt.subplot(1, 2, 2)
plt.plot(indices, energy_values, label="Energy Distance", color="red")
plt.xlabel("Data Index")
plt.ylabel("Energy Distance")
plt.title("Energy Distance Over Time")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.savefig('output/drifty_nex.png')
