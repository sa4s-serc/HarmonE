import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Define model paths
models = {
    "Linear Regression": "output/updated/modified_predictions_linear.csv",
    "SVM": "output/updated/modified_predictions_svm.csv",
    "LSTM": "output/updated/modified_predictions_lstm.csv",
    "Our Approach": "output/runs/predictions_9.csv"
}

# Store results
r2_scores = {}
avg_inference_times = {}
avg_energies = {}

# Read each file and compute metrics
for model, path in models.items():
    df = pd.read_csv(path)
    r2_scores[model] = r2_score(df['true_value'], df['predicted_value'])
    avg_inference_times[model] = df['inference_time'].mean()
    avg_energies[model] = df['energy'].mean()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# R2 Score
axes[0].bar(r2_scores.keys(), r2_scores.values(), color='skyblue')
axes[0].set_title("R² Score")
axes[0].set_ylabel("R²")
axes[0].set_ylim(0, 1)

# Inference Time
axes[1].bar(avg_inference_times.keys(), avg_inference_times.values(), color='lightcoral')
axes[1].set_title("Avg Inference Time (s)")
axes[1].set_ylabel("Time (s)")

# Energy Consumption
axes[2].bar(avg_energies.keys(), avg_energies.values(), color='lightgreen')
axes[2].set_title("Avg Energy Consumption")
axes[2].set_ylabel("Energy (J)")

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('barplot.png')   