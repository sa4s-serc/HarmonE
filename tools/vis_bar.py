import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = "output/results/summary_statistics.csv"
df = pd.read_csv(file_path)

# Define metrics to visualize
metrics = ["mae", "rmse", "mape", "r2_score", "mean_inference_time", "mean_energy"]

# Set up grid layout
num_metrics = len(metrics)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()

# Plot each metric as a bar chart
for i, metric in enumerate(metrics):
    sns.barplot(x="approach", y=metric, data=df, ax=axes[i], palette="viridis")
    axes[i].set_title(metric)
    axes[i].set_xticklabels(df["approach"], rotation=45, ha="right")

plt.tight_layout()

# Save the figure
output_path = "output/summary_statistics.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Visualization saved to {output_path}")
