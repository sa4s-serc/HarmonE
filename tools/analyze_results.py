import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import scipy.stats as stats

# Directories
input_dir = "output/runs/"
output_dir = "output/results/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load all predictions_n.csv files
file_list = sorted(glob.glob(os.path.join(input_dir, "predictions_*.csv")), key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Mapping n to approach name
approach_map = {
    1: "Linear Regression (No Retraining)",
    2: "SVM (No Retraining)",
    3: "LSTM (No Retraining)",
    # Add more mappings as needed when you extend experiments
}

# Load all CSVs into one DataFrame
dfs = []
for file in file_list:
    n = int(file.split('_')[-1].split('.')[0])  # Extract `n`
    df = pd.read_csv(file)
    df["approach"] = approach_map.get(n, f"Unknown_{n}")  # Assign approach name
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# ----------------- DESCRIPTIVE STATISTICS -----------------

from sklearn.metrics import r2_score

def compute_metrics(group):
    y_true = group["true_value"]
    y_pred = group["predicted_value"]
    
    return pd.Series({
        "mean_true_value": y_true.mean(),
        "mean_predicted_value": y_pred.mean(),
        "mae": np.mean(np.abs(y_true - y_pred)),
        "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "r2_score": r2_score(y_true, y_pred),
        "mean_inference_time": group["inference_time"].mean(),
        "mean_energy": group["energy"].mean()
    })

# Group by approach and apply the function
summary_stats = df_all.groupby("approach").apply(compute_metrics).reset_index()

# Save summary statistics
summary_stats.to_csv(os.path.join(output_dir, "summary_statistics.csv"), index=False)

print("Summary statistics saved with R² score added.")

# ----------------- VISUALIZATIONS -----------------

# Scatter Plot: True vs Predicted Values
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_all, x="true_value", y="predicted_value", hue="approach", alpha=0.1)
plt.plot([df_all["true_value"].min(), df_all["true_value"].max()],
         [df_all["true_value"].min(), df_all["true_value"].max()], 'r--')  # Perfect prediction line
plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.title("True vs. Predicted Values")
plt.legend()
plt.savefig(os.path.join(output_dir, "true_vs_predicted.png"))
plt.close()

# Residuals Distribution
df_all["residual"] = df_all["true_value"] - df_all["predicted_value"]
plt.figure(figsize=(8, 6))
sns.histplot(df_all, x="residual", hue="approach", kde=True, bins=30)
plt.axvline(0, color="red", linestyle="--")
plt.title("Residual Distribution")
plt.savefig(os.path.join(output_dir, "residual_distribution.png"))
plt.close()

# Boxplot of Inference Time & Energy Consumption
# Function to remove outliers using Tukey's Fences (IQR method)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal
df_trimmed = remove_outliers(df_all, "inference_time")
df_trimmed = remove_outliers(df_trimmed, "energy")

# Boxplot of Inference Time & Energy Consumption (after better trimming)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(data=df_trimmed, x="approach", y="inference_time", ax=axes[0])
axes[0].set_title("Inference Time Comparison (Outliers Removed)")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

sns.boxplot(data=df_trimmed, x="approach", y="energy", ax=axes[1])
axes[1].set_title("Energy Consumption Comparison (Outliers Removed)")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "inference_time_energy_comparison_filtered.png"))
plt.close()

# ----------------- STATISTICAL ANALYSES -----------------
results_txt = []

# Pairwise comparisons using Wilcoxon test
unique_approaches = df_all["approach"].unique()
for i in range(len(unique_approaches)):
    for j in range(i + 1, len(unique_approaches)):
        a, b = unique_approaches[i], unique_approaches[j]
        error_a = df_all[df_all["approach"] == a]["true_value"] - df_all[df_all["approach"] == a]["predicted_value"]
        error_b = df_all[df_all["approach"] == b]["true_value"] - df_all[df_all["approach"] == b]["predicted_value"]

        stat, p = stats.wilcoxon(error_a, error_b)
        results_txt.append(f"Wilcoxon test between {a} and {b}: p-value = {p:.5f}")

# Correlation between Energy and Inference Time
correlation = df_all[["energy", "inference_time"]].corr()
correlation.to_csv(os.path.join(output_dir, "correlation_analysis.csv"))

results_txt.append("Correlation between Energy and Inference Time:")
results_txt.append(str(correlation))

# ANOVA for error comparison across approaches
df_all["error"] = df_all["true_value"] - df_all["predicted_value"]
anova_result = stats.f_oneway(
    *[df_all[df_all["approach"] == approach]["error"] for approach in unique_approaches]
)
results_txt.append(f"ANOVA p-value for error comparison across approaches: {anova_result.pvalue:.5f}")

# Save results to text file
with open(os.path.join(output_dir, "statistical_analysis_results.txt"), "w") as f:
    f.write("\n".join(results_txt))

print("All results saved in output/results/")
