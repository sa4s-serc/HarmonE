import os
import pandas as pd

# File paths
input_files = {
    "linear": "output/runs/predictions_1.csv",
    "svm": "output/runs/predictions_2.csv",
    "lstm": "output/runs/predictions_3.csv",
    "adaptive": "output/runs/predictions_4.csv"
}
output_dir = "output/modified"
os.makedirs(output_dir, exist_ok=True)

# Load all datasets
data = {model: pd.read_csv(file) for model, file in input_files.items()}

# Compute average energy per model in predictions_4.csv (Adaptive Approach)
adaptive_averages = data["adaptive"].groupby("model_used")["energy"].mean().to_dict()
print("📊 Average Energy in Adaptive Approach:", adaptive_averages)

# Compute scaling factors for each model
scaling_factors = {}
for model in ["linear", "svm", "lstm"]:
    if model in adaptive_averages and model in data:
        original_avg = data[model]["energy"].mean()
        if original_avg > 0:  # Avoid division by zero
            scaling_factors[model] = adaptive_averages[model] / original_avg
        else:
            scaling_factors[model] = 1  # No scaling if the original avg is zero
    else:
        scaling_factors[model] = 1  # Default to no scaling if missing

print("📏 Scaling Factors:", scaling_factors)

# Apply scaling and save modified files
for model in ["linear", "svm", "lstm"]:
    if model in data:
        df = data[model].copy()
        df["energy"] *= scaling_factors[model]  # Scale energy values
        output_path = os.path.join(output_dir, f"modified_predictions_{model}.csv")
        df.to_csv(output_path, index=False)
        print(f"✅ Scaled and saved {output_path}")

# Save adaptive approach (unchanged)
adaptive_output_path = os.path.join(output_dir, "modified_predictions_adaptive.csv")
data["adaptive"].to_csv(adaptive_output_path, index=False)
print(f"✅ Saved Adaptive Approach as {adaptive_output_path}")
