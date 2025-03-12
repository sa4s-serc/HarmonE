import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the models and corresponding filenames
models = {
    "Linear Regression": "output/modified/modified_predictions_linear.csv",
    "SVM": "output/modified/modified_predictions_svm.csv",
    "LSTM": "output/modified/modified_predictions_lstm.csv",
    "Our Approach": "output/modified/modified_predictions_adaptive.csv"
}

# Create plots for each model
plt.figure(figsize=(10, 6))
for model_name, file_path in models.items():
    try:
        # Load data
        df = pd.read_csv(file_path)

        # Compute cumulative sum of energy
        df["cumulative_sum_energy"] = df["energy"].cumsum()

        # Generate threshold line (25,000 * x)
        df["threshold"] = 25000 * np.arange(1, len(df) + 1)

        # Plot each model's cumulative energy
        plt.plot(df.index, df["cumulative_sum_energy"], label=model_name)
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}. Skipping {model_name}.")

# Add the dotted threshold line
plt.plot(df.index, df["threshold"], linestyle="dotted", color="red", label="Threshold (25,000 * x)")

# Labels and Title
plt.xlabel("Entry Index")
plt.ylabel("Cumulative Energy")
plt.title("Cumulative Energy Usage Over Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Save and Show
plt.savefig("cumulative_energy_comparison.png")

