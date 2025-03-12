import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Define the models and corresponding filenames
models = {
    "Linear Regression": "output/updated/modified_predictions_linear.csv",
    "SVM": "output/updated/modified_predictions_svm.csv",
    "LSTM": "output/updated/modified_predictions_lstm.csv",
    # "Switching": "output/runs/predictions_4_1.csv",
    "Switching*": "output/runs/predictions_4_2.csv",
    "Retraining": "output/runs/predictions_8_1.csv",
    # "Retraining*": "output/runs/predictions_8_2.csv",
    # "Our Approach": "output/runs/predictions_9_1.csv",
    "Our Approach*": "output/runs/predictions_9_2.csv",
}

# Create figure
fig = go.Figure()

# Process each model
for model_name, file_path in models.items():
    try:
        # Load data
        df = pd.read_csv(file_path)

        # Compute cumulative sum of energy
        df["cumulative_sum_energy"] = df["energy"].cumsum()

        # Generate threshold line (25,000 * x)
        df["threshold"] = 25000 * np.arange(1, len(df) + 1)

        # Add model line to the plot
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df["cumulative_sum_energy"],
            mode="lines",
            name=model_name
        ))

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}. Skipping {model_name}.")

# Add the threshold line
fig.add_trace(go.Scatter(
    x=df.index, 
    y=df["threshold"], 
    mode="lines",
    line=dict(dash="dot", color="red"),
    name="Threshold (25,000 * x)"
))

# Update layout
fig.update_layout(
    title="Cumulative Energy Usage Over Time",
    xaxis_title="Entry Index",
    yaxis_title="Cumulative Energy",
    legend_title="Models",
    template="plotly_white"
)

# Save the figure
fig.write_image("cumulative_energy_comparison.png")
