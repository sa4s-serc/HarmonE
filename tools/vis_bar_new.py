import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error

# Define model paths
models = {
    # "Linear Regression": "output/updated/modified_predictions_linear.csv",
    # "SVM": "output/updated/modified_predictions_svm.csv",
    # "LSTM": "output/updated/modified_predictions_lstm.csv",
    # "Switching": "output/runs/predictions_4_3.csv",
    # "Switching*": "output/runs/predictions_4_2.csv",
    # "Retraining": "output/runs/predictions_8_1.csv",
    # "Retraining*": "output/runs/predictions_8_2.csv",
    # "Retraining**": "output/runs/predictions_8_3.csv",
    "Our Approach": "output/runs/predictions_9_1.csv",
    "Our Approach*": "output/runs/predictions_9_2.csv",
    "Our Approach**": "output/runs/predictions_9_3.csv",
    "Our Approach***": "output/runs/predictions_9_4.csv",
}

# Store results
r2_scores = {}
avg_inference_times = {}
avg_energies = {}

# Read each file and compute metrics
for model, path in models.items():
    df = pd.read_csv(path)
    r2_scores[model] = mean_squared_error(df['true_value'], df['predicted_value'])
    avg_inference_times[model] = df['inference_time'].mean()
    avg_energies[model] = df['energy'].mean()

# Create a 1-row, 3-column subplot layout
fig = make_subplots(rows=1, cols=3, subplot_titles=["R² Score", "Avg Inference Time (s)", "Avg Energy Consumption"])

# R² Score Plot
fig.add_trace(go.Bar(
    x=list(r2_scores.keys()), 
    y=list(r2_scores.values()), 
    marker_color='skyblue',
    name="R² Score"
), row=1, col=1)

# Inference Time Plot
fig.add_trace(go.Bar(
    x=list(avg_inference_times.keys()), 
    y=list(avg_inference_times.values()), 
    marker_color='lightcoral',
    name="Inference Time"
), row=1, col=2)

# Energy Consumption Plot
fig.add_trace(go.Bar(
    x=list(avg_energies.keys()), 
    y=list(avg_energies.values()), 
    marker_color='lightgreen',
    name="Energy Consumption"
), row=1, col=3)

# Update layout
fig.update_layout(
    title_text="Model Performance Metrics",
    showlegend=False,  # Hide legends since titles indicate what each plot represents
    height=500,
    width=1200
)

# Rotate x-axis labels for better readability
fig.update_xaxes(tickangle=-45)

# Save the figure
fig.write_image("barplot.png")
