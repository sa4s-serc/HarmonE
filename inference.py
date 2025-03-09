import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import pyRAPL
from sklearn.preprocessing import MinMaxScaler

# Ensure directories exist
os.makedirs("knowledge", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Initialize PyRAPL
pyRAPL.setup()
energy_meter = pyRAPL.Measurement("inference")

# ---------------- Load Dataset ----------------
print("Loading synthetic data stream...")

df = pd.read_csv("data/pems/flow_data_test.csv")
data = df["flow"].values

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Create rolling window sequences (assuming sequence length of 10)
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X_stream, y_stream = create_sequences(data_scaled, seq_length)

print("Data stream prepared. Streaming inference begins...")

# ---------------- Define LSTM Model ----------------
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# ---------------- Inference Loop ----------------
# Create a CSV to store predictions
predictions_file = "knowledge/predictions.csv"
print("hi")
if not os.path.exists(predictions_file):
    pd.DataFrame(columns=["true_value", "predicted_value", "model_used", "inference_time", "energy_uJ"]).to_csv(predictions_file, index=False)

for i in range(len(X_stream)):  
    # ---------------- Check Active Model ----------------
    try:
        with open("knowledge/model.csv", "r") as f:
            chosen_model = f.read().strip().lower()  # Read model name (lstm, linear, svm)
    except FileNotFoundError:
        print("Error: knowledge/model.csv not found. Defaulting to LSTM.")
        chosen_model = "lstm"

    print(f"Inference {i+1}/{len(X_stream)}: Using model → {chosen_model.upper()}")

    # ---------------- Load and Use Model ----------------
    X_input = X_stream[i].reshape(1, -1)  # Reshape input for non-LSTM models

    # Start PyRAPL energy measurement
    energy_meter.begin()

    start_time = time.time()
    
    if chosen_model == "lstm":
        lstm_model = LSTMModel()
        lstm_model.load_state_dict(torch.load("models/lstm_model.pth", weights_only=False))
        lstm_model.eval()

        X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(-1)
        prediction = lstm_model(X_tensor).detach().numpy().flatten()[0]

    elif chosen_model == "linear":
        with open("models/lr_model.pkl", "rb") as f:
            lr_model = pickle.load(f)
        prediction = lr_model.predict(X_input)[0]

    elif chosen_model == "svm":
        with open("models/svm_model.pkl", "rb") as f:
            svm_model = pickle.load(f)
        prediction = svm_model.predict(X_input)[0]

    else:
        print(f"Unknown model '{chosen_model}'. Defaulting to LSTM.")
        lstm_model = LSTMModel()
        lstm_model.load_state_dict(torch.load("models/lstm_model.pth"))
        lstm_model.eval()

        X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(-1)
        prediction = lstm_model(X_tensor).detach().numpy().flatten()[0]

    inference_time = time.time() - start_time

    # Stop PyRAPL measurement and get energy usage
    energy_meter.end()
    energy_usage_uJ = energy_meter.result.pkg[0]  # Energy in microjoules (µJ)

    # ---------------- Store Predictions ----------------
    true_value = y_stream[i]
    true_value_actual = scaler.inverse_transform([[true_value]])[0, 0]
    predicted_value_actual = scaler.inverse_transform([[prediction]])[0, 0]

    # Append results to predictions.csv
    pd.DataFrame([[true_value_actual, predicted_value_actual, chosen_model, inference_time, energy_usage_uJ]], 
                 columns=["true_value", "predicted_value", "model_used", "inference_time", "energy_uJ"]).to_csv(
        predictions_file, mode="a", header=False, index=False
    )

    print(f"True: {true_value_actual:.2f}, Predicted: {predicted_value_actual:.2f}, Model: {chosen_model.upper()}, "
          f"Inference Time: {inference_time:.6f} sec, Energy: {energy_usage_uJ} µJ")

    # Simulate real-time streaming delay
    time.sleep(0.2)

print("\nStreaming inference completed. Predictions saved in knowledge/predictions.csv")
