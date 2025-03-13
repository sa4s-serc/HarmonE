import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Ensure directories exist
base_dir = "versionedMR"
os.makedirs(base_dir, exist_ok=True)
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

drift_file = "knowledge/drift.csv"
model_file = "knowledge/model.csv"

def get_next_version(model_name):
    """Finds the next version number for a given model."""
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    existing_versions = [d for d in os.listdir(model_dir) if d.startswith("version_")]

    if existing_versions:
        existing_versions = sorted([int(v.split("_")[-1]) for v in existing_versions])
        return existing_versions[-1] + 1
    return 1

def save_model_and_data(model, model_name, train_data):
    """Saves trained model and its data in `models/` and `versionedMR/`."""
    version = get_next_version(model_name)
    version_path = os.path.join(base_dir, model_name, f"version_{version}")
    os.makedirs(version_path, exist_ok=True)

    # Save model
    if model_name == "lstm":
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), os.path.join(version_path, f"{model_name}.pth"))
    else:
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(version_path, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(model, f)

    # Save training data
    train_data.to_csv(os.path.join(version_path, "data.csv"), index=False)
    print(f"✔ {model_name} saved at {version_path} and {model_path}")

def create_sequences(data, seq_length=5):
    """Creates time series sequences for training."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    """LSTM model architecture"""
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def train_lstm(X_train, y_train):
    """Trains an LSTM model."""
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)

    num_epochs = 50
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    return model

def retrain():
    """Retrains the current model using `drift.csv`."""
    if not os.path.exists(drift_file) or not os.path.exists(model_file):
        print("❌ Missing required files: `drift.csv` or `model.csv`.")
        return

    try:
        drift_data = pd.read_csv(drift_file)["true_value"].values
        with open(model_file, "r") as f:
            model_name = f.read().strip()
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    print(f"🚀 Retraining {model_name} using drift data...")

    # Preprocess data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(drift_data.reshape(-1, 1)).flatten()
    seq_length = 5
    X_train, y_train = create_sequences(data_scaled, seq_length)

    # Train model
    if model_name == "linear":
        model = Ridge(alpha=200)
        model.fit(X_train, y_train)
    elif model_name == "svm":
        model = SVR(kernel="linear", C=0.05, tol=0.16)
        model.fit(X_train, y_train)
    elif model_name == "lstm":
        model = train_lstm(X_train, y_train)
    else:
        print(f"❌ Unknown model type: {model_name}")
        return

    # Inverse transform before saving
    train_data_original = scaler.inverse_transform(data_scaled.reshape(-1, 1)).flatten()
    train_df = pd.DataFrame({"train_data": train_data_original})

    # Save retrained model
    save_model_and_data(model, model_name, train_df)
    print(f"✔ {model_name} retraining completed.")

if __name__ == "__main__":
    retrain()
