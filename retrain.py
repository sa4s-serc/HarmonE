import os
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from cloud_simulation import get_dynamic_price, get_sustainability_score
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Ensure base directories exist
base_dir = "versionedMR"
os.makedirs(base_dir, exist_ok=True)
original_model_dir = "models"
os.makedirs(original_model_dir, exist_ok=True)

JOB_QUEUE_FILE = "jobs_queue.csv"
if not os.path.exists(JOB_QUEUE_FILE):
    pd.DataFrame(columns=["model", "region", "cost", "sustainability", "timestamp"]).to_csv(JOB_QUEUE_FILE, index=False)

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
    """Saves the trained model and training data in both the versioned and original directory."""
    version = get_next_version(model_name)
    version_path = os.path.join(base_dir, model_name, f"version_{version}")
    os.makedirs(version_path, exist_ok=True)

    if model_name == "lstm_model":
        model_path = os.path.join(original_model_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), os.path.join(version_path, f"{model_name}.pth"))
    else:
        model_path = os.path.join(original_model_dir, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(version_path, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(model, f)

    train_data.to_csv(os.path.join(version_path, "data.csv"), index=False)
    print(f"✔ {model_name} saved at {version_path} and {model_path}")

def create_sequences(data, seq_length=5):
    """Create time series sequences for training."""
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
    """Train LSTM model"""
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

def enqueue_retraining(model_name):
    """Adds a retraining request to the queue with cost and sustainability details."""
    region = random.choice(["Amsterdam", "India", "USA"])
    cost = get_dynamic_price(region)
    sustainability = get_sustainability_score(region)

    job = pd.DataFrame([[model_name, region, cost, sustainability, pd.Timestamp.now()]],
                       columns=["model", "region", "cost", "sustainability", "timestamp"])
    
    job.to_csv(JOB_QUEUE_FILE, mode='a', header=False, index=False)
    print(f"✔ Added {model_name} retraining job to queue in {region} (Cost: ${cost}, Sustainability: {sustainability})")

def execute_retraining(model_name):
    """Performs model retraining when conditions are optimal."""
    print(f"🚀 Starting retraining for {model_name}...")

    drift_data = pd.read_csv("knowledge/drift.csv")
    data = drift_data["true_value"].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    seq_length = 5
    X_train, y_train = create_sequences(data_scaled, seq_length)
    train_df = pd.DataFrame({"train_data": data_scaled})

    if model_name == "linear":
        model = Ridge(alpha=256)
        model.fit(X_train, y_train)
        save_model_and_data(model, "lr_model", train_df)

    elif model_name == "svm":
        model = SVR(kernel="linear", C=0.05, tol=0.16)
        model.fit(X_train, y_train)
        save_model_and_data(model, "svm_model", train_df)

    elif model_name == "lstm":
        model = train_lstm(X_train, y_train)
        save_model_and_data(model, "lstm_model", train_df)

    else:
        print(f"❌ Unknown model type: {model_name}")
        return

    print(f"✔ {model_name} retraining completed.")

def main():
    print("🔍 Checking drift and scheduling retraining jobs...")

    try:
        drift_data = pd.read_csv("knowledge/drift.csv")
        with open("knowledge/model.csv", "r") as f:
            model_to_train = f.read().strip()

        if drift_data.empty or model_to_train == "":
            print("❌ Error: One or both input files are empty")
            return

        region = random.choice(["Amsterdam", "India", "USA"])
        cost = get_dynamic_price(region)
        
        if cost < 0.3:  # Threshold to decide immediate retraining vs queuing
            print(f"💰 Low cost detected (${cost}), retraining immediately in {region}.")
            execute_retraining(model_to_train)
        else:
            print(f"📌 Cost too high (${cost}), scheduling retraining in queue.")
            enqueue_retraining(model_to_train)

    except FileNotFoundError as e:
        print(f"❌ Error: {str(e)}")
    except Exception as e:
        print(f"❌ Error during retraining: {str(e)}")

if __name__ == "__main__":
    main()
