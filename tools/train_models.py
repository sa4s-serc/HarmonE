import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Ensure base directories exist
base_dir = "versionedMR"
os.makedirs(base_dir, exist_ok=True)
original_model_dir = "models"
os.makedirs(original_model_dir, exist_ok=True)

def get_next_version(model_name):
    """Finds the next version number for a given model."""
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)  # Ensure model-specific directory exists
    existing_versions = [d for d in os.listdir(model_dir) if d.startswith("version_")]
    
    if existing_versions:
        existing_versions = sorted([int(v.split("_")[-1]) for v in existing_versions])
        return existing_versions[-1] + 1
    return 1  # Start from version_1 if none exists

def save_model_and_data(model, model_name, train_data):
    """Saves the trained model and corresponding training data in both the versioned and original directory."""
    version = get_next_version(model_name)
    version_path = os.path.join(base_dir, model_name, f"version_{version}")
    os.makedirs(version_path, exist_ok=True)

    # Save model in both locations
    if model_name == "lstm":
        model_path = os.path.join(original_model_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), os.path.join(version_path, f"{model_name}.pth"))
    else:
        model_path = os.path.join(original_model_dir, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(version_path, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(model, f)

    # Save training data
    train_data.to_csv(os.path.join(version_path, "data.csv"), index=False)

    print(f"{model_name} saved at {version_path} and {model_path}")

# Load the dataset
# df = pd.read_csv("data/synthetic_data.csv")
# data = df["aggregated_is_iceberg"].values
df = pd.read_csv("data/pems/flow_data_train.csv")
data = df["flow"].values

# Normalize data for LSTM
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Split into train/test (80% train, 20% test)
split_idx = int(len(data) * 0.8)
train_data, test_data = data_scaled[:split_idx], data_scaled[split_idx:]

# Create time series sequences
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# ---------------- LSTM Model ----------------
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

lstm_model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)

# Train LSTM
print("Training LSTM model...")
num_epochs = 50
for epoch in tqdm(range(num_epochs), desc="LSTM Training Progress"):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = lstm_model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

# Save LSTM model with versioning and in the original directory
train_df = pd.DataFrame({"train_data": train_data})  # Convert training data to dataframe
save_model_and_data(lstm_model, "lstm", train_df)

# ---------------- Linear Regression ----------------
print("Training Linear Regression model...")
lr_model = Ridge(alpha=256)
lr_model.fit(X_train, y_train)

# Save Linear Regression model with versioning and in the original directory
save_model_and_data(lr_model, "linear", train_df)

# ---------------- Support Vector Machine (SVM) ----------------
print("Training SVM model...")
svm_model = SVR(kernel="linear", C=0.05, tol=0.16)
svm_model.fit(X_train, y_train)

# Save SVM model with versioning and in the original directory
save_model_and_data(svm_model, "svm", train_df)
