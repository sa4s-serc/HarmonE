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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

print("Loading dataset...")

# Load the dataset
df = pd.read_csv("data/synthetic_data.csv")
data = df["aggregated_is_iceberg"].values

# Normalize data for LSTM
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Split into train/test (80% train, 20% test)
split_idx = int(len(data) * 0.8)
train_data, test_data = data_scaled[:split_idx], data_scaled[split_idx:]

print("Dataset loaded and split into train/test.")

# Create time series sequences
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Prepare training and testing sequences
seq_length = 10
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

print("Time series sequences created.")

# ---------------- LSTM Model ----------------
# class LSTMModel(nn.Module):
#     def __init__(self):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
#         self.fc = nn.Linear(50, 1)

#     def forward(self, x):
#         _, (h_n, _) = self.lstm(x)
#         return self.fc(h_n[-1])

# # Convert to tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# print("LSTM model initialized.")

# # Train LSTM
# lstm_model = LSTMModel()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
# train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)

# print("Training LSTM model...")

# num_epochs = 50
# for epoch in tqdm(range(num_epochs), desc="LSTM Training Progress"):
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         output = lstm_model(X_batch)
#         loss = criterion(output, y_batch)
#         loss.backward()
#         optimizer.step()

# # Save LSTM model
# torch.save(lstm_model.state_dict(), "models/lstm_model.pth")
# print("LSTM training completed and model saved as models/lstm_model.pth.")

# # ---------------- Measure LSTM Inference Time ----------------
# print("Performing inference with LSTM...")
# lstm_model.eval()
# start_time = time.time()
# y_pred_lstm = lstm_model(X_test_tensor).detach().numpy().flatten()
# lstm_inference_time = time.time() - start_time
# print(f"LSTM inference time: {lstm_inference_time:.6f} seconds")

# ---------------- Linear Regression ----------------
print("Training Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Save Linear Regression model
with open("models/lr_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

print("Linear Regression model trained and saved as models/lr_model.pkl.")

# ---------------- Measure Linear Regression Inference Time ----------------
print("Performing inference with Linear Regression...")
start_time = time.time()
y_pred_lr = lr_model.predict(X_test)
lr_inference_time = time.time() - start_time
print(f"Linear Regression inference time: {lr_inference_time:.6f} seconds")

# ---------------- Support Vector Machine (SVM) ----------------
print("Training SVM model...")
svm_model = SVR(kernel="linear", C=1.0, tol=0.01)
svm_model.fit(X_train, y_train)

# Save SVM model
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

print("SVM model trained and saved as models/svm_model.pkl.")

# ---------------- Measure SVM Inference Time ----------------
print("Performing inference with SVM...")
start_time = time.time()
y_pred_svm = svm_model.predict(X_test)
svm_inference_time = time.time() - start_time
print(f"SVM inference time: {svm_inference_time:.6f} seconds")

# Denormalize predictions
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
# y_pred_lstm_actual = scaler.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()
y_pred_lr_actual = scaler.inverse_transform(y_pred_lr.reshape(-1, 1)).flatten()
y_pred_svm_actual = scaler.inverse_transform(y_pred_svm.reshape(-1, 1)).flatten()

# Evaluate models
# mae_lstm = mean_absolute_error(y_test_actual, y_pred_lstm_actual)
mae_lr = mean_absolute_error(y_test_actual, y_pred_lr_actual)
mae_svm = mean_absolute_error(y_test_actual, y_pred_svm_actual)

print("\nModel Evaluation Completed:")
# print(f"LSTM MAE: {mae_lstm:.4f}")
print(f"Linear Regression MAE: {mae_lr:.4f}")
print(f"SVM MAE: {mae_svm:.4f}")

print("\nInference Time Summary:")
# print(f"LSTM Inference Time: {lstm_inference_time:.6f} seconds")
print(f"Linear Regression Inference Time: {lr_inference_time:.6f} seconds")
print(f"SVM Inference Time: {svm_inference_time:.6f} seconds")
