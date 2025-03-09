import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# ---------------- Load Dataset ----------------
print("Loading dataset...")

df = pd.read_csv("data/pems/flow_data_cleaned.csv")
data = df["flow"].values

# Normalize data for LSTM
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Split into train/test (80% train, 20% test)
split_idx = int(len(data) * 0)
train_data, test_data = data_scaled[:split_idx], data_scaled[split_idx:]

print("Dataset loaded and split into train/test.")

# Create time series sequences
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Prepare test sequences
seq_length = 5
X_test, y_test = create_sequences(test_data, seq_length)

print("Test sequences prepared.")

# ---------------- Load LSTM Model ----------------
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

print("Loading LSTM model...")

# Initialize and load trained model
lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load("models/lstm_model.pth"))
lstm_model.eval()

# Convert test data to tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

# Measure LSTM inference time
start_time = time.time()
y_pred_lstm = lstm_model(X_test_tensor).detach().numpy().flatten()
lstm_inference_time = time.time() - start_time

print(f"LSTM model loaded. Inference time: {lstm_inference_time:.6f} seconds.")

# ---------------- Load Linear Regression Model ----------------
print("Loading Linear Regression model...")
with open("models/lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

# Measure Linear Regression inference time
start_time = time.time()
y_pred_lr = lr_model.predict(X_test)
lr_inference_time = time.time() - start_time

print(f"Linear Regression model loaded. Inference time: {lr_inference_time:.6f} seconds.")

# ---------------- Load SVM Model ----------------
print("Loading SVM model...")
with open("models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

# Measure SVM inference time
start_time = time.time()
y_pred_svm = svm_model.predict(X_test)
svm_inference_time = time.time() - start_time

print(f"SVM model loaded. Inference time: {svm_inference_time:.6f} seconds.")

# ---------------- Denormalize Predictions ----------------
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_lstm_actual = scaler.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()
y_pred_lr_actual = scaler.inverse_transform(y_pred_lr.reshape(-1, 1)).flatten()
y_pred_svm_actual = scaler.inverse_transform(y_pred_svm.reshape(-1, 1)).flatten()

# ---------------- Print Sample Predictions ----------------
print("\nSample Predictions (First 10 values):")
print(f"Actual: {y_test_actual[:10]}")
print(f"LSTM Predicted: {y_pred_lstm_actual[:10]}")
print(f"Linear Regression Predicted: {y_pred_lr_actual[:10]}")
print(f"SVM Predicted: {y_pred_svm_actual[:10]}")

# ---------------- Print Inference Time ----------------
print("\nInference Time Summary:")
print(f"LSTM Inference Time: {lstm_inference_time:.6f} seconds")
print(f"Linear Regression Inference Time: {lr_inference_time:.6f} seconds")
print(f"SVM Inference Time: {svm_inference_time:.6f} seconds")

# ---------------- Evaluate Models ----------------

print("\nEvaluation Scores:")

r2_lstm = r2_score(y_test_actual, y_pred_lstm_actual)
r2_lr = r2_score(y_test_actual, y_pred_lr_actual)
r2_svm = r2_score(y_test_actual, y_pred_svm_actual)

print(f"LSTM r2: {r2_lstm:.6f}")
print(f"Linear Regression r2: {r2_lr:.6f}")
print(f"SVM r2: {r2_svm:.6f}")

