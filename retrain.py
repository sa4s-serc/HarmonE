import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def create_sequences(data, seq_length=10):
    """Create time series sequences for training"""
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
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    
    # Initialize model and training parameters
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor), 
        batch_size=16, 
        shuffle=True
    )
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
    
    # Save model
    torch.save(model.state_dict(), "models/lstm_model.pth")
    return model

def main():
    print("Loading drift and model data...")
    try:
        # Read drift and model files
        drift_data = pd.read_csv("knowledge/drift.csv")
        with open("knowledge/model.csv", "r") as f:
            model_to_train = f.read().strip()
        print(f"🔄 Retraining model: {model_to_train.upper()}")
        
        if drift_data.empty or model_to_train == "":
            print("Error: One or both input files are empty")
            return
        
        # Prepare training data
        data = drift_data['true_value'].values
        
        # Normalize data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences for training
        seq_length = 10
        X_train, y_train = create_sequences(data_scaled, seq_length)
        
        # Retrain specified model
        if model_to_train == 'linear':
            print("Retraining Linear Regression model...")
            model = LinearRegression()
            model.fit(X_train, y_train)
            with open("models/lr_model1.pkl", "wb") as f:
                pickle.dump(model, f)
            print("Linear Regression model retrained and saved")
            
        elif model_to_train == 'svm':
            print("Retraining SVM model...")
            model = SVR(kernel="linear", C=1.0, tol=0.01)
            model.fit(X_train, y_train)
            with open("models/svm_model.pkl", "wb") as f:
                pickle.dump(model, f)
            print("SVM model retrained and saved")
            
        elif model_to_train == 'lstm':
            print("Retraining LSTM model...")
            train_lstm(X_train, y_train)
            print("LSTM model retrained and saved")
            
        else:
            print(f"Error: Unknown model type '{model_to_train}'")
            return
        
        print("Retraining completed successfully")
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {str(e)}")
    except Exception as e:
        print(f"Error during retraining: {str(e)}")

if __name__ == "__main__":
    main()