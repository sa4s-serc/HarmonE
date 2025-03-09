import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess

# Set random seed for reproducibility
np.random.seed(42)

# Generate a base time series (ARIMA-like process)
ar_params = np.array([0.75, -0.3, 0.1])  # AR(3) coefficients
ma_params = np.array([0.5, -0.2, 0.1])   # MA(3) coefficients
ar = np.r_[1, -ar_params]  # Include lag=0 coefficient
ma = np.r_[1, ma_params]

arma_process = ArmaProcess(ar, ma)
base_series = arma_process.generate_sample(nsample=50000)  # 500 time steps
base_series = np.abs(base_series) * 10  # Scale for iceberg count (avoid negatives)
base_series = base_series.astype(int)  # Convert to integer counts

# Save as CSV
df = pd.DataFrame({"aggregated_is_iceberg": base_series})
df.to_csv("data/iceberg.csv", index=False)

print("Generated synthetic_data.csv")
