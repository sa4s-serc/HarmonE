# HarmonE

## Prerequisites

- Python 3.8 or higher
- `virtualenv` or `venv` for creating a virtual environment
- `pip` for installing Python packages

## Setup

### 1. Create a Virtual Environment

To avoid conflicts with system-wide packages, create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

- On macOS and Linux:

  ```bash
  source venv/bin/activate
  ```

- On Windows:

  ```bash
  venv\Scripts\activate
  ```

### 2. Install Required Packages

Install the necessary Python packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Set Permissions for pyRAPL

To allow pyRAPL to access energy consumption data, you need to set the appropriate permissions:

```bash
sudo chmod -R a+r /sys/class/powercap/intel-rapl
```

This functionality is currently restricted to Linux systems and may not be supported with other Operating systems.

## Training Models

### 1. Prepare the Models Directory

Before training, ensure the `models/` directory is empty:

```bash
rm -rf models/*
```

### 2. Generate Synthetic Data (Optional)

If you need synthetic data for training, you can generate it using the following script:

```bash
python tools/synthetic_data_generation.py
```

### 3. Train Models

Train your models using the `train_models.py` script. Make sure to run this from the project's root directory:

```bash
python tools/train_models.py
```

## Running the System

### 1. Start the Inference System

To run the main inference system, execute the following command:

```bash
python3 inference.py
```

### 2. Start the Management System

In a separate terminal, start the management system:

```bash
python mape/manage.py
```

## Directory Structure
```
|-- README.md               # Project documentation and setup instructions
|-- data/                   # Directory for storing datasets (raw or processed)
|-- inference.py            # Main script for running the inference system
|-- knowledge/              # Directory containing knowledge files
|   |-- drift.csv           # File tracking model drift information
|   |-- model.csv           # File containing model metadata or configurations
|   `-- predictions.csv     # File storing prediction results
|-- mape/                   # Directory for MAPE-K loop components
|   |-- analyse.py          # Script for analyzing system behavior
|   |-- execute.py          # Script for executing adaptation actions
|   |-- manage.py           # Main script for managing the system
|   |-- monitor.py          # Script for monitoring system performance
|   `-- plan.py             # Script for planning adaptation strategies
|-- models/                 # Directory for storing trained models
|   |-- lr_model.pkl        # Trained Linear Regression model
|   |-- lstm_model.pth      # Trained LSTM model
|   `-- svm_model.pkl       # Trained SVM model
|-- requirements.txt        # List of Python dependencies
|-- retrain.py              # Script for retraining models
|-- tools/                  # Directory containing utility scripts
```