# HarmonE

## 1. Introduction

**HarmonE** is a self-adaptive approach to architecting sustainable MLOps pipelines. It is designed to continuously monitor key performance and energy consumption metrics, adapting its behavior at runtime using the MAPE-K loop. The goal of HarmonE is to maintain both high predictive accuracy and low energy consumption by dynamically managing model switching, selective retraining, and versioned model reuse. This documentation outlines the setup and execution steps for the HarmonE system. 

**Note:** This documentation follows a Linux system, and pyRAPL works only on Intel processors

## 2. Setup

### 2.1 Creating a Virtual Environment

It is recommended to isolate the HarmonE project within a Python virtual environment. To create and activate a virtual environment, run the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.2 Installing Requirements

Once the virtual environment is active, install the necessary Python packages with:

```bash
pip install -r requirements.txt
```

This will install all dependencies required by HarmonE, including libraries for model training, energy profiling, and system monitoring.

### 2.3 Setting Permissions for pyRAPL

To allow the `pyRAPL` library to access energy consumption data, you must grant the appropriate permissions. This is crucial for accurate energy profiling. Run:

```bash
sudo chmod -R a+r /sys/class/powercap/intel-rapl
```

**Important:** This permission adjustment is specific to Linux systems.

### 2.4 Running Cleanup Script

Before starting the system, ensure you run the cleanup script to remove any stale files or data. Execute the script with:

```bash
./cleanup.sh
```

This script removes models, resets CSV files to their header-only state, and writes fresh configuration data to JSON files as specified.

## 3. Starting the Systems

### 3.1 Inference System

To start the main inference system, run the following command:

```bash
python3 inference.py
```

This command launches the inference subsystem, which is responsible for handling real-time predictions using the selected ML models.

### 3.2 Management System

In a separate terminal, start the management system with:

```bash
python mape/manage.py
```

The management system monitors system performance, detects uncertainties, and triggers the appropriate adaptation strategies using the MAPE-K loop.
