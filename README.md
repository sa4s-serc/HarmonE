# HarmonE

HarmonE is a self-adaptive framework for sustainable MLOps pipelines. It integrates a dynamic adaptation mechanism based on the MAPE-K loop to balance energy consumption and predictive accuracy by managing model switching, selective retraining, and versioned model reuse.

This documentation covers setup instructions, baseline configurations, and details about key files and directories used in HarmonE.

---

## 1. Overview

- **HarmonE** continuously monitors system metrics (e.g., prediction accuracy and energy consumption) and adapts model usage at runtime.
- **Model Repository:**  
  - The `models/` folder stores the current versions of the models available for inference.
  - The `versionedMR/` folder archives previous versions of models after retraining.
- **Model Configuration:**  
  - The file `knowledge/model.csv` stores the name of the model currently being used (e.g., `lstm`, `svm`, or `linear`).

---

## 2. Setup Instructions

### 2.1 Environment Setup

1. **Create a Virtual Environment:**  
   Isolate the HarmonE project by creating and activating a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**  
   Install all required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### 2.2 Permissions and Platform Requirements

- **Linux Requirement:**  
  This documentation assumes a Linux system environment.
- **pyRAPL Usage:**  
  The `pyRAPL` library (used for energy measurement) works only on Intel processors. To enable powercap functionality, set the following permissions:
  ```bash
  sudo chmod -R a+r /sys/class/powercap/intel-rapl
  ```

### 2.3 Preparing the Codebase

1. **Cleanup Script:**  
   Run the cleanup script to remove stale models and reset relevant CSV files. Ensure the script is executable:
   ```bash
   chmod +x cleanup.sh
   ./cleanup.sh
   ```

2. **Model Training:**  
   Populate the current model repository and store the first version in the versioned model repository:
   ```bash
   python3 tools/train_models.py
   ```

---

## 3. Baseline Configurations

The system supports nine baseline approaches, categorized into dynamic adaptation and single-model modes. A shell script named `set_approach.sh` is used to select the desired baseline. **Make sure to set execute permissions:**
```bash
chmod +x set_approach.sh
```

### 3.1 Dynamic Adaptation Baselines

- **harmone:**  
  Runs the full HarmonE system with both dynamic model switching (thread `t1` executing `execute_mape`) and drift detection (thread `t2` executing `execute_drift`).
  
- **switch:**  
  Runs only thread `t1` (i.e., only monitoring via `execute_mape`).

- **switch+retrain:**  
  Runs thread `t1` (monitoring) and thread `t3` (periodic retraining).

### 3.2 Single-Model Baselines

These baselines disable dynamic model switching. Instead, the inference system runs a single model defined in `knowledge/model.csv`.

- **Without Retraining:**  
  - `single-lstm`
  - `single-svm`
  - `single-linear`

- **With Retraining:**  
  These options run periodic retraining (thread `t3`) in addition to using a fixed model.
  - `single-lstm+retrain`
  - `single-svm+retrain`
  - `single-linear+retrain`

**Note:** For single-model baselines, the `set_approach.sh` script automatically updates `knowledge/model.csv` to store the model name being used.

---

## 4. Running the System

### 4.1 Running HarmonE (Dynamic Adaptation)

1. **Set the Baseline:**  
   For full dynamic adaptation, run:
   ```bash
   ./set_approach.sh harmone
   ```
2. **Start the Management System:**  
   Then execute:
   ```bash
   python mape/manage.py
   ```
   This will launch the appropriate threads (t1 and t2) for monitoring and drift detection. If python packages are not found, make sure you have entered the virtual enivironment for this terminal as well. Please install other dependencies if prompted.

### 4.2 Running Other Baselines

1. **Adaptive Baselines:**  
   - For `switch`, run:
     ```bash
     ./set_approach.sh switch
     python mape/manage.py
     ```
   - For `switch+retrain`, run:
     ```bash
     ./set_approach.sh switch+retrain
     python mape/manage.py
     ```

2. **Single-Model Baselines:**  
   - **Without Retraining:**  
     To run a single model (e.g., LSTM), set the baseline:
     ```bash
     ./set_approach.sh single-lstm
     ```
     Then, run the inference system without starting `mape/manage.py` (the management system is not launched for these baselines).

   - **With Retraining:**  
     For a single model with retraining (e.g., SVM), run:
     ```bash
     ./set_approach.sh single-svm+retrain
     python mape/manage.py
     ```
     This will update `knowledge/model.csv` to `svm` and run periodic retraining via thread t3.
