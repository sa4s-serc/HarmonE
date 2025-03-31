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
   pip install torch --index-url https://download.pytorch.org/whl/cpu
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

## 3. Working with the PEMS Dataset

### 3.1 Obtaining and Preparing PEMS Data

Due to privacy restrictions, the PEMS dataset cannot be made publicly available directly. However, if you have access to the PEMS dataset from the California Department of Transportation, please follow these steps to integrate the data into HarmonE:

1. **Download the Data:**
   - Obtain the raw CSV files from the California PEMS website. Note that the dataset may originally be in MATLAB format; you might need to convert it to CSV. The available CSV files should include a column named "Flow (Veh/5 Minutes)" which represents the traffic flow measurements.

2. **Place CSV Files in the Project:**
   - Create the folder structure for the raw data:
     ```bash
     mkdir -p data/pems/raw
     ```
   - Place all your PEMS CSV files into the `data/pems/raw/` directory.

3. **Process and Split the Data:**
   - The provided script (`tools/store_pems.py`) will read all CSV files from `data/pems/raw/`, extract the **"Flow (Veh/5 Minutes)"** column, and merge them.
   - The script splits the merged data into two files:
     - `data/pems/flow_data_train.csv` – Contains a smaller portion (default 10%) of the data used for training the first version of models.
     - `data/pems/flow_data_test.csv` – Contains the larger portion (default 90%) of the data used for simulating the system.
   - Run the script with:
     ```bash
     python3 tools/store_pems.py --train_ratio 0.1
     ```
     You can adjust the `--train_ratio` parameter if needed.

**Assumptions:**
- Each CSV file in `data/pems/raw/` has a column labeled exactly as **"Flow (Veh/5 Minutes)"**.
- The training data (the first portion of the merged data) is used for training the initial model version.
- The test data (the remaining data) is used for system simulation and evaluation.

This setup ensures that you have a properly split dataset for both training your models and running the full system simulation in HarmonE.

### 3.2 Inducing Drift in PEMS Test Data

To simulate dta drift in your traffic flow data, you can use the `tools/induce_drift.py` script. This script allows you to interactively define drift regions and specify the magnitude of drift via scale and shift adjustments. The process is visualized with before-and-after plots for easy comparison.

> Note: The HarmonE framework can be used without drift induction—this script is optional for robustness testing.

**Steps:**

1. **Prepare the Data:**  
   Place your PEMS test data file in CSV format at:  
   ```
   data/pems/flow_data_test.csv
   ```
   The CSV should have a column labeled `flow` (if your original column is different, consider renaming it or updating the script).

2. **Run the Drift Induction Script:**  
   Execute the script by running:
   ```bash
   python3 tools/induce_drift.py
   ```
3. **Interactive Prompts:**  
   - The script will first display the original flow data in a plot.
   - You will be prompted to enter the number of drift regions you wish to induce.
   - For each drift region, you will be asked:
     - **Start Index:** The starting row index of the drift region.
     - **End Index:** The ending row index of the drift region.
     - **Scale Factor:** The multiplicative adjustment to apply to the `flow` values.
     - **Shift Amount:** The additive adjustment to apply to the `flow` values.
   - After entering the parameters, the script applies the changes and shows a comparison plot (original vs. drifted).

4. **Confirm Changes:**  
   When prompted, confirm whether to save the drifted data back to the test file. If you decline, no changes will be made.

**Assumptions:**
- The CSV files in the PEMS dataset are expected to have a consistent format.
- The test data file is named `flow_data_test.csv` and is located in the `data/pems/` directory.
- The drift induction is performed on the `flow` column, which should represent the traffic flow (in vehicles per 5 minutes).

This script provides a flexible and visually guided approach to introduce controlled drift into your test data, enabling you to evaluate the robustness of the HarmonE framework under shifting conditions.


## 4 Preparing the Codebase

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

## 5. Approach Configurations

The system supports nine approaches, categorized into adaptative and single-model approaches. A shell script named `set_approach.sh` is used to select the desired approach. **Make sure to set execute permissions:**
```bash
chmod +x set_approach.sh
```
> "Before running any approach configuration, please ensure you have completed the steps outlined in **Section 4: Preparing the Codebase**."


### 5.1 Dynamic Adaptation Baselines

- **harmone:**  
  Runs the full HarmonE system with both dynamic model switching (thread `t1` executing `execute_mape`) and drift detection (thread `t2` executing `execute_drift`).
  
- **switch:**  
  Runs only thread `t1` (i.e., only monitoring via `execute_mape`).

- **switch+retrain:**  
  Runs thread `t1` (monitoring) and thread `t3` (periodic retraining).

### 5.2 Single-Model Baselines

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

## 6. Running the System

### 6.1 Running HarmonE

1. **Set the Baseline:**  
   For full dynamic adaptation, run:
   ```bash
   ./set_approach.sh harmone
   ```

2. **Start the Inference System:**  
   In one terminal (with the virtual environment activated), run:
   ```bash
   python3 inference.py
   ```
   This starts the inference subsystem that uses the models from the `models/` folder.

3. **Start the Management System:**  
   In a separate terminal (with the virtual environment activated), run:
   ```bash
   python mape/manage.py
   ```
   This will launch the appropriate threads (t1 and t2) for monitoring and drift detection.  
   *If Python packages are not found, ensure you have activated your virtual environment and installed all dependencies.*

---

### 6.2 Running Baselines

1. **Adaptive Baselines:**  
   - For **switch**:
     1. Set the baseline:
        ```bash
        ./set_approach.sh switch
        ```
     2. Start the inference system:
        ```bash
        python3 inference.py
        ```
     3. In a separate terminal, start the management system:
        ```bash
        python mape/manage.py
        ```
   - For **switch+retrain**:
     1. Set the baseline:
        ```bash
        ./set_approach.sh switch+retrain
        ```
     2. Start the inference system:
        ```bash
        python3 inference.py
        ```
     3. In a separate terminal, start the management system:
        ```bash
        python mape/manage.py
        ```

2. **Single-Model Baselines:**

   - **Without Retraining:**  
     To run a single model (e.g., LSTM):
     1. Set the baseline:
        ```bash
        ./set_approach.sh single-lstm
        ```
     2. Start the inference system (do not launch the management system):
        ```bash
        python3 inference.py
        ```
   
   - **With Retraining:**  
     For a single model with retraining (e.g., SVM):
     1. Set the baseline:
        ```bash
        ./set_approach.sh single-svm+retrain
        ```
     2. Start the inference system:
        ```bash
        python3 inference.py
        ```
     3. In a separate terminal, start the management system:
        ```bash
        python mape/manage.py
        ```
     This updates `knowledge/model.csv` to `svm` and runs periodic retraining (thread t3).

---