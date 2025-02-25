import time
import pandas as pd
from retrain import train_lstm, save_model_and_data
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import pickle
import torch

JOB_QUEUE_FILE = "jobs_queue.csv"

def get_best_job():
    """Finds the best retraining job based on cost and sustainability."""
    if not os.path.exists(JOB_QUEUE_FILE):
        return None

    jobs = pd.read_csv(JOB_QUEUE_FILE)
    if jobs.empty:
        return None

    # Sort jobs based on cost (lowest first) and sustainability (highest first)
    best_job = jobs.sort_values(by=["cost", "sustainability"], ascending=[True, False]).iloc[0]
    return best_job

def execute_retraining(model_name):
    """Performs model retraining."""
    print(f"🚀 Starting retraining for {model_name}...")

    # Load data
    drift_data = pd.read_csv("knowledge/drift.csv")
    data = drift_data["true_value"].values
    seq_length = 10
    X_train, y_train = create_sequences(data, seq_length)
    train_df = pd.DataFrame({"train_data": data})

    # Retrain Model
    if model_name == "linear":
        model = LinearRegression()
        model.fit(X_train, y_train)
        save_model_and_data(model, "lr_model", train_df)

    elif model_name == "svm":
        model = SVR(kernel="linear", C=1.0, tol=0.01)
        model.fit(X_train, y_train)
        save_model_and_data(model, "svm_model", train_df)

    elif model_name == "lstm":
        model = train_lstm(X_train, y_train)
        save_model_and_data(model, "lstm_model", train_df)

    else:
        print(f"❌ Unknown model type: {model_name}")
        return

    print(f"✔ {model_name} retraining completed.")

    # Remove executed job from queue
    jobs = pd.read_csv(JOB_QUEUE_FILE)
    jobs = jobs[jobs["model"] != model_name]
    jobs.to_csv(JOB_QUEUE_FILE, index=False)

def schedule_retraining():
    """Continuously checks for optimal retraining time."""
    while True:
        print("⏳ Checking for best retraining opportunity...")
        best_job = get_best_job()

        if best_job is not None:
            execute_retraining(best_job["model"])
        else:
            print("🔍 No optimal retraining job found, checking again later.")

        time.sleep(30)  # Wait before checking again

if __name__ == "__main__":
    schedule_retraining()
