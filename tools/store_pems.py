import pandas as pd
import glob
import argparse

def main(train_ratio):
    # Path to your CSV files (please place your PEMS CSV files in this folder)
    file_paths = sorted(glob.glob("data/pems/raw/*.csv"))
    
    if not file_paths:
        print("No CSV files found in data/pems/raw/. Please place your PEMS CSV files there.")
        return

    # Read and concatenate the 'Flow (Veh/5 Minutes)' column from each file
    flow_data = []
    for file in file_paths:
        print(f"Processing file: {file}")
        df = pd.read_csv(file, usecols=["Flow (Veh/5 Minutes)"])
        flow_data.append(df)
    
    # Concatenate all data into a single DataFrame and rename the column to 'flow'
    merged_df = pd.concat(flow_data, ignore_index=True)
    merged_df.rename(columns={"Flow (Veh/5 Minutes)": "flow"}, inplace=True)
    
    total_rows = merged_df.shape[0]
    train_rows = int(total_rows * train_ratio)
    
    # Split into training and test data
    train_df = merged_df.iloc[:train_rows]
    test_df = merged_df.iloc[train_rows:]
    
    # Save training data (used for training the first version of models) and test data (used for system simulation)
    train_df.to_csv("data/pems/flow_data_train.csv", index=False)
    test_df.to_csv("data/pems/flow_data_test.csv", index=False)
    
    print(f"Saved {train_rows} rows to data/pems/flow_data_train.csv")
    print(f"Saved {total_rows - train_rows} rows to data/pems/flow_data_test.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Store and split PEMS flow data CSV files for HarmonE."
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.1,
        help="Proportion of data to be used as training data (default: 0.1). "
             "Training data is used to train the first version of models, while the remaining test data is used to simulate the system."
    )
    args = parser.parse_args()
    main(args.train_ratio)
