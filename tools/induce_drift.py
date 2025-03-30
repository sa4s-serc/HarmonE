import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def induce_drift():
    # Load dataset
    file_path = "data/pems/flow_data_test.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file exists.")
        return
    df.columns = df.columns.str.strip()  # Clean column names

    # Plot original data
    plt.figure(figsize=(12, 6))
    plt.plot(df['flow'], label="Original Flow")
    plt.title("Original Flow Data")
    plt.xlabel("Index")
    plt.ylabel("Flow")
    plt.legend()
    plt.show()

    # Ask the user how many drift regions to induce
    try:
        num_regions = int(input("Enter the number of drift regions to induce: "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return

    drift_params = []
    for i in range(num_regions):
        print(f"\nDrift Region {i+1}:")
        try:
            start = int(input("  Enter start index: "))
            end = int(input("  Enter end index: "))
            scale = float(input("  Enter scale factor (multiplicative adjustment): "))
            shift = float(input("  Enter shift amount (additive adjustment): "))
        except ValueError:
            print("Invalid input. Please enter numeric values for indices, scale, and shift.")
            return
        drift_params.append((start, end, scale, shift))

    # Apply drift modifications
    df_drift = df.copy()
    for start, end, scale, shift in drift_params:
        df_drift.loc[start:end, 'flow'] = df_drift.loc[start:end, 'flow'] * scale + shift

    # Plot updated data for visual comparison
    plt.figure(figsize=(12, 6))
    plt.plot(df['flow'], label="Original Flow", alpha=0.5)
    plt.plot(df_drift['flow'], label="Drifted Flow", color="red")
    plt.title("Flow Data with Induced Drift")
    plt.xlabel("Index")
    plt.ylabel("Flow")
    plt.legend()
    plt.show()

    # Confirm before saving changes
    confirm = input(f"Save the drifted data to {file_path}? (Y/N): ")
    if confirm.lower() == 'y':
        df_drift.to_csv(file_path, index=False)
        print(f"Drifted data saved to {file_path}.")
    else:
        print("Operation cancelled. No changes made.")

if __name__ == "__main__":
    induce_drift()
