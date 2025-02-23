import csv
import numpy as np

# Load the iceberg labels from CSV
with open("data/train.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    labels = np.array([int(row[0]) for row in reader])

# Determine the number of full groups of 10
num_full_groups = len(labels) // 10
remainder = len(labels) % 10

# Aggregate into groups of 10 by counting the number of 1s
aggregated_labels = [np.sum(labels[i * 10 : (i + 1) * 10]) for i in range(num_full_groups)]

# Handle remainder by counting and scaling
if remainder > 0:
    last_count = np.sum(labels[-remainder:]) * (10 / remainder)  # Scale to match a 10-step count
    aggregated_labels.append(round(last_count))  # Round to keep integer values

# Save aggregated values to a new CSV
with open("data/agg_train.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["aggregated_is_iceberg"])  # Header
    for value in aggregated_labels:
        writer.writerow([value])

print("CSV file saved as agg_train.csv")
