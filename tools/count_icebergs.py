import csv

# Open the file as a text file
with open("data/test.json", "r") as f:
    content = f.read()

# Extract "is_iceberg" values while preserving order
labels = []
pos = 0

while True:
    # Find the next occurrence of "is_iceberg":
    pos_0 = content.find('"is_iceberg":0', pos)
    pos_1 = content.find('"is_iceberg":1', pos)

    # If neither is found, we're done
    if pos_0 == -1 and pos_1 == -1:
        break

    # Determine which one appears first
    if pos_0 != -1 and (pos_1 == -1 or pos_0 < pos_1):
        labels.append("0")
        pos = pos_0 + 13  # Move past the found text
    else:
        labels.append("1")
        pos = pos_1 + 13  # Move past the found text

# Write the extracted values to a CSV file
with open("iceberg_labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["is_iceberg"])  # Header
    for label in labels:
        writer.writerow([label])

print("CSV file saved as iceberg_labels.csv")
