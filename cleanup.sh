#!/bin/bash
# Remove all files inside models/ and in subdirectories of versionedMR/
rm -rf models/*
rm -rf versionedMR/*/*

# Keep only the first line of knowledge/predictions.csv
sed -i '2,$d' knowledge/predictions.csv

# Write the specified JSON content to mape_info.json
cat > knowledge/mape_info.json <<EOF
{
    "last_line": 0,
    "current_energy_threshold": 1,
    "linear_version": 1,
    "lstm_version": 1,
    "svm_version": 1,
    "ema_scores": {
        "lstm": 0.82,
        "linear": 0.75,
        "svm": 0.79
    },
    "recovery_cycles": 0
}
EOF

# Keep only the first line of knowledge/drift.csv
sed -i '2,$d' knowledge/drift.csv

# Write the specified JSON content to thresholds.json
cat > knowledge/thresholds.json <<EOF
{
    "min_score": 0.78,
    "max_energy": 1,
    "beta": 0.95,
    "gamma": 0.8,
    "alpha": 0.1,
    "E_m": 0,
    "E_M": 25000
}
EOF

# Empty the file mape_log.csv
> knowledge/mape_log.csv
