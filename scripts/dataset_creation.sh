#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR" || { echo "Failed to change directory"; exit 1; }

# Construct the log file name
LOG_FILE="../logs/run_dataset_creation_$(date '+%Y-%m-%d_%H-%M-%S').log"
mkdir -p ../logs  # Ensure the logs directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Creating dataset..."

python3 ../src/dpo_dataset.py \
    --dataset_name "open-r1/OpenR1-Math-220k" \
    --data_dir ../data/ \
    --method diff
    
echo "Dataset creation complete."