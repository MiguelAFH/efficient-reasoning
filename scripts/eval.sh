#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR" || { echo "Failed to change directory"; exit 1; }

# Construct the log file name
LOG_FILE="../logs/run_model_evaluation_$(date '+%Y-%m-%d_%H-%M-%S').log"
mkdir -p ../logs  # Ensure the logs directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Evaluating models..."
SUITE="efficient-reasoning"

# Run benchmark
helm-run \
    --conf-paths entries.conf \
    --suite $SUITE \
    --max-eval-instances 30

# Summarize benchmark results
helm-summarize --suite $SUITE --schema /share/pi/nigam/users/migufuen/helm/src/helm/benchmark/static/schema_capabilities.yaml
