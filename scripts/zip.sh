#!/bin/bash

#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR" || { echo "Failed to change directory"; exit 1; }

# Construct the log file name
DATE=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_FILE="../logs/zip_results_$DATE.log"
mkdir -p ../logs  # Ensure the logs directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

zip benchmark_output_$DATE.zip -r benchmark_output