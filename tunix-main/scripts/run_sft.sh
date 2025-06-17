# scripts/run_sft.sh
#!/usr/bin/env bash
set -e

# Usage: ./scripts/run_sft.sh [CONFIG] [DATA_DIR] [OUTPUT_DIR]
CONFIG=${1:-configs/sft/sft_tpu.yaml}
DATA_DIR=${2:-/mnt/data/alpaca}
OUTPUT_DIR=${3:-outputs/sft_alpaca}

# Activate environment
source venv/bin/activate

# Launch SFT
python -m tunix.sft.train \
  --config "$CONFIG" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR"

echo "SFT done: logs in $OUTPUT_DIR/logs"
