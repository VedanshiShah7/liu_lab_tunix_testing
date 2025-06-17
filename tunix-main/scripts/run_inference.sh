# scripts/run_inference.sh
#!/usr/bin/env bash
set -e

# Usage: ./scripts/run_inference.sh [CONFIG] [DATA_PATH] [OUTPUT_DIR]
CONFIG=${1:-configs/uncertainty/model_config.yaml}
DATA_PATH=${2:-data/sample.jsonl}
OUTPUT_DIR=${3:-outputs/inference}

# Activate environment
source venv/bin/activate

# Run inference/uncertainty estimation
python -m tunix.run \
  --config "$CONFIG" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR"

echo "Inference done: outputs in $OUTPUT_DIR"
