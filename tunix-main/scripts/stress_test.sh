# scripts/stress_test.sh
#!/usr/bin/env bash
set -e

# Usage: ./scripts/stress_test.sh [CONFIG] [DATA_PATH]
CONFIG=${1:-configs/uncertainty/model_config.yaml}
DATA_PATH=${2:-data/sample.jsonl}

# Activate environment
source venv/bin/activate

# Define stress parameters
BATCH_SIZES=(1 2 4 8 16)
MAX_SEQLENS=(128 256 512)

for bs in "${BATCH_SIZES[@]}"; do
  for sl in "${MAX_SEQLENS[@]}"; do
    OUTDIR=outputs/stress_bs${bs}_sl${sl}
    mkdir -p "$OUTDIR"
    echo "Testing batch_size=$bs, max_seq_length=$sl"
    python -m tunix.run \
      --config "$CONFIG" \
      --data_path "$DATA_PATH" \
      --output_dir "$OUTDIR" \
      --batch_size "$bs" \
      --max_seq_length "$sl" \
      2>&1 | tee "$OUTDIR/log.txt"
  done
done

echo "Stress testing completed."
