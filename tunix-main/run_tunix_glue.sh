#!/bin/bash
set -e

# Activate your Python environment if needed
# source /path/to/your/venv/bin/activate

# Install requirements (adjust if you already installed)
pip install --upgrade pip
pip install tunix datasets transformers evaluate matplotlib pandas

# Make sure output directory exists
mkdir -p sft

echo "Starting multi-task GLUE fine-tuning with TUNiX..."

python run_tunix_glue.py

echo "Done! All outputs saved in sft/ directory."
