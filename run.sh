#!/usr/bin/env bash
cd "$(dirname "$0")"
source venv/bin/activate

echo "[INFO] Launching reinforcement learning training..."

python3 run_training.py

echo "[INFO] Training complete."
