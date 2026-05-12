#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/train_$(date +%Y%m%d_%H%M%S).log"

export NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
export DISPLAY=:0
export SAPIEN_NO_DISPLAY=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

nohup uv run "$SCRIPT_DIR/train_sb3.py" \
    --env sapien \                  # options: sapien, simple
    --lr 3e-4 \                     # default: 3e-4
    --ent-coef 0.05 \               # default: 0.05
    --num-envs 4 \                  # default: 4
    --max-episode-steps 20 \        # default: 20
    --total-timesteps 1e6 \     # default: 1_000_000, 10e6, 
    --eval-freq 5000 \              # default: 1_000
    --n-eval-episodes 5 \           # default: 5
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started with PID $PID"
echo "Logs: $LOG_FILE"
echo "$PID" > "$SCRIPT_DIR/train.pid"