#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/train_$(date +%Y%m%d_%H%M%S).log"

export NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
export DISPLAY=:0
export SAPIEN_NO_DISPLAY=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

ARGS=(
    --env sapien            # options: sapien | simple
    --lr 3e-4               # default: 3e-4
    --ent-coef 0.01         # default: 0.01
    --log-std-init 0.0      # default: 0.0
    --num-envs 4            # default: 4
    --max-episode-steps 20  # default: 20
    --total-timesteps 10e6  # default: 10e6, e.g. 1e6
    --eval-freq 10000       # default: 10000
    --n-eval-episodes 5     # default: 5
    --n-epochs 10           # default: 10
    # --fixed-goal          # flag: fix goal at image centre
)

nohup uv run "$SCRIPT_DIR/train_sb3.py" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started with PID $PID"
echo "Logs: $LOG_FILE"
echo "Tail logs with:  tail -f $LOG_FILE"
echo "$PID" > "$SCRIPT_DIR/train.pid"