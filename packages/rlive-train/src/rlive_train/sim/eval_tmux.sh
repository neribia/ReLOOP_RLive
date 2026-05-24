#!/bin/bash
# Run sim evaluation in the foreground — edit the variables below, then:
#   bash eval_tmux.sh
#
# Typical sim2real workflow:
#   1. Train on server:  bash train_tmux.sh
#   2. Download model from W&B or scp from server
#   3. Set MODEL_PATH below and run this script (sim) or real/eval_tmux.sh (robot)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
export DISPLAY=:0
export SAPIEN_NO_DISPLAY=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# ---------------------------------------------------------------------------
# !! SET THIS to the path of the model you want to evaluate !!
# Example: resources/models/<wandb_run_id>/checkpoints/best_model.zip
MODEL_PATH=""

if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH is not set. Edit eval_tmux.sh and set MODEL_PATH."
    exit 1
fi
# ---------------------------------------------------------------------------

ARGS=(
    --model-path  "$MODEL_PATH"
    --env         sapien          # options: sapien | simple
    --n-episodes  10              # default: 10
    --max-episode-steps 50        # default: 20  — must match training!
    # --run-group  <wandb_run_id> # auto-inferred from model path if omitted
    # --run-name   "my_eval_run"  # default: eval_<run-group>_<env>
    # --fixed-goal                # flag: fix goal at image centre
)

uv run "$SCRIPT_DIR/eval_sb3.py" "${ARGS[@]}"

