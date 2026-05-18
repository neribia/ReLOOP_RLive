#!/bin/bash
# Run training in the foreground — edit the variables below, then:
#   bash train_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
export DISPLAY=:0
export SAPIEN_NO_DISPLAY=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# ---------------------------------------------------------------------------
ARGS=(
    --env sapien              # options: sapien | simple
    --lr 3e-4                 # default: 3e-4
    --ent-coef 0.01           # default: 0.01
    --log-std-init 0.0        # default: 0.0
    --num-envs 4              # default: 4
    --max-episode-steps 20    # default: 20
    --total-timesteps 10e6    # default: 10e6, e.g. 1e6
    --eval-freq 10000         # default: 10000
    --n-eval-episodes 5       # default: 5
    --n-epochs 10             # default: 10
    --checkpoint-freq 100000   # default: same as eval-freq; set higher to save fewer checkpoints
    # --fixed-goal            # flag: fix goal at image centre
)

uv run "$SCRIPT_DIR/train_sb3.py" "${ARGS[@]}"
