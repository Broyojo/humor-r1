#!/usr/bin/env bash
# Launch GRPO training on the local machine.
#
# Defaults pick GPU 1 for the policy + vLLM rollouts and GPU 2 for the
# reward model — both clean A100s on the lab box. Override:
#   POLICY_GPU=3 REWARD_GPU=0 ./scripts/run_grpo.sh
#
# Other useful overrides:
#   MAX_STEPS=10     # smoke test
#   WANDB_MODE=disabled
#   POLICY_MODEL_NAME=Qwen/Qwen3-VL-2B-Thinking
#
# Logs go to logs/grpo_<timestamp>.log.

set -euo pipefail

cd "$(dirname "$0")/.."

POLICY_GPU="${POLICY_GPU:-1}"
REWARD_PHYSICAL_GPU="${REWARD_GPU:-2}"

# Order in CUDA_VISIBLE_DEVICES determines cuda:0 (policy) and cuda:1 (RM).
export CUDA_VISIBLE_DEVICES="${POLICY_GPU},${REWARD_PHYSICAL_GPU}"
export REWARD_GPU=1  # second visible device

export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/scratch/triton_cache}"
mkdir -p logs "$HF_HOME" "$TRITON_CACHE_DIR"

ts=$(date +%Y%m%d_%H%M%S)
log="logs/grpo_${ts}.log"

echo "==> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (policy=cuda:0, RM=cuda:$REWARD_GPU)"
echo "==> Log: $log"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true

uv run python -u scripts/train_grpo_qwen3vl.py "$@" 2>&1 | stdbuf -oL tee "$log"
