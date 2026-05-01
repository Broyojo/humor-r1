#!/usr/bin/env bash
# Orchestrator for the 7-cell eval grid.
# Assumes:
#   - .env has OPENROUTER_API_KEY
#   - GRPO run finished, checkpoint-200 LoRA available for E2b
#   - SFT-instruct LoRA trained for E1a, GRPO-instruct LoRA trained for E2a
#
# Each cell generates captions on dev (validation) + test splits.
# Output: results/captions/{cell}_{split}.jsonl
#
# Run cells with their specific GPU/API config:

set -e
set -a; source .env; set +a

mkdir -p results/captions results/judge

cell() {
  local CELL=$1
  shift
  echo "===================="
  echo "[$(date '+%H:%M:%S')] Cell $CELL"
  echo "===================="
  uv run python scripts/gen_captions.py --cell "$CELL" "$@"
}

cell_api() {
  local CELL=$1
  shift
  echo "===================="
  echo "[$(date '+%H:%M:%S')] Cell $CELL (API)"
  echo "===================="
  uv run python scripts/gen_captions_api.py --cell "$CELL" "$@"
}

# ---- Zero-shot baselines (can run anytime; vLLM uses 1 GPU at a time) ----

run_E0a() {
  cell E0a \
    --base-model Qwen/Qwen3-VL-2B-Instruct \
    --variant no_thinking \
    --num-samples 5 \
    --max-new-tokens 256 \
    --max-model-len 2048
}

run_E0b() {
  cell E0b \
    --base-model Qwen/Qwen3-VL-2B-Thinking \
    --variant thinking \
    --num-samples 5 \
    --max-new-tokens 4096 \
    --max-model-len 6144
}

run_E0c() {
  cell_api E0c --model openai/gpt-5.5 --num-samples 5
}

# ---- Trained policies ----

run_E1a() {
  cell E1a \
    --base-model Qwen/Qwen3-VL-2B-Instruct \
    --lora-dir checkpoints/qwen3vl-2b-sft-instruct-nothink/lora_final \
    --variant no_thinking \
    --num-samples 5 \
    --max-new-tokens 256 \
    --max-model-len 2048
}

run_E1b() {
  cell E1b \
    --base-model "$PWD/checkpoints/qwen3vl-2b-sft-think-merged" \
    --variant thinking \
    --num-samples 5 \
    --max-new-tokens 4096 \
    --max-model-len 6144
}

run_E2a() {
  # GRPO LoRA on Instruct, best checkpoint chosen later from saves at 25/50/75/100.
  cell E2a \
    --base-model Qwen/Qwen3-VL-2B-Instruct \
    --lora-dir "${E2A_LORA_DIR:-checkpoints/qwen3vl-2b-grpo-instruct-nothink/lora_final}" \
    --variant no_thinking \
    --num-samples 5 \
    --max-new-tokens 256 \
    --max-model-len 2048
}

run_E2b() {
  # GRPO LoRA on the Thinking variant (best snapshot before collapse: ckpt-50).
  cell E2b \
    --base-model Qwen/Qwen3-VL-2B-Thinking \
    --lora-dir "${E2B_LORA_DIR:-checkpoints/qwen3vl-2b-grpo-thinking-final/checkpoint-50}" \
    --variant thinking \
    --num-samples 5 \
    --max-new-tokens 4096 \
    --max-model-len 6144
}

# ---- Scoring + metrics ----

score_all() {
  echo "===================="
  echo "[$(date '+%H:%M:%S')] Scoring all cells with RM"
  echo "===================="
  uv run python scripts/score_grid.py \
    --captions-glob 'results/captions/*.jsonl' \
    --reward-model-dir checkpoints/rm-final/final_reward_model \
    --out results/metrics.json
}

# ---- Judging ----

judge_all() {
  for J in anthropic/claude-sonnet-4.6 openai/gpt-5.5; do
    echo "===================="
    echo "[$(date '+%H:%M:%S')] Judge: $J"
    echo "===================="
    uv run python scripts/judge_pairwise.py --judge "$J" \
      --captions-dir results/captions \
      --splits validation test
  done
}

# Dispatch
case "${1:-help}" in
  E0a) run_E0a ;;
  E0b) run_E0b ;;
  E0c) run_E0c ;;
  E1a) run_E1a ;;
  E1b) run_E1b ;;
  E2a) run_E2a ;;
  E2b) run_E2b ;;
  score) score_all ;;
  judge) judge_all ;;
  *) echo "Usage: $0 {E0a|E0b|E0c|E1a|E1b|E2a|E2b|score|judge}"; exit 1 ;;
esac
