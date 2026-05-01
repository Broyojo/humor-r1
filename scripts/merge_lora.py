"""Merge a LoRA adapter into its base model and save the merged model.

    uv run python scripts/merge_lora.py \\
        --base-model Qwen/Qwen3-VL-2B-Instruct \\
        --lora-dir checkpoints/qwen3vl-2b-sft-instruct-nothink/lora_final \\
        --out-dir checkpoints/qwen3vl-2b-sft-instruct-nothink-merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--lora-dir", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading base: {args.base_model}", flush=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"Loading LoRA: {args.lora_dir}", flush=True)
    model = PeftModel.from_pretrained(base, args.lora_dir)
    print("Merging...", flush=True)
    model = model.merge_and_unload()
    print(f"Saving merged model to {out}", flush=True)
    model.save_pretrained(out, safe_serialization=True)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.save_pretrained(out)
    print("Done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
