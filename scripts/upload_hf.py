"""Upload one of our policy checkpoints to the HumorR1 org on HuggingFace.

Each upload includes a generated README describing the cell, base model,
training recipe, and reproducer command.

Usage:
    uv run python scripts/upload_hf.py \\
        --cell E1a \\
        --local-dir checkpoints/qwen3vl-2b-sft-instruct-nothink/lora_final \\
        --base-model Qwen/Qwen3-VL-2B-Instruct \\
        --variant sft_no_thinking
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


CARDS = {
    "sft_no_thinking": {
        "title": "SFT, no thinking (Qwen3-VL-2B-Instruct + LoRA)",
        "summary": (
            "LoRA-adapted Qwen3-VL-2B-Instruct supervised fine-tuned on "
            "the chosen captions of 271 New Yorker contests. The model "
            "emits captions directly inside `<caption>...</caption>` "
            "tags, with no chain-of-thought."
        ),
    },
    "sft_thinking": {
        "title": "SFT, with thinking (Qwen3-VL-2B-Thinking + LoRA, merged)",
        "summary": (
            "LoRA-adapted Qwen3-VL-2B-Thinking supervised fine-tuned on "
            "(image, synthetic thinking, chosen caption) triples, then "
            "merged. Output format: `{thinking}</think>\\n\\n<caption>X</caption>`."
        ),
    },
    "grpo_no_thinking": {
        "title": "GRPO, no thinking (Qwen3-VL-2B-Instruct + LoRA)",
        "summary": (
            "LoRA on Qwen3-VL-2B-Instruct trained via GRPO against the "
            "Bradley-Terry reward model HumorR1/rm-qwen25vl-3b-nodesc. "
            "Captions emitted directly with no thinking trace."
        ),
    },
    "grpo_thinking": {
        "title": "GRPO, with thinking (Qwen3-VL-2B-Thinking + LoRA)",
        "summary": (
            "LoRA on Qwen3-VL-2B-Thinking trained via GRPO against the "
            "Bradley-Terry reward model HumorR1/rm-qwen25vl-3b-nodesc. "
            "Output format: `{thinking}</think>\\n\\n<caption>X</caption>`."
        ),
    },
    "dpo_no_thinking": {
        "title": "DPO, no thinking (Qwen3-VL-2B-Instruct + LoRA)",
        "summary": (
            "LoRA on Qwen3-VL-2B-Instruct trained via Direct Preference "
            "Optimization on 2{,}000 Bradley-Terry preference pairs. "
            "No reward model in the loop at training time. Captions emitted "
            "directly with no thinking trace."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True, help="Cell tag, e.g. E1a")
    p.add_argument("--local-dir", required=True, help="Local checkpoint dir")
    p.add_argument("--base-model", required=True, help="HF base model id")
    p.add_argument("--variant", required=True, choices=list(CARDS.keys()))
    p.add_argument("--repo-id", default=None,
                   help="Override default HumorR1/policy-{cell}-{variant}")
    p.add_argument("--org", default="HumorR1")
    p.add_argument("--private", action="store_true")
    return p.parse_args()


def write_readme(local_dir: Path, cell: str, base_model: str, variant: str,
                 repo_id: str) -> Path:
    info = CARDS[variant]
    is_lora = (local_dir / "adapter_config.json").exists() or \
              (local_dir / "adapter_model.safetensors").exists()
    library_tag = "library_name: peft" if is_lora else "library_name: transformers"
    readme = f"""---
license: apache-2.0
base_model: {base_model}
{library_tag}
tags:
  - vision-language
  - new-yorker
  - humor
  - rlhf
  - {variant.replace('_', '-')}
datasets:
  - yguooo/newyorker_caption_ranking
language:
  - en
---

# humor-r1 — {info["title"]} ({cell})

{info["summary"]}

## Training data

- 271 New Yorker contests, top-rated caption per contest
  (`yguooo/newyorker_caption_ranking`).
- The 60k Bradley-Terry preference pairs underlying the reward model
  (separate split).
- We deliberately do NOT use the dataset's GPT-4o-generated
  Scene/Twist/Location/Entities descriptions in the prompt, since they
  hand-feed scene content to a vision-language model that can already
  see the image; this makes the policy and reward model usable on any
  single-panel cartoon, not just the curated subset.

## How it fits the project

Part of a 2x2 ablation over training method (SFT, GRPO) and output
format (no thinking, thinking) for humor caption generation. See
`HumorR1/rm-qwen25vl-3b-nodesc` for the reward model used to train (and
score) this policy.

## Inference

Backbone: `{base_model}`.
{"This repo is a LoRA adapter; load with `peft.PeftModel.from_pretrained`." if is_lora else "This repo is a merged full model; load with `transformers.AutoModelForCausalLM.from_pretrained`."}

```python
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
{"from vllm.lora.request import LoRARequest" if is_lora else ""}

processor = AutoProcessor.from_pretrained("{base_model}", trust_remote_code=True)
llm = LLM(model="{base_model}", trust_remote_code=True, dtype="bfloat16",
          {("enable_lora=True, max_lora_rank=32, " if is_lora else "")}max_model_len=4096)

# Caption format: <caption>X</caption>; thinking variant prefixes <think>...</think>.
```

## Reward model used during training

- `HumorR1/rm-qwen25vl-3b-nodesc` (held-out pairwise accuracy 0.6635).
"""
    path = local_dir / "README.md"
    path.write_text(readme)
    return path


def main() -> int:
    args = parse_args()
    local = Path(args.local_dir)
    if not local.exists():
        print(f"ERR: {local} does not exist", file=sys.stderr)
        return 1
    repo_id = args.repo_id or f"{args.org}/policy-{args.cell.lower()}-{args.variant.replace('_','-')}"
    print(f"Uploading {local}  -->  {repo_id}", flush=True)
    write_readme(local, args.cell, args.base_model, args.variant, repo_id)

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(local),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"upload {args.cell} ({args.variant})",
        ignore_patterns=["optimizer.pt", "*.tmp", "rng_state.pth", "scheduler.pt"],
    )
    print(f"Done: https://huggingface.co/{repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
