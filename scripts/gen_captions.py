"""Generate captions for one experimental cell on dev + test splits.

Each cell is one (base model, LoRA dir or None, system prompt) tuple.
Saves a JSONL per (cell, split) at results/captions/{cell}_{split}.jsonl
with one row per (contest_number, sample_idx).

Usage:
    uv run python scripts/gen_captions.py \\
        --cell E0a \\
        --base-model Qwen/Qwen3-VL-2B-Instruct \\
        --variant no_thinking \\
        --num-samples 5

For the frontier baseline (E0c) we route through OpenRouter instead of vLLM —
see scripts/gen_captions_api.py.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTION_RE = re.compile(r"<caption>(.*?)</caption>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
IMAGE_MAX_SIDE = 448


SYSTEM_THINKING = (
    "You are a witty cartoon caption writer. Look at the cartoon image, "
    "briefly identify what makes the scene funny, then output exactly one "
    "one-line caption inside <caption>...</caption> tags. The text inside "
    "the tags is the final caption; nothing else counts."
)
SYSTEM_NO_THINKING = (
    "You are a witty cartoon caption writer. Look at the cartoon image and "
    "output exactly one one-line caption inside <caption>...</caption> tags. "
    "Do not include any other text."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True, help="Cell ID, e.g. E0a, E1b")
    p.add_argument("--base-model", required=True,
                   help="HF model name or local path")
    p.add_argument("--lora-dir", type=Path, default=None,
                   help="Optional LoRA adapter dir")
    p.add_argument("--variant", choices=["thinking", "no_thinking"], required=True)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--splits", nargs="+", default=["validation", "test"])
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "results" / "captions")
    p.add_argument("--max-cartoons", type=int, default=0,
                   help="0 = all unique contests in split")
    return p.parse_args()


def maybe_resize(image: Image.Image) -> Image.Image:
    if max(image.size) > IMAGE_MAX_SIDE:
        image = image.copy()
        image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.LANCZOS)
    return image


def build_prompt_text(processor, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_caption(text: str) -> str | None:
    m = CAPTION_RE.search(text)
    if not m:
        return None
    s = m.group(1).strip()
    return s or None


def extract_thinking(text: str) -> str | None:
    m = THINK_RE.search(text)
    if not m:
        return None
    return m.group(1).strip() or None


def load_cartoons(data_root: Path, split: str, max_cartoons: int) -> list[dict]:
    ds_path = data_root / f"caption_sft_{split}"
    ds = load_from_disk(str(ds_path))
    seen: dict[int, dict] = {}
    for row in ds:
        c = int(row["contest_number"])
        if c in seen:
            continue
        img_path = data_root / row["image_path"]
        if not img_path.exists():
            continue
        seen[c] = {
            "contest_number": c,
            "user_prompt": "Write a funny one-line caption for this New Yorker-style cartoon.",
            "image_path": str(img_path),
        }
    cartoons = list(seen.values())
    if max_cartoons > 0:
        cartoons = cartoons[:max_cartoons]
    return cartoons


def make_llm(args) -> LLM:
    kwargs: dict[str, Any] = dict(
        model=args.base_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    if args.lora_dir is not None:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = 32
    return LLM(**kwargs)


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    system = SYSTEM_THINKING if args.variant == "thinking" else SYSTEM_NO_THINKING

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    print(f"Loading vLLM: model={args.base_model} lora={args.lora_dir}", flush=True)
    llm = make_llm(args)
    lora_request = LoRARequest("cell_lora", 1, str(args.lora_dir)) if args.lora_dir else None

    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop=["</caption>"],
        include_stop_str_in_output=True,
    )

    for split in args.splits:
        cartoons = load_cartoons(args.data_root, split, args.max_cartoons)
        print(f"\n[{args.cell}] split={split}: {len(cartoons)} cartoons × {args.num_samples} samples", flush=True)

        prompts: list[dict] = []
        for c in cartoons:
            img = maybe_resize(Image.open(c["image_path"]).convert("RGB"))
            chat = build_prompt_text(processor, system, c["user_prompt"])
            for _ in range(args.num_samples):
                prompts.append({"prompt": chat, "multi_modal_data": {"image": img}})

        outputs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_request)

        out_path = args.out_root / f"{args.cell}_{split}.jsonl"
        with out_path.open("w") as f:
            for c_idx, c in enumerate(cartoons):
                for s_idx in range(args.num_samples):
                    o = outputs[c_idx * args.num_samples + s_idx]
                    txt = o.outputs[0].text
                    row = {
                        "cell": args.cell,
                        "split": split,
                        "contest_number": c["contest_number"],
                        "image_path": c["image_path"],
                        "user_prompt": c["user_prompt"],
                        "sample_idx": s_idx,
                        "completion": txt,
                        "caption": extract_caption(txt),
                        "thinking": extract_thinking(txt),
                        "completion_tokens": len(o.outputs[0].token_ids),
                        "finish_reason": o.outputs[0].finish_reason,
                    }
                    f.write(json.dumps(row) + "\n")
        print(f"  wrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
