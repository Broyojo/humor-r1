"""Synthesize <think>...</think> traces for the SFT dataset.

For each (image, prompt, caption) triple in caption_sft_train, ask the
base Qwen3-VL-2B-Thinking model "why is this caption funny for this
cartoon?" — the model gives a natural 1-3 sentence explanation. We
treat that explanation as the chain-of-thought trace and pair it with
the original (high-rated) caption to form an SFT target of

    {thinking}</think>\n\n<caption>{caption}</caption>

The Qwen3-VL-Thinking chat template already prepends `<think>\\n` to
every assistant turn, so combined with our target the rollouts look
like a clean

    <think>{thinking}</think>\n\n<caption>{caption}</caption>

Run:
    uv run python scripts/synthesize_thinking.py
Outputs: data/caption_sft_train_with_thinking/ (HF dataset on disk)
"""

from __future__ import annotations

import os
# Spawn before vllm import so the worker subprocess starts cleanly.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import re
from pathlib import Path

from datasets import Dataset, load_from_disk
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_ROOT / "caption_sft_train"
OUT_DIR = DATA_ROOT / "caption_sft_train_with_thinking"
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-2B-Thinking")
IMAGE_MAX_SIDE = 448

EXPLAIN_INSTRUCTION = (
    "Look at this New Yorker-style cartoon. A judge has selected the "
    "following one-line caption as the funniest: \"{caption}\". In 1-2 "
    "short sentences, explain what makes that caption funny in context "
    "of the scene."
)


def build_chat(prompt_text: str, caption: str):
    return [
        {"role": "system", "content": "You analyze humor in New Yorker cartoons."},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text + "\n\n" + EXPLAIN_INSTRUCTION.format(caption=caption)},
            ],
        },
    ]


def maybe_resize(image: Image.Image) -> Image.Image:
    if max(image.size) > IMAGE_MAX_SIDE:
        image = image.copy()
        image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.LANCZOS)
    return image


def extract_thinking(raw: str) -> str:
    """Pull the contents of <think>...</think>; if not closed, take everything."""
    m = re.search(r"<think>(.*?)(?:</think>|$)", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()


def main():
    print("loading processor + base model via vLLM ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    print(f"loading {TRAIN_DIR}", flush=True)
    ds = load_from_disk(str(TRAIN_DIR))
    print(f"  rows: {len(ds)}", flush=True)

    prompts = []
    images = []
    for row in ds:
        image = Image.open(DATA_ROOT / row["image_path"]).convert("RGB")
        image = maybe_resize(image)
        chat = build_chat(row["prompt"], row["caption"])
        text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append({"prompt": text, "multi_modal_data": {"image": image}})
        images.append(image)

    sampling = SamplingParams(
        n=1,
        temperature=0.4,
        top_p=0.9,
        max_tokens=512,
        stop=["</think>"],
        include_stop_str_in_output=False,
    )

    print(f"generating {len(prompts)} thinking traces ...", flush=True)
    outputs = llm.generate(prompts, sampling_params=sampling)

    rows_out = []
    for out, row in zip(outputs, ds):
        text = out.outputs[0].text  # everything inside <think>...
        thinking = extract_thinking(text)
        rows_out.append({**row, "thinking": thinking})

    new_ds = Dataset.from_list(rows_out)
    new_ds.save_to_disk(str(OUT_DIR))
    print(f"saved {len(new_ds)} rows to {OUT_DIR}", flush=True)
    # Print 2 samples
    for r in rows_out[:2]:
        print("---")
        print("caption :", r["caption"])
        print("thinking:", r["thinking"][:300])


if __name__ == "__main__":
    main()
