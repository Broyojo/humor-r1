"""Direct Preference Optimization on Qwen3-VL-2B-Instruct using
the BT preference pairs underlying our reward model. Tests whether
preference-direct training beats RM-mediated GRPO and matches SFT.

    uv run python scripts/train_dpo.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset
import pandas as pd
from PIL import Image
from peft import LoraConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import DPOConfig, DPOTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_INSTRUCTION = (
    "You are a witty cartoon caption writer. Look at the cartoon image and "
    "output exactly one one-line caption inside <caption>...</caption> tags. "
    "Do not include any other text."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--bt-pairs", type=Path,
                   default=PROJECT_ROOT / "data" / "bt_pairs_train.parquet")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "checkpoints" / "qwen3vl-2b-dpo-instruct-nothink")
    p.add_argument("--max-pairs", type=int, default=2000,
                   help="Subset of BT pairs to train on (60k full would take hours)")
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--per-device-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=640)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--image-max-side", type=int, default=448)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="dpo-instruct-nothink")
    return p.parse_args()


def build_dataset(args) -> Dataset:
    df = pd.read_parquet(args.bt_pairs)
    if args.max_pairs and len(df) > args.max_pairs:
        df = df.sample(n=args.max_pairs, random_state=args.seed).reset_index(drop=True)
    print(f"DPO pairs: {len(df)}")

    rows = []
    for _, r in df.iterrows():
        img_path = args.data_root / r["image_path"]
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            if max(img.size) > args.image_max_side:
                img.thumbnail((args.image_max_side, args.image_max_side),
                              Image.LANCZOS)
        except Exception:
            continue
        chosen = f"<caption>{r['chosen']}</caption>"
        rejected = f"<caption>{r['rejected']}</caption>"
        # Normalize content to always be a list of typed parts so pyarrow
        # can infer a consistent schema across rows.
        rows.append({
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}]},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Write a funny one-line caption for this New Yorker-style cartoon."},
                ]},
            ],
            "chosen": [{"role": "assistant", "content": [{"type": "text", "text": chosen}]}],
            "rejected": [{"role": "assistant", "content": [{"type": "text", "text": rejected}]}],
            "images": [img],
        })
    print(f"valid rows: {len(rows)}")
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    peft = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Building dataset...")
    train_ds = build_dataset(args)

    cfg = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        beta=args.beta,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        report_to="wandb",
        run_name=args.run_name,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use a copy automatically when peft_config is set
        args=cfg,
        processing_class=processor,
        train_dataset=train_ds,
        peft_config=peft,
    )

    print("Starting DPO training...")
    trainer.train()

    save_path = args.output_dir / "lora_final"
    trainer.save_model(str(save_path))
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
