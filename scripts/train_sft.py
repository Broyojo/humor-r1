"""SFT cold-start for the Qwen3-VL-2B-Thinking captioning policy.

Trains the same LoRA architecture as the GRPO policy on the 813
high-rated cartoon captions in `data/caption_sft_train`. Targets the
output format `<caption>X</caption>` so the GRPO step can refine
quality from a model that already knows the format and the New Yorker
caption distribution.

We use Qwen3-VL-Thinking's chat template with `enable_thinking=False`
during SFT (no thinking trace required) and let GRPO turn thinking on
afterward — the Thinking variant has both modes baked in.

Run:
    uv run python scripts/train_sft.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-2B-Thinking")
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints/qwen3vl-2b-sft-newyorker"

SYSTEM_INSTRUCTION = (
    "You are a witty cartoon caption writer. Look at the cartoon image and "
    "any provided context, then output exactly one one-line caption inside "
    "<caption>...</caption> tags."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--train-data", type=Path, default=DEFAULT_DATA_DIR / "caption_sft_train")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--num-train-epochs", type=float, default=2.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lr-scheduler-type", default="constant")
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--image-max-side", type=int, default=448)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--report-to", default="wandb")
    p.add_argument("--run-name", default=os.environ.get("WANDB_RUN_NAME", "qwen3vl-2b-sft-newyorker"))
    return p.parse_args()


def build_messages(scene_description: str, scene_twist: str, location: str,
                   entities: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]


@dataclass
class SFTCollator:
    processor: Any
    data_root: Path
    image_max_side: int
    max_length: int

    def _load_image(self, image_path: str) -> Image.Image:
        path = Path(image_path)
        if not path.is_absolute():
            path = self.data_root / path
        img = Image.open(path).convert("RGB")
        if max(img.size) > self.image_max_side:
            img.thumbnail((self.image_max_side, self.image_max_side), Image.LANCZOS)
        return img

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Per-example: encode (system+user+assistant=caption) with the chat
        # template in one shot so image-token positions are correct, then mask
        # everything before the assistant content out of the loss.

        per_examples: list[dict] = []
        prompt_lens: list[int] = []  # number of leading tokens to mask

        tokenizer = self.processor.tokenizer
        eos_id = tokenizer.eos_token_id

        for ex in features:
            image = self._load_image(ex["image_path"])
            base_msgs = build_messages(
                ex["scene_description"], ex["scene_twist"],
                ex["location"], ex["entities"], ex["prompt"],
            )
            target = f"<caption>{ex['caption']}</caption>"

            # 1) Render the prompt text only (assistant turn opened).
            prompt_text = self.processor.apply_chat_template(
                base_msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            # 2) Tokenize prompt+image with the processor to get aligned
            #    pixel_values + input_ids.
            prompt_inputs = self.processor(
                text=[prompt_text], images=[image],
                padding=False, truncation=False, return_tensors="pt",
            )
            prompt_ids = prompt_inputs["input_ids"][0]

            # Append the assistant content + EOS as plain text tokens.
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            full_ids = torch.cat(
                [prompt_ids, torch.tensor(target_ids + [eos_id], dtype=torch.long)],
                dim=0,
            )
            attn = torch.ones_like(full_ids)
            labels = full_ids.clone()
            labels[: len(prompt_ids)] = -100

            # Truncate from the right (prompt + caption fits well under
            # max_length on 448 px images, but be defensive).
            if full_ids.size(0) > self.max_length:
                full_ids = full_ids[: self.max_length]
                attn = attn[: self.max_length]
                labels = labels[: self.max_length]

            # Extend mm_token_type_ids by zeros for the appended target
            # tokens (those are not multimodal).
            mm_ids = None
            if "mm_token_type_ids" in prompt_inputs:
                p_mm = prompt_inputs["mm_token_type_ids"][0]
                tail = torch.zeros(
                    full_ids.size(0) - p_mm.size(0), dtype=p_mm.dtype
                )
                mm_ids = torch.cat([p_mm, tail], dim=0)

            per_examples.append({
                "input_ids": full_ids,
                "attention_mask": attn,
                "labels": labels,
                "mm_token_type_ids": mm_ids,
                "pixel_values": prompt_inputs["pixel_values"],
                "image_grid_thw": prompt_inputs["image_grid_thw"],
            })
            prompt_lens.append(int(prompt_inputs["input_ids"].shape[1]))

        # Pad token-level fields to the longest in the batch.
        max_len = max(ex["input_ids"].size(0) for ex in per_examples)
        pad_id = tokenizer.pad_token_id or eos_id
        B = len(per_examples)

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        has_mm = any(ex.get("mm_token_type_ids") is not None for ex in per_examples)
        mm_token_type_ids = (
            torch.zeros((B, max_len), dtype=torch.long) if has_mm else None
        )
        for i, ex in enumerate(per_examples):
            n = ex["input_ids"].size(0)
            input_ids[i, :n] = ex["input_ids"]
            attention_mask[i, :n] = ex["attention_mask"]
            labels[i, :n] = ex["labels"]
            if mm_token_type_ids is not None and ex["mm_token_type_ids"] is not None:
                mm_token_type_ids[i, :n] = ex["mm_token_type_ids"]

        pixel_values = torch.cat([ex["pixel_values"] for ex in per_examples], dim=0)
        image_grid_thw = torch.cat([ex["image_grid_thw"] for ex in per_examples], dim=0)

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        if mm_token_type_ids is not None:
            out["mm_token_type_ids"] = mm_token_type_ids
        return out


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    print(f"loading processor + model: {args.model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print(f"loading dataset from {args.train_data}", flush=True)
    train_ds = load_from_disk(str(args.train_data))
    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    print(f"  train rows: {len(train_ds)}", flush=True)

    collator = SFTCollator(
        processor=processor,
        data_root=args.data_root,
        image_max_side=args.image_max_side,
        max_length=args.max_length,
    )

    optim_name = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        remove_unused_columns=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=[] if str(args.report_to).lower() == "none" else args.report_to,
        run_name=args.run_name,
        dataloader_num_workers=2,
        optim=optim_name,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()

    final = args.output_dir / "lora_final"
    trainer.model.save_pretrained(str(final))
    processor.save_pretrained(str(final))
    print(f"saved SFT LoRA to {final}", flush=True)


if __name__ == "__main__":
    main()
