"""Train a Qwen3-VL reward model on Bradley-Terry caption pairs.

This script fine-tunes a Qwen3-VL vision-language backbone with a scalar reward
head. Each training example contains the same cartoon image and prompt paired
with a `chosen` and `rejected` caption. Training optimizes the Bradley-Terry
pairwise preference loss:

    -log(sigmoid(score(chosen) - score(rejected)))

The learned reward model can later be used for ranking, reranking, or as a
reward signal in downstream RLHF/RLAIF-style training.

Why a custom Trainer/model?
- This is a multimodal model.
- We want a scalar reward head on top of the text hidden states.
- Standard text-only reward model trainers do not directly fit this setup.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch import nn
from transformers import (
    AutoModel,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")
DEFAULT_DATA_DIR = Path("./data")
CKPT_ROOT = Path(os.environ.get("CKPT_ROOT", str(PROJECT_ROOT / "checkpoints")))
DEFAULT_OUTPUT_DIR = CKPT_ROOT / "qwen3-vl-reward-model"
DEFAULT_MAX_LENGTH = 384
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_MAX_TRAIN_SAMPLES = 20_000
DEFAULT_MAX_EVAL_SAMPLES = 2_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Qwen3-VL reward model on Bradley-Terry caption pairs."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Base Qwen3-VL checkpoint.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_DATA_DIR / "bt_pairs_train.parquet",
        help="Training Bradley-Terry pairs parquet file.",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_DATA_DIR / "bt_pairs_validation.parquet",
        help="Validation Bradley-Terry pairs parquet file.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory used to resolve relative image paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints and final adapters.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum sequence length after tokenization.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging frequency.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Evaluation frequency in steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Checkpoint save frequency in steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Max number of checkpoints to keep.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=DEFAULT_MAX_TRAIN_SAMPLES,
        help="Cap on the number of training pairs to use for faster iteration.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=DEFAULT_MAX_EVAL_SAMPLES,
        help="Cap on the number of eval pairs to use for faster iteration.",
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("DTYPE", "bfloat16"),
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation passed to `from_pretrained`.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=DEFAULT_LORA_RANK,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=DEFAULT_LORA_ALPHA,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing.",
    )
    parser.add_argument(
        "--report-to",
        default="wandb",
        help="Trainer `report_to` target. Use `none` to disable logging backends.",
    )
    parser.add_argument(
        "--run-name",
        default=os.environ.get("WANDB_RUN_NAME", "qwen3-vl-reward-model"),
        help="Run name for logging.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        default="cosine",
        help="LR scheduler (cosine|constant|linear|...).",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (default 1.0).",
    )
    parser.add_argument(
        "--image-max-side",
        type=int,
        default=448,
        help="Resize cartoons so the long edge is this many px before processing.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def build_messages(prompt: str, caption: str) -> list[dict[str, Any]]:
    # `prompt` is intentionally ignored — the original dataset prompt carries
    # GPT-4o-generated Scene/Twist/Location/Entities annotations that are not
    # available for OOD cartoons. We want the RM to score (image, caption)
    # using just the image so it generalizes to any single-panel cartoon.
    text = (
        "Write a funny one-line caption for this New Yorker-style cartoon.\n\n"
        f"Candidate caption: {caption}\n\n"
        "Judge how funny this caption is for the cartoon."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]


def infer_hidden_size(config: Any) -> int:
    candidates = [
        getattr(config, "hidden_size", None),
        getattr(getattr(config, "text_config", None), "hidden_size", None),
        getattr(getattr(config, "language_config", None), "hidden_size", None),
        getattr(getattr(config, "llm_config", None), "hidden_size", None),
    ]
    for value in candidates:
        if value is not None:
            return int(value)

    raise ValueError(
        f"Could not infer hidden size from config type {type(config).__name__}: {config}"
    )


def extract_last_hidden_state(outputs: Any) -> torch.Tensor:
    """Best-effort extraction of token-level hidden states from model outputs.

    Different multimodal models sometimes expose hidden states slightly
    differently. This helper tries the common locations first.
    """
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state

    if isinstance(outputs, dict):
        if "last_hidden_state" in outputs and outputs["last_hidden_state"] is not None:
            return outputs["last_hidden_state"]

        for key in ["hidden_states", "text_hidden_states"]:
            if key in outputs and outputs[key] is not None:
                value = outputs[key]
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    return value[-1]
                if torch.is_tensor(value):
                    return value

    for attr in ["hidden_states", "text_hidden_states"]:
        if hasattr(outputs, attr):
            value = getattr(outputs, attr)
            if value is None:
                continue
            if isinstance(value, (list, tuple)) and len(value) > 0:
                return value[-1]
            if torch.is_tensor(value):
                return value

    if isinstance(outputs, (list, tuple)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
        return outputs[0]

    raise ValueError(
        f"Could not extract last hidden state from outputs of type {type(outputs).__name__}"
    )


@dataclass
class PreferenceCollator:
    processor: Any
    data_root: Path
    max_length: int
    text_budget: int = 160
    image_max_side: int = 448

    def _load_image(self, image_path: str) -> Image.Image:
        path = Path(image_path)
        if not path.is_absolute():
            path = self.data_root / path
        img = Image.open(path).convert("RGB")
        # Resize so the long edge is `image_max_side`. The Qwen2.5-VL
        # processor emits ~1 image token per (28*28) pixel patch; without
        # this knock-down a 600x500 cartoon expands to ~750 image tokens.
        # Capping the long edge at 448 brings it to ~256 tokens — cuts
        # forward time in roughly half on a 3B VLM.
        if max(img.size) > self.image_max_side:
            img.thumbnail((self.image_max_side, self.image_max_side), Image.LANCZOS)
        return img

    def _truncate_text_segment(self, text: str, max_tokens: int) -> str:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return text

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
        )
        return tokenizer.decode(token_ids, skip_special_tokens=False)

    def _encode_batch(self, texts: list[str], images: list[Image.Image]) -> dict[str, torch.Tensor]:
        model_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        return dict(model_inputs)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = []
        images = []

        for example in features:
            image = self._load_image(example["image_path"])
            for caption_key in ("chosen", "rejected"):
                prompt = self._truncate_text_segment(str(example["prompt"]), self.text_budget)
                caption = self._truncate_text_segment(
                    str(example[caption_key]),
                    max(32, self.text_budget // 2),
                )
                messages = build_messages(prompt, caption)
                texts.append(
                    self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
                images.append(image.copy())

        batch = self._encode_batch(texts, images)
        batch["pair_batch_size"] = torch.tensor(len(features), dtype=torch.long)
        return batch


class Qwen3VLRewardModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype,
        attn_implementation: str,
        lora_r: int,
        lora_alpha: int,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        self.model_name = model_name

        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )

        hidden_size = infer_hidden_size(self.backbone.config)
        self.score_head = nn.Linear(hidden_size, 1, bias=False)
        # Zero-init: initial reward = 0 for everything, so BT loss starts at
        # log(2) and gradient signal is purely about ranking. Default Kaiming
        # init gave initial rewards ~50 in earlier runs and the optimizer
        # spent the first many steps just rescaling the head.
        with torch.no_grad():
            self.score_head.weight.zero_()

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        if gradient_checkpointing:
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()

    def _pool_last_text_token(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        last_token_idx = attention_mask.long().sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, last_token_idx]
        return pooled

    def _score(self, **model_inputs) -> torch.Tensor:
        outputs = self.backbone(
            **model_inputs,
            return_dict=True,
        )

        hidden_states = extract_last_hidden_state(outputs)
        attention_mask = model_inputs["attention_mask"]
        pooled = self._pool_last_text_token(hidden_states, attention_mask)
        pooled = pooled.to(self.score_head.weight.dtype)
        rewards = self.score_head(pooled).squeeze(-1)
        return rewards

    def forward(self, **batch):
        batch.pop("pair_batch_size", None)
        rewards = self._score(**batch)
        chosen_rewards = rewards[0::2]
        rejected_rewards = rewards[1::2]
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        return {
            "loss": loss,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
        }

    def save_pretrained(self, save_directory: str | os.PathLike[str]):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        backbone_dir = save_dir / "backbone_adapter"
        self.backbone.save_pretrained(backbone_dir)
        torch.save(self.score_head.state_dict(), save_dir / "reward_head.pt")

        with open(save_dir / "reward_model_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_model_name": self.model_name,
                    "score_head_shape": list(self.score_head.weight.shape),
                },
                f,
                indent=2,
            )


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["loss"].detach()

        if prediction_loss_only:
            return loss, None, None

        predictions = torch.stack(
            [outputs["chosen_rewards"], outputs["rejected_rewards"]],
            dim=1,
        ).detach()
        return loss, predictions, None


def compute_metrics(eval_prediction):
    predictions = eval_prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    chosen = predictions[:, 0]
    rejected = predictions[:, 1]

    accuracy = float(np.mean(chosen > rejected))
    margin = float(np.mean(chosen - rejected))
    return {
        "preference_accuracy": accuracy,
        "reward_margin": margin,
    }


def load_pair_dataset(path: Path, max_samples: int | None):
    dataset = load_dataset("parquet", data_files=str(path))["train"]
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset
    
def _resolve_image_path(data_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if not path.is_absolute():
        path = data_root / path
    return path


@torch.no_grad()
def score_single_example(
    model: Qwen3VLRewardModel,
    processor: Any,
    data_root: Path,
    prompt: str,
    caption: str,
    image_path: str,
    device: torch.device,
    text_budget: int = 160,
) -> float:
    tokenizer = getattr(processor, "tokenizer", None)

    def truncate_text_segment(text: str, max_tokens: int) -> str:
        if tokenizer is None:
            return text
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
        )
        return tokenizer.decode(token_ids, skip_special_tokens=False)

    prompt = truncate_text_segment(str(prompt), text_budget)
    caption = truncate_text_segment(str(caption), max(32, text_budget // 2))

    image = Image.open(_resolve_image_path(data_root, image_path)).convert("RGB")

    messages = build_messages(prompt, caption)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    model_inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    model_inputs = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in model_inputs.items()
    }

    reward = model._score(**model_inputs)
    return float(reward.item())


@torch.no_grad()
def evaluate_reward_model(
    model: Qwen3VLRewardModel,
    processor: Any,
    dataset,
    data_root: Path,
    text_budget: int = 160,
) -> dict[str, float]:
    device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()

    chosen_scores = []
    rejected_scores = []

    for example in dataset:
        chosen_score = score_single_example(
            model=model,
            processor=processor,
            data_root=data_root,
            prompt=example["prompt"],
            caption=example["chosen"],
            image_path=example["image_path"],
            device=device,
            text_budget=text_budget,
        )
        rejected_score = score_single_example(
            model=model,
            processor=processor,
            data_root=data_root,
            prompt=example["prompt"],
            caption=example["rejected"],
            image_path=example["image_path"],
            device=device,
            text_budget=text_budget,
        )

        chosen_scores.append(chosen_score)
        rejected_scores.append(rejected_score)

    chosen_scores = np.array(chosen_scores, dtype=np.float64)
    rejected_scores = np.array(rejected_scores, dtype=np.float64)
    margins = chosen_scores - rejected_scores

    preference_accuracy = float(np.mean(margins > 0.0))
    reward_margin = float(np.mean(margins))
    bt_loss = float(np.mean(-np.log(1.0 / (1.0 + np.exp(-margins)))))

    metrics = {
        "preference_accuracy": preference_accuracy,
        "reward_margin": reward_margin,
        "bt_loss": bt_loss,
        "num_examples": int(len(chosen_scores)),
    }

    if model_was_training:
        model.train()

    return metrics


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    train_dataset = load_pair_dataset(args.train_file, args.max_train_samples)

    eval_dataset = None
    if args.eval_file.exists():
        eval_dataset = load_pair_dataset(args.eval_file, args.max_eval_samples)

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if getattr(processor, "tokenizer", None) is not None:
        tokenizer = processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3VLRewardModel(
        model_name=args.model_name,
        dtype=get_torch_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
    )

    optim_name = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        remove_unused_columns=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        report_to=[] if str(args.report_to).lower() == "none" else args.report_to,
        run_name=args.run_name,
        dataloader_num_workers=4,
        gradient_checkpointing=False,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        optim=optim_name,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
    )

    collator = PreferenceCollator(
        processor=processor,
        data_root=args.data_root,
        max_length=args.max_length,
        image_max_side=args.image_max_side,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    trainer.train()

    if eval_dataset is not None:
        metrics = evaluate_reward_model(
            model=trainer.model,
            processor=processor,
            dataset=eval_dataset,
            data_root=args.data_root,
        )
        print("Manual evaluation metrics:")
        print(json.dumps(metrics, indent=2))

    final_dir = args.output_dir / "final_reward_model"
    trainer.model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir / "processor")
    print(f"Saved reward model artifacts to {final_dir}")



if __name__ == "__main__":
    main()
