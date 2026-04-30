"""Inference wrapper for the trained Bradley-Terry reward model.

Loads the artifacts written by `scripts/train_reward_model.py`:
  - `backbone_adapter/`       PEFT LoRA adapter on top of the base Qwen2.5-VL.
  - `reward_head.pt`          Linear(hidden_size, 1) state dict.
  - `reward_model_config.json` Records `base_model_name`.

Used by `scripts/eval_reward_model.py` for held-out preference accuracy and
by the GRPO trainer's reward callback to score policy rollouts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoProcessor


def _build_messages(prompt: str, caption: str) -> list[dict[str, Any]]:
    # `prompt` is intentionally ignored — must match training-time
    # build_messages in scripts/train_reward_model.py. The RM scores
    # (image, caption) using the image alone so it generalizes OOD.
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


def _truncate(text: str, tokenizer, max_tokens: int) -> str:
    if tokenizer is None:
        return text
    ids = tokenizer.encode(
        text, add_special_tokens=False, truncation=True, max_length=max_tokens
    )
    return tokenizer.decode(ids, skip_special_tokens=False)


def _extract_last_hidden_state(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if isinstance(outputs, dict) and outputs.get("last_hidden_state") is not None:
        return outputs["last_hidden_state"]
    for attr in ("hidden_states", "text_hidden_states"):
        value = getattr(outputs, attr, None) or (
            outputs.get(attr) if isinstance(outputs, dict) else None
        )
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and value:
            return value[-1]
        if torch.is_tensor(value):
            return value
    if isinstance(outputs, (list, tuple)) and outputs and torch.is_tensor(outputs[0]):
        return outputs[0]
    raise ValueError(f"Could not find last_hidden_state on {type(outputs).__name__}")


def _infer_hidden_size(config: Any) -> int:
    for cfg in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "language_config", None),
        getattr(config, "llm_config", None),
    ):
        size = getattr(cfg, "hidden_size", None) if cfg is not None else None
        if size is not None:
            return int(size)
    raise ValueError(f"Could not infer hidden size from {type(config).__name__}")


@dataclass
class LoadedRewardModel:
    backbone: nn.Module          # PEFT-wrapped base + adapter
    score_head: nn.Linear
    processor: Any
    base_model_name: str
    device: torch.device
    dtype: torch.dtype
    text_budget: int = 160

    def to(self, device: torch.device | str) -> "LoadedRewardModel":
        device = torch.device(device)
        self.backbone.to(device)
        self.score_head.to(device)
        self.device = device
        return self

    def eval(self) -> "LoadedRewardModel":
        self.backbone.eval()
        self.score_head.eval()
        return self


def load_reward_model(
    artifact_dir: str | Path,
    *,
    base_model_name: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    attn_implementation: str = "sdpa",
) -> LoadedRewardModel:
    """Materialize the RM on the requested device, ready for scoring."""
    artifact_dir = Path(artifact_dir)
    config_path = artifact_dir / "reward_model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    with config_path.open() as f:
        cfg = json.load(f)

    base_name = base_model_name or cfg["base_model_name"]

    backbone = AutoModel.from_pretrained(
        base_name,
        dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )

    adapter_dir = artifact_dir / "backbone_adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing {adapter_dir}")
    backbone = PeftModel.from_pretrained(backbone, str(adapter_dir), is_trainable=False)

    hidden_size = _infer_hidden_size(backbone.config)
    score_head = nn.Linear(hidden_size, 1, bias=False)
    score_head.load_state_dict(torch.load(artifact_dir / "reward_head.pt", map_location="cpu"))
    score_head = score_head.to(dtype=dtype)

    processor_dir = artifact_dir / "processor"
    has_processor_files = processor_dir.exists() and any(processor_dir.iterdir())
    processor_source = str(processor_dir) if has_processor_files else base_name
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    if getattr(processor, "tokenizer", None) is not None:
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    device = torch.device(device)
    backbone.to(device).eval()
    score_head.to(device).eval()

    return LoadedRewardModel(
        backbone=backbone,
        score_head=score_head,
        processor=processor,
        base_model_name=base_name,
        device=device,
        dtype=dtype,
    )


def _prepare_batch_inputs(
    rm: LoadedRewardModel,
    images: Sequence[Image.Image],
    prompts: Sequence[str],
    captions: Sequence[str],
) -> dict[str, torch.Tensor]:
    tokenizer = getattr(rm.processor, "tokenizer", None)
    texts = []
    pil_images = []
    for image, prompt, caption in zip(images, prompts, captions, strict=True):
        prompt = _truncate(str(prompt), tokenizer, rm.text_budget)
        caption = _truncate(str(caption), tokenizer, max(32, rm.text_budget // 2))
        messages = _build_messages(prompt, caption)
        texts.append(
            rm.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        )
        pil_images.append(image.convert("RGB") if image.mode != "RGB" else image)

    inputs = rm.processor(
        text=texts,
        images=pil_images,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    inputs = {
        k: (v.to(rm.device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }
    return inputs


@torch.no_grad()
def score(
    rm: LoadedRewardModel,
    image: Image.Image,
    prompt: str,
    caption: str,
) -> float:
    """Score a single (image, prompt, caption) triple. Returns scalar reward."""
    return score_batch(rm, [image], [prompt], [caption])[0]


@torch.no_grad()
def score_batch(
    rm: LoadedRewardModel,
    images: Sequence[Image.Image],
    prompts: Sequence[str],
    captions: Sequence[str],
) -> list[float]:
    """Score a batch and return per-item rewards. All sequences must align."""
    if not images:
        return []

    inputs = _prepare_batch_inputs(rm, images, prompts, captions)
    outputs = rm.backbone(**inputs, return_dict=True)
    hidden_states = _extract_last_hidden_state(outputs)
    attention_mask = inputs["attention_mask"]

    last_idx = attention_mask.long().sum(dim=1) - 1
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    pooled = hidden_states[batch_idx, last_idx].to(rm.score_head.weight.dtype)
    rewards = rm.score_head(pooled).squeeze(-1)
    return [float(r) for r in rewards.detach().cpu().tolist()]


@torch.no_grad()
def score_pair(
    rm: LoadedRewardModel,
    image: Image.Image,
    prompt: str,
    chosen: str,
    rejected: str,
) -> tuple[float, float]:
    """Score chosen + rejected for a single example (one batched forward pass)."""
    chosen_score, rejected_score = score_batch(
        rm,
        [image, image],
        [prompt, prompt],
        [chosen, rejected],
    )
    return chosen_score, rejected_score


@torch.no_grad()
def bt_loss(chosen: torch.Tensor | float, rejected: torch.Tensor | float) -> float:
    if not torch.is_tensor(chosen):
        chosen = torch.tensor(chosen, dtype=torch.float32)
    if not torch.is_tensor(rejected):
        rejected = torch.tensor(rejected, dtype=torch.float32)
    return float(-F.logsigmoid(chosen - rejected).mean())
