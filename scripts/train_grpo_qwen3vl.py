"""GRPO training of Qwen3-VL-2B-Thinking against the trained Bradley-Terry RM.

Pipeline:
  1. Policy = Qwen3-VL-2B-Thinking with LoRA, trained via TRL GRPOTrainer.
  2. Rollouts = vLLM colocate on the same GPU as the trainer (1 GPU each.)
  3. Rewards = combination of:
       - format reward: 1.0 if the completion contains <caption>...</caption>
       - humor reward:  scalar from the Bradley-Terry RM scoring the
                        extracted caption against the cartoon image+prompt
  4. Dataset = New Yorker caption_sft_train (one row per (image, prompt));
     we drop the SFT target caption since GRPO doesn't need it.
  5. The RM (Qwen2.5-VL-3B + LoRA + scalar head) lives on a separate GPU
     so it doesn't fight the policy for memory.

Notes:
  - Qwen3-VL-Thinking's chat template *automatically* prepends `<think>\\n`
    after the assistant turn, so reasoning is mandatory. The policy emits
    `<think>...</think>` then the visible response; we extract <caption>
    from that response.
  - RM and policy are different model families (Qwen2.5-VL-3B vs Qwen3-VL-2B);
    they don't share weights and don't need to.
  - Reward function is sync (calls into the RM via score_batch). For 4
    generations / step / 1 prompt, it's fast enough — bigger batches will
    want async or batched inference.

Run interactively (single node, multi-GPU):
    uv run python scripts/train_grpo_qwen3vl.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig
from PIL import Image
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent))
from reward_model import LoadedRewardModel, load_reward_model, score_batch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_MODEL_NAME = os.environ.get("POLICY_MODEL_NAME", "Qwen/Qwen3-VL-2B-Thinking")
REWARD_MODEL_DIR = Path(
    os.environ.get(
        "REWARD_MODEL_DIR",
        str(Path.home() / "scratch/humor-r1/final_reward_model"),
    )
)
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data")))
TRAIN_DATA_DIR = DATA_ROOT / "caption_sft_train"

CKPT_ROOT = Path(
    os.environ.get("CKPT_ROOT", str(Path.home() / "scratch/humor-r1/checkpoints"))
)
OUTPUT_DIR = CKPT_ROOT / "qwen3vl-2b-grpo-newyorker"

REWARD_GPU = int(os.environ.get("REWARD_GPU", "1"))  # second visible A100
USE_VLLM = os.environ.get("USE_VLLM", "1") == "1"

# When CUDA_VISIBLE_DEVICES is not set, default to using GPU 0 (policy) and
# GPU 1 (RM) on the local machine. Override by exporting CUDA_VISIBLE_DEVICES
# upstream — REWARD_GPU is then an index into the *visible* devices.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
        "DEFAULT_VISIBLE", "0,1"
    )

LORA_RANK = int(os.environ.get("LORA_RANK", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))

# Thinking budget. Qwen3-VL-Thinking is verbose — at 768 we observed
# clipped_ratio=1.0 (every completion ran out before closing </think>).
# Give it real room to reason and emit the caption.
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "2048"))
# Image patches dominate the prompt length. Empirically a 600x600 cartoon
# at the processor's default max_pixels expanded to ~9k tokens. We resize
# images to IMAGE_MAX_SIDE before passing to the processor (see
# build_dataset()), which knocks the image to ~1k tokens.
IMAGE_MAX_SIDE = int(os.environ.get("IMAGE_MAX_SIDE", "448"))
MAX_MODEL_LENGTH = int(os.environ.get("MAX_MODEL_LENGTH", "4096"))

SYSTEM_INSTRUCTION = (
    "You are a witty cartoon caption writer. Look at the cartoon image and "
    "any provided context, then briefly identify what makes the scene funny. "
    "Keep your thinking to a few short sentences — do not enumerate many "
    "options. End your reasoning, then output exactly one one-line caption "
    "inside <caption>...</caption> tags. The text inside the tags is the "
    "final caption; nothing else counts."
)

CAPTION_RE = re.compile(r"<caption>(.*?)</caption>", re.DOTALL)


# Singleton-ish RM handle. Loaded once in main() and closed-over by the reward fns.
_RM: LoadedRewardModel | None = None


def extract_caption(completion_text: str) -> str | None:
    """Pull the contents of the first <caption>...</caption> block, or None."""
    match = CAPTION_RE.search(completion_text)
    if match:
        text = match.group(1).strip()
        return text or None
    return None


def completion_to_text(completion: Any) -> str:
    """GRPO passes completions as either plain strings or message lists."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return str(completion)


def format_reward(completions, **kwargs) -> list[float]:
    """1.0 if the completion has a non-empty <caption> block, else 0.0."""
    return [
        1.0 if extract_caption(completion_to_text(c)) else 0.0
        for c in completions
    ]


def _resolve_image_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else DATA_ROOT / p


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = 2.718281828459045 ** (-x)
        return 1.0 / (1.0 + z)
    z = 2.718281828459045 ** x
    return z / (1.0 + z)


def humor_reward(completions, image_path, prompt_text, **kwargs) -> list[float | None]:
    """Bradley-Terry reward in [0, 1]: sigmoid(RM_score(caption | image, prompt)).

    Returns None for completions without a <caption> tag; TRL converts those
    to NaN and `nansum` skips them when aggregating with the format reward.
    Net effect: missing-caption completions are scored only by the format
    reward (= 0), captioned completions get format=1 plus humor.

    Raw RM scores can be very negative; sigmoid keeps the humor reward in
    [0, 1] so a bad caption is always at least as good as no caption.

    `image_path` and `prompt_text` are forwarded by TRL from the dataset
    columns of the same name (each list is aligned with `completions`).
    """
    assert _RM is not None, "Reward model must be loaded before humor_reward is called."

    images: list[Image.Image] = []
    prompts: list[str] = []
    captions: list[str] = []
    keep_indices: list[int] = []

    for i, (completion, img_path, prompt) in enumerate(
        zip(completions, image_path, prompt_text, strict=True)
    ):
        text = completion_to_text(completion)
        caption = extract_caption(text)
        if caption is None:
            continue
        try:
            image = Image.open(_resolve_image_path(img_path)).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        images.append(image)
        prompts.append(str(prompt))
        captions.append(caption)
        keep_indices.append(i)

    rewards: list[float | None] = [None] * len(completions)
    if images:
        scores = score_batch(_RM, images, prompts, captions)
        for idx, s in zip(keep_indices, scores):
            rewards[idx] = _sigmoid(float(s))
    return rewards


def build_dataset() -> Dataset:
    """Load caption_sft_train and reshape into GRPO prompts.

    Multiple captions per cartoon collapse to one prompt-only row per cartoon.
    """
    if not TRAIN_DATA_DIR.exists():
        raise FileNotFoundError(
            f"{TRAIN_DATA_DIR} not found. Run scripts/download_data.py first."
        )

    print("  loading caption_sft_train from disk...", flush=True)
    sft = load_from_disk(str(TRAIN_DATA_DIR))
    print(f"  loaded {len(sft)} rows", flush=True)

    print("  collapsing to one row per contest...", flush=True)
    seen: dict[int, dict[str, Any]] = {}
    for row in sft:
        contest = int(row["contest_number"])
        if contest in seen:
            continue
        seen[contest] = {
            "image_path": row["image_path"],
            "prompt_text": row["prompt"],
        }
    print(f"  unique contests: {len(seen)}", flush=True)

    # Note: we do NOT preload PIL images into a column. Storing PIL.Image in
    # Dataset.from_list triggers HF Datasets to PNG-encode each image at
    # construction time, which is single-threaded and slow (~10s per call
    # for hundreds of images, on top of pyarrow overhead). Instead we keep
    # only the path; load_image_column() below uses set_transform to open
    # images on-demand inside the dataloader.
    print("  building rows (paths only)...", flush=True)
    rows = []
    for contest, info in seen.items():
        image_path = _resolve_image_path(info["image_path"])
        if not image_path.exists():
            continue

        # TRL injects an image placeholder before the first user message
        # using prepare_multimodal_messages when an `image` column exists;
        # see trl/data_utils.py.
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": info["prompt_text"]},
        ]

        rows.append(
            {
                "prompt": messages,
                "image_path": str(image_path),
                "prompt_text": info["prompt_text"],
                "contest_number": contest,
            }
        )

    print(f"  building Dataset from {len(rows)} rows...", flush=True)
    dataset = Dataset.from_list(rows)

    # Load images lazily and downscale them — this avoids the slow PNG
    # round-trip that Dataset.from_list does for PIL.Image columns and
    # caps the image-patch count fed to the policy. Default IMAGE_MAX_SIDE
    # of 448 produces ~1k image tokens; the processor's default of
    # 12.8M pixels would balloon to 6-8k image tokens for 600px cartoons.
    def _add_image(batch):
        images = []
        for p in batch["image_path"]:
            img = Image.open(p).convert("RGB")
            if max(img.size) > IMAGE_MAX_SIDE:
                img.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.LANCZOS)
            images.append(img)
        batch["image"] = images
        return batch

    print("  attaching lazy image loader...", flush=True)
    dataset.set_transform(_add_image)
    print("  done", flush=True)
    return dataset


def main() -> int:
    global _RM
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    print(f"Loading reward model on cuda:{REWARD_GPU}...")
    _RM = load_reward_model(
        REWARD_MODEL_DIR,
        dtype=torch.bfloat16,
        device=f"cuda:{REWARD_GPU}",
    )
    print(f"  RM base: {_RM.base_model_name}")

    print("Building training dataset...")
    train_dataset = build_dataset()
    print(f"  unique cartoons: {len(train_dataset)}")
    if len(train_dataset) == 0:
        raise RuntimeError("No training examples — check data/ symlink.")

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        # Preserve `image_path` / `prompt_text` columns so the reward fn sees them.
        # Default is True, which would drop everything outside ["prompt","image","images"].
        remove_unused_columns=False,
        model_init_kwargs={"dtype": "bfloat16"},
        # Optimizer / schedule. LoRA RL benefits from ~10x typical FullFT LR
        # (thinkingmachines.ai/blog/lora).
        learning_rate=5e-5,
        weight_decay=0.001,
        lr_scheduler_type="constant",
        optim="adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        # GRPO knobs.
        temperature=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=int(os.environ.get("MAX_STEPS", "500")),
        save_steps=200,
        logging_steps=1,
        # KL stays modest — Qwen3-VL-Thinking produces structured think/answer
        # blocks already, so we don't want to drift far from that prior.
        beta=0.04,
        epsilon=0.2,
        epsilon_high=0.28,
        delta=1.5,
        loss_type="cispo",
        mask_truncated_completions=True,
        # Reward weights: format is a strong scaffold (it fully covers the
        # no-caption case via NaN propagation in nansum); humor is the actual
        # signal in [0, 1]. The two are roughly comparable in magnitude so
        # a small humor bump can outweigh formatting noise once the policy
        # reliably emits tags.
        reward_weights=[1.0, 1.0],
        # Rollouts via vLLM colocate on the trainer's GPU.
        use_vllm=USE_VLLM,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
        vllm_max_model_length=MAX_MODEL_LENGTH,
        # Logging.
        report_to="wandb",
        run_name=os.environ.get("WANDB_RUN_NAME", "qwen3vl-2b-grpo-newyorker"),
        logging_first_step=True,
        log_completions=True,
        num_completions_to_print=2,
    )

    trainer = GRPOTrainer(
        model=POLICY_MODEL_NAME,
        reward_funcs=[format_reward, humor_reward],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )
    trainer.train()

    lora_dir = OUTPUT_DIR / "lora_final"
    trainer.model.save_pretrained(str(lora_dir))
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(str(lora_dir))
    print(f"Saved policy LoRA adapters to {lora_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
