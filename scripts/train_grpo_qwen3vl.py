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
        str(PROJECT_ROOT / "checkpoints/rm-baseline-20k-fa2/final_reward_model"),
    )
)
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data")))
TRAIN_DATA_DIR = DATA_ROOT / "caption_sft_train"

CKPT_ROOT = Path(
    os.environ.get("CKPT_ROOT", str(PROJECT_ROOT / "checkpoints"))
)
OUTPUT_DIR = CKPT_ROOT / "qwen3vl-2b-grpo-newyorker"

# Single-A100 setup: RM colocates with the policy on GPU 0. Override
# REWARD_GPU=1 if you have a second card.
REWARD_GPU = int(os.environ.get("REWARD_GPU", "0"))
USE_VLLM = os.environ.get("USE_VLLM", "1") == "1"

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("DEFAULT_VISIBLE", "0")

LORA_RANK = int(os.environ.get("LORA_RANK", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))

# Thinking budget. Qwen3-VL-Thinking is verbose — at 768 we observed
# clipped_ratio=1.0 (every completion ran out before closing </think>).
# Give it real room to reason and emit the caption.
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "3072"))
# Image patches dominate the prompt length. Empirically a 600x600 cartoon
# at the processor's default max_pixels expanded to ~9k tokens. We resize
# images to IMAGE_MAX_SIDE before passing to the processor (see
# build_dataset()), which knocks the image to ~1k tokens.
IMAGE_MAX_SIDE = int(os.environ.get("IMAGE_MAX_SIDE", "448"))
MAX_MODEL_LENGTH = int(os.environ.get("MAX_MODEL_LENGTH", "5120"))

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


def humor_reward(completions, image_path, prompt_text, **kwargs) -> list[float]:
    """Single combined reward in [0, 1]:
        - 0.0 if no <caption>...</caption> block (encodes the format scaffold)
        - sigmoid(RM_score(caption | image, prompt)) otherwise

    This replaces the earlier (format_reward, humor_reward) pair. The format
    reward was a scaffold the policy maxed out within ~50 steps and then
    contributed zero gradient; folding the format check into "no caption -> 0"
    keeps the reward in [0, 1] (friendly to GRPO's group-relative advantage
    normalization) and still teaches the format via the implicit floor.

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

    # Reward shaping (env-tunable):
    #   USE_SIGMOID=1 (default):
    #     - no caption:  reward = 0.0
    #     - has caption: reward = FORMAT_BONUS + (1-FORMAT_BONUS) * sigmoid(RM)
    #   USE_SIGMOID=0:
    #     - no caption:  reward = NO_CAPTION_PENALTY (default -5.0)
    #     - has caption: reward = raw RM_score
    #
    # We default to sigmoid for stability, but raw RM gives ~20× larger
    # within-group reward variance, so the GRPO advantages are non-trivial
    # even when the policy is producing similar-quality captions across the
    # group. Use raw RM when training reward stalls under sigmoid.
    use_sigmoid = os.environ.get("USE_SIGMOID", "1") == "1"
    format_bonus = float(os.environ.get("FORMAT_BONUS", "0.0"))
    no_cap_penalty = float(os.environ.get("NO_CAPTION_PENALTY", "-5.0"))
    if use_sigmoid:
        rewards: list[float] = [0.0] * len(completions)
    else:
        rewards = [no_cap_penalty] * len(completions)
    if images:
        scores = score_batch(_RM, images, prompts, captions)
        for idx, s in zip(keep_indices, scores):
            if use_sigmoid:
                rewards[idx] = format_bonus + (1.0 - format_bonus) * _sigmoid(float(s))
            else:
                rewards[idx] = float(s)
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
            # rm_prompt: the full prompt with Scene/Twist/Location/Entities
            # context — the RM was trained with this and expects it.
            "rm_prompt": row["prompt"],
        }
    print(f"  unique contests: {len(seen)}", flush=True)

    # Policy prompt: NO scene description / twist / location / entities. The
    # policy is a vision-language model — it should be looking at the image,
    # not reading a textual description of the image. We pass the same
    # generic instruction for every cartoon so the policy can't shortcut
    # via the prompt text.
    POLICY_USER_PROMPT = (
        "Write a funny one-line caption for this New Yorker-style cartoon."
    )

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
            {"role": "user", "content": POLICY_USER_PROMPT},
        ]

        rows.append(
            {
                "prompt": messages,
                "image_path": str(image_path),
                # `prompt_text` is what humor_reward forwards to the RM —
                # keep the full description-augmented prompt the RM was
                # trained on, NOT the clean policy prompt. This way the RM
                # scores captions in the distribution it was calibrated for.
                "prompt_text": info["rm_prompt"],
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
        model_init_kwargs={
            "dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
        },
        # Optimizer / schedule — match the RM training recipe.
        # LoRA RL benefits from ~10x typical FullFT LR (thinkingmachines.ai/blog/lora).
        learning_rate=float(os.environ.get("LR", "2e-4")),
        weight_decay=0.0,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        optim="adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        # GRPO knobs. eff_prompts_per_step = per_device * accum = 8.
        # num_generations=8 → 64 rollouts per optimizer step.
        temperature=1.0,
        per_device_train_batch_size=int(os.environ.get("PER_DEVICE_BATCH", "1")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "8")),
        num_generations=int(os.environ.get("NUM_GENERATIONS", "8")),
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=int(os.environ.get("MAX_STEPS", "500")),
        save_steps=int(os.environ.get("SAVE_STEPS", "100")),
        logging_steps=1,
        # KL stays modest — Qwen3-VL-Thinking produces structured think/answer
        # blocks already, so we don't want to drift far from that prior.
        beta=0.04,
        epsilon=0.2,
        epsilon_high=0.28,
        delta=1.5,
        loss_type="cispo",
        # We end completions via the </caption> stop string (vLLM SamplingParams.stop)
        # so they don't emit EOS and TRL flags them all as "truncated". Disable
        # the mask so we keep the gradient signal from those completions.
        mask_truncated_completions=False,
        # Single combined reward (humor_reward); see its docstring.
        # No reward_weights needed when there's a single reward function.
        # Stop generation as soon as the closing </caption> tag is emitted —
        # otherwise Qwen3-VL-Thinking rambles through the whole 3072-token
        # budget without committing. include_stop_str_in_output keeps the
        # closing tag in the completion so our `<caption>(.*?)</caption>`
        # regex still matches.
        generation_kwargs={
            "stop": ["</caption>"],
            "include_stop_str_in_output": True,
        },
        # Rollouts via vLLM colocate on the trainer's GPU.
        # When the RM lives on the same GPU (single-A100 setup), drop this to
        # ~0.30 to leave room for the RM's weights+activations.
        use_vllm=USE_VLLM,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=float(os.environ.get("VLLM_MEM", "0.30")),
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
        reward_funcs=humor_reward,
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
