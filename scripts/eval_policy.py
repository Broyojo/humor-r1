"""Generate captions with the trained policy and score them with the BT reward model.

Compares two policies:
  - base: Qwen3-VL-2B-Thinking with no adapter
  - lora: same backbone with the GRPO LoRA adapter merged in

Uses vLLM for fast batched generation: all (cartoon × sample) prompts go in
one batched call per policy, instead of one HF generate() per sample. On 90
prompts × 2048 max tokens this is the difference between ~5 minutes and
~75 minutes.

Run after a training run lands at $CKPT_ROOT/qwen3vl-2b-grpo-newyorker/lora_final.

    uv run python scripts/eval_policy.py --num-samples 3 --max-cartoons 30
"""

from __future__ import annotations

import argparse
import json
import os
# vLLM forks a worker subprocess; if torch is imported before, the fork
# carries a tainted CUDA state and crashes ("Cannot re-initialize CUDA in
# forked subprocess"). Switch to spawn before any torch/vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, str(Path(__file__).parent))
from reward_model import load_reward_model, score_batch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LORA_DIR = Path(
    os.environ.get(
        "POLICY_LORA_DIR",
        str(PROJECT_ROOT / "checkpoints/qwen3vl-2b-grpo-newyorker/lora_final"),
    )
)
DEFAULT_RM_DIR = Path(
    os.environ.get(
        "REWARD_MODEL_DIR",
        str(PROJECT_ROOT / "checkpoints/rm-baseline-20k-fa2/final_reward_model"),
    )
)
DEFAULT_TEST_DATA = PROJECT_ROOT / "data" / "caption_sft_test"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

POLICY_MODEL_NAME = os.environ.get("POLICY_MODEL_NAME", "Qwen/Qwen3-VL-2B-Thinking")
SYSTEM_INSTRUCTION = (
    "You are a witty cartoon caption writer. Look at the cartoon image and "
    "any provided context, then briefly identify what makes the scene funny. "
    "Keep your thinking to a few short sentences — do not enumerate many "
    "options. End your reasoning, then output exactly one one-line caption "
    "inside <caption>...</caption> tags. The text inside the tags is the "
    "final caption; nothing else counts."
)
CAPTION_RE = re.compile(r"<caption>(.*?)</caption>", re.DOTALL)
IMAGE_MAX_SIDE = int(os.environ.get("IMAGE_MAX_SIDE", "448"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--reward-model-dir", type=Path, default=DEFAULT_RM_DIR)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-cartoons", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--policy-gpu", type=int, default=0)
    parser.add_argument("--reward-gpu", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=5120)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--skip-base", action="store_true",
                        help="Only evaluate the LoRA policy.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    p = Path(image_path)
    return p if p.is_absolute() else data_root / p


def maybe_resize(image: Image.Image) -> Image.Image:
    if max(image.size) > IMAGE_MAX_SIDE:
        image = image.copy()
        image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.LANCZOS)
    return image


def build_prompt_text(processor, prompt_text: str) -> str:
    """Apply chat template producing the prompt prefix (without the image
    payload, which vLLM passes through `multi_modal_data`)."""
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
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


def make_llm(model_name: str, lora_dir: Path | None, *,
             gpu_id: int, max_model_len: int, gpu_memory_utilization: float) -> LLM:
    # vLLM honors CUDA_VISIBLE_DEVICES; we set it before constructing.
    # `enable_lora=True` with `lora_dir` lets us load the adapter directly.
    kwargs: dict[str, Any] = dict(
        model=model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    if lora_dir is not None:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = 32
    llm = LLM(**kwargs)
    return llm


def generate_for_policy(
    label: str,
    llm: LLM,
    processor,
    cartoons: list[dict],
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    lora_dir: Path | None,
) -> list[dict]:
    print(f"\n=== Generating with policy: {label} ===", flush=True)
    prompt_text = [build_prompt_text(processor, c["prompt_text"]) for c in cartoons]
    images = [maybe_resize(Image.open(c["_image_path_resolved"]).convert("RGB"))
              for c in cartoons]

    # Build N=num_samples copies of each prompt so vLLM batches them all.
    prompts = []
    for i, c in enumerate(cartoons):
        for _ in range(num_samples):
            prompts.append({
                "prompt": prompt_text[i],
                "multi_modal_data": {"image": images[i]},
            })

    sampling = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_new_tokens,
        # Match training: stop the second </caption> closes so we don't burn
        # the whole budget on rambling that never commits.
        stop=["</caption>"],
        include_stop_str_in_output=True,
    )

    lora_request = None
    if lora_dir is not None:
        lora_request = LoRARequest("policy_lora", 1, str(lora_dir))

    print(f"  prompts queued: {len(prompts)}", flush=True)
    outputs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_request)

    rows = []
    for c_idx, c in enumerate(cartoons):
        completions = []
        for s in range(num_samples):
            out = outputs[c_idx * num_samples + s]
            completions.append(out.outputs[0].text)
        rows.append({
            "contest_number": c["contest_number"],
            "prompt_text": c["prompt_text"],
            "completions": completions,
            "_image_path_resolved": c["_image_path_resolved"],
        })
    return rows


def score_rows(rows: list[dict], rm, num_samples: int, label: str) -> dict:
    print(f"\n=== Scoring {label} captions with RM ===", flush=True)
    images: list[Image.Image] = []
    prompts: list[str] = []
    captions: list[str] = []
    placement: list[tuple[int, int]] = []  # (row_idx, sample_idx)

    for r_i, r in enumerate(rows):
        img = Image.open(resolve_image_path_rownext(r)).convert("RGB")
        for s_i, c_text in enumerate(r["completions"]):
            cap = extract_caption(c_text)
            r.setdefault("captions", []).append(cap)
            if cap is None:
                r.setdefault("scores", []).append(None)
                continue
            images.append(img)
            prompts.append(r["prompt_text"])
            captions.append(cap)
            placement.append((r_i, s_i))
            r.setdefault("scores", []).append(None)  # backfilled below

    if images:
        # Chunk to avoid OOM: a single big score_batch tries to allocate
        # `len(images) * seq_len * hidden_size` activations and OOMs once we
        # have ~90 captions with images attached on a single A100 sharing
        # memory with vLLM remnants.
        chunk = int(os.environ.get("SCORE_BATCH", "8"))
        scores: list[float] = []
        for i in range(0, len(images), chunk):
            scores.extend(
                score_batch(
                    rm,
                    images[i : i + chunk],
                    prompts[i : i + chunk],
                    captions[i : i + chunk],
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for (r_i, s_i), sc in zip(placement, scores, strict=True):
            rows[r_i]["scores"][s_i] = float(sc)

    valid = [s for r in rows for s in r["scores"] if s is not None]
    fail_zero = [(s if s is not None else 0.0) for r in rows for s in r["scores"]]
    emitted = sum(1 for r in rows for c in r["captions"] if c is not None)
    total = sum(len(r["captions"]) for r in rows)

    metrics = {
        "label": label,
        "num_cartoons": len(rows),
        "num_samples_per_cartoon": num_samples,
        "format_emit_rate": emitted / max(total, 1),
        "rm_score_mean_emitted": float(np.mean(valid)) if valid else float("nan"),
        "rm_score_std_emitted": float(np.std(valid)) if valid else float("nan"),
        "rm_score_se_emitted": float(np.std(valid) / np.sqrt(len(valid))) if valid else float("nan"),
        "rm_score_mean_with_fail_zero": float(np.mean(fail_zero)),
        "n_emitted": int(emitted),
        "n_total": int(total),
        "rows": rows,
    }
    print(f"  format emit rate : {metrics['format_emit_rate']:.3f}  ({emitted}/{total})", flush=True)
    print(f"  RM mean (emitted): {metrics['rm_score_mean_emitted']:.4f} ± {metrics['rm_score_se_emitted']:.4f} SE", flush=True)
    print(f"  RM mean (fail=0) : {metrics['rm_score_mean_with_fail_zero']:.4f}", flush=True)
    return metrics


def resolve_image_path_rownext(row: dict) -> Path:
    return Path(row["_image_path_resolved"])


def main() -> int:
    args = parse_args()

    # Pin policy GPU for vLLM. RM goes on a separate GPU after we tear down vLLM.
    visible_before = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_before is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.policy_gpu},{args.reward_gpu}"

    test_ds = load_from_disk(str(args.test_data))
    seen: dict[int, dict] = {}
    for row in test_ds:
        c = int(row["contest_number"])
        if c not in seen:
            p = resolve_image_path(args.data_root, row["image_path"])
            if not p.exists():
                continue
            seen[c] = {
                "contest_number": c,
                "image_path": row["image_path"],
                "prompt_text": row["prompt"],
                "_image_path_resolved": str(p),
            }
    cartoons = list(seen.values())
    if args.max_cartoons > 0:
        cartoons = cartoons[: args.max_cartoons]
    print(f"Evaluating on {len(cartoons)} test cartoons (×{args.num_samples} samples each)", flush=True)

    # We have to construct vLLM with `enable_lora=True` for the LoRA pass so
    # `lora_request=...` is accepted. Building one LLM that supports LoRA and
    # using base = no LoRA request, lora = with LoRA request, would be ideal,
    # but vLLM still loads the base model identically in either case, so we
    # just build it once with enable_lora=True.
    print(f"\nLoading vLLM policy on cuda:{args.policy_gpu} (enable_lora=True)...", flush=True)
    processor = AutoProcessor.from_pretrained(POLICY_MODEL_NAME, trust_remote_code=True)
    llm = make_llm(
        POLICY_MODEL_NAME,
        lora_dir=args.lora_dir,
        gpu_id=args.policy_gpu,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    all_metrics: dict[str, Any] = {}

    if not args.skip_base:
        base_rows = generate_for_policy(
            "base", llm, processor, cartoons,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            lora_dir=None,
        )
        all_metrics["base"] = {"rows": base_rows}  # filled in by score_rows

    lora_rows = generate_for_policy(
        "lora", llm, processor, cartoons,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        lora_dir=args.lora_dir,
    )

    # Tear down vLLM before loading the reward model so RM can claim GPU memory.
    # `del llm` alone leaves the EngineCore subprocess holding ~60 GB; we have
    # to destroy the distributed env explicitly.
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:  # noqa: BLE001
        print(f"  vllm teardown warn: {e}", flush=True)
    del llm
    import gc; gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"\nLoading reward model on cuda:{args.reward_gpu}...", flush=True)
    rm = load_reward_model(args.reward_model_dir, dtype=torch.bfloat16,
                            device=f"cuda:{args.reward_gpu}")

    if not args.skip_base:
        all_metrics["base"] = score_rows(all_metrics["base"]["rows"], rm,
                                          args.num_samples, "base")
    all_metrics["lora"] = score_rows(lora_rows, rm, args.num_samples, "lora")

    print("\n=== Summary ===", flush=True)
    for k in ("format_emit_rate", "rm_score_mean_emitted", "rm_score_se_emitted",
              "rm_score_mean_with_fail_zero"):
        for label, m in all_metrics.items():
            v = m.get(k, float("nan"))
            print(f"  {label:>4s}  {k:30s}: {v:.4f}", flush=True)

    if not args.skip_base:
        delta = (all_metrics["lora"]["rm_score_mean_emitted"]
                  - all_metrics["base"]["rm_score_mean_emitted"])
        pooled_se = float(np.sqrt(
            all_metrics["lora"]["rm_score_se_emitted"]**2
            + all_metrics["base"]["rm_score_se_emitted"]**2
        ))
        print(f"\n  delta (lora - base) RM mean: {delta:+.4f} ± {pooled_se:.4f} SE  "
              f"({delta/pooled_se:+.2f}σ)" if pooled_se > 0 else "", flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nWrote full results to {args.output}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
