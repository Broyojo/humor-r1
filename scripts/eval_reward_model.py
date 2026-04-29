"""Evaluate the trained Bradley-Terry reward model on held-out preference pairs.

Loads the artifacts from `--reward-model-dir` (default: the friend's checkpoint
at `~/scratch/humor-r1/final_reward_model/`) and scores chosen vs rejected
captions on `--eval-file` (default: `data/bt_pairs_validation.parquet`).

Reports:
  - preference_accuracy : fraction of pairs where score(chosen) > score(rejected)
  - reward_margin       : mean of score(chosen) - score(rejected)
  - bt_loss             : -log sigmoid(margin), the BT training loss
  - score statistics    : mean/std/min/max of chosen and rejected scores

The headline number is preference_accuracy. Random is 0.5; a usable RL signal
should be comfortably above 0.65 on the validation set.

Throughput note: image preprocessing in the Qwen2.5-VL processor is CPU-bound
and single-threaded per call, so we use a DataLoader with worker processes to
prefetch and a generous batch size.

Run via:
    uv run python scripts/eval_reward_model.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from reward_model import LoadedRewardModel, _prepare_batch_inputs, load_reward_model  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RM_DIR = Path(
    os.environ.get(
        "REWARD_MODEL_DIR",
        str(Path.home() / "scratch/humor-r1/final_reward_model"),
    )
)
DEFAULT_EVAL_FILE = PROJECT_ROOT / "data" / "bt_pairs_validation.parquet"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-model-dir", type=Path, default=DEFAULT_RM_DIR)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Cap on eval pairs (set to 0 for all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Pairs per forward pass (each pair scores chosen+rejected = 2*BS rows).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes for parallel image preprocessing.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--report-by-sigma",
        action="store_true",
        help="Bucket accuracy by sigma_gap (clean preference pairs vs noisy).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to dump full metrics JSON.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    p = Path(image_path)
    return p if p.is_absolute() else data_root / p


@dataclass
class _Row:
    """Loaded row, ready to be passed to the RM. Workers produce these."""
    chosen_caption: str
    rejected_caption: str
    prompt_text: str
    image: Image.Image
    sigma_gap: float


class PairDataset(Dataset):
    """Loads pairs from the parquet, opening images on demand in workers."""

    def __init__(self, parquet_path: Path, data_root: Path, max_samples: int):
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_path))
        self.rows = table.to_pylist()
        if max_samples and max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> _Row | None:
        row = self.rows[idx]
        path = resolve_image_path(self.data_root, row["image_path"])
        if not path.exists():
            return None
        try:
            image = Image.open(path).convert("RGB")
            image.load()  # force read so the worker decodes off the main thread
        except Exception:  # noqa: BLE001
            return None
        sigma = row.get("sigma_gap")
        return _Row(
            chosen_caption=str(row["chosen"]),
            rejected_caption=str(row["rejected"]),
            prompt_text=str(row["prompt"]),
            image=image,
            sigma_gap=float(sigma) if sigma is not None else float("nan"),
        )


def _collate(batch: list[_Row | None]) -> list[_Row]:
    return [r for r in batch if r is not None]


@torch.no_grad()
def score_pairs_batch(
    rm: LoadedRewardModel,
    rows: list[_Row],
) -> tuple[list[float], list[float]]:
    """Score chosen and rejected for a batch of pairs in one forward pass."""
    images: list[Image.Image] = []
    prompts: list[str] = []
    captions: list[str] = []
    for r in rows:
        images.extend([r.image, r.image])
        prompts.extend([r.prompt_text, r.prompt_text])
        captions.extend([r.chosen_caption, r.rejected_caption])

    inputs = _prepare_batch_inputs(rm, images, prompts, captions)
    outputs = rm.backbone(**inputs, return_dict=True)
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        hidden = outputs.last_hidden_state
    elif isinstance(outputs, dict) and outputs.get("last_hidden_state") is not None:
        hidden = outputs["last_hidden_state"]
    else:
        raise ValueError(f"No last_hidden_state on {type(outputs).__name__}")

    attn = inputs["attention_mask"]
    last = attn.long().sum(dim=1) - 1
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    pooled = hidden[batch_idx, last].to(rm.score_head.weight.dtype)
    rewards = rm.score_head(pooled).squeeze(-1).detach().cpu().tolist()
    chosen = rewards[0::2]
    rejected = rewards[1::2]
    return chosen, rejected


def main() -> int:
    args = parse_args()

    if not args.reward_model_dir.exists():
        raise FileNotFoundError(args.reward_model_dir)
    if not args.eval_file.exists():
        raise FileNotFoundError(args.eval_file)

    print(f"Loading reward model from {args.reward_model_dir}...", flush=True)
    rm = load_reward_model(
        args.reward_model_dir,
        dtype=torch_dtype(args.dtype),
        device=args.device,
    )
    print(f"  base_model     : {rm.base_model_name}", flush=True)
    print(f"  device         : {rm.device}", flush=True)
    print(f"  dtype          : {rm.dtype}", flush=True)

    print(f"Loading eval pairs from {args.eval_file}...", flush=True)
    dataset = PairDataset(args.eval_file, args.data_root, args.max_samples)
    print(f"  num pairs      : {len(dataset)}", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )

    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    sigma_gaps: list[float] = []
    skipped = 0

    t0 = time.time()
    log_every = max(1, len(loader) // 20)

    for step, batch in enumerate(loader):
        if not batch:
            skipped += args.batch_size
            continue
        chosen, rejected = score_pairs_batch(rm, batch)
        chosen_scores.extend(chosen)
        rejected_scores.extend(rejected)
        sigma_gaps.extend(r.sigma_gap for r in batch)
        skipped += args.batch_size - len(batch)

        if step % log_every == 0 or step == len(loader) - 1:
            n = len(chosen_scores)
            elapsed = time.time() - t0
            margins = np.array(chosen_scores) - np.array(rejected_scores)
            acc = float(np.mean(margins > 0.0)) if n > 0 else float("nan")
            margin_mean = float(np.mean(margins)) if n > 0 else float("nan")
            rate = n / elapsed if elapsed > 0 else 0.0
            print(
                f"  [{step + 1:3d} / {len(loader)}] pairs={n:5d}  "
                f"acc={acc:.4f}  margin={margin_mean:+.4f}  "
                f"rate={rate:.1f} pairs/s  elapsed={elapsed:5.1f}s",
                flush=True,
            )

    chosen_arr = np.array(chosen_scores, dtype=np.float64)
    rejected_arr = np.array(rejected_scores, dtype=np.float64)
    sigma_arr = np.array(sigma_gaps, dtype=np.float64)
    margins = chosen_arr - rejected_arr

    metrics: dict[str, float | int] = {
        "num_pairs_scored": int(len(margins)),
        "num_pairs_skipped": int(skipped),
        "preference_accuracy": float(np.mean(margins > 0.0)),
        "reward_margin_mean": float(np.mean(margins)),
        "reward_margin_std": float(np.std(margins)),
        "bt_loss": float(np.mean(np.log1p(np.exp(-margins)))),
        "chosen_mean": float(np.mean(chosen_arr)),
        "chosen_std": float(np.std(chosen_arr)),
        "rejected_mean": float(np.mean(rejected_arr)),
        "rejected_std": float(np.std(rejected_arr)),
        "score_min": float(np.min(np.concatenate([chosen_arr, rejected_arr]))),
        "score_max": float(np.max(np.concatenate([chosen_arr, rejected_arr]))),
    }

    if args.report_by_sigma and np.any(np.isfinite(sigma_arr)):
        buckets = [(3.0, 5.0), (5.0, 8.0), (8.0, np.inf)]
        for lo, hi in buckets:
            mask = (sigma_arr >= lo) & (sigma_arr < hi) & np.isfinite(sigma_arr)
            if mask.any():
                bucket_acc = float(np.mean(margins[mask] > 0.0))
                metrics[f"acc_sigma_{lo}_{hi}"] = bucket_acc
                metrics[f"n_sigma_{lo}_{hi}"] = int(mask.sum())

    print("\n=== Reward model evaluation ===", flush=True)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:24s}: {v:.4f}", flush=True)
        else:
            print(f"  {k:24s}: {v}", flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nWrote metrics to {args.output}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
