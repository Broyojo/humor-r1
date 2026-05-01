"""Score one or more cell JSONLs with the trained RM and compute caption metrics.

Loads each cell's captions (one JSONL per (cell, split)), scores the captions
with the BT reward model, and computes diversity, length, format, and
truncation metrics. Writes one merged JSON of per-cell metrics plus per-row
scored JSONLs back next to the inputs.

Usage:
    uv run python scripts/score_grid.py \\
        --captions-glob 'results/captions/*.jsonl' \\
        --reward-model-dir checkpoints/rm-final/final_reward_model
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_model import load_reward_model, score_batch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"\w+|[^\w\s]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--captions-glob", default="results/captions/*.jsonl")
    p.add_argument("--reward-model-dir", type=Path,
                   default=PROJECT_ROOT / "checkpoints" / "rm-final" / "final_reward_model")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "metrics.json")
    p.add_argument("--score-batch", type=int, default=4)
    p.add_argument("--max-tokens-per-caption", type=int, default=80)
    return p.parse_args()


def tokenize(s: str) -> list[str]:
    return WORD_RE.findall(s.lower())


def distinct_n(captions: list[str], n: int) -> float:
    total = 0
    seen: set[tuple[str, ...]] = set()
    for c in captions:
        toks = tokenize(c)
        for i in range(len(toks) - n + 1):
            seen.add(tuple(toks[i : i + n]))
            total += 1
    return len(seen) / max(total, 1)


def self_bleu_within_cartoon(rows_by_contest: dict[int, list[str]]) -> float:
    """Mean self-BLEU-2 within each cartoon (over its samples).

    A blunt diversity proxy: for each generated caption per cartoon, BLEU-2
    against the other samples for that cartoon. Lower = more diverse per prompt.
    """
    from collections import Counter

    def bleu2(hyp: list[str], refs: list[list[str]]) -> float:
        if len(hyp) < 2 or not refs:
            return 0.0
        hyp_bigrams = Counter(zip(hyp, hyp[1:]))
        ref_max: Counter = Counter()
        for r in refs:
            if len(r) < 2:
                continue
            rb = Counter(zip(r, r[1:]))
            for k, v in rb.items():
                if v > ref_max[k]:
                    ref_max[k] = v
        clipped = sum(min(c, ref_max[k]) for k, c in hyp_bigrams.items())
        return clipped / max(sum(hyp_bigrams.values()), 1)

    scores = []
    for _, caps in rows_by_contest.items():
        toks_list = [tokenize(c) for c in caps]
        for i, toks in enumerate(toks_list):
            others = [t for j, t in enumerate(toks_list) if j != i and t]
            if not others:
                continue
            scores.append(bleu2(toks, others))
    return float(np.mean(scores)) if scores else 0.0


def cell_split_from_path(path: Path) -> tuple[str, str]:
    name = path.stem  # e.g. "E0a_validation"
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return name, "unknown"


def score_one_file(path: Path, rm, score_batch_size: int, max_caption_tok: int) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cell, split = cell_split_from_path(path)

    captions = [r.get("caption") for r in rows]
    completions = [r.get("completion", "") for r in rows]
    finish_reasons = [r.get("finish_reason", "") for r in rows]
    completion_tokens = [r.get("completion_tokens") for r in rows]

    n_total = len(rows)
    n_emitted = sum(1 for c in captions if c)
    n_truncated = sum(1 for fr, c in zip(finish_reasons, captions)
                      if (fr == "length" or (fr is None and c is None)))

    # RM scoring on emitted captions only
    images: list[Image.Image] = []
    prompts: list[str] = []
    caps: list[str] = []
    placement: list[int] = []
    img_cache: dict[str, Image.Image] = {}
    for i, r in enumerate(rows):
        cap = r.get("caption")
        if not cap:
            continue
        ip = r["image_path"]
        if ip not in img_cache:
            img_cache[ip] = Image.open(ip).convert("RGB")
        images.append(img_cache[ip])
        prompts.append(r.get("user_prompt", ""))
        # Truncate caption to a sane token budget for RM.
        toks = cap.split()
        caps.append(" ".join(toks[:max_caption_tok]))
        placement.append(i)

    rm_scores: list[float] = []
    if images:
        for i in range(0, len(images), score_batch_size):
            batch = score_batch(rm,
                                 images[i : i + score_batch_size],
                                 prompts[i : i + score_batch_size],
                                 caps[i : i + score_batch_size])
            rm_scores.extend([float(b) for b in batch])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Write per-row scores back
    scored_rows = []
    score_iter = iter(rm_scores)
    placement_set = set(placement)
    for i, r in enumerate(rows):
        sr = dict(r)
        sr["rm_score"] = next(score_iter) if i in placement_set else None
        scored_rows.append(sr)
    scored_path = path.with_suffix(".scored.jsonl")
    with scored_path.open("w") as f:
        for sr in scored_rows:
            f.write(json.dumps(sr) + "\n")

    # Per-cartoon aggregates
    rows_by_contest: dict[int, list[str]] = {}
    for r in rows:
        if r.get("caption"):
            rows_by_contest.setdefault(int(r["contest_number"]), []).append(r["caption"])

    valid_caps = [c for c in captions if c]
    cap_lens = [len(tokenize(c)) for c in valid_caps]

    rm_arr = np.array(rm_scores, dtype=np.float64) if rm_scores else np.array([])
    metrics = {
        "cell": cell,
        "split": split,
        "n_total": n_total,
        "n_emitted": n_emitted,
        "n_truncated": int(n_truncated),
        "format_rate": n_emitted / max(n_total, 1),
        "truncation_rate": n_truncated / max(n_total, 1),
        "rm_mean": float(rm_arr.mean()) if rm_arr.size else float("nan"),
        "rm_std": float(rm_arr.std()) if rm_arr.size else float("nan"),
        "rm_se": float(rm_arr.std() / math.sqrt(rm_arr.size)) if rm_arr.size else float("nan"),
        "rm_p25": float(np.percentile(rm_arr, 25)) if rm_arr.size else float("nan"),
        "rm_p50": float(np.percentile(rm_arr, 50)) if rm_arr.size else float("nan"),
        "rm_p75": float(np.percentile(rm_arr, 75)) if rm_arr.size else float("nan"),
        "caption_len_mean": float(np.mean(cap_lens)) if cap_lens else float("nan"),
        "caption_len_p90": float(np.percentile(cap_lens, 90)) if cap_lens else float("nan"),
        "distinct_1": distinct_n(valid_caps, 1),
        "distinct_2": distinct_n(valid_caps, 2),
        "distinct_3": distinct_n(valid_caps, 3),
        "self_bleu2": self_bleu_within_cartoon(rows_by_contest),
        "scored_path": str(scored_path),
    }
    return metrics


def main() -> int:
    args = parse_args()
    paths = sorted(Path().glob(args.captions_glob))
    if not paths:
        print(f"No captions matched {args.captions_glob}", file=sys.stderr)
        return 1
    paths = [p for p in paths if not p.name.endswith(".scored.jsonl")]
    print(f"Scoring {len(paths)} caption files...", flush=True)

    print(f"Loading RM from {args.reward_model_dir}", flush=True)
    rm = load_reward_model(args.reward_model_dir, dtype=torch.bfloat16, device="cuda")

    all_metrics = []
    for p in paths:
        print(f"\n--- {p.name} ---", flush=True)
        m = score_one_file(p, rm, args.score_batch, args.max_tokens_per_caption)
        print(f"  emit={m['format_rate']:.2f} trunc={m['truncation_rate']:.2f} "
              f"RM={m['rm_mean']:+.3f}±{m['rm_se']:.3f}  d1={m['distinct_1']:.3f} "
              f"d2={m['distinct_2']:.3f} self-BLEU2={m['self_bleu2']:.3f} "
              f"len_p90={m['caption_len_p90']:.0f}", flush=True)
        all_metrics.append(m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nWrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
