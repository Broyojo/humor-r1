"""Pairwise LLM-as-judge over experimental cells using OpenRouter.

For each (split, cartoon, ordered cell pair), shows a frontier judge the image
and 2 candidate captions and asks which is funnier. Each pair is judged twice
(positions A/B swapped) to control for position bias.

Outputs:
- results/judge/{judge}_{split}_pairs.jsonl   per-pair raw judgements
- results/judge/{judge}_{split}_summary.json  per-cell win rates and BT scores
- results/judge_calibration.json              judge accuracy on chosen vs rejected pairs

Usage:
    uv run python scripts/judge_pairwise.py --calibrate --judge openai/gpt-5.5
    uv run python scripts/judge_pairwise.py --judge openai/gpt-5.5 \\
        --captions-dir results/captions \\
        --splits validation test
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINNER_RE = re.compile(r'"winner"\s*:\s*"([AB])"', re.IGNORECASE)


PAIRWISE_SYSTEM = (
    "You are an expert humor evaluator specializing in New Yorker cartoon "
    "captions. You will see a cartoon image and two candidate captions. "
    "Pick the funnier caption.\n\n"
    "A great New Yorker caption is unexpected, witty, concise, and rewards "
    "the viewer for noticing details in the cartoon. Avoid captions that "
    "merely describe the scene, are too long, or rely on cliché.\n\n"
    'Reply with ONLY one JSON object: {"winner": "A" or "B"}.'
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--judge", default="anthropic/claude-sonnet-4.6",
                   help="OpenRouter model slug (faster/cheaper than gpt-5.5; pass openai/gpt-5.5 for cross-validation)")
    p.add_argument("--captions-dir", type=Path,
                   default=PROJECT_ROOT / "results" / "captions")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "results" / "judge")
    p.add_argument("--splits", nargs="+", default=["validation", "test"])
    p.add_argument("--max-cartoons-per-split", type=int, default=0,
                   help="0 = all unique contests with captions in every cell")
    p.add_argument("--calibrate", action="store_true",
                   help="Run judge-vs-human calibration on BT pairs (chosen vs rejected) and exit")
    p.add_argument("--calibrate-n", type=int, default=100)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Bumped to 1024 to leave room for reasoning models (gpt-5.5).")
    return p.parse_args()


def encode_image(path: str | Path, max_side: int = 448) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_openrouter(judge: str, messages: list, max_tokens: int,
                    max_retries: int = 4) -> str:
    import httpx
    api_key = os.environ["OPENROUTER_API_KEY"]
    payload = {
        "model": judge,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": messages,
    }
    # Suppress lengthy CoT for reasoning models so a low max_tokens still
    # leaves budget for the JSON output.
    if "gpt-5" in judge or "o1" in judge or "o3" in judge:
        payload["reasoning"] = {"effort": "minimal"}
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Broyojo/humor-r1",
                    "X-Title": "humor-r1 judge",
                },
                json=payload,
                timeout=60.0,
            )
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"http {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"openrouter failed: {last_exc}")


def parse_winner(text: str) -> str | None:
    m = WINNER_RE.search(text)
    if m:
        return m.group(1).upper()
    # Fallback: last A or B character
    cleaned = re.sub(r"[^AB]", "", text.upper())
    if cleaned:
        return cleaned[-1]
    return None


def judge_pair(judge: str, image_b64: str, caption_a: str, caption_b: str,
               max_tokens: int) -> str | None:
    user_text = (
        f"Caption A: {caption_a}\n\nCaption B: {caption_b}\n\n"
        "Which is funnier? Respond with only the JSON."
    )
    messages = [
        {"role": "system", "content": PAIRWISE_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            }},
        ]},
    ]
    text = call_openrouter(judge, messages, max_tokens)
    return parse_winner(text)


def calibrate(args) -> int:
    """Run judge on BT (chosen, rejected) pairs from val to measure accuracy."""
    import pandas as pd
    rng = random.Random(args.seed)
    val = pd.read_parquet(args.data_root / "bt_pairs_validation.parquet")
    val = val.sample(n=min(args.calibrate_n, len(val)), random_state=args.seed).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir.parent / "judge_calibration.json"

    correct = 0
    total = 0
    pos_a_correct = 0; pos_a_total = 0
    pos_b_correct = 0; pos_b_total = 0
    rows = []
    for i, row in val.iterrows():
        img_path = args.data_root / row["image_path"]
        if not img_path.exists():
            continue
        try:
            img_b64 = encode_image(img_path)
        except Exception as e:
            print(f"  skip {img_path}: {e}", flush=True)
            continue
        # Randomize position
        chosen_first = rng.random() < 0.5
        cap_a = row["chosen"] if chosen_first else row["rejected"]
        cap_b = row["rejected"] if chosen_first else row["chosen"]
        winner = judge_pair(args.judge, img_b64, cap_a, cap_b, args.max_tokens)
        if winner is None:
            continue
        # Judge "winner": A or B → which underlying caption?
        judge_picks_chosen = (winner == "A" and chosen_first) or (winner == "B" and not chosen_first)
        rows.append({
            "contest_number": int(row["contest_number"]),
            "chosen_first": chosen_first,
            "winner": winner,
            "judge_picks_chosen": judge_picks_chosen,
        })
        total += 1
        if judge_picks_chosen:
            correct += 1
        if winner == "A":
            pos_a_total += 1
            if judge_picks_chosen:
                pos_a_correct += 1
        else:
            pos_b_total += 1
            if judge_picks_chosen:
                pos_b_correct += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}] acc={correct/max(total,1):.3f}", flush=True)

    summary = {
        "judge": args.judge,
        "n": total,
        "accuracy": correct / max(total, 1),
        "position_A_pick_rate": (pos_a_total / max(total, 1)),
        "accuracy_when_picked_A": pos_a_correct / max(pos_a_total, 1),
        "accuracy_when_picked_B": pos_b_correct / max(pos_b_total, 1),
    }
    print(f"\n=== Calibration ({args.judge}) ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out = {"summary": summary, "rows": rows}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


def collect_captions(captions_dir: Path, split: str) -> dict[str, dict[int, list[str]]]:
    """Returns {cell: {contest_number: [caption, ...]}}, only emitted captions."""
    out: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(captions_dir.glob(f"*_{split}.jsonl")):
        if path.name.endswith(".scored.jsonl"):
            continue
        cell = path.stem.rsplit("_", 1)[0]
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("caption"):
                out[cell][int(r["contest_number"])].append(r["caption"])
    return {k: dict(v) for k, v in out.items()}


def pick_one_per_cell(captions_by_cell: dict, contests: list[int],
                      seed: int) -> dict[int, dict[str, str]]:
    """For each contest, pick one random caption per cell that has any."""
    rng = random.Random(seed)
    out: dict[int, dict[str, str]] = {}
    for c in contests:
        picks: dict[str, str] = {}
        for cell, by_contest in captions_by_cell.items():
            caps = by_contest.get(c, [])
            if caps:
                picks[cell] = rng.choice(caps)
        if len(picks) >= 2:
            out[c] = picks
    return out


def fit_bt_scores(pair_results: list[dict]) -> dict[str, float]:
    """MLE Bradley-Terry from pairwise winner records."""
    cells = sorted({r["cell_a"] for r in pair_results} |
                   {r["cell_b"] for r in pair_results})
    idx = {c: i for i, c in enumerate(cells)}
    # Wins matrix: W[i,j] = times i beat j
    W = np.zeros((len(cells), len(cells)), dtype=np.float64)
    for r in pair_results:
        if r.get("winner_cell") is None:
            continue
        i = idx[r["winner_cell"]]
        j = idx[r["cell_a"] if r["winner_cell"] == r["cell_b"] else r["cell_b"]]
        W[i, j] += 1
    # Iterative MLE (MM algorithm, Hunter 2004)
    p = np.ones(len(cells)) / len(cells)
    for _ in range(500):
        p_new = np.zeros_like(p)
        denom_sum = W + W.T
        for i in range(len(cells)):
            num = W[i].sum()
            den = 0.0
            for j in range(len(cells)):
                if i == j:
                    continue
                if denom_sum[i, j] > 0:
                    den += denom_sum[i, j] / (p[i] + p[j])
            p_new[i] = num / max(den, 1e-12)
        p_new /= p_new.sum() if p_new.sum() > 0 else 1
        if np.allclose(p, p_new, atol=1e-7):
            break
        p = p_new
    # Convert to log-odds (BT score)
    p = np.clip(p, 1e-12, 1.0)
    log_p = np.log(p)
    log_p -= log_p.mean()
    return {c: float(log_p[idx[c]]) for c in cells}


def run_pairwise(args) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"\n=== Pairwise judging: split={split} judge={args.judge} ===", flush=True)
        captions = collect_captions(args.captions_dir, split)
        cells = sorted(captions.keys())
        if len(cells) < 2:
            print(f"  not enough cells with captions: {cells}", flush=True)
            continue
        # Use only contests that have captions in EVERY cell.
        contest_sets = [set(by.keys()) for by in captions.values()]
        common = sorted(set.intersection(*contest_sets))
        if args.max_cartoons_per_split > 0:
            common = common[: args.max_cartoons_per_split]
        print(f"  cells={cells}  common_cartoons={len(common)}", flush=True)

        picks = pick_one_per_cell(captions, common, args.seed)
        print(f"  picks per contest: {len(picks)} contests covered", flush=True)

        out_path = args.out_dir / f"{args.judge.replace('/', '__')}_{split}_pairs.jsonl"
        results: list[dict] = []
        with out_path.open("w") as f:
            for ci, contest in enumerate(common):
                if contest not in picks:
                    continue
                # Need image
                # Look up image_path from any caption file
                image_path = None
                for cell in cells:
                    p_path = args.captions_dir / f"{cell}_{split}.jsonl"
                    if not p_path.exists():
                        continue
                    for line in p_path.read_text().splitlines():
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        if int(r["contest_number"]) == contest:
                            image_path = r["image_path"]
                            break
                    if image_path:
                        break
                if not image_path:
                    continue
                try:
                    img_b64 = encode_image(image_path)
                except Exception as e:
                    print(f"  skip contest {contest}: {e}", flush=True)
                    continue

                cell_caps = picks[contest]
                cell_list = sorted(cell_caps.keys())
                # All ordered pairs (i, j) where i != j → judge with A=i_cap, B=j_cap.
                # That gives us position-counterbalanced data: each unordered pair
                # is judged twice (once with each cell in position A).
                for i in range(len(cell_list)):
                    for j in range(len(cell_list)):
                        if i == j:
                            continue
                        cell_a, cell_b = cell_list[i], cell_list[j]
                        cap_a, cap_b = cell_caps[cell_a], cell_caps[cell_b]
                        winner = judge_pair(args.judge, img_b64, cap_a, cap_b, args.max_tokens)
                        winner_cell = cell_a if winner == "A" else (cell_b if winner == "B" else None)
                        rec = {
                            "split": split,
                            "contest_number": contest,
                            "cell_a": cell_a, "cell_b": cell_b,
                            "caption_a": cap_a, "caption_b": cap_b,
                            "winner": winner,
                            "winner_cell": winner_cell,
                        }
                        results.append(rec)
                        f.write(json.dumps(rec) + "\n")
                if (ci + 1) % 5 == 0:
                    print(f"  [{ci+1}/{len(common)}] judged so far: {len(results)} pairs", flush=True)
        print(f"  wrote {out_path} ({len(results)} pairs)", flush=True)

        # Aggregate: per-cell win rate vs each other cell.
        from collections import Counter
        wins: Counter = Counter()  # (cell_a, cell_b) -> count where cell_a won
        plays: Counter = Counter()
        for r in results:
            if r["winner_cell"] is None:
                continue
            plays[(r["cell_a"], r["cell_b"])] += 1
            if r["winner_cell"] == r["cell_a"]:
                wins[(r["cell_a"], r["cell_b"])] += 1
        per_cell_winrate: dict[str, float] = {}
        for cell in cells:
            wins_total = sum(v for (a, _), v in wins.items() if a == cell)
            plays_total = sum(v for (a, _), v in plays.items() if a == cell)
            per_cell_winrate[cell] = wins_total / max(plays_total, 1)

        bt = fit_bt_scores(results)
        summary = {
            "judge": args.judge,
            "split": split,
            "n_pairs": len(results),
            "n_cells": len(cells),
            "win_rate_per_cell": per_cell_winrate,
            "bt_score_per_cell": bt,
        }
        s_path = args.out_dir / f"{args.judge.replace('/', '__')}_{split}_summary.json"
        s_path.write_text(json.dumps(summary, indent=2))
        print(f"  per-cell win rate: {per_cell_winrate}", flush=True)
        print(f"  BT scores: {bt}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.calibrate:
        return calibrate(args)
    return run_pairwise(args)


if __name__ == "__main__":
    sys.exit(main())
