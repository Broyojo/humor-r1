"""Non-circular best-of-N analysis: pick RM-best caption per (cell, cartoon),
then judge the selections pairwise with Sonnet 4.6 to verify the BoN gain
isn't circular.

Output: results/judge/bon_<judge>_<split>_summary.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from judge_pairwise import (  # noqa: E402
    PAIRWISE_SYSTEM, judge_pair, encode_image, fit_bt_scores
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--judge", default="anthropic/claude-sonnet-4.6")
    p.add_argument("--captions-dir", type=Path,
                   default=PROJECT_ROOT / "results" / "captions")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "results" / "judge")
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--max-tokens", type=int, default=1024)
    return p.parse_args()


def load_bon_picks(captions_dir: Path, split: str) -> tuple[dict, dict]:
    """Returns ({cell: {contest: best_caption}}, {contest: image_path}).

    "Best" = highest rm_score across the cell's N samples for that cartoon.
    """
    picks: dict[str, dict[int, str]] = defaultdict(dict)
    images: dict[int, str] = {}
    for path in sorted(captions_dir.glob(f"*_{split}.scored.jsonl")):
        cell = path.stem.replace(f"_{split}.scored", "")
        # Group by contest, find best
        by_contest: dict[int, list[tuple[float, str, str]]] = defaultdict(list)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cap = r.get("caption")
            score = r.get("rm_score")
            if not cap or score is None:
                continue
            c = int(r["contest_number"])
            by_contest[c].append((float(score), cap, r["image_path"]))
            if c not in images:
                images[c] = r["image_path"]
        for c, items in by_contest.items():
            items.sort(key=lambda t: -t[0])
            picks[cell][c] = items[0][1]
    return dict(picks), images


def run_split(args, split: str) -> dict:
    print(f"\n=== BoN judging: split={split} judge={args.judge} ===", flush=True)
    picks, images = load_bon_picks(args.captions_dir, split)
    cells = sorted(picks.keys())
    common = sorted(set.intersection(*(set(d.keys()) for d in picks.values())))
    print(f"  cells={cells}  common_cartoons={len(common)}")

    results = []
    out_path = args.out_dir / f"bon_{args.judge.replace('/', '__')}_{split}_pairs.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ci, contest in enumerate(common):
            try:
                img_b64 = encode_image(images[contest])
            except Exception as e:
                print(f"  skip {contest}: {e}")
                continue
            for i in range(len(cells)):
                for j in range(len(cells)):
                    if i == j:
                        continue
                    a, b = cells[i], cells[j]
                    cap_a, cap_b = picks[a][contest], picks[b][contest]
                    winner = judge_pair(args.judge, img_b64, cap_a, cap_b, args.max_tokens)
                    winner_cell = a if winner == "A" else (b if winner == "B" else None)
                    rec = {
                        "split": split, "contest_number": contest,
                        "cell_a": a, "cell_b": b,
                        "caption_a": cap_a, "caption_b": cap_b,
                        "winner": winner, "winner_cell": winner_cell,
                    }
                    results.append(rec)
                    f.write(json.dumps(rec) + "\n")
            if (ci + 1) % 5 == 0:
                print(f"  [{ci+1}/{len(common)}] {len(results)} pairs")
    print(f"  wrote {out_path} ({len(results)} pairs)")

    bt = fit_bt_scores(results)
    # Per-cell win rate
    wins: dict[tuple[str, str], int] = defaultdict(int)
    plays: dict[tuple[str, str], int] = defaultdict(int)
    for r in results:
        if r["winner_cell"] is None:
            continue
        plays[(r["cell_a"], r["cell_b"])] += 1
        if r["winner_cell"] == r["cell_a"]:
            wins[(r["cell_a"], r["cell_b"])] += 1
    per_cell_winrate = {}
    for cell in cells:
        wt = sum(v for (a, _), v in wins.items() if a == cell)
        pt = sum(v for (a, _), v in plays.items() if a == cell)
        per_cell_winrate[cell] = wt / max(pt, 1)

    summary = {
        "judge": args.judge, "split": split,
        "n_pairs": len(results),
        "win_rate_per_cell": per_cell_winrate,
        "bt_score_per_cell": bt,
    }
    s_path = args.out_dir / f"bon_{args.judge.replace('/', '__')}_{split}_summary.json"
    s_path.write_text(json.dumps(summary, indent=2))
    print(f"  per-cell BoN-judge win rate: {per_cell_winrate}")
    print(f"  per-cell BoN-judge BT score: {bt}")
    return summary


def main() -> int:
    args = parse_args()
    for split in args.splits:
        run_split(args, split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
