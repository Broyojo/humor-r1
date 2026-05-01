"""Recompute metrics.json from already-scored JSONL files without re-running
the RM. Reads results/captions/*.scored.jsonl and aggregates per-cell stats.

Useful when we add a cell mid-run and the metrics.json got overwritten by
a partial re-score.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"\w+|[^\w\s]")


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
    from collections import Counter

    def bleu2(hyp: list[str], refs: list[list[str]]) -> float:
        if len(hyp) < 2 or not refs:
            return 0.0
        hb = Counter(zip(hyp, hyp[1:]))
        ref_max: Counter = Counter()
        for r in refs:
            if len(r) < 2:
                continue
            rb = Counter(zip(r, r[1:]))
            for k, v in rb.items():
                if v > ref_max[k]:
                    ref_max[k] = v
        clipped = sum(min(c, ref_max[k]) for k, c in hb.items())
        return clipped / max(sum(hb.values()), 1)

    scores = []
    for _, caps in rows_by_contest.items():
        toks_list = [tokenize(c) for c in caps]
        for i, toks in enumerate(toks_list):
            others = [t for j, t in enumerate(toks_list) if j != i and t]
            if not others:
                continue
            scores.append(bleu2(toks, others))
    return float(np.mean(scores)) if scores else 0.0


def compute_one(scored_path: Path, raw_path: Path) -> dict:
    cell, split = scored_path.stem.replace(".scored", "").rsplit("_", 1)
    rows = [json.loads(l) for l in scored_path.read_text().splitlines() if l.strip()]
    raw_rows = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]

    captions = [r.get("caption") for r in raw_rows]
    finish_reasons = [r.get("finish_reason") for r in raw_rows]

    n_total = len(raw_rows)
    n_emitted = sum(1 for c in captions if c)
    n_truncated = sum(
        1 for fr, c in zip(finish_reasons, captions)
        if (fr == "length" or (fr is None and c is None))
    )

    rm_scores = [float(r["rm_score"]) for r in rows
                 if r.get("rm_score") is not None]
    rm_arr = np.array(rm_scores, dtype=np.float64) if rm_scores else np.array([])

    rows_by_contest: dict[int, list[str]] = defaultdict(list)
    for r in raw_rows:
        if r.get("caption"):
            rows_by_contest[int(r["contest_number"])].append(r["caption"])

    valid_caps = [c for c in captions if c]
    cap_lens = [len(tokenize(c)) for c in valid_caps]

    return {
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


def main() -> int:
    cap_dir = PROJECT_ROOT / "results" / "captions"
    out_path = PROJECT_ROOT / "results" / "metrics.json"
    rows = []
    for scored in sorted(cap_dir.glob("*.scored.jsonl")):
        raw = scored.with_suffix("").with_suffix(".jsonl")
        # ".scored.jsonl" → strip .scored, leaving .jsonl
        raw = scored.parent / scored.name.replace(".scored.jsonl", ".jsonl")
        if not raw.exists():
            print(f"  skipping {scored.name}: raw not found", file=sys.stderr)
            continue
        m = compute_one(scored, raw)
        print(f"  {m['cell']} {m['split']:<11s}  RM={m['rm_mean']:+.3f}±{m['rm_se']:.3f}"
              f"  fmt={m['format_rate']:.2f}  d2={m['distinct_2']:.3f}  sb={m['self_bleu2']:.3f}")
        rows.append(m)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
