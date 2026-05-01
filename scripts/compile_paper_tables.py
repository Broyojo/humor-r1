"""Compile RM scoring + judge results into LaTeX tables for the paper.

Reads:
  results/metrics.json
  results/judge/<judge>_<split>_summary.json
  results/judge/<judge>_<split>_pairs.jsonl  (for win-rate vs base computation)
Writes:
  results/paper_tables.tex
  results/numbers.json   (raw per-cell numbers for the writeup script)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JUDGE = "anthropic/claude-sonnet-4.6"
JUDGE_TAG = JUDGE.replace("/", "__")
CELLS = ["E0a", "E0b", "E0c", "E1a", "E1b", "E2a", "E2b", "E3"]
CELL_LABELS = {
    "E0a": "Qwen3-VL-2B-Instruct (zero-shot)",
    "E0b": "Qwen3-VL-2B-Thinking (zero-shot)",
    "E0c": "GPT-5.5 (frontier zero-shot)",
    "E1a": "SFT, no thinking",
    "E1b": "SFT, thinking",
    "E2a": "SFT$\\to$GRPO, no thinking",
    "E2b": "GRPO, thinking (ckpt-50)",
    "E3":  "DPO, no thinking",
}
BASE_FOR_WINRATE = "E0a"


def load_metrics() -> dict[tuple[str, str], dict]:
    with (PROJECT_ROOT / "results" / "metrics.json").open() as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        out[(r["cell"], r["split"])] = r
    return out


def load_judge_summary(split: str) -> dict:
    p = PROJECT_ROOT / "results" / "judge" / f"{JUDGE_TAG}_{split}_summary.json"
    with p.open() as f:
        return json.load(f)


def load_judge_pairs(split: str) -> list[dict]:
    p = PROJECT_ROOT / "results" / "judge" / f"{JUDGE_TAG}_{split}_pairs.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def winrate_vs_base(pairs: list[dict], cell: str, base: str) -> tuple[float, int, int]:
    wins = 0
    plays = 0
    for r in pairs:
        if r.get("winner_cell") is None:
            continue
        a, b = r["cell_a"], r["cell_b"]
        if {a, b} != {cell, base}:
            continue
        plays += 1
        if r["winner_cell"] == cell:
            wins += 1
    return (wins / plays if plays else 0.0), wins, plays


def fmt_se(mean: float, se: float, sigma: int = 3) -> str:
    return f"${mean:+.{sigma}f} \\pm {se:.{sigma}f}$"


def fmt_pct(p: float, n: int = 1) -> str:
    return f"{p*100:.{n}f}\\%"


def make_main_table(metrics: dict, judge_val: dict, judge_test: dict,
                    pairs_val: list[dict], pairs_test: list[dict]) -> str:
    rows = []
    for c in CELLS:
        m_val = metrics.get((c, "validation"), {})
        m_test = metrics.get((c, "test"), {})
        bt_val = judge_val["bt_score_per_cell"].get(c, float("nan"))
        bt_test = judge_test["bt_score_per_cell"].get(c, float("nan"))
        if c == BASE_FOR_WINRATE:
            wr_val_str = "--"
            wr_test_str = "--"
        else:
            wr_val, _, _ = winrate_vs_base(pairs_val, c, BASE_FOR_WINRATE)
            wr_test, _, _ = winrate_vs_base(pairs_test, c, BASE_FOR_WINRATE)
            wr_val_str = fmt_pct(wr_val)
            wr_test_str = fmt_pct(wr_test)
        rows.append({
            "cell": c,
            "rm_val": fmt_se(m_val.get("rm_mean", float("nan")), m_val.get("rm_se", float("nan"))),
            "rm_test": fmt_se(m_test.get("rm_mean", float("nan")), m_test.get("rm_se", float("nan"))),
            "wr_val": wr_val_str,
            "wr_test": wr_test_str,
            "bt_val": f"${bt_val:+.2f}$",
            "bt_test": f"${bt_test:+.2f}$",
        })
    body = ""
    for r in rows:
        body += (f"{r['cell']} & {r['rm_val']} & {r['rm_test']} & "
                 f"{r['wr_val']} & {r['wr_test']} & "
                 f"{r['bt_val']} & {r['bt_test']} \\\\\n")
    return body


def make_diversity_table(metrics: dict) -> str:
    body = ""
    for c in CELLS:
        m = metrics.get((c, "test"), {})
        body += (f"{c} & "
                 f"{m.get('distinct_1', 0):.3f} & "
                 f"{m.get('distinct_2', 0):.3f} & "
                 f"{m.get('distinct_3', 0):.3f} & "
                 f"{m.get('self_bleu2', 0):.3f} & "
                 f"{m.get('caption_len_p90', 0):.0f} & "
                 f"{m.get('format_rate', 0)*100:.0f}\\% & "
                 f"{m.get('truncation_rate', 0)*100:.0f}\\% \\\\\n")
    return body


def main() -> None:
    metrics = load_metrics()
    judge_val = load_judge_summary("validation")
    judge_test = load_judge_summary("test")
    pairs_val = load_judge_pairs("validation")
    pairs_test = load_judge_pairs("test")

    main_body = make_main_table(metrics, judge_val, judge_test,
                                 pairs_val, pairs_test)
    div_body = make_diversity_table(metrics)

    main_tex = (
        "\\begin{tabular}{l rr cc cc}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c}{RM score} & \\multicolumn{2}{c}{Win\\% vs E0a (Sonnet)} & \\multicolumn{2}{c}{Sonnet BT} \\\\\n"
        " \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Cell & dev & test & dev & test & dev & test \\\\\n"
        "\\midrule\n"
        + main_body
        + "\\bottomrule\n\\end{tabular}\n"
    )
    div_tex = (
        "\\begin{tabular}{l ccc c c c c}\n"
        "\\toprule\n"
        "Cell & d-1 & d-2 & d-3 & SB-2 & len-p90 & fmt\\% & trunc\\% \\\\\n"
        "\\midrule\n"
        + div_body
        + "\\bottomrule\n\\end{tabular}\n"
    )

    out_path = PROJECT_ROOT / "results" / "paper_tables.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "% Auto-generated by compile_paper_tables.py — do not edit by hand.\n\n"
        "%% Main results table (RM score + judge):\n"
        f"{main_tex}\n"
        "%% Diversity / format table (test split):\n"
        f"{div_tex}\n"
    )
    print(f"Wrote {out_path}")
    print()
    print("=== MAIN ===")
    print(main_tex)
    print()
    print("=== DIVERSITY (test) ===")
    print(div_tex)

    # Also dump numbers.json for the writeup script
    numbers = {
        "cells": CELLS,
        "labels": CELL_LABELS,
        "rm": {(c, s): {
            "mean": metrics.get((c, s), {}).get("rm_mean"),
            "se": metrics.get((c, s), {}).get("rm_se"),
            "n": metrics.get((c, s), {}).get("n_emitted"),
        } for c in CELLS for s in ("validation", "test")},
        "diversity": {c: metrics.get((c, "test"), {}) for c in CELLS},
        "judge_bt_val": judge_val["bt_score_per_cell"],
        "judge_bt_test": judge_test["bt_score_per_cell"],
        "judge_wr_overall_val": judge_val["win_rate_per_cell"],
        "judge_wr_overall_test": judge_test["win_rate_per_cell"],
        "winrate_vs_base_test": {c: winrate_vs_base(pairs_test, c, BASE_FOR_WINRATE)[0]
                                  for c in CELLS if c != BASE_FOR_WINRATE},
        "winrate_vs_base_val": {c: winrate_vs_base(pairs_val, c, BASE_FOR_WINRATE)[0]
                                 for c in CELLS if c != BASE_FOR_WINRATE},
        "n_pairs_test": judge_test["n_pairs"],
        "n_pairs_val": judge_val["n_pairs"],
    }
    # Convert tuple keys to strings for JSON
    numbers["rm"] = {f"{c}|{s}": v for (c, s), v in numbers["rm"].items()}
    (PROJECT_ROOT / "results" / "numbers.json").write_text(
        json.dumps(numbers, indent=2)
    )
    print("Wrote results/numbers.json")


if __name__ == "__main__":
    main()
