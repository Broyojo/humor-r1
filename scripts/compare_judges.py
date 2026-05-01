"""Compare Sonnet and Opus judge rankings for cross-validation.

After both judges finish, computes per-cell BT scores from each, plus
Spearman correlation across cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def spearmanr(x, y):
    """Lightweight Spearman rho without scipy."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    if n < 3:
        return rho, float("nan")
    t = rho * np.sqrt((n - 2) / max(1 - rho ** 2, 1e-12))
    # Two-sided p approximation via Student-t (n-2 df) -> normal for large n.
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return rho, float(p)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JUDGES = ["anthropic/claude-sonnet-4.6", "anthropic/claude-opus-4.7"]


def main() -> int:
    out_lines = []
    for split in ("validation", "test"):
        out_lines.append(f"\n## Split: {split}")
        bt_per_judge = {}
        for j in JUDGES:
            tag = j.replace("/", "__")
            sp = PROJECT_ROOT / "results" / "judge" / f"{tag}_{split}_summary.json"
            if not sp.exists():
                out_lines.append(f"  {j}: summary missing ({sp.name})")
                continue
            data = json.loads(sp.read_text())
            bt_per_judge[j] = data["bt_score_per_cell"]
            out_lines.append(f"  {j}: cells={len(data['bt_score_per_cell'])}  n_pairs={data['n_pairs']}")
        if len(bt_per_judge) < 2:
            continue
        common = sorted(set.intersection(*(set(d.keys()) for d in bt_per_judge.values())))
        s = np.array([bt_per_judge[JUDGES[0]][c] for c in common])
        o = np.array([bt_per_judge[JUDGES[1]][c] for c in common])
        rho, p = spearmanr(list(s), list(o))
        out_lines.append(f"  Spearman(Sonnet, Opus) over cells {common}: rho={rho:.3f} p={p:.4f}")
        out_lines.append(f"  Pearson:  r={float(np.corrcoef(s, o)[0, 1]):.3f}")
        out_lines.append(f"  cell-by-cell BT scores:")
        out_lines.append(f"    {'cell':<5s}  {'sonnet':>8s}  {'opus':>8s}")
        for c in common:
            out_lines.append(f"    {c:<5s}  {bt_per_judge[JUDGES[0]][c]:>+8.3f}  {bt_per_judge[JUDGES[1]][c]:>+8.3f}")

    print("\n".join(out_lines))
    out_path = PROJECT_ROOT / "results" / "cross_judge.txt"
    out_path.write_text("\n".join(out_lines))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    main()
