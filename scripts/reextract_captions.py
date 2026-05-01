"""Re-extract captions from saved completion JSONL files using the latest
extraction logic in gen_captions.extract_caption. Use after updating the
extractor without re-running expensive generation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_captions import extract_caption  # noqa: E402


def main() -> int:
    paths = sorted(Path("results/captions").glob("*.jsonl"))
    paths = [p for p in paths if not p.name.endswith(".scored.jsonl")]
    # Cells whose models are supposed to emit thinking; for these, only count
    # captions that came from <caption> tags (no first-line fallback). The
    # fallback is for plain Instruct-style models that just emit one line.
    THINKING_CELLS = {"E0b", "E1b", "E2b"}
    for p in paths:
        cell = p.stem.split("_")[0]
        is_thinking = cell in THINKING_CELLS
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        changed = 0
        for r in rows:
            new = extract_caption(r.get("completion") or "",
                                  allow_fallback=not is_thinking)
            if new != r.get("caption"):
                r["caption"] = new
                changed += 1
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        emitted = sum(1 for r in rows if r.get("caption"))
        print(f"  {p.name}: total={len(rows)} emit={emitted} ({emitted/max(len(rows),1):.0%}) changed={changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
