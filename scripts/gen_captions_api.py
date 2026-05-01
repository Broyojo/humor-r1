"""Frontier-model zero-shot baseline via OpenRouter (E0c).

Generates captions on dev + test by calling a vision-capable model through
OpenRouter's OpenAI-compatible API. No vLLM, no GPU.

    uv run python scripts/gen_captions_api.py --cell E0c \\
        --judge-model openai/gpt-4o --num-samples 5
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

from datasets import load_from_disk
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTION_RE = re.compile(r"<caption>(.*?)</caption>", re.DOTALL)
IMAGE_MAX_SIDE = 448

SYSTEM_NO_THINKING = (
    "You are a witty cartoon caption writer. Look at the cartoon image and "
    "output exactly one one-line caption inside <caption>...</caption> tags. "
    "Do not include any other text."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True)
    p.add_argument("--model", default="openai/gpt-5.5",
                   help="OpenRouter model slug (frontier baseline; default openai/gpt-5.5)")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--splits", nargs="+", default=["validation", "test"])
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "results" / "captions")
    p.add_argument("--max-cartoons", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=128)
    return p.parse_args()


def encode_image(path: Path, max_side: int = IMAGE_MAX_SIDE) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_openrouter(model: str, system: str, user_text: str, image_b64: str,
                    temperature: float, max_tokens: int,
                    max_retries: int = 4) -> str:
    import httpx
    api_key = os.environ["OPENROUTER_API_KEY"]
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"
                }},
            ]},
        ],
    }
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Broyojo/humor-r1",
                    "X-Title": "humor-r1 eval",
                },
                json=payload,
                timeout=60.0,
            )
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"http {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"openrouter call failed after {max_retries} retries: {last_exc}")


def extract_caption(text: str) -> str | None:
    m = CAPTION_RE.search(text)
    if m:
        s = m.group(1).strip()
        if s:
            return s
    # Fallback: if model didn't use tags, treat the first stripped line as caption.
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return line or None


def load_cartoons(data_root: Path, split: str, max_cartoons: int) -> list[dict]:
    ds = load_from_disk(str(data_root / f"caption_sft_{split}"))
    seen: dict[int, dict] = {}
    for row in ds:
        c = int(row["contest_number"])
        if c in seen:
            continue
        p = data_root / row["image_path"]
        if not p.exists():
            continue
        seen[c] = {"contest_number": c, "image_path": str(p)}
    out = list(seen.values())
    if max_cartoons > 0:
        out = out[:max_cartoons]
    return out


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    user_text = "Write a funny one-line caption for this New Yorker-style cartoon."

    for split in args.splits:
        cartoons = load_cartoons(args.data_root, split, args.max_cartoons)
        print(f"[{args.cell}] split={split}: {len(cartoons)} cartoons × {args.num_samples} samples ({args.model})", flush=True)
        out_path = args.out_root / f"{args.cell}_{split}.jsonl"
        with out_path.open("w") as f:
            for c_idx, c in enumerate(cartoons):
                img_b64 = encode_image(Path(c["image_path"]))
                for s_idx in range(args.num_samples):
                    text = call_openrouter(
                        args.model, SYSTEM_NO_THINKING, user_text, img_b64,
                        args.temperature, args.max_tokens,
                    )
                    row = {
                        "cell": args.cell,
                        "split": split,
                        "contest_number": c["contest_number"],
                        "image_path": c["image_path"],
                        "user_prompt": user_text,
                        "sample_idx": s_idx,
                        "completion": text,
                        "caption": extract_caption(text),
                        "thinking": None,
                        "completion_tokens": None,
                        "finish_reason": "api",
                    }
                    f.write(json.dumps(row) + "\n")
                if (c_idx + 1) % 10 == 0:
                    print(f"  [{split}] {c_idx + 1}/{len(cartoons)}", flush=True)
        print(f"  wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
