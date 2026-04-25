import argparse
import math
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset, load_from_disk

SEED = 42
MIN_VOTES = 10
MIN_CAPTIONS_PER_CONTEST = 20
TOP_CAPTIONS_PER_CARTOON = 3
SIGMA_THRESHOLD = 3.0
MAX_BT_PAIRS_PER_CONTEST = 1000
MAX_PAIR_SAMPLING_ATTEMPTS_MULTIPLIER = 200
DEFAULT_OUTPUT_DIR = Path("./data")
GENERATION_INSTRUCTION = (
    "Write a funny one-line caption for this New Yorker-style cartoon."
)

random.seed(SEED)
np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download the New Yorker caption dataset, save images, create a "
            "caption-generation dataset, and format Bradley-Terry pairs."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where images and processed datasets will be written.",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=MIN_VOTES,
        help="Drop captions with fewer than this many votes.",
    )
    parser.add_argument(
        "--min-captions-per-contest",
        type=int,
        default=MIN_CAPTIONS_PER_CONTEST,
        help="Require at least this many surviving captions before using a contest.",
    )
    parser.add_argument(
        "--top-captions-per-cartoon",
        type=int,
        default=TOP_CAPTIONS_PER_CARTOON,
        help="How many top captions to keep per image for the caption model dataset.",
    )
    parser.add_argument(
        "--sigma-threshold",
        type=float,
        default=SIGMA_THRESHOLD,
        help="Minimum sigma separation required for a Bradley-Terry pair.",
    )
    parser.add_argument(
        "--max-bt-pairs-per-contest",
        type=int,
        default=MAX_BT_PAIRS_PER_CONTEST,
        help="Cap on preference pairs sampled from a single contest.",
    )
    return parser.parse_args()


def download_all():
    print("=" * 60)
    print("Step 1: Downloading datasets")
    print("=" * 60)

    print("  Downloading ratings (1_rating)...")
    ratings = load_dataset("yguooo/newyorker_caption_ranking", "1_rating")

    print("  Downloading descriptions (2_gpt4o_description)...")
    descriptions = load_dataset("yguooo/newyorker_caption_ranking", "2_gpt4o_description")

    print("  Downloading cartoons (3_cartoons parquet)...")
    cartoons = load_dataset(
        "parquet",
        data_files={
            "train": (
                "hf://datasets/yguooo/newyorker_caption_ranking/"
                "cartoons/train-00000-of-00001.parquet"
            ),
            "validation": (
                "hf://datasets/yguooo/newyorker_caption_ranking/"
                "cartoons/validation-00000-of-00001.parquet"
            ),
            "test": (
                "hf://datasets/yguooo/newyorker_caption_ranking/"
                "cartoons/test-00000-of-00001.parquet"
            ),
        },
    )

    return ratings, descriptions, cartoons


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item is not None)
    return str(value)


def build_context(row):
    return {
        "description": normalize_text(row.get("canny")),
        "unusual": normalize_text(row.get("uncanny")),
        "location": normalize_text(row.get("location")),
        "entities": normalize_text(row.get("entities")),
    }


def build_prompt(context):
    prompt_lines = [GENERATION_INSTRUCTION]
    if context["description"]:
        prompt_lines.append(f"Scene: {context['description']}")
    if context["unusual"]:
        prompt_lines.append(f"Twist: {context['unusual']}")
    if context["location"]:
        prompt_lines.append(f"Location: {context['location']}")
    if context["entities"]:
        prompt_lines.append(f"Entities: {context['entities']}")
    return "\n".join(prompt_lines)


def find_image_column(split):
    for candidate in ("image", "cartoon", "img", "picture"):
        if candidate in split.column_names:
            return candidate
    return None


def extract_contest_number(row, image_col=None):
    contest_number = row.get("contest_number")
    if contest_number is not None:
        return int(contest_number)

    if image_col is None:
        return None

    image = row.get(image_col)
    if isinstance(image, dict):
        image_path = image.get("path")
        if image_path:
            stem = Path(str(image_path)).stem
            if stem.isdigit():
                return int(stem)

    return None


def write_image_file(image, image_path):
    if hasattr(image, "save"):
        image.save(image_path)
        return True

    if isinstance(image, bytes):
        image_path.write_bytes(image)
        return True

    if isinstance(image, dict):
        image_bytes = image.get("bytes")
        if image_bytes is not None:
            image_path.write_bytes(image_bytes)
            return True

    return False


def save_images(cartoons, descriptions, image_dir):
    print("\n" + "=" * 60)
    print("Step 2: Saving cartoon images")
    print("=" * 60)

    image_dir.mkdir(parents=True, exist_ok=True)
    available_images = set()
    saved = 0

    for split_name, split in cartoons.items():
        print(f"  Processing {split_name} split ({len(split)} cartoons)...")
        image_col = find_image_column(split)
        print(f"    Columns: {split.column_names}")

        if image_col is None:
            print(f"    WARNING: Could not find an image column for {split_name}")
            continue

        description_rows = descriptions[split_name]
        if len(description_rows) != len(split):
            print(
                f"    WARNING: description/cartoon length mismatch for {split_name} "
                f"({len(description_rows)} vs {len(split)})"
            )

        for idx, row in enumerate(split):
            contest_number = extract_contest_number(row, image_col=image_col)
            if contest_number is None and idx < len(description_rows):
                description_contest = description_rows[idx].get("contest_number")
                if description_contest is not None:
                    contest_number = int(description_contest)

            image = row.get(image_col)
            if contest_number is None or image is None:
                continue

            image_path = image_dir / f"{contest_number}.png"
            if not image_path.exists():
                if write_image_file(image, image_path):
                    saved += 1

            available_images.add(int(contest_number))

    print(f"  Saved {saved} new images to {image_dir}/")
    print(f"  Total image files available: {len(available_images)}")
    return available_images


def index_descriptions(description_split):
    indexed = {}
    for row in description_split:
        contest_number = row.get("contest_number")
        if contest_number is None:
            continue
        indexed[int(contest_number)] = row
    return indexed


def get_relative_image_path(contest_number):
    return f"images/{contest_number}.png"


def get_standard_error(row):
    std = row.get("std")
    if std is None:
        std = row.get("precision")
    votes = row.get("votes")
    if std is None:
        return None
    if votes is None or votes <= 0:
        return float(std)
    return float(std)


def is_valid_pair(chosen_row, rejected_row, sigma_threshold):
    chosen_mean = float(chosen_row["mean"])
    rejected_mean = float(rejected_row["mean"])
    mean_gap = chosen_mean - rejected_mean
    if mean_gap <= 0:
        return False, mean_gap, None

    chosen_se = get_standard_error(chosen_row)
    rejected_se = get_standard_error(rejected_row)
    if chosen_se is None or rejected_se is None:
        return True, mean_gap, None

    pooled_se = math.sqrt(chosen_se**2 + rejected_se**2)
    sigma_gap = mean_gap / pooled_se if pooled_se > 0 else float("inf")
    return sigma_gap >= sigma_threshold, mean_gap, sigma_gap


def build_caption_rows_for_contest(contest_number, rows, description_row, args):
    filtered = [row for row in rows if row.get("votes", 0) >= args.min_votes]
    if len(filtered) < args.min_captions_per_contest:
        return []

    filtered.sort(key=lambda row: float(row["mean"]), reverse=True)
    context = build_context(description_row)
    prompt = build_prompt(context)
    examples = []

    for rank, row in enumerate(filtered[: args.top_captions_per_cartoon], start=1):
        examples.append(
            {
                "contest_number": contest_number,
                "image_path": get_relative_image_path(contest_number),
                "prompt": prompt,
                "caption": row["caption"],
                "label_rank": rank,
                "caption_mean": float(row["mean"]),
                "caption_votes": int(row["votes"]),
                "scene_description": context["description"],
                "scene_twist": context["unusual"],
                "location": context["location"],
                "entities": context["entities"],
            }
        )

    return examples


def build_bt_rows_for_contest(contest_number, rows, description_row, args):
    filtered = [row for row in rows if row.get("votes", 0) >= args.min_votes]
    if len(filtered) < args.min_captions_per_contest:
        return []

    filtered.sort(key=lambda row: float(row["mean"]), reverse=True)
    context = build_context(description_row)
    prompt = build_prompt(context)
    max_pairs = args.max_bt_pairs_per_contest
    if max_pairs <= 0:
        return []

    pair_rows = []
    seen_index_pairs = set()
    max_attempts = max(max_pairs * MAX_PAIR_SAMPLING_ATTEMPTS_MULTIPLIER, len(filtered) * 20)
    attempts = 0
    n = len(filtered)

    while len(pair_rows) < max_pairs and attempts < max_attempts:
        attempts += 1
        chosen_idx = random.randrange(0, n - 1)
        rejected_idx = random.randrange(chosen_idx + 1, n)
        pair_key = (chosen_idx, rejected_idx)
        if pair_key in seen_index_pairs:
            continue
        seen_index_pairs.add(pair_key)

        chosen_row = filtered[chosen_idx]
        rejected_row = filtered[rejected_idx]
        keep_pair, mean_gap, sigma_gap = is_valid_pair(
            chosen_row, rejected_row, args.sigma_threshold
        )
        if not keep_pair:
            continue

        pair_rows.append(
            {
                "contest_number": contest_number,
                "image_path": get_relative_image_path(contest_number),
                "prompt": prompt,
                "chosen": chosen_row["caption"],
                "rejected": rejected_row["caption"],
                "chosen_mean": float(chosen_row["mean"]),
                "rejected_mean": float(rejected_row["mean"]),
                "chosen_votes": int(chosen_row["votes"]),
                "rejected_votes": int(rejected_row["votes"]),
                "mean_gap": mean_gap,
                "sigma_gap": sigma_gap,
                "scene_description": context["description"],
                "scene_twist": context["unusual"],
                "location": context["location"],
                "entities": context["entities"],
            }
        )

    return pair_rows


def build_processed_rows(ratings_split, descriptions_by_contest, available_images, split_name, args):
    print(f"\n  Building processed rows for {split_name}...")
    caption_rows = []
    bt_rows = []

    current_contest = None
    current_rows = []

    def flush_contest(contest_number, rows):
        if contest_number is None or contest_number not in available_images:
            return
        description_row = descriptions_by_contest.get(contest_number)
        if description_row is None:
            return

        caption_rows.extend(
            build_caption_rows_for_contest(contest_number, rows, description_row, args)
        )
        bt_rows.extend(
            build_bt_rows_for_contest(contest_number, rows, description_row, args)
        )

    for row in ratings_split:
        contest_number = row.get("contest_number")
        if contest_number is None:
            continue
        contest_number = int(contest_number)

        if current_contest is None:
            current_contest = contest_number

        if contest_number != current_contest:
            flush_contest(current_contest, current_rows)
            current_contest = contest_number
            current_rows = []

        current_rows.append(row)

    flush_contest(current_contest, current_rows)

    print(f"    Total caption rows: {len(caption_rows):,}")
    print(f"    Total Bradley-Terry pairs: {len(bt_rows):,}")
    return caption_rows, bt_rows


def save_dataset(rows, save_path):
    if not rows:
        return False
    dataset = Dataset.from_list(rows)
    dataset.save_to_disk(str(save_path))
    return True


def save_rows_to_parquet(rows, save_path, chunk_size=5000):
    if not rows:
        return False

    writer = None
    try:
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            table = pa.Table.from_pylist(chunk)
            if writer is None:
                writer = pq.ParquetWriter(save_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    return True


def summarize_dataset(path):
    if not path.exists():
        return None
    if path.is_file() and path.suffix == ".parquet":
        return pq.ParquetFile(path).metadata.num_rows
    return len(load_from_disk(str(path)))


def main():
    args = parse_args()
    output_dir = args.output_dir
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    ratings, descriptions, cartoons = download_all()
    available_images = save_images(cartoons, descriptions, image_dir)

    print("\n" + "=" * 60)
    print("Step 3: Creating processed datasets")
    print("=" * 60)

    split_names = [split for split in ("train", "validation", "test") if split in ratings]

    for split_name in split_names:
        descriptions_by_contest = index_descriptions(descriptions[split_name])
        caption_rows, bt_rows = build_processed_rows(
            ratings[split_name],
            descriptions_by_contest,
            available_images,
            split_name,
            args,
        )
        caption_save_path = output_dir / f"caption_sft_{split_name}"
        if save_dataset(caption_rows, caption_save_path):
            print(f"    Saved caption dataset to {caption_save_path}/")
            sample = caption_rows[0]
            print(f"    Sample caption row: {sample['image_path']} -> {sample['caption']}")
        else:
            print(f"    WARNING: No caption rows for {split_name}")

        bt_save_path = output_dir / f"bt_pairs_{split_name}.parquet"
        if save_rows_to_parquet(bt_rows, bt_save_path):
            print(f"    Saved Bradley-Terry dataset to {bt_save_path}")
            sample = bt_rows[0]
            print(
                "    Sample BT pair: "
                f"{sample['chosen']} || {sample['rejected']}"
            )
        else:
            print(f"    WARNING: No Bradley-Terry pairs for {split_name}")

    print("\n" + "=" * 60)
    print("Done! Final summary")
    print("=" * 60)
    print(f"  Images: {len(list(image_dir.glob('*.png'))):,} in {image_dir}/")

    for split_name in split_names:
        caption_path = output_dir / f"caption_sft_{split_name}"
        bt_path = output_dir / f"bt_pairs_{split_name}.parquet"
        caption_count = summarize_dataset(caption_path)
        bt_count = summarize_dataset(bt_path)

        if caption_count is not None:
            print(f"  {split_name:12s} caption rows: {caption_count:,}")
        if bt_count is not None:
            print(f"  {split_name:12s} BT pairs:    {bt_count:,}")


if __name__ == "__main__":
    main()
