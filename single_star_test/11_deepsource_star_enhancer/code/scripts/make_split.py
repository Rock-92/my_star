from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed 11 DeepSource-style train/val/test split.")
    parser.add_argument("--data-root", type=Path, default=Path("single_star_test/data/data_model"))
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=300)
    parser.add_argument("--val-count", type=int, default=30)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-column", default="group")
    return parser.parse_args()


def load_rows(manifest: Path) -> list[dict[str, str]]:
    if not manifest.exists():
        raise FileNotFoundError(f"missing manifest: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def take_round_robin(
    buckets: dict[str, list[dict[str, str]]],
    count: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        rng.shuffle(keys)
        for key in keys:
            if len(selected) >= count:
                break
            if buckets[key]:
                selected.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) < count:
        remaining = [row for bucket in buckets.values() for row in bucket]
        rng.shuffle(remaining)
        need = count - len(selected)
        selected.extend(remaining[:need])
        selected_ids = {id(row) for row in selected}
        for key in keys:
            buckets[key] = [row for row in buckets[key] if id(row) not in selected_ids]
    if len(selected) != count:
        raise RuntimeError(f"requested {count} rows, got {len(selected)}")
    return selected


def assign_split(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    total = int(args.train_count) + int(args.val_count) + int(args.test_count)
    if len(rows) < total:
        raise RuntimeError(f"need {total} rows, only have {len(rows)}")
    rng = random.Random(int(args.seed))
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(args.stratify_column, ""))].append(dict(row))
    for bucket in buckets.values():
        rng.shuffle(bucket)

    train = take_round_robin(buckets, int(args.train_count), rng)
    val = take_round_robin(buckets, int(args.val_count), rng)
    test = take_round_robin(buckets, int(args.test_count), rng)
    split_rows: list[dict[str, str]] = []
    for split_name, group_rows in (("train", train), ("val", val), ("test", test)):
        for row in group_rows:
            row["split11"] = split_name
            split_rows.append(row)
    return split_rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.data_root / "manifest.csv")
    split_rows = assign_split(rows, args)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split11"] + [key for key in rows[0].keys() if key != "split11"]
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)
    counts = {name: sum(row["split11"] == name for row in split_rows) for name in ("train", "val", "test")}
    summary = {
        "data_root": str(args.data_root),
        "out_csv": str(args.out_csv),
        "seed": int(args.seed),
        "counts": counts,
        "stratify_column": args.stratify_column,
    }
    (args.out_csv.with_suffix(".json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
