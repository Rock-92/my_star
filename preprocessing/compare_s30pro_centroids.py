from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.mask_generator import read_gray_image, tetra3_like_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare single-frame star extraction with stacked-image extraction.",
    )
    parser.add_argument("--root", type=Path, default=Path("data/data_S30Pro"))
    parser.add_argument("--ext", default=".fit", help="Image extension to use, e.g. .fit or .jpg.")
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--filtsize", type=int, default=25)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--max-area", type=int, default=100)
    parser.add_argument("--background-mode", default="local_mean")
    parser.add_argument("--fit-channel-mode", default="mean", choices=("mean", "first", "max", "luma"))
    parser.add_argument(
        "--include-all-singles",
        action="store_true",
        help="Do not filter _sub frames by exposure/filter/date parsed from the stacked filename.",
    )
    return parser.parse_args()


def formal_suffix(value: str) -> str:
    value = value.strip().lower()
    if not value.startswith("."):
        value = "." + value
    return value


def image_files(directory: Path, suffix: str) -> list[Path]:
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == suffix)
    if suffix in {".jpg", ".jpeg"}:
        files = [path for path in files if not path.name.endswith("_thn.jpg")]
    return files


def stack_metadata(path: Path) -> dict[str, Any]:
    match = re.search(r"^Stacked_(\d+)_(.+?)_(\d+(?:\.\d+)s)_([^_]+)_(\d{8})-", path.name)
    if not match:
        return {}
    return {
        "expected_singles": int(match.group(1)),
        "target": match.group(2),
        "exposure": match.group(3),
        "filter": match.group(4),
        "date": match.group(5),
    }


def matching_singles(stack_file: Path, singles: list[Path], include_all: bool) -> tuple[list[Path], int | None]:
    metadata = stack_metadata(stack_file)
    expected = metadata.get("expected_singles")
    if include_all or not metadata:
        return singles, expected

    token = f"_{metadata['exposure']}_{metadata['filter']}_{metadata['date']}-"
    filtered = [path for path in singles if token in path.name]
    if filtered:
        singles = filtered
    if expected is not None and len(singles) > expected:
        singles = singles[:expected]
    return singles, expected


def count_stars(path: Path, args: argparse.Namespace) -> int:
    image = read_gray_image(path, fit_channel_mode=args.fit_channel_mode)
    result = tetra3_like_mask(
        image,
        sigma=args.sigma,
        filtsize=args.filtsize,
        background_mode=args.background_mode,
        min_area=args.min_area,
        max_area=args.max_area,
    )
    return int(len(result.centroids_yx))


def summarize_group(stack_dir: Path, args: argparse.Namespace, suffix: str) -> dict[str, Any] | None:
    sub_dir = stack_dir.parent / f"{stack_dir.name}_sub"
    if not sub_dir.exists():
        return None

    stack_files = image_files(stack_dir, suffix)
    single_files = image_files(sub_dir, suffix)
    if not stack_files or not single_files:
        return None

    stack_file = stack_files[0]
    single_files, expected = matching_singles(stack_file, single_files, args.include_all_singles)
    stack_count = count_stars(stack_file, args)
    single_counts = [count_stars(path, args) for path in single_files]
    percentages = [count / stack_count * 100.0 if stack_count else 0.0 for count in single_counts]

    return {
        "set": stack_dir.name,
        "stack_file": stack_file.name,
        "stack_count": stack_count,
        "expected_singles": expected,
        "used_singles": len(single_counts),
        "single_count_mean": statistics.mean(single_counts) if single_counts else 0.0,
        "single_count_median": statistics.median(single_counts) if single_counts else 0.0,
        "single_count_min": min(single_counts) if single_counts else 0,
        "single_count_max": max(single_counts) if single_counts else 0,
        "pct_mean": statistics.mean(percentages) if percentages else 0.0,
        "pct_median": statistics.median(percentages) if percentages else 0.0,
        "pct_min": min(percentages) if percentages else 0.0,
        "pct_max": max(percentages) if percentages else 0.0,
        "_single_count_sum": sum(single_counts),
        "_percentages": percentages,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    suffix = formal_suffix(args.ext)
    groups = []
    all_percentages = []
    weighted_single_count = 0
    weighted_stack_count = 0

    stack_dirs = sorted(
        [path for path in args.root.iterdir() if path.is_dir() and not path.name.endswith("_sub")],
        key=lambda path: path.name,
    )
    for stack_dir in stack_dirs:
        group = summarize_group(stack_dir, args, suffix)
        if group is None:
            continue
        percentages = group.pop("_percentages")
        single_count_sum = group.pop("_single_count_sum")
        groups.append(group)
        all_percentages.extend(percentages)
        weighted_single_count += int(single_count_sum)
        weighted_stack_count += int(group["stack_count"] * group["used_singles"])

    output = {
        "root": str(args.root),
        "ext": suffix,
        "params": {
            "sigma": args.sigma,
            "filtsize": args.filtsize,
            "background_mode": args.background_mode,
            "min_area": args.min_area,
            "max_area": args.max_area,
            "fit_channel_mode": args.fit_channel_mode,
        },
        "groups": groups,
        "overall": {
            "single_images": sum(int(group["used_singles"]) for group in groups),
            "mean_pct_unweighted": statistics.mean(all_percentages) if all_percentages else 0.0,
            "median_pct_unweighted": statistics.median(all_percentages) if all_percentages else 0.0,
            "weighted_pct_by_counts": (
                weighted_single_count / weighted_stack_count * 100.0 if weighted_stack_count else 0.0
            ),
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
