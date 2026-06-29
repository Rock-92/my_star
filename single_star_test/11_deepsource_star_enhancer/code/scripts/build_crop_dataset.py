from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "single_star_test" / "11_deepsource_star_enhancer" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from deepsource_star.data import DeepSourceStarDataset  # noqa: E402


def default_data_root() -> Path:
    candidates = [
        REPO_ROOT / "data_model",
        REPO_ROOT / "single_star_test" / "data" / "data_model",
    ]
    for path in candidates:
        if (path / "manifest.csv").exists():
            return path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute fixed DeepSource crop datasets.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-split-reason", default="train")
    parser.add_argument("--val-split-reason", default="frame_holdout")
    parser.add_argument("--test-split-reason", default="coord_holdout")
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--val-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=200)
    parser.add_argument("--crops-per-image", type=int, default=20)
    parser.add_argument("--target-mode", choices=["deepsource", "gaussian", "delta"], default="deepsource")
    parser.add_argument("--gaussian-sigma", type=float, default=1.2)
    parser.add_argument("--triangle-radius", type=float, default=6.0)
    parser.add_argument("--background-level", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_dataset(args: argparse.Namespace, split_reason: str, count: int) -> DeepSourceStarDataset:
    return DeepSourceStarDataset(
        data_root=args.data_root,
        split_reason=split_reason,
        count=count,
        crop_size=args.crop_size,
        crops_per_image=args.crops_per_image,
        target_mode=args.target_mode,
        gaussian_sigma=args.gaussian_sigma,
        triangle_radius=args.triangle_radius,
        background_level=args.background_level,
        alpha=args.alpha,
        seed=args.seed,
    )


def write_split(out_dir: Path, split_name: str, dataset: DeepSourceStarDataset) -> dict[str, object]:
    images = np.empty((len(dataset), 1, dataset.crop_size, dataset.crop_size), dtype=np.float32)
    targets = np.empty_like(images)
    sample_ids: list[str] = []
    rows = []
    for index in tqdm(range(len(dataset)), desc=f"build {split_name}", dynamic_ncols=True):
        item = dataset[index]
        image = item["image"].numpy().astype(np.float32)
        target = item["target"].numpy().astype(np.float32)
        sample_id = str(item["sample_id"])
        images[index] = image
        targets[index] = target
        sample_ids.append(sample_id)
        rows.append({"index": index, "sample_id": sample_id})

    out_path = out_dir / f"{split_name}.npz"
    np.savez(out_path, images=images, targets=targets, sample_ids=np.asarray(sample_ids, dtype="U64"))
    with (out_dir / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    return {
        "split": split_name,
        "path": str(out_path),
        "images": len(dataset.rows),
        "crops": len(dataset),
        "shape": list(images.shape),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = [
        ("train", args.train_split_reason, args.train_samples),
        ("val", args.val_split_reason, args.val_samples),
        ("test", args.test_split_reason, args.test_samples),
    ]
    summary = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "splits": [],
    }
    for split_name, split_reason, count in splits:
        dataset = make_dataset(args, split_reason, count)
        summary["splits"].append(write_split(args.out_dir, split_name, dataset))
    (args.out_dir / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
