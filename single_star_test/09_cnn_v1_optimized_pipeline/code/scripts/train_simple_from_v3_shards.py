from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


class SimpleCandidateScorer(nn.Module):
    """The original 01 single-channel CNN architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the original simple CNN on the optimized sharded candidate "
            "dataset. Class 2 is positive; class 0/1 are negatives."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=31)
    parser.add_argument("--pos-neg-ratio", type=float, default=1.0)
    parser.add_argument("--score-thresholds", default="0.05:0.95:0.01")
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_thresholds(text: str) -> list[float]:
    text = str(text).strip()
    if ":" in text and "," not in text:
        start, stop, step = (float(value) for value in text.split(":"))
        return np.arange(start, stop + step * 0.5, step, dtype=np.float64).tolist()
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def load_metadata(data_dir: Path) -> dict[str, object]:
    path = data_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"missing metadata.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def shard_paths(data_dir: Path, metadata: dict[str, object], key: str) -> list[Path]:
    values = metadata.get(key, [])
    if not isinstance(values, list):
        raise ValueError(f"metadata field {key!r} must be a list")
    paths = [(data_dir / str(value)).resolve() for value in values]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing shard(s): {missing[:3]}")
    return paths


def center_crop_channel(arrays: dict[str, np.ndarray], patch_size: int) -> np.ndarray:
    if "patches_small" in arrays:
        patches = arrays["patches_small"].astype(np.float32)
        return patches[:, :1]

    patches_large = arrays["patches_large"].astype(np.float32)
    context = patches_large.shape[-1]
    radius = patch_size // 2
    center = context // 2
    return patches_large[
        :,
        :1,
        center - radius : center + radius + 1,
        center - radius : center + radius + 1,
    ].astype(np.float32)


class ShardBinaryDataset(IterableDataset):
    def __init__(
        self,
        paths: list[Path],
        patch_size: int,
        seed: int,
        pos_neg_ratio: float,
        balance: bool,
    ) -> None:
        self.paths = paths
        self.patch_size = int(patch_size)
        self.seed = int(seed)
        self.pos_neg_ratio = float(pos_neg_ratio)
        self.balance = bool(balance)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        rng = np.random.default_rng(self.seed + self.epoch * 1009 + worker_id)
        order = rng.permutation(len(self.paths))[worker_id::worker_count]
        for shard_index in order:
            with np.load(self.paths[int(shard_index)]) as data:
                arrays = {key: data[key] for key in data.files}
            patches = center_crop_channel(arrays, self.patch_size)
            labels = (arrays["classes"].astype(np.int64) == 2).astype(np.float32)

            if self.balance:
                pos = np.flatnonzero(labels > 0.5)
                neg = np.flatnonzero(labels <= 0.5)
                if len(pos) and len(neg):
                    neg_count = min(len(neg), max(1, int(round(len(pos) * self.pos_neg_ratio))))
                    neg = rng.choice(neg, size=neg_count, replace=False)
                    indexes = np.concatenate([pos, neg])
                else:
                    indexes = np.arange(len(labels))
            else:
                indexes = np.arange(len(labels))
            rng.shuffle(indexes)
            for index in indexes:
                index = int(index)
                yield torch.from_numpy(patches[index]), torch.tensor(labels[index], dtype=torch.float32)


def load_eval_arrays(paths: list[Path], patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    patch_parts = []
    label_parts = []
    for path in paths:
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        patch_parts.append(center_crop_channel(arrays, patch_size))
        label_parts.append((arrays["classes"].astype(np.int64) == 2).astype(np.float32))
    if not patch_parts:
        raise ValueError("no validation shards")
    return np.concatenate(patch_parts, axis=0), np.concatenate(label_parts, axis=0)


def evaluate_patch_f1(
    model: nn.Module,
    patches: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.from_numpy(patches[start : start + batch_size]).to(device)
            scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    score_array = np.concatenate(scores)
    best = {"threshold": -1.0, "patch_f1": -1.0, "precision": 0.0, "recall": 0.0}
    positives = labels > 0.5
    for threshold in thresholds:
        pred = score_array >= float(threshold)
        tp = int(np.logical_and(pred, positives).sum())
        fp = int(np.logical_and(pred, ~positives).sum())
        fn = int(np.logical_and(~pred, positives).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["patch_f1"]:
            best = {
                "threshold": float(threshold),
                "patch_f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
    return best


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.data_dir)
    train_paths = shard_paths(args.data_dir, metadata, "train_shards")
    val_paths = shard_paths(args.data_dir, metadata, "val_shards")

    train_dataset = ShardBinaryDataset(
        train_paths,
        patch_size=args.patch_size,
        seed=args.seed,
        pos_neg_ratio=args.pos_neg_ratio,
        balance=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_patches, val_labels = load_eval_arrays(val_paths, args.patch_size)
    thresholds = parse_thresholds(args.score_thresholds)

    model = SimpleCandidateScorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        losses = []
        for patches, labels in train_loader:
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(patches), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate_patch_f1(model, val_patches, val_labels, thresholds, device, args.eval_batch_size)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            **metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"patch_f1={metrics['patch_f1']:.4f} p={metrics['precision']:.4f} "
            f"r={metrics['recall']:.4f} t={metrics['threshold']:.2f}",
            flush=True,
        )

        checkpoint = {
            "model": model.state_dict(),
            "model_type": "legacy",
            "input_channels": 1,
            "feature_dim": 0,
            "patch_size": args.patch_size,
            "source_data_dir": str(args.data_dir),
            "source_schema_version": metadata.get("schema_version"),
            "candidate_methods": metadata.get("candidate_methods"),
            "dedup_radius_px": metadata.get("dedup_radius_px"),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        }
        torch.save(checkpoint, args.out_dir / "candidate_scorer_last.pt")
        if metrics["patch_f1"] > best_f1:
            best_f1 = metrics["patch_f1"]
            torch.save(checkpoint, args.out_dir / "candidate_scorer_best.pt")

    torch.save({"history": history}, args.out_dir / "history.pt")
    with (args.out_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()) if history else ["epoch"])
        writer.writeheader()
        writer.writerows(history)
    (args.out_dir / "summary.json").write_text(
        json.dumps({"best_patch_f1": best_f1, "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
