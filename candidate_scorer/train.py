from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candidate_scorer.evaluate import evaluate_models, load_rows, parse_thresholds, select_rows
from candidate_scorer.model import CenterAwareScorer
from candidate_scorer.pipeline import add_center_channels
from star_unet.evaluate import resolve_path
from star_unet.train import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the center-aware dual-scale candidate scorer.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/candidate_scorer_v2"))
    parser.add_argument("--hard-negative-data-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=Path("star_unet/config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/candidate_scorer_v2"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-width", type=int, default=24)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--quality-loss-weight", type=float, default=0.5)
    parser.add_argument("--offset-loss-weight", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--coord-eval-interval", type=int, default=1)
    parser.add_argument("--coord-eval-split-reason", default="frame_holdout")
    parser.add_argument("--coord-eval-count", type=int, default=0)
    parser.add_argument("--score-thresholds", default="0.05:0.95:0.02")
    parser.add_argument("--nms-radius-px", type=float, default=3.5)
    parser.add_argument("--match-radius-px", type=float, default=4.0)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--candidate-cache-dir", type=Path, default=Path("data/candidate_cache"))
    return parser.parse_args()


class BalancedShardDataset(IterableDataset):
    def __init__(
        self,
        shard_paths: list[Path],
        normalization: dict[str, object],
        seed: int,
        balance: bool,
        patch_size: int,
    ) -> None:
        self.shard_paths = shard_paths
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.maximum(np.asarray(normalization["std"], dtype=np.float32), 1e-6)
        self.seed = int(seed)
        self.balance = bool(balance)
        self.patch_size = int(patch_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        rng = np.random.default_rng(self.seed + self.epoch + worker_id * 100003)
        order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.shard_paths))
        order = order[worker_id::worker_count]
        for shard_index in order:
            with np.load(self.shard_paths[int(shard_index)]) as data:
                arrays = {key: data[key] for key in data.files}
            classes = arrays["classes"].astype(np.int64)
            if self.balance:
                class_indexes = [np.flatnonzero(classes == class_id) for class_id in range(3)]
                nonempty = [indexes for indexes in class_indexes if len(indexes)]
                target_count = max(len(indexes) for indexes in nonempty)
                indexes = np.concatenate([
                    rng.choice(values, size=target_count, replace=len(values) < target_count)
                    for values in nonempty
                ])
            else:
                indexes = np.arange(len(classes))
            rng.shuffle(indexes)
            numeric = (
                arrays["numeric_features"].astype(np.float32) - self.mean
            ) / self.std
            for index in indexes:
                index = int(index)
                large_base = arrays["patches_large"][index].astype(np.float32)
                if large_base.shape[0] == 3:
                    context = large_base.shape[-1]
                    radius = self.patch_size // 2
                    center = context // 2
                    small_base = large_base[
                        :, center - radius : center + radius + 1, center - radius : center + radius + 1
                    ]
                    small_patch = add_center_channels(small_base[None])[0]
                    large_patch = add_center_channels(large_base[None])[0]
                else:
                    large_patch = large_base
                    small_patch = arrays["patches_small"][index].astype(np.float32)
                yield (
                    torch.from_numpy(small_patch),
                    torch.from_numpy(large_patch),
                    torch.from_numpy(numeric[index]),
                    torch.tensor(classes[index], dtype=torch.long),
                    torch.tensor(arrays["quality"][index], dtype=torch.float32),
                    torch.from_numpy(arrays["offsets_yx"][index].astype(np.float32)),
                )


def focal_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, gamma: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none")
    probability = torch.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
    return (((1.0 - probability) ** gamma) * ce).mean()


def compute_loss(
    output: dict[str, torch.Tensor],
    classes: torch.Tensor,
    quality: torch.Tensor,
    offsets: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    class_loss = focal_cross_entropy(output["class_logits"], classes, args.focal_gamma)
    quality_loss = F.binary_cross_entropy_with_logits(output["quality_logit"], quality)
    positive = classes == 2
    if torch.any(positive):
        offset_loss = F.smooth_l1_loss(output["offset_yx"][positive], offsets[positive])
    else:
        offset_loss = output["offset_yx"].sum() * 0.0
    total = (
        class_loss
        + args.quality_loss_weight * quality_loss
        + args.offset_loss_weight * offset_loss
    )
    return total, {
        "class_loss": float(class_loss.detach()),
        "quality_loss": float(quality_loss.detach()),
        "offset_loss": float(offset_loss.detach()),
    }


def validation_loss(
    model: CenterAwareScorer,
    dataset: BalancedShardDataset,
    batch_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    total_loss = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for small, large, numeric, classes, quality, offsets in loader:
            small, large, numeric = small.to(device), large.to(device), numeric.to(device)
            classes, quality, offsets = classes.to(device), quality.to(device), offsets.to(device)
            loss, _ = compute_loss(model(small, large, numeric), classes, quality, offsets, args)
            total_loss += float(loss) * len(classes)
            count += len(classes)
    return total_loss / max(count, 1)


def _eval_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dedup_radius_px=2.5,
        target_threshold=0.5,
        min_distance=3.0,
        match_radius_px=args.match_radius_px,
        nms_radius_px=args.nms_radius_px,
        edge_margin_px=128,
        batch_size=args.eval_batch_size,
        cache_dir=args.candidate_cache_dir,
        no_cache=False,
        bootstrap_samples=0,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dir = resolve_path(args.data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", 0)) not in (2, 3):
        raise ValueError("center-aware training requires schema_version=2 or 3 dataset")
    train_shards = [data_dir / value for value in metadata["train_shards"]]
    val_shards = [data_dir / value for value in metadata["val_shards"]]
    if args.hard_negative_data_dir is not None:
        hard_dir = resolve_path(args.hard_negative_data_dir)
        hard_meta = json.loads((hard_dir / "metadata.json").read_text(encoding="utf-8"))
        train_shards.extend(hard_dir / value for value in hard_meta["train_shards"])

    normalization = metadata["numeric_normalization"]
    train_dataset = BalancedShardDataset(
        train_shards, normalization, args.seed, balance=True, patch_size=int(metadata["patch_size"])
    )
    val_dataset = BalancedShardDataset(
        val_shards, normalization, args.seed + 10000, balance=False,
        patch_size=int(metadata["patch_size"]),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = resolve_device(args.device)
    model = CenterAwareScorer(
        input_channels=int(metadata["input_channels"]),
        feature_dim=int(metadata["numeric_feature_dim"]),
        width=args.model_width,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    start_epoch = 1
    best_f1 = -1.0
    if args.resume is not None:
        checkpoint = torch.load(resolve_path(args.resume), map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_f1 = float(checkpoint.get("best_coord_f1", -1.0))

    data_root = resolve_path(args.data_root)
    coord_rows = select_rows(
        load_rows(data_root, args.coord_eval_split_reason), "", args.coord_eval_count
    )
    config = load_config(resolve_path(args.config))
    thresholds = parse_thresholds(args.score_thresholds)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total = 0
        for batch_index, (small, large, numeric, classes, quality, offsets) in enumerate(loader, start=1):
            small = small.to(device, non_blocking=True)
            large = large.to(device, non_blocking=True)
            numeric = numeric.to(device, non_blocking=True)
            classes = classes.to(device, non_blocking=True)
            quality = quality.to(device, non_blocking=True)
            offsets = offsets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss, parts = compute_loss(model(small, large, numeric), classes, quality, offsets, args)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(classes)
            total += len(classes)
            if args.log_interval > 0 and batch_index % args.log_interval == 0:
                print(
                    f"  epoch={epoch} batch={batch_index} loss={total_loss / max(total, 1):.4f}"
                    f" class={parts['class_loss']:.4f} quality={parts['quality_loss']:.4f}"
                    f" offset={parts['offset_loss']:.4f}",
                    flush=True,
                )
        scheduler.step()
        val_loss = validation_loss(model, val_dataset, args.batch_size, device, args)
        checkpoint = {
            "schema_version": int(metadata["schema_version"]),
            "model_type": "center_aware_v2",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "input_channels": int(metadata["input_channels"]),
            "feature_dim": int(metadata["numeric_feature_dim"]),
            "model_width": args.model_width,
            "patch_size": int(metadata["patch_size"]),
            "context_patch_size": int(metadata["context_patch_size"]),
            "candidate_methods": str(metadata["candidate_methods"]),
            "dedup_radius_px": float(metadata["dedup_radius_px"]),
            "numeric_normalization": normalization,
            "train_loss": total_loss / max(total, 1),
            "val_loss": val_loss,
            "best_coord_f1": best_f1,
        }
        coord_best = None
        if args.coord_eval_interval > 0 and epoch % args.coord_eval_interval == 0:
            summary, report = evaluate_models(
                [(model, checkpoint)],
                coord_rows,
                config,
                thresholds,
                str(metadata["candidate_methods"]),
                _eval_namespace(args),
                device,
                data_root=data_root,
            )
            coord_best = report["best"]
            current_f1 = float(coord_best["micro_f1"])
            checkpoint["coord_best"] = coord_best
            checkpoint["selected_score_threshold"] = float(coord_best["score_threshold"])
            if current_f1 > best_f1:
                best_f1 = current_f1
                checkpoint["best_coord_f1"] = best_f1
                torch.save(checkpoint, out_dir / "candidate_scorer_best.pt")
        torch.save(checkpoint, out_dir / "candidate_scorer_last.pt")
        row = {
            "epoch": epoch,
            "train_loss": checkpoint["train_loss"],
            "val_loss": val_loss,
            "coord_best": coord_best,
        }
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[epoch {epoch:02d}] train_loss={checkpoint['train_loss']:.4f}"
            f" val_loss={val_loss:.4f}"
            f" coord_f1={float(coord_best['micro_f1']) if coord_best else -1:.3f}"
            f" threshold={coord_best['score_threshold'] if coord_best else None}",
            flush=True,
        )

    flat_rows = []
    for row in history:
        best = row["coord_best"] or {}
        flat_rows.append({
            "epoch": row["epoch"],
            "train_loss": row["train_loss"],
            "val_loss": row["val_loss"],
            "micro_f1": best.get("micro_f1"),
            "macro_f1": best.get("macro_f1"),
            "precision": best.get("precision"),
            "recall": best.get("recall"),
            "score_threshold": best.get("score_threshold"),
        })
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
