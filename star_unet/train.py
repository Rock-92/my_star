from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import StarMapDataset
from .losses import SegmentationLoss, segmentation_metrics
from .model import UNet


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args()
    config_path = Path(pre_args.config)
    defaults = load_config(config_path)

    parser = argparse.ArgumentParser(
        description="Train U-Net for star segmentation masks.",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--train-dir",
        default=defaults.get("train_dir"),
        required=defaults.get("train_dir") is None,
        help="Directory with images/ and masks/ labels.",
    )
    parser.add_argument("--val-dir", default=defaults.get("val_dir"), help="Optional validation directory with the same layout.")
    parser.add_argument("--out-dir", default=defaults.get("out_dir", "runs/star_unet"), help="Directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 100))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 4))
    parser.add_argument("--val-batch-size", type=int, default=defaults.get("val_batch_size", defaults.get("batch_size", 4)))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 1e-3))
    parser.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 5e-4))
    parser.add_argument("--bce-weight", type=float, default=defaults.get("bce_weight", 1.0))
    parser.add_argument("--dice-weight", type=float, default=defaults.get("dice_weight", 1.0))
    parser.add_argument("--positive-weight", type=float, default=defaults.get("positive_weight"))
    parser.add_argument("--loss-mode", default=defaults.get("loss_mode", "bce_dice"), choices=("bce_dice", "error_focused"))
    parser.add_argument("--false-positive-weight", type=float, default=defaults.get("false_positive_weight", 10.0))
    parser.add_argument("--false-negative-weight", type=float, default=defaults.get("false_negative_weight", 10.0))
    parser.add_argument("--target-positive-threshold", type=float, default=defaults.get("target_positive_threshold", 0.5))
    parser.add_argument("--target-negative-threshold", type=float, default=defaults.get("target_negative_threshold", 0.05))
    parser.add_argument("--hard-negative-threshold", type=float, default=defaults.get("hard_negative_threshold", 0.1))
    parser.add_argument("--features", default=defaults.get("features", "32,64"), help="Comma-separated U-Net channel sizes.")
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 0))
    parser.add_argument("--device", default=defaults.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--amp", action="store_true", default=bool(defaults.get("amp", False)), help="Use mixed precision on CUDA.")
    parser.add_argument("--fit-channel-mode", default=defaults.get("fit_channel_mode", "mean"), choices=("mean", "first", "max", "luma"))
    norm_defaults = defaults.get("image_normalization", {}) if isinstance(defaults.get("image_normalization", {}), dict) else {}
    parser.add_argument("--norm-mode", default=norm_defaults.get("mode", "percentile"), choices=("percentile", "minmax", "none", "raw"))
    parser.add_argument("--norm-lower-percentile", type=float, default=norm_defaults.get("lower_percentile", 0.5))
    parser.add_argument("--norm-upper-percentile", type=float, default=norm_defaults.get("upper_percentile", 99.8))
    parser.add_argument("--scheduler-step-size", type=int, default=defaults.get("scheduler_step_size", 20))
    parser.add_argument("--scheduler-gamma", type=float, default=defaults.get("scheduler_gamma", 0.5))
    parser.add_argument("--log-interval", type=int, default=defaults.get("log_interval", 25))
    parser.set_defaults(augmentation=defaults.get("augmentation", {}))
    parser.set_defaults(crop=defaults.get("crop", {}))
    parser.set_defaults(val_crop=defaults.get("val_crop", {}))
    args = parser.parse_args()
    args.image_normalization = {
        "mode": args.norm_mode,
        "lower_percentile": args.norm_lower_percentile,
        "upper_percentile": args.norm_upper_percentile,
    }
    return args


def parse_features(value: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v.strip()) for v in str(value).split(",") if v.strip())


def resolve_device(name: str) -> torch.device:
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_loader(args: argparse.Namespace, root: str, shuffle: bool, pin_memory: bool, batch_size: int) -> DataLoader:
    augmentation = args.augmentation if shuffle else None
    crop = args.crop if shuffle else args.val_crop
    dataset = StarMapDataset(
        root,
        augmentation=augmentation,
        fit_channel_mode=args.fit_channel_mode,
        image_normalization=args.image_normalization,
        crop=crop,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )


def run_epoch(
    model: UNet,
    loader: DataLoader,
    criterion: SegmentationLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    phase: str = "train",
    log_interval: int = 0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "bce_loss": 0.0, "dice_loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    count = 0

    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(image)
                loss, parts = criterion(output, mask)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        metrics = segmentation_metrics(output.detach(), mask.detach())
        batch_size = image.size(0)
        count += batch_size
        for key in ("loss", "bce_loss", "dice_loss"):
            totals[key] += float(parts[key].item()) * batch_size
        for key, value in metrics.items():
            totals[key] += float(value) * batch_size

        if log_interval > 0 and (batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches):
            print(
                f"  {phase} batch {batch_index}/{total_batches}"
                f" loss {parts['loss'].item():.4f}"
                f" f1 {metrics['f1']:.4f}",
                flush=True,
            )

    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model: UNet, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = parse_features(args.features)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    model = UNet(in_channels=1, out_channels=1, features=features).to(device)
    criterion = SegmentationLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        positive_weight=args.positive_weight,
        mode=args.loss_mode,
        false_positive_weight=args.false_positive_weight,
        false_negative_weight=args.false_negative_weight,
        target_positive_threshold=args.target_positive_threshold,
        target_negative_threshold=args.target_negative_threshold,
        hard_negative_threshold=args.hard_negative_threshold,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    if not scaler.is_enabled():
        scaler = None

    train_loader = make_loader(args, args.train_dir, shuffle=True, pin_memory=pin_memory, batch_size=args.batch_size)
    val_loader = (
        make_loader(args, args.val_dir, shuffle=False, pin_memory=pin_memory, batch_size=args.val_batch_size)
        if args.val_dir
        else None
    )
    print(
        f"[start] device={device}, train_samples={len(train_loader.dataset)}, "
        f"val_samples={len(val_loader.dataset) if val_loader is not None else 0}, "
        f"epochs={args.epochs}, batch_size={args.batch_size}, val_batch_size={args.val_batch_size}",
        flush=True,
    )

    best_score = -1.0
    history = []
    config = vars(args) | {"features": features}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        print(f"[epoch {epoch:03d}] training", flush=True)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            phase="train",
            log_interval=args.log_interval,
        )
        scheduler.step()

        row = {"epoch": epoch, "train": train_metrics}
        if val_loader is not None:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[epoch {epoch:03d}] validation", flush=True)
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    phase="val",
                    log_interval=args.log_interval,
                )
            row["val"] = val_metrics
            score = val_metrics["f1"]
        else:
            score = train_metrics["f1"]

        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, row)
        if score > best_score:
            best_score = score
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, row)

        val_text = ""
        if "val" in row:
            val_text = (
                f" | val loss {row['val']['loss']:.4f}"
                f" f1 {row['val']['f1']:.4f}"
            )
        print(
            f"epoch {epoch:03d}"
            f" | train loss {train_metrics['loss']:.4f}"
            f" f1 {train_metrics['f1']:.4f}"
            f"{val_text}"
        )


if __name__ == "__main__":
    main()
