from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "single_star_test" / "11_deepsource_star_enhancer" / "code"
UNET_CODE = REPO_ROOT / "single_star_test" / "00_unet_heatmap" / "code"
V3_CODE = REPO_ROOT / "single_star_test" / "07_cnn_v3_center_aware" / "code"
ARCHIVE_ROOT = REPO_ROOT / "single_star_test"
for path in (CODE_ROOT, V3_CODE, UNET_CODE, ARCHIVE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - cloud env normally has tqdm, fallback keeps script dependency-light.
    def tqdm(iterable, **kwargs):
        return iterable

from deepsource_star.data import DeepSourceStarDataset, PrecomputedCropDataset  # noqa: E402
from deepsource_star.model import DeepSourceEnhancer  # noqa: E402
from candidate_scorer.pipeline import generate_candidates  # noqa: E402
from star_unet.postprocess import detection_metrics, heatmap_to_centroids  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Train a DeepSource-style star enhancer on data_model.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--crop-data-dir", type=Path, default=None, help="Optional directory with train/val/test.npz crops.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, default=None, help="Optional 11-specific split CSV with split11 column.")
    parser.add_argument("--train-split-reason", default="train")
    parser.add_argument("--val-split-reason", default="frame_holdout")
    parser.add_argument("--test-split-reason", default="coord_holdout")
    parser.add_argument("--train-split-name", default="train")
    parser.add_argument("--val-split-name", default="val")
    parser.add_argument("--test-split-name", default="test")
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
    parser.add_argument("--target-weighting", choices=["none", "stack_photometry"], default="stack_photometry")
    parser.add_argument("--target-min-amplitude", type=float, default=0.25)
    parser.add_argument("--target-max-amplitude", type=float, default=1.0)
    parser.add_argument("--target-min-radius", type=float, default=2.0)
    parser.add_argument("--target-max-radius", type=float, default=8.0)
    parser.add_argument("--target-aperture-radius", type=float, default=5.0)
    parser.add_argument("--target-annulus-inner", type=float, default=8.0)
    parser.add_argument("--target-annulus-outer", type=float, default=14.0)
    parser.add_argument("--filters", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--output-activation", choices=["relu", "sigmoid", "none"], default="relu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-dao-method", default="daofind:5.0")
    parser.add_argument("--val-dao-dedup-radius-px", type=float, default=2.5)
    parser.add_argument("--val-target-threshold", type=float, default=0.5)
    parser.add_argument("--val-min-distance", type=float, default=3.0)
    parser.add_argument("--val-match-radius-px", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; using cpu", flush=True)
        name = "cpu"
    return torch.device(name)


def split_name_for_reason(args: argparse.Namespace, split_reason: str) -> str:
    if split_reason == args.train_split_reason:
        return args.train_split_name
    if split_reason == args.val_split_reason:
        return args.val_split_name
    if split_reason == args.test_split_reason:
        return args.test_split_name
    return split_reason


def make_dataset(args: argparse.Namespace, split_reason: str, count: int) -> DeepSourceStarDataset:
    return DeepSourceStarDataset(
        data_root=args.data_root,
        split_reason=split_reason,
        count=count,
        split_csv=args.split_csv,
        split_name=split_name_for_reason(args, split_reason),
        crop_size=args.crop_size,
        crops_per_image=args.crops_per_image,
        target_mode=args.target_mode,
        gaussian_sigma=args.gaussian_sigma,
        triangle_radius=args.triangle_radius,
        background_level=args.background_level,
        alpha=args.alpha,
        target_weighting=args.target_weighting,
        target_min_amplitude=args.target_min_amplitude,
        target_max_amplitude=args.target_max_amplitude,
        target_min_radius=args.target_min_radius,
        target_max_radius=args.target_max_radius,
        target_aperture_radius=args.target_aperture_radius,
        target_annulus_inner=args.target_annulus_inner,
        target_annulus_outer=args.target_annulus_outer,
        seed=args.seed,
    )


def make_datasets(args: argparse.Namespace):
    if args.crop_data_dir is not None:
        return (
            PrecomputedCropDataset(args.crop_data_dir, "train"),
            PrecomputedCropDataset(args.crop_data_dir, "val"),
            PrecomputedCropDataset(args.crop_data_dir, "test"),
        )
    return (
        make_dataset(args, args.train_split_reason, args.train_samples),
        make_dataset(args, args.val_split_reason, args.val_samples),
        make_dataset(args, args.test_split_reason, args.test_samples),
    )


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    progress = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True)
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        pred = model(image)
        loss = criterion(pred, target)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{losses[-1]:.6f}")
    return float(np.mean(losses)) if losses else float("nan")


def aggregate_detection_metrics(rows: list[dict[str, float | int | None]]) -> dict[str, float | int]:
    if not rows:
        return {
            "val_dao_precision": 0.0,
            "val_dao_recall": 0.0,
            "val_dao_f1": 0.0,
            "val_dao_pred_count": 0,
            "val_dao_target_count": 0,
            "val_dao_matched_count": 0,
            "val_dao_false_count": 0,
            "val_dao_missed_count": 0,
        }
    pred = int(sum(int(row["pred_count"]) for row in rows))
    target = int(sum(int(row["target_count"]) for row in rows))
    matched = int(sum(int(row["matched_count"]) for row in rows))
    false = int(sum(int(row["false_count"]) for row in rows))
    missed = int(sum(int(row["missed_count"]) for row in rows))
    precision = matched / pred if pred else 0.0
    recall = matched / target if target else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "val_dao_precision": float(precision),
        "val_dao_recall": float(recall),
        "val_dao_f1": float(f1),
        "val_dao_pred_count": pred,
        "val_dao_target_count": target,
        "val_dao_matched_count": matched,
        "val_dao_false_count": false,
        "val_dao_missed_count": missed,
    }


def evaluate_val_with_daofind(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    desc: str,
) -> dict[str, float | int]:
    model.eval()
    losses: list[float] = []
    metric_rows: list[dict[str, float | int | None]] = []
    progress = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True)
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        pred = model(image)
        loss = criterion(pred, target)
        losses.append(float(loss.detach().cpu()))

        pred_np = pred.detach().cpu().numpy()[:, 0]
        target_np = target.detach().cpu().numpy()[:, 0]
        for pred_map, target_map in zip(pred_np, target_np):
            pred_centroids = generate_candidates(
                pred_map.astype(np.float32),
                args.val_dao_method,
                dedup_radius_px=args.val_dao_dedup_radius_px,
            ).centroids_yx
            target_centroids = heatmap_to_centroids(
                target_map.astype(np.float32),
                threshold=args.val_target_threshold,
                min_distance=args.val_min_distance,
            )
            metric_rows.append(
                detection_metrics(pred_centroids, target_centroids, radius_px=args.val_match_radius_px)
            )
        if hasattr(progress, "set_postfix"):
            running = aggregate_detection_metrics(metric_rows)
            progress.set_postfix(
                loss=f"{losses[-1]:.6f}",
                p=f"{running['val_dao_precision']:.4f}",
                r=f"{running['val_dao_recall']:.4f}",
                f1=f"{running['val_dao_f1']:.4f}",
            )

    metrics = aggregate_detection_metrics(metric_rows)
    metrics["val_loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    train_ds, val_ds, test_ds = make_datasets(args)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = DeepSourceEnhancer(
        filters=args.filters,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        output_activation=args.output_activation,
    ).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    metadata = {
        "args": jsonable_args(args),
        "train_images": len(train_ds.rows),
        "val_images": len(val_ds.rows),
        "test_images": len(test_ds.rows),
        "train_patches": len(train_ds),
        "val_patches": len(val_ds),
        "test_patches": len(test_ds),
        "model": "DeepSourceEnhancer",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[data] train_images={len(train_ds.rows)} train_patches={len(train_ds)} "
        f"val_images={len(val_ds.rows)} val_patches={len(val_ds)} "
        f"test_images={len(test_ds.rows)} test_patches={len(test_ds)} device={device}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            desc=f"epoch {epoch}/{args.epochs} train",
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_metrics = evaluate_val_with_daofind(
                model,
                val_loader,
                criterion,
                device,
                args,
                desc=f"epoch {epoch}/{args.epochs} val+dao5",
            )
            test_loss = run_epoch(
                model,
                test_loader,
                criterion,
                device,
                desc=f"epoch {epoch}/{args.epochs} test",
                optimizer=None,
            )
        val_loss = float(val_metrics["val_loss"])
        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics, "test_loss": test_loss}
        history.append(record)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "metadata": metadata,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                },
                args.out_dir / "deepsource_star_best.pt",
            )
        torch.save(
            {
                "model": model.state_dict(),
                "metadata": metadata,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            },
            args.out_dir / "deepsource_star_last.pt",
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} test_loss={test_loss:.6f} "
            f"val_dao5_p={float(val_metrics['val_dao_precision']):.4f} "
            f"val_dao5_r={float(val_metrics['val_dao_recall']):.4f} "
            f"val_dao5_f1={float(val_metrics['val_dao_f1']):.4f}",
            flush=True,
        )

    with (args.out_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "test_loss",
                "val_dao_precision",
                "val_dao_recall",
                "val_dao_f1",
                "val_dao_pred_count",
                "val_dao_target_count",
                "val_dao_matched_count",
                "val_dao_false_count",
                "val_dao_missed_count",
            ],
        )
        writer.writeheader()
        writer.writerows(history)
    (args.out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
