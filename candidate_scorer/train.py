from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candidate_scorer.model import CandidateScorer
from candidate_scorer.evaluate import (
    extract_patches,
    make_feature_image,
    method_args,
    score_patches,
)
from candidate_scorer.build_dataset import crop_at, crop_origins
from preprocessing.mask_generator import generate_mask
from star_unet.dataset import _normalize_image, _read_input_image, _read_mask_image
from star_unet.evaluate import load_manifest_samples
from star_unet.postprocess import detection_metrics, heatmap_to_centroids
from star_unet.train import load_config
from star_unet.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the candidate scorer on prebuilt patch data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/candidate_scorer_sigma2p5"))
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=Path("star_unet/config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/candidate_scorer_sigma2p5"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=31)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to continue training from.")
    parser.add_argument("--log-interval", type=int, default=100, help="Print training progress every N batches. 0 disables batch logs.")
    parser.add_argument("--coord-eval-count", type=int, default=5, help="Number of validation frames used for coordinate-level F1 each epoch.")
    parser.add_argument("--coord-eval-crop-size", type=int, default=1024, help="Centered crop size for coordinate-level F1. 0 uses full frames.")
    parser.add_argument("--coord-eval-crop-mode", choices=("center", "random"), default="center")
    parser.add_argument("--coord-eval-crops-per-image", type=int, default=1)
    parser.add_argument("--coord-eval-interval", type=int, default=1, help="Run coordinate-level F1 every N epochs.")
    parser.add_argument("--candidate-sigma", type=float, default=2.5)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--match-radius-px", type=float, default=4.0)
    return parser.parse_args()


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["patches"].astype(np.float32), data["labels"].astype(np.float32)


def validation_loss(
    model: nn.Module,
    patches: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    criterion: nn.Module,
) -> float:
    total_loss = 0.0
    total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            batch = torch.from_numpy(patches[start : start + batch_size]).to(device)
            batch_labels = torch.from_numpy(labels[start : start + batch_size]).to(device)
            logits = model(batch)
            total_loss += float(criterion(logits, batch_labels).item()) * len(batch_labels)
            total += len(batch_labels)
    return total_loss / max(total, 1)


def spread_subset(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or count >= len(rows):
        return rows
    indexes = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[int(index)] for index in indexes]


def coordinate_eval(
    model: CandidateScorer,
    samples: list[dict[str, str]],
    config: dict[str, object],
    input_channels: int,
    thresholds: list[float],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    fit_channel_mode = str(config.get("fit_channel_mode", "mean"))
    image_norm = config.get("image_normalization", {})
    image_norm = image_norm if isinstance(image_norm, dict) else {}
    totals = {
        threshold: {"pred_count": 0, "target_count": 0, "matched_count": 0, "false_count": 0, "missed_count": 0}
        for threshold in thresholds
    }

    rng = np.random.default_rng(10_000 + len(samples))
    for sample in samples:
        image_path = Path(sample.get("image_out") or sample.get("single_fits") or "")
        mask_path = Path(sample.get("mask_out") or "")
        image_path = image_path if image_path.is_absolute() else REPO_ROOT / image_path
        mask_path = mask_path if mask_path.is_absolute() else REPO_ROOT / mask_path

        full_raw = _read_input_image(image_path, fit_channel_mode)
        full_mask = _read_mask_image(mask_path)
        origins = crop_origins(
            full_raw.shape,
            args.coord_eval_crop_size,
            args.coord_eval_crop_mode,
            args.coord_eval_crops_per_image,
            rng,
        )
        for y0, x0 in origins:
            raw = crop_at(full_raw, args.coord_eval_crop_size, y0, x0)
            image = _normalize_image(raw, image_norm)
            target_heatmap = crop_at(full_mask, args.coord_eval_crop_size, y0, x0)
            target_yx = heatmap_to_centroids(
                target_heatmap,
                threshold=args.target_threshold,
                min_distance=args.min_distance,
            )
            candidate_yx = generate_mask(raw, "daofind_like", method_args(args.candidate_sigma)).centroids_yx
            feature_image = make_feature_image(raw, image, input_channels)
            patches = extract_patches(feature_image, candidate_yx, args.patch_size)
            scores = score_patches(model, patches, device, args.batch_size)

            for threshold in thresholds:
                metrics = detection_metrics(candidate_yx[scores >= threshold], target_yx, radius_px=args.match_radius_px)
                for key in totals[threshold]:
                    totals[threshold][key] += int(metrics[key])

    rows = []
    for threshold, row in totals.items():
        pred = row["pred_count"]
        target = row["target_count"]
        matched = row["matched_count"]
        precision = matched / pred if pred else 0.0
        recall = matched / target if target else 0.0
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "score_threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                **row,
            }
        )
    return max(rows, key=lambda row: row["f1"])


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir if args.data_dir.is_absolute() else REPO_ROOT / args.data_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y = load_npz(data_dir / "train_patches.npz")
    val_x, val_y = load_npz(data_dir / "val_patches.npz")
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_config(config_path)
    data_root = args.data_root if args.data_root.is_absolute() else REPO_ROOT / args.data_root
    coord_samples = spread_subset(load_manifest_samples(data_root, "val"), args.coord_eval_count)
    device = resolve_device(args.device)
    input_channels = int(train_x.shape[1])
    model = CandidateScorer(input_channels=input_channels).to(device)
    if args.resume is not None:
        resume_path = args.resume if args.resume.is_absolute() else REPO_ROOT / args.resume
        checkpoint = torch.load(resume_path, map_location=device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"[resume] loaded {resume_path}")
    neg = int(np.sum(train_y <= 0.5))
    pos = int(np.sum(train_y > 0.5))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, neg / max(pos, 1))], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    thresholds = [float(item.strip()) for item in args.score_thresholds.split(",") if item.strip()]
    history = []
    best_coord_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        total_batches = len(loader)
        for batch_index, (patches, labels) in enumerate(loader, start=1):
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(patches), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total += len(labels)
            if args.log_interval > 0 and (
                batch_index == 1 or batch_index % args.log_interval == 0 or batch_index == total_batches
            ):
                running_loss = total_loss / max(total, 1)
                percent = 100.0 * batch_index / max(total_batches, 1)
                print(
                    f"  train batch {batch_index}/{total_batches}"
                    f" ({percent:.1f}%)"
                    f" loss={running_loss:.4f}",
                    flush=True,
                )
        val_loss = validation_loss(model, val_x, val_y, device, args.batch_size, criterion)
        if args.coord_eval_interval > 0 and epoch % args.coord_eval_interval == 0:
            best = coordinate_eval(
                model,
                coord_samples,
                config,
                input_channels,
                thresholds,
                args,
                device,
            )
        else:
            best = {
                "score_threshold": None,
                "precision": 0.0,
                "recall": 0.0,
                "f1": -1.0,
                "pred_count": 0,
                "target_count": 0,
                "matched_count": 0,
                "false_count": 0,
                "missed_count": 0,
            }
        train_loss = total_loss / max(total, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "coord_best": best})
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "coord_best": best,
            "input_channels": input_channels,
        }
        torch.save(checkpoint, out_dir / "candidate_scorer_last.pt")
        torch.save(checkpoint, out_dir / "candidate_scorer.pt")
        if float(best["f1"]) > best_coord_f1:
            best_coord_f1 = float(best["f1"])
            torch.save(checkpoint, out_dir / "candidate_scorer_best.pt")
            best_note = " saved_best"
        else:
            best_note = ""
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f}"
            f" val_loss={val_loss:.4f}"
            f" coord_t={best['score_threshold']}"
            f" coord_f1={best['f1']:.3f}"
            f" p={best['precision']:.3f}"
            f" r={best['recall']:.3f}"
            f"{best_note}"
        )

    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        final_rows = [row["coord_best"] | {"epoch": row["epoch"], "train_loss": row["train_loss"], "val_loss": row["val_loss"]} for row in history]
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
