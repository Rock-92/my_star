from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.mask_generator import generate_mask
from star_unet.dataset import _normalize_image, _read_input_image, _read_mask_image
from star_unet.evaluate import baseline_args, load_manifest_samples, resolve_path
from star_unet.postprocess import detection_metrics, heatmap_to_centroids
from star_unet.train import load_config, resolve_device


@dataclass
class CandidateSet:
    patches: np.ndarray
    labels: np.ndarray
    centroids_yx: np.ndarray
    target_yx: np.ndarray
    sample_id: str


class CandidateScorer(nn.Module):
    def __init__(self, patch_size: int) -> None:
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
    parser = argparse.ArgumentParser(description="Train a small CNN to rescore low-threshold DAOFind candidates.")
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=Path("star_unet/config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/candidate_scorer_sigma2p5"))
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=31)
    parser.add_argument("--positive-radius-px", type=float, default=4.0)
    parser.add_argument("--ignore-radius-px", type=float, default=6.0)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--max-train-samples", type=int, default=120)
    parser.add_argument("--max-val-samples", type=int, default=40)
    parser.add_argument("--neg-pos-ratio", type=int, default=3)
    parser.add_argument("--max-negatives-per-image", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--score-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def center_crop(array: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return array
    h, w = array.shape[:2]
    crop_h = min(size, h)
    crop_w = min(size, w)
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    return array[y0 : y0 + crop_h, x0 : x0 + crop_w]


def spread_subset(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    indexes = np.linspace(0, len(rows) - 1, limit, dtype=int)
    return [rows[int(index)] for index in indexes]


def scorer_args(sigma: float) -> SimpleNamespace:
    args = SimpleNamespace(**vars(baseline_args()))
    args.sigma = float(sigma)
    return args


def extract_patches(image: np.ndarray, centroids_yx: np.ndarray, patch_size: int) -> np.ndarray:
    radius = patch_size // 2
    padded = np.pad(image, radius, mode="reflect")
    patches = []
    for cy, cx in np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2)):
        y = int(round(float(cy) - 0.5)) + radius
        x = int(round(float(cx) - 0.5)) + radius
        patch = padded[y - radius : y + radius + 1, x - radius : x + radius + 1]
        patches.append(patch.astype(np.float32))
    if not patches:
        return np.empty((0, 1, patch_size, patch_size), dtype=np.float32)
    return np.asarray(patches, dtype=np.float32)[:, None, :, :]


def label_candidates(
    candidate_yx: np.ndarray,
    target_yx: np.ndarray,
    positive_radius_px: float,
    ignore_radius_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_yx = np.asarray(candidate_yx, dtype=np.float32).reshape((-1, 2))
    target_yx = np.asarray(target_yx, dtype=np.float32).reshape((-1, 2))
    if len(candidate_yx) == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=bool)
    if len(target_yx) == 0:
        return np.zeros(len(candidate_yx), dtype=np.float32), np.ones(len(candidate_yx), dtype=bool)

    tree = cKDTree(np.column_stack((target_yx[:, 1], target_yx[:, 0])))
    distances, _ = tree.query(np.column_stack((candidate_yx[:, 1], candidate_yx[:, 0])), k=1)
    labels = (distances <= float(positive_radius_px)).astype(np.float32)
    keep = (distances <= float(positive_radius_px)) | (distances >= float(ignore_radius_px))
    return labels, keep


def build_candidate_set(
    sample: dict[str, str],
    config: dict[str, object],
    data_args: argparse.Namespace,
) -> CandidateSet:
    fit_channel_mode = str(config.get("fit_channel_mode", "mean"))
    image_norm = config.get("image_normalization", {})
    sample_id = str(sample.get("sample_id") or Path(sample.get("image_out", "")).stem)
    image_path = resolve_path(sample.get("image_out") or sample.get("single_fits") or "")
    mask_path = resolve_path(sample.get("mask_out") or "")
    raw = center_crop(_read_input_image(image_path, fit_channel_mode), data_args.crop_size)
    image = center_crop(_normalize_image(raw, image_norm if isinstance(image_norm, dict) else {}), data_args.crop_size)
    mask = center_crop(_read_mask_image(mask_path), data_args.crop_size)
    target_yx = heatmap_to_centroids(mask, threshold=data_args.target_threshold, min_distance=data_args.min_distance)
    candidates = generate_mask(raw, "daofind_like", scorer_args(data_args.sigma)).centroids_yx
    labels, keep = label_candidates(
        candidates,
        target_yx,
        positive_radius_px=data_args.positive_radius_px,
        ignore_radius_px=data_args.ignore_radius_px,
    )
    candidates = candidates[keep]
    labels = labels[keep]
    patches = extract_patches(image, candidates, data_args.patch_size)
    return CandidateSet(patches=patches, labels=labels, centroids_yx=candidates, target_yx=target_yx, sample_id=sample_id)


def sample_training_arrays(candidate_sets: list[CandidateSet], neg_pos_ratio: int, max_negatives_per_image: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    patch_parts = []
    label_parts = []
    for item in candidate_sets:
        pos_idx = np.flatnonzero(item.labels > 0.5)
        neg_idx = np.flatnonzero(item.labels <= 0.5)
        max_neg = min(len(neg_idx), max(max_negatives_per_image, len(pos_idx) * neg_pos_ratio))
        if len(neg_idx) > max_neg:
            neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
        keep_idx = np.concatenate([pos_idx, neg_idx])
        if len(keep_idx) == 0:
            continue
        rng.shuffle(keep_idx)
        patch_parts.append(item.patches[keep_idx])
        label_parts.append(item.labels[keep_idx])
    return np.concatenate(patch_parts, axis=0), np.concatenate(label_parts, axis=0)


def score_candidates(model: nn.Module, patches: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    if len(patches) == 0:
        return np.empty((0,), dtype=np.float32)
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.from_numpy(patches[start : start + batch_size]).to(device)
            logits = model(batch)
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def evaluate_candidate_sets(
    model: nn.Module,
    candidate_sets: list[CandidateSet],
    thresholds: list[float],
    device: torch.device,
    batch_size: int,
    match_radius_px: float,
) -> list[dict[str, object]]:
    accum = {
        threshold: {"pred_count": 0, "target_count": 0, "matched_count": 0, "false_count": 0, "missed_count": 0}
        for threshold in thresholds
    }
    for item in candidate_sets:
        scores = score_candidates(model, item.patches, device, batch_size=batch_size)
        for threshold in thresholds:
            pred_yx = item.centroids_yx[scores >= threshold]
            metrics = detection_metrics(pred_yx, item.target_yx, radius_px=match_radius_px)
            for key in accum[threshold]:
                accum[threshold][key] += int(metrics[key])

    rows = []
    for threshold, row in accum.items():
        pred = row["pred_count"]
        target = row["target_count"]
        matched = row["matched_count"]
        precision = matched / pred if pred else 0.0
        recall = matched / target if target else 0.0
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "score_threshold": threshold,
                **row,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_rate": row["false_count"] / pred if pred else 0.0,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    data_root = resolve_path(args.data_root)
    train_rows = spread_subset(load_manifest_samples(data_root, "train"), args.max_train_samples)
    val_rows = spread_subset(load_manifest_samples(data_root, "val"), args.max_val_samples)

    print(f"[data] building candidates sigma={args.sigma} train={len(train_rows)} val={len(val_rows)}")
    train_sets = [build_candidate_set(row, config, args) for row in train_rows]
    val_sets = [build_candidate_set(row, config, args) for row in val_rows]
    train_x, train_y = sample_training_arrays(
        train_sets,
        neg_pos_ratio=args.neg_pos_ratio,
        max_negatives_per_image=args.max_negatives_per_image,
        seed=args.seed,
    )
    pos = int(np.sum(train_y > 0.5))
    neg = int(np.sum(train_y <= 0.5))
    print(f"[data] patches={len(train_y)} pos={pos} neg={neg}")

    device = resolve_device(args.device)
    model = CandidateScorer(args.patch_size).to(device)
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1))], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for patches, labels in loader:
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(patches)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total += len(labels)
        eval_rows = evaluate_candidate_sets(
            model,
            val_sets,
            thresholds=[float(item) for item in args.score_thresholds.split(",") if item.strip()],
            device=device,
            batch_size=args.batch_size,
            match_radius_px=args.positive_radius_px,
        )
        best = max(eval_rows, key=lambda row: row["f1"])
        record = {"epoch": epoch, "loss": total_loss / max(total, 1), "best": best}
        history.append(record)
        print(
            f"[epoch {epoch:02d}] loss={record['loss']:.4f} "
            f"best_t={best['score_threshold']} f1={best['f1']:.3f} "
            f"precision={best['precision']:.3f} recall={best['recall']:.3f}"
        )

    final_rows = evaluate_candidate_sets(
        model,
        val_sets,
        thresholds=[float(item) for item in args.score_thresholds.split(",") if item.strip()],
        device=device,
        batch_size=args.batch_size,
        match_radius_px=args.positive_radius_px,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "patch_size": args.patch_size,
            "sigma": args.sigma,
            "config": vars(args),
        },
        out_dir / "candidate_scorer.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(final_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
