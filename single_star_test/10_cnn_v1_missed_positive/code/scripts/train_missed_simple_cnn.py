from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[4]
UNET_CODE = REPO_ROOT / "single_star_test" / "00_unet_heatmap" / "code"
V3_CODE = REPO_ROOT / "single_star_test" / "07_cnn_v3_center_aware" / "code"
ARCHIVE_ROOT = REPO_ROOT / "single_star_test"
for path in (V3_CODE, UNET_CODE, ARCHIVE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from candidate_scorer.pipeline import extract_patches, generate_candidates  # noqa: E402
from star_unet.dataset import _normalize_image, _read_input_image, _read_mask_image  # noqa: E402
from star_unet.postprocess import detection_metrics, heatmap_to_centroids  # noqa: E402
from star_unet.train import load_config, resolve_device  # noqa: E402


class SimpleCandidateScorer(nn.Module):
    def __init__(self, patch_size: int = 31) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
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


@dataclass
class FrameData:
    sample_id: str
    patches: np.ndarray
    candidates_yx: np.ndarray
    targets_yx: np.ndarray
    missed_positive_index: np.ndarray
    negative_index: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the simple CNN on missed positives from a previous "
            "full-frame evaluation, using 1:1 missed-positive/negative samples."
        )
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "single_star_test/data/data_model")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "single_star_test/00_unet_heatmap/code/star_unet/config.json",
    )
    parser.add_argument(
        "--source-eval-dir",
        type=Path,
        default=REPO_ROOT / "single_star_test/result_analysis/09_eval_coord10",
        help="Directory containing report.json and per_sample.csv from the 09 full-frame eval.",
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-methods", default="daofind:2.5")
    parser.add_argument("--dedup-radius-px", type=float, default=2.5)
    parser.add_argument("--patch-size", type=int, default=31)
    parser.add_argument("--positive-radius-px", type=float, default=4.0)
    parser.add_argument("--negative-radius-px", type=float, default=8.0)
    parser.add_argument("--match-radius-px", type=float, default=4.0)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--source-threshold", type=float, default=None)
    parser.add_argument("--sample-ids", default="", help="Optional comma-separated override for eval sample ids.")
    parser.add_argument("--max-positives", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--score-thresholds", default="0.30:0.95:0.01")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_thresholds(text: str) -> list[float]:
    text = str(text).strip()
    if ":" in text and "," not in text:
        start, stop, step = (float(value) for value in text.split(":"))
        return np.arange(start, stop + step * 0.5, step, dtype=np.float64).tolist()
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def resolve_data_path(data_root: Path, path_text: str | Path) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    if path.is_absolute():
        return path
    parts = list(path.parts)
    if "data_model" in parts:
        return data_root.joinpath(*parts[parts.index("data_model") + 1 :])
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return data_root / path


def load_manifest_rows(data_root: Path) -> list[dict[str, str]]:
    manifest = data_root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"missing manifest: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_eval_source(source_eval_dir: Path, sample_ids: str) -> tuple[list[str], float]:
    report = json.loads((source_eval_dir / "report.json").read_text(encoding="utf-8"))
    threshold = float(report["best_cnn_by_mean_f1"]["threshold"])
    if sample_ids.strip():
        ids = [item.strip() for item in sample_ids.split(",") if item.strip()]
        return ids, threshold
    with (source_eval_dir / "per_sample.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    best_method = str(report["best_cnn_by_mean_f1"]["method"])
    ids = sorted({row["sample_id"] for row in rows if row.get("method") == best_method})
    if not ids:
        raise ValueError(f"could not find sample ids for {best_method} in {source_eval_dir / 'per_sample.csv'}")
    return ids, threshold


def load_model(path: Path, device: torch.device) -> SimpleCandidateScorer:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"unsupported checkpoint: {path}")
    patch_size = int(checkpoint.get("patch_size", 31))
    model = SimpleCandidateScorer(patch_size=patch_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def score_model(
    model: nn.Module,
    patches: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not len(patches):
        return np.empty((0,), dtype=np.float32)
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.from_numpy(patches[start : start + batch_size]).to(device)
            scores.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def target_matches(pred_yx: np.ndarray, targets_yx: np.ndarray, radius_px: float) -> set[int]:
    pred = np.asarray(pred_yx, dtype=np.float32).reshape((-1, 2))
    targets = np.asarray(targets_yx, dtype=np.float32).reshape((-1, 2))
    if not len(pred) or not len(targets):
        return set()
    tree = cKDTree(np.column_stack((pred[:, 1], pred[:, 0])))
    matched: set[int] = set()
    for target_index, (ty, tx) in enumerate(targets):
        if tree.query_ball_point([float(tx), float(ty)], r=float(radius_px)):
            matched.add(int(target_index))
    return matched


def candidate_distances(candidates_yx: np.ndarray, targets_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(candidates_yx, dtype=np.float32).reshape((-1, 2))
    targets = np.asarray(targets_yx, dtype=np.float32).reshape((-1, 2))
    if not len(candidates) or not len(targets):
        return (
            np.full(len(candidates), np.inf, dtype=np.float32),
            np.full(len(candidates), -1, dtype=np.int64),
        )
    tree = cKDTree(np.column_stack((targets[:, 1], targets[:, 0])))
    distances, nearest = tree.query(np.column_stack((candidates[:, 1], candidates[:, 0])), k=1)
    return distances.astype(np.float32), nearest.astype(np.int64)


def build_frame(
    row: dict[str, str],
    data_root: Path,
    config: dict[str, Any],
    source_model: nn.Module,
    source_threshold: float,
    args: argparse.Namespace,
    device: torch.device,
) -> FrameData:
    fit_channel_mode = str(config.get("fit_channel_mode", "mean"))
    image_norm = config.get("image_normalization", {})
    sample_id = str(row["sample_id"])
    image_path = resolve_data_path(data_root, row.get("image_out") or row.get("single_fits") or "")
    mask_path = resolve_data_path(data_root, row.get("mask_out") or "")
    raw = _read_input_image(image_path, fit_channel_mode)
    normalized = _normalize_image(raw, image_norm if isinstance(image_norm, dict) else {})
    mask = _read_mask_image(mask_path)
    targets = heatmap_to_centroids(mask, threshold=args.target_threshold, min_distance=args.min_distance)
    candidates = generate_candidates(raw, args.candidate_methods, args.dedup_radius_px).centroids_yx
    patches = extract_patches(normalized[None].astype(np.float32), candidates, args.patch_size)[:, :1]
    source_scores = score_model(source_model, patches, device, args.eval_batch_size)
    recalled_targets = target_matches(candidates[source_scores >= source_threshold], targets, args.match_radius_px)
    distances, nearest = candidate_distances(candidates, targets)

    missed_indices: list[int] = []
    for target_index in range(len(targets)):
        if target_index in recalled_targets:
            continue
        nearby = np.flatnonzero((nearest == target_index) & (distances <= float(args.positive_radius_px)))
        if len(nearby):
            missed_indices.append(int(nearby[np.argmin(distances[nearby])]))
    negative_index = np.flatnonzero(distances >= float(args.negative_radius_px)).astype(np.int64)
    return FrameData(
        sample_id=sample_id,
        patches=patches,
        candidates_yx=candidates,
        targets_yx=targets,
        missed_positive_index=np.asarray(sorted(set(missed_indices)), dtype=np.int64),
        negative_index=negative_index,
    )


def sample_training_arrays(
    frames: list[FrameData],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    positives = np.concatenate(
        [frame.patches[frame.missed_positive_index] for frame in frames if len(frame.missed_positive_index)],
        axis=0,
    )
    negatives = np.concatenate(
        [frame.patches[frame.negative_index] for frame in frames if len(frame.negative_index)],
        axis=0,
    )
    if len(positives) > int(args.max_positives):
        positives = positives[rng.choice(len(positives), size=int(args.max_positives), replace=False)]
    neg_count = min(len(negatives), len(positives))
    negatives = negatives[rng.choice(len(negatives), size=neg_count, replace=False)]
    patches = np.concatenate([positives, negatives], axis=0)
    labels = np.concatenate([
        np.ones(len(positives), dtype=np.float32),
        np.zeros(len(negatives), dtype=np.float32),
    ])
    order = rng.permutation(len(labels))
    return patches[order], labels[order], {"positives": int(len(positives)), "negatives": int(len(negatives))}


def evaluate_frames(
    model: nn.Module,
    frames: list[FrameData],
    thresholds: list[float],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        metrics = []
        for frame in frames:
            scores = score_model(model, frame.patches, device, args.eval_batch_size)
            metrics.append(detection_metrics(frame.candidates_yx[scores >= threshold], frame.targets_yx, args.match_radius_px))
        rows.append({
            "score_threshold": float(threshold),
            "mean_precision": float(np.mean([m["precision"] for m in metrics])),
            "mean_recall": float(np.mean([m["recall"] for m in metrics])),
            "mean_f1": float(np.mean([m["f1"] for m in metrics])),
            "mean_pred_count": float(np.mean([m["pred_count"] for m in metrics])),
        })
    return rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids, report_threshold = load_eval_source(args.source_eval_dir, args.sample_ids)
    source_threshold = float(args.source_threshold) if args.source_threshold is not None else report_threshold
    rows_by_id = {row["sample_id"]: row for row in load_manifest_rows(args.data_root)}
    rows = [rows_by_id[sample_id] for sample_id in sample_ids]
    config = load_config(args.config)
    device = resolve_device(args.device)
    source_model = load_model(args.source_checkpoint, device)

    print(f"[source] frames={len(rows)} threshold={source_threshold:.4f} checkpoint={args.source_checkpoint}", flush=True)
    frames = []
    for index, row in enumerate(rows, start=1):
        frame = build_frame(row, args.data_root, config, source_model, source_threshold, args, device)
        frames.append(frame)
        print(
            f"[frame {index}/{len(rows)}] {frame.sample_id}: "
            f"missed_pos={len(frame.missed_positive_index)} negatives={len(frame.negative_index)}",
            flush=True,
        )

    patches, labels, counts = sample_training_arrays(frames, args, rng)
    if counts["positives"] == 0 or counts["negatives"] == 0:
        raise RuntimeError(f"not enough training data: {counts}")
    print(f"[data] positives={counts['positives']} negatives={counts['negatives']} total={len(labels)}", flush=True)

    model = SimpleCandidateScorer(args.patch_size).to(device)
    model.load_state_dict(source_model.state_dict())
    loader = DataLoader(
        TensorDataset(torch.from_numpy(patches), torch.from_numpy(labels)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    thresholds = parse_thresholds(args.score_thresholds)
    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_patches, batch_labels in loader:
            batch_patches = batch_patches.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_patches), batch_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        eval_rows = evaluate_frames(model, frames, thresholds, args, device)
        best = max(eval_rows, key=lambda row: float(row["mean_f1"]))
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "best": best}
        history.append(record)
        if float(best["mean_f1"]) > best_f1:
            best_f1 = float(best["mean_f1"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "patch_size": args.patch_size,
                    "input_channels": 1,
                    "model_type": "legacy_simple",
                    "source_checkpoint": str(args.source_checkpoint),
                    "source_eval_dir": str(args.source_eval_dir),
                    "source_threshold": source_threshold,
                    "sample_counts": counts,
                    "best": best,
                    "epoch": epoch,
                },
                args.out_dir / "candidate_scorer_best.pt",
            )
        print(
            f"epoch={epoch} loss={record['loss']:.4f} "
            f"mean_f1={best['mean_f1']:.4f} p={best['mean_precision']:.4f} "
            f"r={best['mean_recall']:.4f} t={best['score_threshold']:.2f}",
            flush=True,
        )

    final_rows = evaluate_frames(model, frames, thresholds, args, device)
    torch.save(
        {
            "model": model.state_dict(),
            "patch_size": args.patch_size,
            "input_channels": 1,
            "model_type": "legacy_simple",
            "source_checkpoint": str(args.source_checkpoint),
            "source_eval_dir": str(args.source_eval_dir),
            "source_threshold": source_threshold,
            "sample_counts": counts,
            "best": max(final_rows, key=lambda row: float(row["mean_f1"])),
            "epoch": args.epochs,
        },
        args.out_dir / "candidate_scorer_last.pt",
    )
    (args.out_dir / "summary.json").write_text(json.dumps(final_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "sample_ids": sample_ids,
                "source_threshold": source_threshold,
                "sample_counts": counts,
                "frames": [
                    {
                        "sample_id": frame.sample_id,
                        "missed_positive_count": int(len(frame.missed_positive_index)),
                        "negative_count": int(len(frame.negative_index)),
                    }
                    for frame in frames
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"[done] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
