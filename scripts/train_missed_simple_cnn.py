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

from candidate_scorer.pipeline import resolve_data_path
from preprocessing.mask_generator import generate_mask
from scripts.train_candidate_scorer import CandidateScorer, extract_patches, score_candidates
from star_unet.dataset import _normalize_image, _read_input_image, _read_mask_image
from star_unet.evaluate import baseline_args, load_manifest_samples, resolve_path
from star_unet.postprocess import detection_metrics, heatmap_to_centroids
from star_unet.train import load_config, resolve_device


@dataclass
class CandidateFrame:
    sample_id: str
    patches: np.ndarray
    centroids_yx: np.ndarray
    target_yx: np.ndarray
    positive_index: np.ndarray
    negative_index: np.ndarray
    missed_positive_index: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Specialize the original simple CNN on true stars missed by an existing scorer."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=Path("star_unet/config.json"))
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/simple_cnn_missed_sigma2p5"))
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--patch-size", type=int, default=31)
    parser.add_argument("--positive-radius-px", type=float, default=4.0)
    parser.add_argument("--negative-radius-px", type=float, default=8.0)
    parser.add_argument("--match-radius-px", type=float, default=4.0)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--source-threshold", type=float, default=0.7)
    parser.add_argument("--train-split-reason", default="train")
    parser.add_argument("--val-split-reason", default="frame_holdout")
    parser.add_argument("--train-samples", type=int, default=160)
    parser.add_argument("--val-samples", type=int, default=12)
    parser.add_argument("--max-positives", type=int, default=20000)
    parser.add_argument("--neg-pos-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--score-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def spread_subset(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    indexes = np.linspace(0, len(rows) - 1, limit, dtype=int)
    return [rows[int(index)] for index in indexes]


def load_rows(data_root: Path, split_reason: str) -> list[dict[str, str]]:
    rows = load_manifest_samples(data_root, "train") + load_manifest_samples(data_root, "val")
    wanted = {part.strip() for part in split_reason.split(",") if part.strip()}
    return [row for row in rows if row.get("split_reason", row.get("split", "")) in wanted]


def scorer_args(sigma: float) -> SimpleNamespace:
    args = SimpleNamespace(**vars(baseline_args()))
    args.sigma = float(sigma)
    return args


def load_simple_checkpoint(path: Path, patch_size: int, device: torch.device) -> CandidateScorer:
    checkpoint = torch.load(resolve_path(path), map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = CandidateScorer(patch_size).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _candidate_target_distances(candidates: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(candidates) or not len(targets):
        return (
            np.full(len(candidates), np.inf, dtype=np.float32),
            np.full(len(candidates), -1, dtype=np.int64),
        )
    tree = cKDTree(np.column_stack((targets[:, 1], targets[:, 0])))
    distances, target_index = tree.query(np.column_stack((candidates[:, 1], candidates[:, 0])), k=1)
    return distances.astype(np.float32), target_index.astype(np.int64)


def _matched_targets(pred_yx: np.ndarray, targets: np.ndarray, radius_px: float) -> set[int]:
    if not len(pred_yx) or not len(targets):
        return set()
    tree = cKDTree(np.column_stack((pred_yx[:, 1], pred_yx[:, 0])))
    matched: set[int] = set()
    for target_index, (ty, tx) in enumerate(targets):
        if tree.query_ball_point([float(tx), float(ty)], r=float(radius_px)):
            matched.add(target_index)
    return matched


def build_frame(
    row: dict[str, str],
    data_root: Path,
    config: dict[str, object],
    source_model: CandidateScorer,
    args: argparse.Namespace,
    device: torch.device,
) -> CandidateFrame:
    fit_channel_mode = str(config.get("fit_channel_mode", "mean"))
    image_norm = config.get("image_normalization", {})
    sample_id = str(row.get("sample_id") or Path(row.get("image_out", "")).stem)
    image_path = resolve_data_path(data_root, row.get("image_out") or row.get("single_fits") or "")
    mask_path = resolve_data_path(data_root, row.get("mask_out") or "")
    raw = _read_input_image(image_path, fit_channel_mode)
    image = _normalize_image(raw, image_norm if isinstance(image_norm, dict) else {})
    mask = _read_mask_image(mask_path)
    targets = heatmap_to_centroids(mask, threshold=args.target_threshold, min_distance=args.min_distance)
    candidates = generate_mask(raw, "daofind_like", scorer_args(args.sigma)).centroids_yx.astype(np.float32)
    patches = extract_patches(image, candidates, args.patch_size)
    scores = score_candidates(source_model, patches, device, batch_size=args.eval_batch_size)
    pred_yx = candidates[scores >= float(args.source_threshold)]
    recalled_targets = _matched_targets(pred_yx, targets, args.match_radius_px)

    distances, nearest_target = _candidate_target_distances(candidates, targets)
    positive_index = np.flatnonzero(distances <= float(args.positive_radius_px))
    negative_index = np.flatnonzero(distances >= float(args.negative_radius_px))
    missed_positive_index = np.asarray(
        [
            int(index)
            for index in positive_index
            if int(nearest_target[index]) >= 0 and int(nearest_target[index]) not in recalled_targets
        ],
        dtype=np.int64,
    )
    return CandidateFrame(
        sample_id=sample_id,
        patches=patches,
        centroids_yx=candidates,
        target_yx=targets,
        positive_index=positive_index.astype(np.int64),
        negative_index=negative_index.astype(np.int64),
        missed_positive_index=missed_positive_index,
    )


def sample_training_data(
    frames: list[CandidateFrame],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    positive_parts = []
    negative_parts = []
    for frame in frames:
        if len(frame.missed_positive_index):
            positive_parts.append(frame.patches[frame.missed_positive_index])
        if len(frame.negative_index):
            negative_parts.append(frame.patches[frame.negative_index])
    positives = np.concatenate(positive_parts, axis=0) if positive_parts else np.empty((0, 1, args.patch_size, args.patch_size), dtype=np.float32)
    negatives = np.concatenate(negative_parts, axis=0) if negative_parts else np.empty((0, 1, args.patch_size, args.patch_size), dtype=np.float32)
    if len(positives) > args.max_positives:
        positives = positives[rng.choice(len(positives), size=args.max_positives, replace=False)]
    negative_count = min(len(negatives), int(round(len(positives) * float(args.neg_pos_ratio))))
    if len(negatives) > negative_count:
        negatives = negatives[rng.choice(len(negatives), size=negative_count, replace=False)]
    patches = np.concatenate([positives, negatives], axis=0)
    labels = np.concatenate([
        np.ones(len(positives), dtype=np.float32),
        np.zeros(len(negatives), dtype=np.float32),
    ])
    order = rng.permutation(len(labels))
    return patches[order], labels[order], {"positives": int(len(positives)), "negatives": int(len(negatives))}


def evaluate_frames(
    model: CandidateScorer,
    frames: list[CandidateFrame],
    thresholds: list[float],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    totals = {
        threshold: {"pred_count": 0, "target_count": 0, "matched_count": 0, "false_count": 0, "missed_count": 0}
        for threshold in thresholds
    }
    for frame in frames:
        scores = score_candidates(model, frame.patches, device, batch_size=args.eval_batch_size)
        for threshold in thresholds:
            metrics = detection_metrics(frame.centroids_yx[scores >= threshold], frame.target_yx, radius_px=args.match_radius_px)
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
        rows.append({
            "score_threshold": threshold,
            **row,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_path(args.data_root)
    config = load_config(resolve_path(args.config))
    device = resolve_device(args.device)
    source_model = load_simple_checkpoint(args.source_checkpoint, args.patch_size, device)

    train_rows = spread_subset(load_rows(data_root, args.train_split_reason), args.train_samples)
    val_rows = spread_subset(load_rows(data_root, args.val_split_reason), args.val_samples)
    print(f"[build] train_frames={len(train_rows)} val_frames={len(val_rows)} sigma={args.sigma}")
    train_frames = [build_frame(row, data_root, config, source_model, args, device) for row in train_rows]
    val_frames = [build_frame(row, data_root, config, source_model, args, device) for row in val_rows]
    patches, labels, counts = sample_training_data(train_frames, args, rng)
    print(f"[data] positives={counts['positives']} negatives={counts['negatives']} total={len(labels)}")
    if counts["positives"] == 0 or counts["negatives"] == 0:
        raise RuntimeError("not enough missed positives or negatives to train")

    model = CandidateScorer(args.patch_size).to(device)
    model.load_state_dict(source_model.state_dict())
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(patches), torch.from_numpy(labels)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    thresholds = [float(item) for item in args.score_thresholds.split(",") if item.strip()]
    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch_patches, batch_labels in loader:
            batch_patches = batch_patches.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_patches), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_labels)
            total += len(batch_labels)
        rows = evaluate_frames(model, val_frames, thresholds, args, device)
        best = max(rows, key=lambda row: float(row["f1"]))
        record = {"epoch": epoch, "loss": total_loss / max(total, 1), "best": best}
        history.append(record)
        if float(best["f1"]) > best_f1:
            best_f1 = float(best["f1"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "patch_size": args.patch_size,
                    "input_channels": 1,
                    "model_type": "legacy_simple",
                    "config": vars(args),
                    "best": best,
                    "epoch": epoch,
                },
                out_dir / "candidate_scorer_best.pt",
            )
        print(
            f"[epoch {epoch:02d}] loss={record['loss']:.4f} "
            f"f1={best['f1']:.3f} p={best['precision']:.3f} r={best['recall']:.3f} "
            f"t={best['score_threshold']}",
            flush=True,
        )

    final_rows = evaluate_frames(model, val_frames, thresholds, args, device)
    torch.save(
        {
            "model": model.state_dict(),
            "patch_size": args.patch_size,
            "input_channels": 1,
            "model_type": "legacy_simple",
            "config": vars(args),
            "best": max(final_rows, key=lambda row: float(row["f1"])),
            "epoch": args.epochs,
        },
        out_dir / "candidate_scorer_last.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(final_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)
    metadata = {
        "source_checkpoint": str(args.source_checkpoint),
        "train_frames": len(train_rows),
        "val_frames": len(val_rows),
        "sample_counts": counts,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
