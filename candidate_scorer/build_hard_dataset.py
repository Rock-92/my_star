from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candidate_scorer.build_dataset import (
    build_split,
    crop_at,
    crop_origins,
    daofind_args,
    extract_patches,
    label_candidates,
    make_feature_image,
    spread_subset,
    write_split,
)
from candidate_scorer.evaluate import score_patches
from candidate_scorer.model import CandidateScorer
from preprocessing.mask_generator import generate_mask
from star_unet.dataset import _normalize_image, _read_input_image, _read_mask_image
from star_unet.evaluate import load_manifest_samples, resolve_path
from star_unet.postprocess import heatmap_to_centroids
from star_unet.train import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build single-channel candidate scorer data with mined hard negatives.")
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=Path("star_unet/config.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/candidate_scorer_sigma2p5_hardneg"))
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=31)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=1)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--val-samples", type=int, default=0)
    parser.add_argument("--hard-negatives-per-image", type=int, default=800)
    parser.add_argument("--random-negatives-per-image", type=int, default=0)
    parser.add_argument("--positive-radius-px", type=float, default=4.0)
    parser.add_argument("--ignore-radius-px", type=float, default=6.0)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_scorer(checkpoint_path: Path, input_channels: int, device: torch.device) -> CandidateScorer:
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location=device)
    checkpoint_channels = int(checkpoint.get("input_channels", input_channels)) if isinstance(checkpoint, dict) else input_channels
    if checkpoint_channels != input_channels:
        raise ValueError(f"checkpoint input_channels={checkpoint_channels}, requested channels={input_channels}")
    model = CandidateScorer(input_channels=input_channels).to(device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_hard_train_split(
    rows: list[dict[str, str]],
    config: dict[str, object],
    args: argparse.Namespace,
    model: CandidateScorer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    rng = np.random.default_rng(args.seed)
    fit_channel_mode = str(config.get("fit_channel_mode", "mean"))
    image_norm = config.get("image_normalization", {})
    image_norm = image_norm if isinstance(image_norm, dict) else {}

    patch_parts = []
    label_parts = []
    centroid_parts = []
    meta_rows = []

    for index, sample in enumerate(rows, start=1):
        sample_id = str(sample.get("sample_id") or Path(sample.get("image_out", "")).stem)
        image_path = resolve_path(sample.get("image_out") or sample.get("single_fits") or "")
        mask_path = resolve_path(sample.get("mask_out") or "")
        full_raw = _read_input_image(image_path, fit_channel_mode)
        full_mask = _read_mask_image(mask_path)
        origins = crop_origins(full_raw.shape, args.crop_size, "center", 1, rng)
        y0, x0 = origins[0]
        raw = crop_at(full_raw, args.crop_size, y0, x0)
        image = _normalize_image(raw, image_norm)
        feature_image = make_feature_image(raw, image, args.channels)
        target_heatmap = crop_at(full_mask, args.crop_size, y0, x0)
        target_yx = heatmap_to_centroids(
            target_heatmap,
            threshold=args.target_threshold,
            min_distance=args.min_distance,
        )
        result = generate_mask(raw, "daofind_like", daofind_args(args.sigma))
        labels, keep = label_candidates(
            result.centroids_yx,
            target_yx,
            positive_radius_px=args.positive_radius_px,
            ignore_radius_px=args.ignore_radius_px,
        )
        centroids = result.centroids_yx[keep]
        labels = labels[keep]
        patches = extract_patches(feature_image, centroids, args.patch_size)

        pos_idx = np.flatnonzero(labels > 0.5)
        neg_idx = np.flatnonzero(labels <= 0.5)
        hard_idx = np.empty((0,), dtype=np.int64)
        random_idx = np.empty((0,), dtype=np.int64)

        if len(neg_idx):
            neg_scores = score_patches(model, patches[neg_idx], device, args.batch_size)
            hard_count = min(len(neg_idx), int(args.hard_negatives_per_image))
            if hard_count:
                hard_local = np.argsort(-neg_scores)[:hard_count]
                hard_idx = neg_idx[hard_local]

            remaining = np.setdiff1d(neg_idx, hard_idx, assume_unique=False)
            random_count = min(len(remaining), int(args.random_negatives_per_image))
            if random_count:
                random_idx = rng.choice(remaining, size=random_count, replace=False)

        chosen = np.concatenate([pos_idx, hard_idx, random_idx])
        if len(chosen):
            rng.shuffle(chosen)
        patch_parts.append(patches[chosen])
        label_parts.append(labels[chosen].astype(np.float32))
        centroid_parts.append(centroids[chosen].astype(np.float32))

        meta = {
            "split": "train",
            "sample_id": sample_id,
            "target_count": int(len(target_yx)),
            "candidate_count": int(len(result.centroids_yx)),
            "kept_count": int(len(chosen)),
            "positive_count": int(len(pos_idx)),
            "hard_negative_count": int(len(hard_idx)),
            "random_negative_count": int(len(random_idx)),
            "negative_count": int(len(hard_idx) + len(random_idx)),
        }
        meta_rows.append(meta)
        print(
            f"[train] {index}/{len(rows)} {sample_id}:"
            f" pos={meta['positive_count']}"
            f" hard_neg={meta['hard_negative_count']}"
            f" random_neg={meta['random_negative_count']}"
        )

    return (
        np.concatenate(patch_parts, axis=0),
        np.concatenate(label_parts, axis=0),
        np.concatenate(centroid_parts, axis=0),
        meta_rows,
    )


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(resolve_path(args.config))
    data_root = resolve_path(args.data_root)
    train_rows = spread_subset(load_manifest_samples(data_root, "train"), args.train_samples)
    val_rows = spread_subset(load_manifest_samples(data_root, "val"), args.val_samples)

    device = resolve_device(args.device)
    model = load_scorer(args.checkpoint, input_channels=args.channels, device=device)
    train_x, train_y, train_centroids, train_meta = build_hard_train_split(train_rows, config, args, model, device)

    val_args = argparse.Namespace(**vars(args))
    val_args.max_negatives_per_image = 10**9
    val_x, val_y, val_centroids, val_meta = build_split(val_rows, config, val_args, "val")

    write_split(out_dir, "train", train_x, train_y, train_centroids)
    write_split(out_dir, "val", val_x, val_y, val_centroids)

    metadata = {
        "sigma": args.sigma,
        "channels": args.channels,
        "crop_size": args.crop_size,
        "patch_size": args.patch_size,
        "checkpoint": str(resolve_path(args.checkpoint)),
        "hard_negatives_per_image": args.hard_negatives_per_image,
        "random_negatives_per_image": args.random_negatives_per_image,
        "train_patch_count": int(len(train_y)),
        "train_positive_count": int(np.sum(train_y > 0.5)),
        "train_negative_count": int(np.sum(train_y <= 0.5)),
        "val_patch_count": int(len(val_y)),
        "val_positive_count": int(np.sum(val_y > 0.5)),
        "val_negative_count": int(np.sum(val_y <= 0.5)),
        "samples": train_meta + val_meta,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list((train_meta + val_meta)[0].keys()))
        writer.writeheader()
        writer.writerows(train_meta + val_meta)
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
