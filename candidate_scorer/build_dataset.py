from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.mask_generator import generate_mask, subtract_background
from star_unet.dataset import _normalize_image, _read_input_image, _read_mask_image
from star_unet.evaluate import baseline_args, load_manifest_samples, resolve_path
from star_unet.postprocess import heatmap_to_centroids
from star_unet.train import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build patch data for the candidate scorer.")
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=Path("star_unet/config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/candidate_scorer_sigma2p5"))
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--crop-mode", choices=("center", "random"), default="center")
    parser.add_argument("--crops-per-image", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=31)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=1)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--positive-radius-px", type=float, default=4.0)
    parser.add_argument("--ignore-radius-px", type=float, default=6.0)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--max-negatives-per-image", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def spread_subset(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    indexes = np.linspace(0, len(rows) - 1, limit, dtype=int)
    return [rows[int(index)] for index in indexes]


def crop_at(array: np.ndarray, size: int, y0: int, x0: int) -> np.ndarray:
    if size <= 0:
        return array
    h, w = array.shape[:2]
    crop_h = min(size, h)
    crop_w = min(size, w)
    return array[y0 : y0 + crop_h, x0 : x0 + crop_w]


def crop_origins(shape: tuple[int, int], size: int, mode: str, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if size <= 0:
        return [(0, 0)]
    h, w = shape[:2]
    crop_h = min(size, h)
    crop_w = min(size, w)
    if mode == "center":
        return [(max(0, (h - crop_h) // 2), max(0, (w - crop_w) // 2))]
    origins = []
    for _ in range(max(1, int(count))):
        y0 = int(rng.integers(0, max(h - crop_h + 1, 1)))
        x0 = int(rng.integers(0, max(w - crop_w + 1, 1)))
        origins.append((y0, x0))
    return origins


def daofind_args(sigma: float) -> SimpleNamespace:
    args = SimpleNamespace(**vars(baseline_args()))
    args.sigma = float(sigma)
    return args


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


def normalize_feature(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, [0.5, 99.8])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((np.nan_to_num(image, nan=float(low)) - float(low)) / (float(high) - float(low)), 0.0, 1.0).astype(np.float32)


def make_feature_image(raw: np.ndarray, normalized: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        return normalized[None, :, :].astype(np.float32)
    residual = subtract_background(raw, "local_median", 25)
    matched = ndi.gaussian_filter(residual, sigma=max(3.0 / 2.3548, 0.5))
    return np.stack(
        [
            normalized,
            normalize_feature(residual),
            normalize_feature(matched),
        ],
        axis=0,
    ).astype(np.float32)


def extract_patches(feature_image: np.ndarray, centroids_yx: np.ndarray, patch_size: int) -> np.ndarray:
    radius = patch_size // 2
    padded = np.pad(feature_image, ((0, 0), (radius, radius), (radius, radius)), mode="reflect")
    patches = []
    for cy, cx in np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2)):
        y = int(round(float(cy) - 0.5)) + radius
        x = int(round(float(cx) - 0.5)) + radius
        patches.append(padded[:, y - radius : y + radius + 1, x - radius : x + radius + 1].astype(np.float32))
    if not patches:
        return np.empty((0, feature_image.shape[0], patch_size, patch_size), dtype=np.float32)
    return np.asarray(patches, dtype=np.float32)


def build_split(
    rows: list[dict[str, str]],
    config: dict[str, object],
    args: argparse.Namespace,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    rng = np.random.default_rng(args.seed + (0 if split == "train" else 1000))
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

        sample_patch_parts = []
        sample_label_parts = []
        sample_centroid_parts = []
        target_total = 0
        candidate_total = 0
        positive_total = 0
        negative_total = 0
        origins = crop_origins(full_raw.shape, args.crop_size, args.crop_mode, args.crops_per_image, rng)
        for crop_index, (y0, x0) in enumerate(origins):
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

            pos_idx = np.flatnonzero(labels > 0.5)
            neg_idx = np.flatnonzero(labels <= 0.5)
            if split == "train" and len(neg_idx) > args.max_negatives_per_image:
                neg_idx = rng.choice(neg_idx, size=args.max_negatives_per_image, replace=False)
            chosen = np.concatenate([pos_idx, neg_idx])
            if len(chosen):
                rng.shuffle(chosen)
                centroids = centroids[chosen]
                labels = labels[chosen]

            patches = extract_patches(feature_image, centroids, args.patch_size)
            sample_patch_parts.append(patches)
            sample_label_parts.append(labels.astype(np.float32))
            sample_centroid_parts.append(centroids.astype(np.float32))
            target_total += int(len(target_yx))
            candidate_total += int(len(result.centroids_yx))
            positive_total += int(np.sum(labels > 0.5))
            negative_total += int(np.sum(labels <= 0.5))

        patches = np.concatenate(sample_patch_parts, axis=0)
        labels = np.concatenate(sample_label_parts, axis=0)
        centroids = np.concatenate(sample_centroid_parts, axis=0)
        patch_parts.append(patches)
        label_parts.append(labels.astype(np.float32))
        centroid_parts.append(centroids.astype(np.float32))
        meta_rows.append(
            {
                "split": split,
                "sample_id": sample_id,
                "crop_count": int(len(origins)),
                "target_count": target_total,
                "candidate_count": candidate_total,
                "kept_count": int(len(labels)),
                "positive_count": positive_total,
                "negative_count": negative_total,
            }
        )
        print(f"[{split}] {index}/{len(rows)} {sample_id}: pos={meta_rows[-1]['positive_count']} neg={meta_rows[-1]['negative_count']}")

    return (
        np.concatenate(patch_parts, axis=0),
        np.concatenate(label_parts, axis=0),
        np.concatenate(centroid_parts, axis=0),
        meta_rows,
    )


def write_split(out_dir: Path, split: str, patches: np.ndarray, labels: np.ndarray, centroids_yx: np.ndarray) -> None:
    np.savez_compressed(
        out_dir / f"{split}_patches.npz",
        patches=patches.astype(np.float16),
        labels=labels.astype(np.float32),
        centroids_yx=centroids_yx.astype(np.float32),
    )


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    data_root = resolve_path(args.data_root)

    train_rows = spread_subset(load_manifest_samples(data_root, "train"), args.train_samples)
    val_rows = spread_subset(load_manifest_samples(data_root, "val"), args.val_samples)
    train_x, train_y, train_centroids, train_meta = build_split(train_rows, config, args, "train")
    val_x, val_y, val_centroids, val_meta = build_split(val_rows, config, args, "val")

    write_split(out_dir, "train", train_x, train_y, train_centroids)
    write_split(out_dir, "val", val_x, val_y, val_centroids)
    metadata = {
        "sigma": args.sigma,
        "crop_size": args.crop_size,
        "crop_mode": args.crop_mode,
        "crops_per_image": args.crops_per_image,
        "patch_size": args.patch_size,
        "channels": args.channels,
        "positive_radius_px": args.positive_radius_px,
        "ignore_radius_px": args.ignore_radius_px,
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
