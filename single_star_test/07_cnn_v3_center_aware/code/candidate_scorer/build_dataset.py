from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candidate_scorer.pipeline import (
    extract_patches,
    generate_candidates,
    make_center_aware_features,
    resolve_data_path,
)
from preprocessing.mask_generator import subtract_background
from star_unet.dataset import _normalize_image, _read_input_image, _read_mask_image
from star_unet.evaluate import load_manifest_samples, resolve_path
from star_unet.postprocess import heatmap_to_centroids
from star_unet.train import load_config


SCHEMA_VERSION = 3
NUMERIC_FEATURE_NAMES = [
    "raw_center",
    "norm_center",
    "residual_center",
    "matched_center",
    "local_mean",
    "local_std",
    "snr",
    "x_norm",
    "y_norm",
    "edge_distance_norm",
    "candidate_response",
    "local_q90",
    "moment_xx",
    "moment_yy",
    "moment_xy",
    "source_count",
]


def resolve_cli_path(path: Path) -> Path:
    """Resolve CLI paths from the current working directory first.

    The archived code imports helpers from the old U-Net directory, whose
    resolve_path treats paths as relative to that code folder. For the
    single_star_test archive we want commands run from repo root to resolve
    paths such as single_star_test/data/data_model literally.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists() or path.parts[:1] == ("single_star_test",):
        return cwd_path
    return resolve_path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build resumable sharded data for the center-aware candidate scorer.")
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=Path("star_unet/config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/candidate_scorer_v2"))
    parser.add_argument("--candidate-methods", default="daofind:2.0,daofind:2.5,sextractor:1.5")
    parser.add_argument("--dedup-radius-px", type=float, default=2.5)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--crop-mode", choices=("center", "random", "stratified", "full"), default="stratified")
    parser.add_argument("--crops-per-image", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=31)
    parser.add_argument("--context-patch-size", type=int, default=63)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--val-samples", type=int, default=0)
    parser.add_argument("--train-split-reason", default="train")
    parser.add_argument("--val-split-reason", default="frame_holdout")
    parser.add_argument("--positive-radius-px", type=float, default=4.0)
    parser.add_argument("--ignore-radius-px", type=float, default=6.0)
    parser.add_argument("--soft-label-sigma-px", type=float, default=2.0)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--max-negatives-per-crop", type=int, default=300)
    parser.add_argument("--max-offcenter-per-crop", type=int, default=100)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--resume", action="store_true")
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


def crop_at(array: np.ndarray, size: int, y0: int, x0: int) -> np.ndarray:
    if size <= 0:
        return array
    return array[y0 : y0 + min(size, array.shape[0]), x0 : x0 + min(size, array.shape[1])]


def crop_origins(
    shape: tuple[int, int],
    size: int,
    mode: str,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if size <= 0 or mode == "full":
        return [(0, 0)]
    h, w = shape[:2]
    ch, cw = min(size, h), min(size, w)
    max_y, max_x = max(h - ch, 0), max(w - cw, 0)
    if mode == "center":
        return [(max_y // 2, max_x // 2)]
    if mode == "random":
        return [
            (int(rng.integers(0, max_y + 1)), int(rng.integers(0, max_x + 1)))
            for _ in range(max(1, count))
        ]
    anchors = [
        (0, 0),
        (0, max_x),
        (max_y, 0),
        (max_y, max_x),
        (max_y // 2, max_x // 2),
    ]
    origins = anchors[: min(len(anchors), max(1, count))]
    while len(origins) < max(1, count):
        origins.append((int(rng.integers(0, max_y + 1)), int(rng.integers(0, max_x + 1))))
    return origins


def label_candidates(
    candidate_yx: np.ndarray,
    target_yx: np.ndarray,
    positive_radius_px: float,
    ignore_radius_px: float,
    soft_label_sigma_px: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.asarray(candidate_yx, dtype=np.float32).reshape((-1, 2))
    targets = np.asarray(target_yx, dtype=np.float32).reshape((-1, 2))
    classes = np.zeros(len(candidates), dtype=np.int64)
    quality = np.zeros(len(candidates), dtype=np.float32)
    offsets = np.zeros((len(candidates), 2), dtype=np.float32)
    distances = np.full(len(candidates), np.inf, dtype=np.float32)
    if not len(candidates) or not len(targets):
        return classes, quality, offsets, distances

    target_tree = cKDTree(np.column_stack((targets[:, 1], targets[:, 0])))
    distances[:], _ = target_tree.query(np.column_stack((candidates[:, 1], candidates[:, 0])), k=1)
    classes[distances < float(ignore_radius_px)] = 1

    candidate_tree = cKDTree(np.column_stack((candidates[:, 1], candidates[:, 0])))
    edges: list[tuple[float, int, int]] = []
    for target_index, (ty, tx) in enumerate(targets):
        nearby = candidate_tree.query_ball_point([float(tx), float(ty)], r=float(positive_radius_px))
        for candidate_index in nearby:
            distance = float(np.linalg.norm(candidates[candidate_index] - targets[target_index]))
            edges.append((distance, candidate_index, target_index))
    edges.sort(key=lambda row: (row[0], row[1], row[2]))
    used_candidates: set[int] = set()
    used_targets: set[int] = set()
    sigma = max(float(soft_label_sigma_px), 1e-6)
    for distance, candidate_index, target_index in edges:
        if candidate_index in used_candidates or target_index in used_targets:
            continue
        used_candidates.add(candidate_index)
        used_targets.add(target_index)
        classes[candidate_index] = 2
        quality[candidate_index] = float(np.exp(-(distance**2) / (2.0 * sigma**2)))
        offsets[candidate_index] = targets[target_index] - candidates[candidate_index]
        distances[candidate_index] = distance
    return classes, quality, offsets, distances


def candidate_numeric_features(
    raw: np.ndarray,
    normalized: np.ndarray,
    centroids_yx: np.ndarray,
    patch_size: int,
    response: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
) -> np.ndarray:
    centroids = np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2))
    if not len(centroids):
        return np.empty((0, len(NUMERIC_FEATURE_NAMES)), dtype=np.float32)
    residual = subtract_background(raw, "local_mean", 25)
    matched = ndi.gaussian_filter(residual, sigma=max(3.0 / 2.3548, 0.5))
    radius = patch_size // 2
    h, w = raw.shape
    rows = []
    response = np.zeros(len(centroids), dtype=np.float32) if response is None else response
    source_mask = np.ones(len(centroids), dtype=np.int64) if source_mask is None else source_mask
    for index, (cy, cx) in enumerate(centroids):
        y = int(np.clip(round(float(cy) - 0.5), 0, h - 1))
        x = int(np.clip(round(float(cx) - 0.5), 0, w - 1))
        patch = residual[max(0, y - radius) : min(h, y + radius + 1), max(0, x - radius) : min(w, x + radius + 1)]
        local_mean = float(np.mean(patch)) if patch.size else 0.0
        local_std = float(np.std(patch)) if patch.size else 1.0
        local_std = local_std if np.isfinite(local_std) and local_std > 1e-6 else 1.0
        positive = np.clip(patch - local_mean, 0.0, None).astype(np.float64)
        total = float(positive.sum())
        if total > 0:
            yy, xx = np.mgrid[: patch.shape[0], : patch.shape[1]].astype(np.float64)
            yy -= (patch.shape[0] - 1) * 0.5
            xx -= (patch.shape[1] - 1) * 0.5
            mxx = float(np.sum(xx * xx * positive) / total)
            myy = float(np.sum(yy * yy * positive) / total)
            mxy = float(np.sum(xx * yy * positive) / total)
        else:
            mxx = myy = mxy = 0.0
        edge = min(y, x, h - 1 - y, w - 1 - x)
        rows.append(
            [
                raw[y, x],
                normalized[y, x],
                residual[y, x],
                matched[y, x],
                local_mean,
                local_std,
                (residual[y, x] - local_mean) / local_std,
                x / max(w - 1, 1),
                y / max(h - 1, 1),
                edge / max(min(h, w) * 0.5, 1.0),
                response[index],
                float(np.percentile(patch, 90)) if patch.size else 0.0,
                mxx,
                myy,
                mxy,
                bin(int(source_mask[index])).count("1"),
            ]
        )
    return np.nan_to_num(np.asarray(rows, dtype=np.float32))


def _select_training_indices(
    classes: np.ndarray,
    max_negatives: int,
    max_offcenter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    parts = [np.flatnonzero(classes == 2)]
    for class_id, limit in ((1, max_offcenter), (0, max_negatives)):
        indexes = np.flatnonzero(classes == class_id)
        if limit >= 0 and len(indexes) > limit:
            indexes = rng.choice(indexes, size=limit, replace=False)
        parts.append(indexes)
    chosen = np.concatenate(parts)
    rng.shuffle(chosen)
    return chosen


def build_sample(
    sample: dict[str, str],
    data_root: Path,
    config: dict[str, object],
    args: argparse.Namespace,
    split: str,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    sample_id = str(sample.get("sample_id") or Path(sample.get("image_out", "")).stem)
    fit_channel_mode = str(config.get("fit_channel_mode", "mean"))
    image_norm = config.get("image_normalization", {})
    image_norm = image_norm if isinstance(image_norm, dict) else {}
    full_raw = _read_input_image(
        resolve_data_path(data_root, sample.get("image_out") or sample.get("single_fits") or ""),
        fit_channel_mode,
    )
    full_mask = _read_mask_image(resolve_data_path(data_root, sample.get("mask_out") or ""))
    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "patches_large", "classes", "quality", "offsets_yx",
            "centroids_yx", "numeric_features", "source_mask", "match_distance",
        )
    }
    totals = {"target_count": 0, "candidate_count": 0, "class_0": 0, "class_1": 0, "class_2": 0}
    origins = crop_origins(full_raw.shape, args.crop_size, args.crop_mode, args.crops_per_image, rng)
    for y0, x0 in origins:
        raw = crop_at(full_raw, args.crop_size, y0, x0)
        normalized = _normalize_image(raw, image_norm)
        targets = heatmap_to_centroids(
            crop_at(full_mask, args.crop_size, y0, x0),
            threshold=args.target_threshold,
            min_distance=args.min_distance,
        )
        candidate_set = generate_candidates(raw, args.candidate_methods, args.dedup_radius_px)
        classes, quality, offsets, distances = label_candidates(
            candidate_set.centroids_yx,
            targets,
            args.positive_radius_px,
            args.ignore_radius_px,
            args.soft_label_sigma_px,
        )
        chosen = np.arange(len(classes))
        if split in {"train", "val"}:
            chosen = _select_training_indices(
                classes, args.max_negatives_per_crop, args.max_offcenter_per_crop, rng
            )
        centroids = candidate_set.centroids_yx[chosen]
        features = make_center_aware_features(raw, normalized)
        parts["patches_large"].append(
            extract_patches(features, centroids, args.context_patch_size).astype(np.float16)
        )
        parts["classes"].append(classes[chosen])
        parts["quality"].append(quality[chosen])
        parts["offsets_yx"].append(offsets[chosen])
        parts["centroids_yx"].append(centroids)
        parts["numeric_features"].append(candidate_numeric_features(
            raw, normalized, centroids, args.patch_size,
            candidate_set.response[chosen], candidate_set.source_mask[chosen],
        ))
        parts["source_mask"].append(candidate_set.source_mask[chosen])
        parts["match_distance"].append(distances[chosen])
        totals["target_count"] += len(targets)
        totals["candidate_count"] += len(candidate_set.centroids_yx)
        for class_id in range(3):
            totals[f"class_{class_id}"] += int(np.sum(classes[chosen] == class_id))
    arrays = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    meta = {
        "split": split,
        "sample_id": sample_id,
        "group": str(sample.get("group", "")),
        "split_reason": str(sample.get("split_reason", "")),
        "crop_count": len(origins),
        "kept_count": int(len(arrays["classes"])),
        **totals,
    }
    return arrays, meta


def _config_fingerprint(args: argparse.Namespace) -> str:
    excluded = {"resume", "out_dir"}
    payload = {key: str(value) for key, value in vars(args).items() if key not in excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


class ShardWriter:
    def __init__(self, out_dir: Path, split: str, shard_size: int, start_index: int = 0) -> None:
        self.shard_dir = out_dir / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_size = max(1, int(shard_size))
        self.index = start_index
        self.parts: dict[str, list[np.ndarray]] = {}
        self.count = 0

    def add(self, arrays: dict[str, np.ndarray]) -> None:
        sample_count = len(arrays["classes"])
        if self.count and self.count + sample_count > self.shard_size:
            self.flush()
        for key, value in arrays.items():
            self.parts.setdefault(key, []).append(value)
        self.count += sample_count
        if self.count >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.count:
            return
        arrays = {key: np.concatenate(value, axis=0) for key, value in self.parts.items()}
        path = self.shard_dir / f"{self.split}_{self.index:06d}.npz"
        _atomic_savez(path, arrays)
        self.index += 1
        self.parts = {}
        self.count = 0


def _numeric_normalization(shards: list[Path]) -> dict[str, object]:
    total = 0
    sum_x = np.zeros(len(NUMERIC_FEATURE_NAMES), dtype=np.float64)
    sum_x2 = np.zeros(len(NUMERIC_FEATURE_NAMES), dtype=np.float64)
    for path in shards:
        with np.load(path) as data:
            values = data["numeric_features"].astype(np.float64)
        total += len(values)
        sum_x += values.sum(axis=0)
        sum_x2 += np.square(values).sum(axis=0)
    mean = sum_x / max(total, 1)
    variance = np.maximum(sum_x2 / max(total, 1) - mean * mean, 1e-12)
    return {
        "mean": mean.astype(float).tolist(),
        "std": np.sqrt(variance).astype(float).tolist(),
        "names": NUMERIC_FEATURE_NAMES,
    }


def main() -> None:
    args = parse_args()
    out_dir = resolve_cli_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "build_state.json"
    fingerprint = _config_fingerprint(args)
    state = {"schema_version": SCHEMA_VERSION, "fingerprint": fingerprint, "completed": {"train": [], "val": []}}
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError(f"{out_dir} already contains build state; pass --resume or choose another --out-dir")
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("resume configuration does not match the existing dataset fingerprint")
        state = existing
    config = load_config(resolve_cli_path(args.config))
    data_root = resolve_cli_path(args.data_root)
    split_rows = {
        "train": spread_subset(load_rows(data_root, args.train_split_reason), args.train_samples),
        "val": spread_subset(load_rows(data_root, args.val_split_reason), args.val_samples),
    }
    all_meta: list[dict[str, object]] = []
    samples_csv = out_dir / "samples.csv"
    if samples_csv.exists():
        with samples_csv.open("r", newline="", encoding="utf-8") as handle:
            all_meta = list(csv.DictReader(handle))

    for split, rows in split_rows.items():
        existing_shards = sorted((out_dir / "shards").glob(f"{split}_*.npz")) if (out_dir / "shards").exists() else []
        writer = ShardWriter(out_dir, split, args.shard_size, len(existing_shards))
        completed = set(state["completed"][split])
        for index, sample in enumerate(rows, start=1):
            sample_id = str(sample.get("sample_id") or "")
            if sample_id in completed:
                continue
            sample_seed = int(hashlib.sha256(
                f"{args.seed}|{split}|{sample_id}".encode("utf-8")
            ).hexdigest()[:16], 16) % (2**32)
            arrays, meta = build_sample(
                sample, data_root, config, args, split, np.random.default_rng(sample_seed)
            )
            writer.add(arrays)
            writer.flush()
            all_meta.append(meta)
            state["completed"][split].append(sample_id)
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[{split}] {index}/{len(rows)} {sample_id}:"
                f" true={meta['class_2']} offcenter={meta['class_1']} background={meta['class_0']}",
                flush=True,
            )
        writer.flush()

    train_shards = sorted((out_dir / "shards").glob("train_*.npz"))
    val_shards = sorted((out_dir / "shards").glob("val_*.npz"))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "candidate_methods": args.candidate_methods,
        "dedup_radius_px": args.dedup_radius_px,
        "patch_size": args.patch_size,
        "context_patch_size": args.context_patch_size,
        "input_channels": 6,
        "numeric_feature_dim": len(NUMERIC_FEATURE_NAMES),
        "numeric_normalization": _numeric_normalization(train_shards),
        "train_split_reason": args.train_split_reason,
        "val_split_reason": args.val_split_reason,
        "train_shards": [str(path.relative_to(out_dir)) for path in train_shards],
        "val_shards": [str(path.relative_to(out_dir)) for path in val_shards],
        "samples": all_meta,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if all_meta:
        fieldnames = list(all_meta[0].keys())
        with samples_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_meta)
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
