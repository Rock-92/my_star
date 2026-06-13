from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candidate_scorer.build_dataset import (
    SCHEMA_VERSION,
    ShardWriter,
    candidate_numeric_features,
    crop_at,
    crop_origins,
    label_candidates,
    load_rows,
    spread_subset,
)
from candidate_scorer.evaluate import load_models, score_candidates
from candidate_scorer.pipeline import (
    extract_patches,
    generate_candidates,
    make_center_aware_features,
    resolve_data_path,
)
from star_unet.dataset import _normalize_image, _read_input_image, _read_mask_image
from star_unet.evaluate import resolve_path
from star_unet.postprocess import heatmap_to_centroids
from star_unet.train import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine full-frame false positives into v2 training shards.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=Path("star_unet/config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/candidate_scorer_v2_hard"))
    parser.add_argument("--split-reason", default="train")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--crop-mode", choices=("stratified", "random", "center", "full"), default="stratified")
    parser.add_argument("--crops-per-image", type=int, default=5)
    parser.add_argument("--hard-background-per-crop", type=int, default=800)
    parser.add_argument("--hard-offcenter-per-crop", type=int, default=400)
    parser.add_argument("--positive-radius-px", type=float, default=4.0)
    parser.add_argument("--ignore-radius-px", type=float, default=6.0)
    parser.add_argument("--soft-label-sigma-px", type=float, default=2.0)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    models = load_models([args.checkpoint], device)
    checkpoint = models[0][1]
    if checkpoint.get("model_type") != "center_aware_v2":
        raise ValueError("hard-negative mining requires a center_aware_v2 checkpoint")
    methods = str(checkpoint["candidate_methods"])
    patch_size = int(checkpoint["patch_size"])
    context_size = int(checkpoint["context_patch_size"])
    dedup_radius = float(checkpoint.get("dedup_radius_px", 2.5))
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "build_state.json"
    fingerprint = hashlib.sha256(
        json.dumps({**vars(args), "checkpoint": str(resolve_path(args.checkpoint))}, default=str, sort_keys=True).encode()
    ).hexdigest()[:16]
    state = {"schema_version": SCHEMA_VERSION, "fingerprint": fingerprint, "completed": []}
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError("hard dataset already exists; pass --resume")
        if existing["fingerprint"] != fingerprint:
            raise ValueError("resume configuration fingerprint mismatch")
        state = existing

    config = load_config(resolve_path(args.config))
    channel_mode = str(config.get("fit_channel_mode", "mean"))
    image_norm = config.get("image_normalization", {})
    image_norm = image_norm if isinstance(image_norm, dict) else {}
    data_root = resolve_path(args.data_root)
    rows = spread_subset(load_rows(data_root, args.split_reason), args.samples)
    existing_shards = sorted((out_dir / "shards").glob("train_*.npz")) if (out_dir / "shards").exists() else []
    writer = ShardWriter(out_dir, "train", 10**9, len(existing_shards))
    sample_meta = []
    for row_index, sample in enumerate(rows, start=1):
        sample_id = str(sample["sample_id"])
        if sample_id in state["completed"]:
            continue
        seed = int(hashlib.sha256(f"{args.seed}|{sample_id}".encode()).hexdigest()[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        full_raw = _read_input_image(
            resolve_data_path(data_root, sample.get("image_out") or sample.get("single_fits") or ""),
            channel_mode,
        )
        full_mask = _read_mask_image(resolve_data_path(data_root, sample.get("mask_out") or ""))
        parts: dict[str, list[np.ndarray]] = {key: [] for key in (
            "patches_large", "classes", "quality", "offsets_yx",
            "centroids_yx", "numeric_features", "source_mask", "match_distance",
        )}
        for y0, x0 in crop_origins(
            full_raw.shape, args.crop_size, args.crop_mode, args.crops_per_image, rng
        ):
            raw = crop_at(full_raw, args.crop_size, y0, x0)
            normalized = _normalize_image(raw, image_norm)
            targets = heatmap_to_centroids(
                crop_at(full_mask, args.crop_size, y0, x0),
                threshold=args.target_threshold,
                min_distance=args.min_distance,
            )
            candidates = generate_candidates(raw, methods, dedup_radius)
            classes, quality, offsets, distances = label_candidates(
                candidates.centroids_yx, targets, args.positive_radius_px,
                args.ignore_radius_px, args.soft_label_sigma_px,
            )
            scores, _ = score_candidates(
                models, raw, normalized, candidates.centroids_yx,
                candidates.source_mask, candidates.response, args.batch_size, device,
            )
            selected_parts = [np.flatnonzero(classes == 2)]
            for class_id, limit in ((1, args.hard_offcenter_per_crop), (0, args.hard_background_per_crop)):
                indexes = np.flatnonzero(classes == class_id)
                indexes = indexes[np.argsort(-scores[indexes])[:limit]]
                selected_parts.append(indexes)
            chosen = np.concatenate(selected_parts)
            rng.shuffle(chosen)
            centroids = candidates.centroids_yx[chosen]
            feature_image = make_center_aware_features(raw, normalized)
            parts["patches_large"].append(
                extract_patches(feature_image, centroids, context_size).astype(np.float16)
            )
            parts["classes"].append(classes[chosen])
            parts["quality"].append(quality[chosen])
            parts["offsets_yx"].append(offsets[chosen])
            parts["centroids_yx"].append(centroids)
            parts["numeric_features"].append(candidate_numeric_features(
                raw, normalized, centroids, patch_size,
                candidates.response[chosen], candidates.source_mask[chosen],
            ))
            parts["source_mask"].append(candidates.source_mask[chosen])
            parts["match_distance"].append(distances[chosen])
        arrays = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
        writer.add(arrays)
        writer.flush()
        meta = {
            "sample_id": sample_id,
            "count": len(arrays["classes"]),
            "true": int(np.sum(arrays["classes"] == 2)),
            "offcenter": int(np.sum(arrays["classes"] == 1)),
            "background": int(np.sum(arrays["classes"] == 0)),
        }
        sample_meta.append(meta)
        state["completed"].append(sample_id)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[{row_index}/{len(rows)}] {sample_id}: {meta}", flush=True)

    shard_paths = sorted((out_dir / "shards").glob("train_*.npz"))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "candidate_methods": methods,
        "dedup_radius_px": dedup_radius,
        "patch_size": patch_size,
        "context_patch_size": context_size,
        "input_channels": int(checkpoint["input_channels"]),
        "numeric_feature_dim": int(checkpoint["feature_dim"]),
        "numeric_normalization": checkpoint["numeric_normalization"],
        "train_shards": [str(path.relative_to(out_dir)) for path in shard_paths],
        "val_shards": [],
        "samples": sample_meta,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
