from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candidate_scorer.build_dataset import candidate_numeric_features
from candidate_scorer.model import CenterAwareScorer, build_model_from_checkpoint
from candidate_scorer.pipeline import (
    add_center_channels,
    extract_patches,
    generate_candidates,
    make_center_aware_features,
    resolve_data_path,
    score_nms,
)
from star_unet.dataset import _normalize_image, _read_input_image, _read_mask_image
from star_unet.evaluate import load_manifest_samples, resolve_path
from star_unet.postprocess import detection_metrics, heatmap_to_centroids
from star_unet.train import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-frame candidate scorer evaluation.")
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=Path("star_unet/config.json"))
    parser.add_argument("--checkpoints", default="")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Backward-compatible single checkpoint.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/candidate_scorer_eval"))
    parser.add_argument("--split-reason", default="frame_holdout")
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--candidate-methods", default="")
    parser.add_argument("--dedup-radius-px", type=float, default=2.5)
    parser.add_argument("--score-thresholds", default="0.05:0.95:0.01")
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--nms-radius-px", type=float, default=3.5)
    parser.add_argument(
        "--score-mode",
        choices=("class", "class_quality", "quality"),
        default="class_quality",
        help="Candidate score used for thresholding.",
    )
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--match-radius-px", type=float, default=4.0)
    parser.add_argument("--edge-margin-px", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/candidate_cache"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_thresholds(text: str) -> list[float]:
    text = str(text).strip()
    if ":" in text and "," not in text:
        start, stop, step = (float(value) for value in text.split(":"))
        return np.arange(start, stop + step * 0.5, step, dtype=np.float64).tolist()
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def load_rows(data_root: Path, split_reason: str) -> list[dict[str, str]]:
    rows = load_manifest_samples(data_root, "train") + load_manifest_samples(data_root, "val")
    wanted = {value.strip() for value in split_reason.split(",") if value.strip()}
    return [row for row in rows if row.get("split_reason", row.get("split", "")) in wanted]


def select_rows(rows: list[dict[str, str]], sample_ids: str, count: int) -> list[dict[str, str]]:
    if sample_ids.strip():
        by_id = {str(row["sample_id"]): row for row in rows}
        wanted = [value.strip() for value in sample_ids.split(",") if value.strip()]
        return [by_id[value] for value in wanted]
    if count <= 0 or count >= len(rows):
        return rows
    indexes = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[int(index)] for index in indexes]


def normalize_numeric(features: np.ndarray, normalization: dict[str, object] | None) -> np.ndarray:
    if not normalization or not len(features):
        return features.astype(np.float32)
    mean = np.asarray(normalization.get("mean", []), dtype=np.float32)
    std = np.asarray(normalization.get("std", []), dtype=np.float32)
    if len(mean) != features.shape[1]:
        raise ValueError("numeric feature normalization dimension mismatch")
    return ((features - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def load_models(paths: list[Path], device: torch.device) -> list[tuple[torch.nn.Module, dict[str, object]]]:
    models = []
    for path in paths:
        checkpoint = torch.load(resolve_path(path), map_location=device)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(f"unsupported checkpoint format: {path}")
        model = build_model_from_checkpoint(checkpoint).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append((model, checkpoint))
    return models


def _candidate_cache_path(
    cache_dir: Path,
    sample_id: str,
    methods: str,
    dedup_radius: float,
    target_threshold: float,
    min_distance: float,
) -> Path:
    key = hashlib.sha256(
        f"v2-local-mean|{methods}|{dedup_radius:g}|{target_threshold:g}|{min_distance:g}".encode("utf-8")
    ).hexdigest()[:12]
    return cache_dir / key / f"{sample_id}.npz"


def load_or_build_candidates(
    raw: np.ndarray,
    target_heatmap: np.ndarray,
    sample_id: str,
    methods: str,
    dedup_radius: float,
    cache_dir: Path,
    use_cache: bool,
    target_threshold: float,
    min_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = _candidate_cache_path(
        cache_dir, sample_id, methods, dedup_radius, target_threshold, min_distance
    )
    if use_cache and path.exists():
        with np.load(path) as data:
            return (
                data["centroids_yx"].astype(np.float32),
                data["source_mask"].astype(np.int64),
                data["response"].astype(np.float32),
                data["target_yx"].astype(np.float32),
            )
    candidates = generate_candidates(raw, methods, dedup_radius)
    targets = heatmap_to_centroids(target_heatmap, threshold=target_threshold, min_distance=min_distance)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            centroids_yx=candidates.centroids_yx,
            source_mask=candidates.source_mask,
            response=candidates.response,
            target_yx=targets,
        )
    return candidates.centroids_yx, candidates.source_mask, candidates.response, targets


def score_candidates(
    models: list[tuple[torch.nn.Module, dict[str, object]]],
    raw: np.ndarray,
    normalized: np.ndarray,
    centroids_yx: np.ndarray,
    source_mask: np.ndarray,
    response: np.ndarray,
    batch_size: int,
    device: torch.device,
    score_mode: str = "class_quality",
) -> tuple[np.ndarray, np.ndarray]:
    if not len(centroids_yx):
        return np.empty((0,), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    scores_by_model = []
    offsets_by_model = []
    feature_image = make_center_aware_features(raw, normalized)
    raw_numeric = candidate_numeric_features(
        raw, normalized, centroids_yx, 31, response=response, source_mask=source_mask
    )
    for model, checkpoint in models:
        if not isinstance(model, CenterAwareScorer):
            raise ValueError("full-frame v2 evaluator requires center_aware_v2 checkpoints")
        patch_size = int(checkpoint.get("patch_size", 31))
        context_size = int(checkpoint.get("context_patch_size", 63))
        numeric = raw_numeric
        if patch_size != 31:
            numeric = candidate_numeric_features(
                raw, normalized, centroids_yx, patch_size,
                response=response, source_mask=source_mask,
            )
        numeric = normalize_numeric(numeric, checkpoint.get("numeric_normalization"))
        model_scores = []
        model_offsets = []
        with torch.no_grad():
            for start in range(0, len(centroids_yx), batch_size):
                batch_centroids = centroids_yx[start : start + batch_size]
                small = add_center_channels(extract_patches(feature_image, batch_centroids, patch_size))
                large = add_center_channels(extract_patches(feature_image, batch_centroids, context_size))
                output = model(
                    torch.from_numpy(small).to(device),
                    torch.from_numpy(large).to(device),
                    torch.from_numpy(numeric[start : start + batch_size]).to(device),
                )
                class_prob = torch.softmax(output["class_logits"], dim=1)[:, 2]
                quality = torch.sigmoid(output["quality_logit"])
                if score_mode == "class":
                    score = class_prob
                elif score_mode == "quality":
                    score = quality
                elif score_mode == "class_quality":
                    score = class_prob * quality.sqrt()
                else:
                    raise ValueError(f"unsupported score mode: {score_mode}")
                model_scores.append(score.cpu().numpy())
                model_offsets.append(output["offset_yx"].cpu().numpy())
        scores_by_model.append(np.concatenate(model_scores))
        offsets_by_model.append(np.concatenate(model_offsets))
    return (
        np.mean(np.stack(scores_by_model, axis=0), axis=0).astype(np.float32),
        np.mean(np.stack(offsets_by_model, axis=0), axis=0).astype(np.float32),
    )


def _edge_subset(points: np.ndarray, shape: tuple[int, int], margin: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    if not len(points):
        return points
    h, w = shape
    keep = (
        (points[:, 0] < margin) | (points[:, 1] < margin)
        | (points[:, 0] >= h - margin) | (points[:, 1] >= w - margin)
    )
    return points[keep]


def _aggregate(per_sample: list[dict[str, object]], threshold: float) -> dict[str, object]:
    rows = [sample["thresholds"][f"{threshold:.6f}"] for sample in per_sample]
    pred = sum(int(row["pred_count"]) for row in rows)
    target = sum(int(row["target_count"]) for row in rows)
    matched = sum(int(row["matched_count"]) for row in rows)
    precision = matched / pred if pred else 0.0
    recall = matched / target if target else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    macro_f1 = float(np.mean([float(row["f1"]) for row in rows])) if rows else 0.0
    return {
        "score_threshold": threshold,
        "pred_count": pred,
        "target_count": target,
        "matched_count": matched,
        "false_count": pred - matched,
        "missed_count": target - matched,
        "precision": precision,
        "recall": recall,
        "micro_f1": f1,
        "macro_f1": macro_f1,
    }


def _bootstrap_ci(per_sample: list[dict[str, object]], threshold: float, count: int, seed: int) -> list[float]:
    if not per_sample or count <= 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    values = []
    key = f"{threshold:.6f}"
    for _ in range(count):
        selected = rng.integers(0, len(per_sample), size=len(per_sample))
        rows = [per_sample[int(index)]["thresholds"][key] for index in selected]
        pred = sum(int(row["pred_count"]) for row in rows)
        target = sum(int(row["target_count"]) for row in rows)
        matched = sum(int(row["matched_count"]) for row in rows)
        precision = matched / pred if pred else 0.0
        recall = matched / target if target else 0.0
        values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return np.percentile(values, [2.5, 97.5]).astype(float).tolist()


def evaluate_models(
    models: list[tuple[torch.nn.Module, dict[str, object]]],
    rows: list[dict[str, str]],
    config: dict[str, object],
    thresholds: list[float],
    candidate_methods: str,
    args: argparse.Namespace,
    device: torch.device,
    data_root: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    fit_channel_mode = str(config.get("fit_channel_mode", "mean"))
    image_norm = config.get("image_normalization", {})
    image_norm = image_norm if isinstance(image_norm, dict) else {}
    per_sample = []
    data_root = resolve_path("data/data_model") if data_root is None else data_root
    cache_dir = resolve_path(args.cache_dir)
    for index, sample in enumerate(rows, start=1):
        started = time.perf_counter()
        sample_id = str(sample.get("sample_id") or "")
        raw = _read_input_image(
            resolve_data_path(data_root, sample.get("image_out") or sample.get("single_fits") or ""),
            fit_channel_mode,
        )
        normalized = _normalize_image(raw, image_norm)
        mask = _read_mask_image(resolve_data_path(data_root, sample.get("mask_out") or ""))
        centroids, source_mask, response, targets = load_or_build_candidates(
            raw, mask, sample_id, candidate_methods, args.dedup_radius_px,
            cache_dir, not args.no_cache, args.target_threshold, args.min_distance,
        )
        scores, offsets = score_candidates(
            models, raw, normalized, centroids, source_mask, response, args.batch_size,
            device, args.score_mode,
        )
        corrected = centroids + offsets
        threshold_rows = {}
        edge_rows = {}
        for threshold in thresholds:
            selected = np.flatnonzero(scores >= threshold)
            kept_local = score_nms(corrected[selected], scores[selected], args.nms_radius_px)
            predictions = corrected[selected[kept_local]]
            threshold_rows[f"{threshold:.6f}"] = detection_metrics(
                predictions, targets, radius_px=args.match_radius_px
            )
            edge_rows[f"{threshold:.6f}"] = detection_metrics(
                _edge_subset(predictions, raw.shape, args.edge_margin_px),
                _edge_subset(targets, raw.shape, args.edge_margin_px),
                radius_px=args.match_radius_px,
            )
        oracle = detection_metrics(centroids, targets, radius_px=args.match_radius_px)
        per_sample.append({
            "sample_id": sample_id,
            "group": str(sample.get("group", "")),
            "candidate_count": len(centroids),
            "target_count": len(targets),
            "oracle_recall": oracle["recall"],
            "elapsed_seconds": time.perf_counter() - started,
            "thresholds": threshold_rows,
            "edge_thresholds": edge_rows,
        })
        print(
            f"[{index}/{len(rows)}] {sample_id}: candidates={len(centroids)}"
            f" oracle_r={oracle['recall']:.3f} time={per_sample[-1]['elapsed_seconds']:.2f}s",
            flush=True,
        )
    summary_rows = [_aggregate(per_sample, threshold) for threshold in thresholds]
    best = dict(max(summary_rows, key=lambda row: float(row["micro_f1"])))
    best_threshold = float(best["score_threshold"])
    best["micro_f1_ci95"] = _bootstrap_ci(per_sample, best_threshold, args.bootstrap_samples, args.seed)
    best["candidate_oracle_recall"] = (
        sum(float(row["oracle_recall"]) * int(row["target_count"]) for row in per_sample)
        / max(sum(int(row["target_count"]) for row in per_sample), 1)
    )
    best["mean_candidates"] = float(np.mean([row["candidate_count"] for row in per_sample]))
    best["mean_seconds"] = float(np.mean([row["elapsed_seconds"] for row in per_sample]))
    best["score_mode"] = args.score_mode
    group_rows = {}
    for group in sorted({str(row["group"]) for row in per_sample}):
        selected = [row for row in per_sample if str(row["group"]) == group]
        group_rows[group] = _aggregate(selected, best_threshold)
    edge_samples = [
        {**row, "thresholds": row["edge_thresholds"]}
        for row in per_sample
    ]
    best["edge_metrics"] = _aggregate(edge_samples, best_threshold)
    best["groups"] = group_rows
    return summary_rows, {"best": best, "per_sample": per_sample}


def main() -> None:
    args = parse_args()
    checkpoint_text = args.checkpoints.strip()
    if not checkpoint_text and args.checkpoint is not None:
        checkpoint_text = str(args.checkpoint)
    paths = [Path(value.strip()) for value in checkpoint_text.split(",") if value.strip()]
    if not paths:
        raise ValueError("--checkpoints or --checkpoint is required")
    device = resolve_device(args.device)
    models = load_models(paths, device)
    first_checkpoint = models[0][1]
    candidate_methods = args.candidate_methods or str(
        first_checkpoint.get("candidate_methods", "daofind:2.0,daofind:2.5,sextractor:1.5")
    )
    data_root = resolve_path(args.data_root)
    rows = select_rows(load_rows(data_root, args.split_reason), args.sample_ids, args.count)
    thresholds = (
        [float(args.fixed_threshold)]
        if args.fixed_threshold is not None
        else parse_thresholds(args.score_thresholds)
    )
    summary_rows, report = evaluate_models(
        models, rows, load_config(resolve_path(args.config)), thresholds, candidate_methods, args, device,
        data_root=data_root,
    )
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "thresholds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(report["best"], ensure_ascii=False, indent=2))
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
