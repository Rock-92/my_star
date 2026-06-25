from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

from preprocessing.mask_generator import METHODS, generate_mask
from .dataset import _normalize_image, _read_input_image, _read_mask_image
from .postprocess import (
    detection_metrics,
    heatmap_to_centroids,
    heatmap_to_uint8,
    normalize_for_display,
    write_triplet,
)
from .predict import load_model, predict_heatmap
from .train import load_config, resolve_device


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate U-Net and baseline star extractors on data_model validation data.")
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/star_unet/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/star_unet/eval"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5")
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--match-radius-px", type=float, default=4.0)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of tiles to run per model forward pass.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Accepted for command compatibility; this evaluator reads samples sequentially.",
    )
    parser.add_argument("--skip-baselines", action="store_true", help="Do not run traditional baseline extractors.")
    parser.add_argument(
        "--cache-predictions",
        action="store_true",
        help="Save and reuse model heatmaps so threshold sweeps can run without another forward pass.",
    )
    parser.add_argument("--prediction-cache-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def parse_thresholds(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("at least one threshold is required")
    return values


def resolve_path(path_text: str | Path) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_manifest_samples(data_root: Path, split: str) -> list[dict[str, str]]:
    manifest = data_root / "manifest.csv"
    if manifest.exists():
        rows = []
        with manifest.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("split") == split:
                    rows.append(row)
        return rows

    image_dir = data_root / split / "images"
    mask_dir = data_root / split / "masks"
    if not image_dir.exists():
        raise FileNotFoundError(f"missing split image directory: {image_dir}")
    rows = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        mask_path = next((mask_dir / f"{image_path.stem}{suffix}" for suffix in (".png", ".tif", ".tiff", ".jpg") if (mask_dir / f"{image_path.stem}{suffix}").exists()), None)
        if mask_path is None:
            continue
        rows.append(
            {
                "sample_id": image_path.stem,
                "split": split,
                "group": "",
                "single_fits": str(image_path),
                "image_out": str(image_path),
                "mask_out": str(mask_path),
                "split_reason": "unknown",
            }
        )
    return rows


def baseline_args() -> SimpleNamespace:
    return SimpleNamespace(
        sigma=None,
        filtsize=25,
        background_mode="local_mean",
        min_area=5,
        max_area=100,
        max_axis_ratio=None,
        no_binary_open=False,
        mesh_size=64,
        mesh_filter_size=3,
        filter_sigma=1.0,
        fwhm=3.0,
        peak_window=None,
        mask_radius=None,
        radius_mode="gaussian",
        min_mask_radius=None,
        max_mask_radius=None,
        radius_scale=1.0,
        min_separation=None,
        max_peaks=None,
        exclude_border=None,
    )


def add_summary(acc: dict[str, dict[str, Any]], key: str, metrics: dict[str, Any]) -> None:
    row = acc.setdefault(
        key,
        {
            "pred_count": 0,
            "target_count": 0,
            "matched_count": 0,
            "false_count": 0,
            "missed_count": 0,
            "samples": 0,
        },
    )
    for field in ("pred_count", "target_count", "matched_count", "false_count", "missed_count"):
        row[field] += int(metrics[field])
    row["samples"] += 1


def finalize_summary(acc: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, row in sorted(acc.items()):
        pred = int(row["pred_count"])
        target = int(row["target_count"])
        matched = int(row["matched_count"])
        precision = matched / pred if pred else 0.0
        recall = matched / target if target else 0.0
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                **row,
                "method": key,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_rate": int(row["false_count"]) / pred if pred else 0.0,
            }
        )
    return rows


def prediction_cache_path(out_dir: Path, args: argparse.Namespace, sample_id: str) -> Path:
    cache_dir = resolve_path(args.prediction_cache_dir) if args.prediction_cache_dir is not None else out_dir / "prediction_cache"
    return cache_dir / f"{sample_id}.npy"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    thresholds = parse_thresholds(args.thresholds)
    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_manifest_samples(data_root, args.split)
    if not samples:
        raise RuntimeError(f"no samples found for split {args.split!r} under {data_root}")

    device = resolve_device(args.device or config.get("device", "auto"))
    model = None
    checkpoint = resolve_path(args.checkpoint)
    if not args.skip_model:
        if checkpoint.exists():
            model = load_model(checkpoint, config, device)
        else:
            print(f"[warn] checkpoint not found, skipping U-Net: {checkpoint}")

    fit_channel_mode = str(config.get("fit_channel_mode", "mean"))
    image_norm = config.get("image_normalization", {})
    bargs = baseline_args()
    summary_acc: dict[str, dict[str, Any]] = {}
    summary_by_reason: dict[str, dict[str, Any]] = {}
    per_sample: list[dict[str, Any]] = []
    preview_written = 0

    for index, sample in enumerate(samples, start=1):
        image_path = resolve_path(sample.get("image_out") or sample.get("single_fits") or "")
        mask_path = resolve_path(sample.get("mask_out") or "")
        split_reason = sample.get("split_reason") or "unknown"
        raw = _read_input_image(image_path, fit_channel_mode)
        image = _normalize_image(raw, image_norm)
        target_heatmap = _read_mask_image(mask_path)
        target_centroids = heatmap_to_centroids(
            target_heatmap,
            threshold=args.target_threshold,
            min_distance=args.min_distance,
        )

        sample_result: dict[str, Any] = {
            "sample_id": sample.get("sample_id", image_path.stem),
            "image": str(image_path),
            "mask": str(mask_path),
            "group": sample.get("group", ""),
            "split_reason": split_reason,
            "target_count": int(len(target_centroids)),
            "methods": [],
        }

        if not args.skip_baselines:
            for method in METHODS:
                result = generate_mask(raw, method, bargs)
                metrics = detection_metrics(result.centroids_yx, target_centroids, radius_px=args.match_radius_px)
                metrics = {"method": method, **metrics}
                sample_result["methods"].append(metrics)
                add_summary(summary_acc, method, metrics)
                add_summary(summary_by_reason, f"{method}/{split_reason}", metrics)

        model_heatmap = None
        if model is not None:
            sample_id = str(sample.get("sample_id") or image_path.stem)
            cache_path = prediction_cache_path(out_dir, args, sample_id)
            if args.cache_predictions and cache_path.exists():
                model_heatmap = np.load(cache_path).astype(np.float32)
            else:
                model_heatmap = predict_heatmap(
                    model,
                    image,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                    fit_channel_mode=fit_channel_mode,
                    device=device,
                    batch_size=args.batch_size,
                )
                if args.cache_predictions:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache_path, model_heatmap.astype(np.float16))
            for threshold in thresholds:
                pred_centroids = heatmap_to_centroids(
                    model_heatmap,
                    threshold=threshold,
                    min_distance=args.min_distance,
                )
                method_name = f"unet_t{threshold:.2f}"
                metrics = detection_metrics(pred_centroids, target_centroids, radius_px=args.match_radius_px)
                metrics = {"method": method_name, "threshold": threshold, **metrics}
                sample_result["methods"].append(metrics)
                add_summary(summary_acc, method_name, metrics)
                add_summary(summary_by_reason, f"{method_name}/{split_reason}", metrics)

            if preview_written < args.preview_count:
                preview_path = out_dir / "previews" / f"{image_path.stem}_triplet.jpg"
                write_triplet(preview_path, normalize_for_display(raw), target_heatmap, model_heatmap)
                Image.fromarray(heatmap_to_uint8(model_heatmap)).save(out_dir / "previews" / f"{image_path.stem}_pred_heatmap.png")
                preview_written += 1

        per_sample.append(sample_result)
        print(f"[eval] {index}/{len(samples)} {image_path.name}: target={len(target_centroids)}")

    summary = finalize_summary(summary_acc)
    reason_summary = finalize_summary(summary_by_reason)
    unet_rows = [row for row in summary if row["method"].startswith("unet_t")]
    best_unet = max(unet_rows, key=lambda row: row["f1"], default=None)
    output = {
        "config": {
            "data_root": str(data_root),
            "split": args.split,
            "checkpoint": str(checkpoint) if model is not None else None,
            "thresholds": thresholds,
            "target_threshold": float(args.target_threshold),
            "min_distance": float(args.min_distance),
            "match_radius_px": float(args.match_radius_px),
            "tile_size": int(args.tile_size),
            "tile_overlap": int(args.tile_overlap),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "skip_baselines": bool(args.skip_baselines),
            "cache_predictions": bool(args.cache_predictions),
            "prediction_cache_dir": str(resolve_path(args.prediction_cache_dir))
            if args.prediction_cache_dir is not None
            else str(out_dir / "prediction_cache"),
        },
        "best_unet": best_unet,
        "summary": summary,
        "summary_by_split_reason": reason_summary,
        "per_sample": per_sample,
    }
    (out_dir / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "per_sample.json").write_text(json.dumps(per_sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_dir / 'summary.json'}")
    if best_unet is not None:
        print(f"[best] {best_unet['method']} f1={best_unet['f1']:.4f} recall={best_unet['recall']:.4f} false={best_unet['false_rate']:.4f}")


if __name__ == "__main__":
    main()
