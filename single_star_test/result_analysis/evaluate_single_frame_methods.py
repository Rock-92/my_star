from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate traditional extractors and CNN V1 on full-frame coord_holdout data."
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "single_star_test/data/data_model")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "single_star_test/00_unet_heatmap/code/star_unet/config.json",
    )
    parser.add_argument("--split-reason", default="coord_holdout")
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--sample-ids", default="")
    parser.add_argument(
        "--methods",
        default="daofind5=daofind:5.0,daofind4p5=daofind:4.5,log3p2=log:3.2,alog3p2=alog:3.2",
        help="Comma-separated name=method:sigma entries.",
    )
    parser.add_argument("--cnn-checkpoint", type=Path, default=None)
    parser.add_argument("--cnn-candidate-methods", default="daofind:2.5")
    parser.add_argument("--cnn-thresholds", default="0.30:0.95:0.01")
    parser.add_argument("--dedup-radius-px", type=float, default=2.5)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--match-radius-px", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "single_star_test/result_analysis/single_frame_test_extractors",
    )
    return parser.parse_args()


def parse_thresholds(text: str) -> list[float]:
    text = str(text).strip()
    if ":" in text and "," not in text:
        start, stop, step = (float(value) for value in text.split(":"))
        return np.arange(start, stop + step * 0.5, step, dtype=np.float64).tolist()
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def parse_methods(text: str) -> list[tuple[str, str]]:
    rows = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, spec = item.split("=", 1)
        else:
            spec = item
            name = spec.replace(":", "")
        rows.append((name.strip(), spec.strip()))
    return rows


def load_manifest_rows(data_root: Path, split_reason: str) -> list[dict[str, str]]:
    manifest = data_root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"missing manifest: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    wanted = {item.strip() for item in split_reason.split(",") if item.strip()}
    return [row for row in rows if row.get("split_reason", row.get("split", "")) in wanted]


def select_rows(rows: list[dict[str, str]], sample_ids: str, count: int) -> list[dict[str, str]]:
    if sample_ids.strip():
        by_id = {str(row["sample_id"]): row for row in rows}
        return [by_id[item.strip()] for item in sample_ids.split(",") if item.strip()]
    if count <= 0 or count >= len(rows):
        return rows
    indexes = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[int(index)] for index in indexes]


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


def metrics_with_context(
    method: str,
    sample_id: str,
    pred_yx: np.ndarray,
    target_yx: np.ndarray,
    seconds: float,
) -> dict[str, Any]:
    row = detection_metrics(pred_yx, target_yx, radius_px=args_global.match_radius_px)
    return {
        "method": method,
        "sample_id": sample_id,
        **row,
        "seconds": seconds,
    }


def load_cnn(checkpoint_path: Path, device: torch.device) -> tuple[SimpleCandidateScorer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"unsupported CNN checkpoint: {checkpoint_path}")
    patch_size = int(checkpoint.get("patch_size", 31))
    model = SimpleCandidateScorer(patch_size=patch_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def score_cnn(
    model: nn.Module,
    normalized: np.ndarray,
    centroids_yx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not len(centroids_yx):
        return np.empty((0,), dtype=np.float32)
    feature_image = normalized[None].astype(np.float32)
    patches = extract_patches(feature_image, centroids_yx, 31)[:, :1]
    scores = []
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.from_numpy(patches[start : start + batch_size]).to(device)
            scores.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    pred = int(sum(int(row["pred_count"]) for row in rows))
    target = int(sum(int(row["target_count"]) for row in rows))
    matched = int(sum(int(row["matched_count"]) for row in rows))
    false = int(sum(int(row["false_count"]) for row in rows))
    missed = int(sum(int(row["missed_count"]) for row in rows))
    micro_p = matched / pred if pred else 0.0
    micro_r = matched / target if target else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    return {
        "image_count": len(rows),
        "mean_precision": float(np.mean([float(row["precision"]) for row in rows])),
        "mean_recall": float(np.mean([float(row["recall"]) for row in rows])),
        "mean_f1": float(np.mean([float(row["f1"]) for row in rows])),
        "mean_pred_count": float(np.mean([float(row["pred_count"]) for row in rows])),
        "mean_target_count": float(np.mean([float(row["target_count"]) for row in rows])),
        "mean_seconds": float(np.mean([float(row["seconds"]) for row in rows])),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "pred_count": pred,
        "target_count": target,
        "matched_count": matched,
        "false_count": false,
        "missed_count": missed,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_cnn_checkpoint() -> Path:
    candidates = [
        REPO_ROOT
        / "single_star_test/01_cnn_v1_simple_sigma2p5/results/candidate_scorer_sigma2p5_full_resume60/candidate_scorer_best.pt",
        REPO_ROOT
        / "single_star_test/01_cnn_v1_simple_sigma2p5/results/candidate_scorer_sigma2p5_full_resume60/candidate_scorer.pt",
        REPO_ROOT
        / "single_star_test/01_cnn_v1_simple_sigma2p5/results/candidate_scorer_sigma2p5_full/candidate_scorer.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("could not find a CNN V1 checkpoint; pass --cnn-checkpoint")


def main() -> None:
    global args_global
    args = parse_args()
    args_global = args
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    image_norm = config.get("image_normalization", {})
    fit_channel_mode = str(config.get("fit_channel_mode", "mean"))
    rows = select_rows(load_manifest_rows(args.data_root, args.split_reason), args.sample_ids, args.count)
    device = resolve_device(args.device)
    cnn_path = args.cnn_checkpoint or default_cnn_checkpoint()
    cnn_model, _ = load_cnn(cnn_path, device)
    thresholds = parse_thresholds(args.cnn_thresholds)
    method_specs = parse_methods(args.methods)

    per_sample: list[dict[str, Any]] = []
    cnn_threshold_rows: dict[float, list[dict[str, Any]]] = {threshold: [] for threshold in thresholds}

    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id") or index)
        image_path = resolve_data_path(args.data_root, row.get("image_out") or row.get("single_fits") or "")
        mask_path = resolve_data_path(args.data_root, row.get("mask_out") or "")
        raw = _read_input_image(image_path, fit_channel_mode)
        normalized = _normalize_image(raw, image_norm if isinstance(image_norm, dict) else {})
        mask = _read_mask_image(mask_path)
        targets = heatmap_to_centroids(mask, threshold=args.target_threshold, min_distance=args.min_distance)

        for method_name, method_spec in method_specs:
            start = time.perf_counter()
            candidates = generate_candidates(raw, method_spec, dedup_radius_px=args.dedup_radius_px).centroids_yx
            seconds = time.perf_counter() - start
            metrics = metrics_with_context(method_name, sample_id, candidates, targets, seconds)
            per_sample.append(metrics)

        start = time.perf_counter()
        candidates = generate_candidates(
            raw,
            args.cnn_candidate_methods,
            dedup_radius_px=args.dedup_radius_px,
        ).centroids_yx
        scores = score_cnn(cnn_model, normalized, candidates, device, args.batch_size)
        score_seconds = time.perf_counter() - start
        for threshold in thresholds:
            selected = candidates[scores >= float(threshold)]
            metrics = metrics_with_context(
                f"cnn_v1_t{threshold:.2f}",
                sample_id,
                selected,
                targets,
                score_seconds,
            )
            cnn_threshold_rows[threshold].append(metrics)
        print(f"[{index}/{len(rows)}] {sample_id}: targets={len(targets)} cnn_candidates={len(candidates)}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    for method_name, _ in method_specs:
        summary_rows.append({"method": method_name, **aggregate([row for row in per_sample if row["method"] == method_name])})
    for threshold, metric_rows in cnn_threshold_rows.items():
        summary_rows.append({"method": f"cnn_v1_t{threshold:.2f}", "threshold": threshold, **aggregate(metric_rows)})
        per_sample.extend(metric_rows)
    best_cnn = max(
        (row for row in summary_rows if str(row["method"]).startswith("cnn_v1_")),
        key=lambda row: float(row["mean_f1"]),
    )

    write_csv(args.out_dir / "per_sample.csv", per_sample)
    write_csv(args.out_dir / "summary.csv", summary_rows)
    report = {
        "split_reason": args.split_reason,
        "image_count": len(rows),
        "cnn_checkpoint": str(cnn_path),
        "cnn_candidate_methods": args.cnn_candidate_methods,
        "best_cnn_by_mean_f1": best_cnn,
        "summary": summary_rows,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["best_cnn_by_mean_f1"], ensure_ascii=False, indent=2))
    print(f"[done] wrote {args.out_dir}")


args_global: argparse.Namespace


if __name__ == "__main__":
    main()
