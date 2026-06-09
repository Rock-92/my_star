from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candidate_scorer.model import CandidateScorer
from preprocessing.mask_generator import generate_mask, subtract_background
from star_unet.dataset import _normalize_image, _read_input_image, _read_mask_image
from star_unet.evaluate import baseline_args, load_manifest_samples, resolve_path
from star_unet.postprocess import detection_metrics, heatmap_to_centroids, normalize_for_display
from star_unet.train import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coordinate-level evaluation for DAOFind + CNN candidate scorer.")
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=Path("star_unet/config.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/candidate_scorer_sigma2p5_full/candidate_scorer.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/candidate_scorer_sigma2p5_full/eval_spotcheck"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--candidate-sigma", type=float, default=2.5)
    parser.add_argument("--baseline-sigma", type=float, default=5.0)
    parser.add_argument("--score-thresholds", default="0.7,0.8,0.9")
    parser.add_argument("--patch-size", type=int, default=31)
    parser.add_argument("--crop-size", type=int, default=0, help="Optional centered crop for fast debugging. 0 means full frame.")
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--match-radius-px", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preview-count", type=int, default=3)
    return parser.parse_args()


def pick_spread(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or count >= len(rows):
        return rows
    indexes = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[int(index)] for index in indexes]


def pick_by_ids(rows: list[dict[str, str]], sample_ids: str) -> list[dict[str, str]]:
    wanted = [part.strip() for part in sample_ids.split(",") if part.strip()]
    by_id = {str(row.get("sample_id") or Path(row.get("image_out", "")).stem): row for row in rows}
    missing = [sample_id for sample_id in wanted if sample_id not in by_id]
    if missing:
        raise KeyError(f"sample id(s) not found: {', '.join(missing)}")
    return [by_id[sample_id] for sample_id in wanted]


def method_args(sigma: float) -> SimpleNamespace:
    args = SimpleNamespace(**vars(baseline_args()))
    args.sigma = float(sigma)
    return args


def center_crop(array: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    if size <= 0:
        return array, (0, 0)
    h, w = array.shape[:2]
    crop_h = min(size, h)
    crop_w = min(size, w)
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    return array[y0 : y0 + crop_h, x0 : x0 + crop_w], (y0, x0)


def normalize_feature(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, [0.5, 99.8])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((np.nan_to_num(image, nan=float(low)) - float(low)) / (float(high) - float(low)), 0.0, 1.0).astype(np.float32)


def make_feature_image(raw: np.ndarray, normalized: np.ndarray, input_channels: int) -> np.ndarray:
    if input_channels == 1:
        return normalized[None, :, :].astype(np.float32)
    if input_channels != 3:
        raise ValueError(f"unsupported input_channels={input_channels}")
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


def score_patches(model: CandidateScorer, patches: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    if len(patches) == 0:
        return np.empty((0,), dtype=np.float32)
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.from_numpy(patches[start : start + batch_size]).to(device)
            scores.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def add_summary(acc: dict[str, dict[str, int]], method: str, metrics: dict[str, object]) -> None:
    row = acc.setdefault(method, {"pred_count": 0, "target_count": 0, "matched_count": 0, "false_count": 0, "missed_count": 0, "samples": 0})
    for key in ("pred_count", "target_count", "matched_count", "false_count", "missed_count"):
        row[key] += int(metrics[key])
    row["samples"] += 1


def finalize(acc: dict[str, dict[str, int]]) -> list[dict[str, object]]:
    rows = []
    for method, row in sorted(acc.items()):
        pred = row["pred_count"]
        target = row["target_count"]
        matched = row["matched_count"]
        precision = matched / pred if pred else 0.0
        recall = matched / target if target else 0.0
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({**row, "method": method, "precision": precision, "recall": recall, "f1": f1, "false_rate": row["false_count"] / pred if pred else 0.0})
    return rows


def draw_panel(image: np.ndarray, title: str, target_yx: np.ndarray, pred_yx: np.ndarray | None, max_side: int = 900) -> Image.Image:
    base = Image.fromarray(normalize_for_display(image)).convert("RGB")
    scale = min(max_side / base.width, max_side / base.height, 1.0)
    if scale < 1.0:
        base = base.resize((int(base.width * scale), int(base.height * scale)), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(base)
    for y, x in np.asarray(target_yx, dtype=np.float32).reshape((-1, 2)):
        sx = float(x) * scale
        sy = float(y) * scale
        r = 4
        draw.ellipse((sx - r, sy - r, sx + r, sy + r), outline=(0, 255, 80), width=1)
    if pred_yx is not None:
        for y, x in np.asarray(pred_yx, dtype=np.float32).reshape((-1, 2)):
            sx = float(x) * scale
            sy = float(y) * scale
            r = 6
            draw.line((sx - r, sy, sx + r, sy), fill=(255, 60, 60), width=1)
            draw.line((sx, sy - r, sx, sy + r), fill=(255, 60, 60), width=1)

    band_h = 22
    canvas = Image.new("RGB", (base.width, base.height + band_h), (18, 18, 18))
    canvas.paste(base, (0, band_h))
    ImageDraw.Draw(canvas).text((6, 5), title, fill=(240, 240, 240), font=ImageFont.load_default())
    return canvas


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    data_root = resolve_path(args.data_root)
    samples = load_manifest_samples(data_root, args.split)
    selected = pick_by_ids(samples, args.sample_ids) if args.sample_ids.strip() else pick_spread(samples, args.count)
    thresholds = [float(item.strip()) for item in args.score_thresholds.split(",") if item.strip()]

    device = resolve_device(args.device)
    checkpoint = torch.load(resolve_path(args.checkpoint), map_location=device)
    input_channels = int(checkpoint.get("input_channels", 1)) if isinstance(checkpoint, dict) else 1
    model = CandidateScorer(input_channels=input_channels).to(device)
    model.load_state_dict(checkpoint["model"])

    fit_channel_mode = str(config.get("fit_channel_mode", "mean"))
    image_norm = config.get("image_normalization", {})
    image_norm = image_norm if isinstance(image_norm, dict) else {}
    per_sample = []
    summary_acc: dict[str, dict[str, int]] = {}
    preview_written = 0

    for index, sample in enumerate(selected, start=1):
        sample_id = str(sample.get("sample_id") or Path(sample.get("image_out", "")).stem)
        image_path = resolve_path(sample.get("image_out") or sample.get("single_fits") or "")
        mask_path = resolve_path(sample.get("mask_out") or "")
        raw, (y0, x0) = center_crop(_read_input_image(image_path, fit_channel_mode), args.crop_size)
        image, _ = center_crop(_normalize_image(raw, image_norm), args.crop_size)
        feature_image = make_feature_image(raw, image, input_channels)
        target_heatmap, _ = center_crop(_read_mask_image(mask_path), args.crop_size)
        target_yx = heatmap_to_centroids(target_heatmap, threshold=args.target_threshold, min_distance=args.min_distance)

        baseline_yx = generate_mask(raw, "daofind_like", method_args(args.baseline_sigma)).centroids_yx
        candidate_yx = generate_mask(raw, "daofind_like", method_args(args.candidate_sigma)).centroids_yx
        patches = extract_patches(feature_image, candidate_yx, args.patch_size)
        scores = score_patches(model, patches, device, args.batch_size)

        sample_rows = []
        baseline_metrics = {"method": f"daofind_s{args.baseline_sigma:g}", **detection_metrics(baseline_yx, target_yx, radius_px=args.match_radius_px)}
        raw_candidate_metrics = {"method": f"daofind_s{args.candidate_sigma:g}_raw", **detection_metrics(candidate_yx, target_yx, radius_px=args.match_radius_px)}
        for metrics in (baseline_metrics, raw_candidate_metrics):
            sample_rows.append(metrics)
            add_summary(summary_acc, str(metrics["method"]), metrics)

        best_pred_yx = None
        best_metrics = None
        for threshold in thresholds:
            pred_yx = candidate_yx[scores >= threshold]
            metrics = {
                "method": f"cnn_s{args.candidate_sigma:g}_t{threshold:.2f}",
                "score_threshold": threshold,
                **detection_metrics(pred_yx, target_yx, radius_px=args.match_radius_px),
            }
            sample_rows.append(metrics)
            add_summary(summary_acc, str(metrics["method"]), metrics)
            if best_metrics is None or float(metrics["f1"]) > float(best_metrics["f1"]):
                best_metrics = metrics
                best_pred_yx = pred_yx

        per_sample.append(
            {
                "sample_id": sample_id,
                "image": str(image_path),
                "crop_origin_yx": [y0, x0],
                "target_count": int(len(target_yx)),
                "methods": sample_rows,
            }
        )
        print(f"[{index}/{len(selected)}] {sample_id}: target={len(target_yx)} baseline_f1={baseline_metrics['f1']:.3f} best_cnn={best_metrics['method']} f1={best_metrics['f1']:.3f}")

        if preview_written < args.preview_count:
            panels = [
                draw_panel(raw, f"{sample_id} target={len(target_yx)}", target_yx, None),
                draw_panel(raw, f"DAO s{args.baseline_sigma:g} f1={baseline_metrics['f1']:.3f}", target_yx, baseline_yx),
                draw_panel(raw, f"{best_metrics['method']} f1={best_metrics['f1']:.3f}", target_yx, best_pred_yx),
            ]
            canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), (0, 0, 0))
            x = 0
            for panel in panels:
                canvas.paste(panel, (x, 0))
                x += panel.width
            canvas.save(out_dir / f"{sample_id}_compare.jpg", quality=92)
            preview_written += 1

    summary = finalize(summary_acc)
    (out_dir / "per_sample.json").write_text(json.dumps(per_sample, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
