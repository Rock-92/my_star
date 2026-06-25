from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.mask_generator import generate_mask
from star_unet.dataset import _read_input_image, _read_mask_image
from star_unet.evaluate import baseline_args, load_manifest_samples, resolve_path
from star_unet.postprocess import detection_metrics, heatmap_to_centroids
from star_unet.train import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe whether lower DAOFind sigma improves candidate recall.")
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=Path("star_unet/config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/star_unet/daofind_sigma_probe"))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--sigmas", default="5,4,3,2.5,2,1.5")
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--match-radius-px", type=float, default=4.0)
    return parser.parse_args()


def pick_spread(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count >= len(rows):
        return rows
    indexes = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[int(index)] for index in indexes]


def pick_by_ids(rows: list[dict[str, str]], sample_ids: str) -> list[dict[str, str]]:
    wanted = [part.strip() for part in sample_ids.split(",") if part.strip()]
    by_id = {str(row.get("sample_id") or Path(row.get("image_out", "")).stem): row for row in rows}
    return [by_id[item] for item in wanted]


def center_crop(array: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return array
    h, w = array.shape[:2]
    crop_h = min(size, h)
    crop_w = min(size, w)
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    return array[y0 : y0 + crop_h, x0 : x0 + crop_w]


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    fit_channel_mode = str(config.get("fit_channel_mode", "mean"))
    data_root = resolve_path(args.data_root)
    samples = load_manifest_samples(data_root, "val")
    selected = pick_by_ids(samples, args.sample_ids) if args.sample_ids.strip() else pick_spread(samples, args.count)
    sigmas = [float(item.strip()) for item in args.sigmas.split(",") if item.strip()]

    rows: list[dict[str, object]] = []
    for sample in selected:
        sample_id = str(sample.get("sample_id") or Path(sample.get("image_out", "")).stem)
        image_path = resolve_path(sample.get("image_out") or sample.get("single_fits") or "")
        mask_path = resolve_path(sample.get("mask_out") or "")
        raw = center_crop(_read_input_image(image_path, fit_channel_mode), args.crop_size)
        target_heatmap = center_crop(_read_mask_image(mask_path), args.crop_size)
        target_yx = heatmap_to_centroids(target_heatmap, threshold=args.target_threshold, min_distance=args.min_distance)

        for sigma in sigmas:
            bargs = SimpleNamespace(**vars(baseline_args()))
            bargs.sigma = sigma
            result = generate_mask(raw, "daofind_like", bargs)
            metrics = detection_metrics(result.centroids_yx, target_yx, radius_px=args.match_radius_px)
            rows.append(
                {
                    "sample_id": sample_id,
                    "sigma": sigma,
                    "target_count": int(len(target_yx)),
                    **metrics,
                    "raw_peaks": result.debug.get("raw_peaks"),
                    "threshold": result.debug.get("threshold"),
                    "noise": result.debug.get("noise"),
                }
            )
        print(f"[done] {sample_id}: target={len(target_yx)}")

    summary: list[dict[str, object]] = []
    for sigma in sigmas:
        sigma_rows = [row for row in rows if row["sigma"] == sigma]
        pred = sum(int(row["pred_count"]) for row in sigma_rows)
        target = sum(int(row["target_count"]) for row in sigma_rows)
        matched = sum(int(row["matched_count"]) for row in sigma_rows)
        false = sum(int(row["false_count"]) for row in sigma_rows)
        precision = matched / pred if pred else 0.0
        recall = matched / target if target else 0.0
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        summary.append(
            {
                "sigma": sigma,
                "samples": len(sigma_rows),
                "pred_count": pred,
                "target_count": target,
                "matched_count": matched,
                "false_count": false,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_rate": false / pred if pred else 0.0,
            }
        )

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "per_sample": rows}, indent=2), encoding="utf-8")
    print(f"[wrote] {out_dir}")
    for row in summary:
        print(
            f"sigma={row['sigma']}: pred={row['pred_count']} matched={row['matched_count']} "
            f"precision={row['precision']:.3f} recall={row['recall']:.3f} f1={row['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
