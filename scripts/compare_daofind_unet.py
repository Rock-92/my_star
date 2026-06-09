from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.mask_generator import generate_mask
from star_unet.dataset import _normalize_image, _read_input_image, _read_mask_image
from star_unet.evaluate import baseline_args, load_manifest_samples, resolve_path
from star_unet.postprocess import detection_metrics, heatmap_to_centroids, normalize_for_display
from star_unet.predict import load_model, predict_heatmap
from star_unet.train import load_config, resolve_device
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DAOFIND-like baseline and U-Net on a few validation frames.")
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=Path("star_unet/config.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/star_unet/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/star_unet/daofind_compare"))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--sample-ids", default="", help="Comma-separated sample ids. Overrides --count when provided.")
    parser.add_argument("--crop-size", type=int, default=0, help="Use a centered square crop for a faster spot check.")
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5")
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--match-radius-px", type=float, default=4.0)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def parse_thresholds(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def pick_spread(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count >= len(rows):
        return rows
    indexes = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[int(index)] for index in indexes]


def pick_by_ids(rows: list[dict[str, str]], sample_ids: str) -> list[dict[str, str]]:
    wanted = [part.strip() for part in sample_ids.split(",") if part.strip()]
    if not wanted:
        return []
    by_id = {str(row.get("sample_id") or Path(row.get("image_out", "")).stem): row for row in rows}
    missing = [sample_id for sample_id in wanted if sample_id not in by_id]
    if missing:
        raise KeyError(f"sample id(s) not found in validation split: {', '.join(missing)}")
    return [by_id[sample_id] for sample_id in wanted]


def center_crop(array: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    if size <= 0:
        return array, (0, 0)
    h, w = array.shape[:2]
    crop_h = min(size, h)
    crop_w = min(size, w)
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    return array[y0 : y0 + crop_h, x0 : x0 + crop_w], (y0, x0)


def direct_predict_crop(model, image: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)[None, None]).to(device)
    with torch.no_grad():
        logits = model(tensor)
        heatmap = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return heatmap.astype(np.float32)


def draw_points(
    image: np.ndarray,
    title: str,
    target_yx: np.ndarray,
    pred_yx: np.ndarray | None = None,
    max_side: int = 900,
) -> Image.Image:
    image_u8 = normalize_for_display(image)
    pil = Image.fromarray(image_u8).convert("RGB")
    scale = min(max_side / pil.width, max_side / pil.height, 1.0)
    if scale < 1.0:
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.Resampling.BILINEAR)

    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    for y, x in np.asarray(target_yx).reshape((-1, 2)):
        sx, sy = float(x) * scale, float(y) * scale
        r = 4
        draw.ellipse((sx - r, sy - r, sx + r, sy + r), outline=(0, 255, 80), width=1)

    if pred_yx is not None:
        for y, x in np.asarray(pred_yx).reshape((-1, 2)):
            sx, sy = float(x) * scale, float(y) * scale
            r = 6
            draw.line((sx - r, sy, sx + r, sy), fill=(255, 60, 60), width=1)
            draw.line((sx, sy - r, sx, sy + r), fill=(255, 60, 60), width=1)

    band_h = 22
    canvas = Image.new("RGB", (pil.width, pil.height + band_h), (18, 18, 18))
    canvas.paste(pil, (0, band_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(240, 240, 240), font=font)
    return canvas


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    data_root = resolve_path(args.data_root)
    rows = load_manifest_samples(data_root, "val")
    selected = pick_by_ids(rows, args.sample_ids) if args.sample_ids.strip() else pick_spread(rows, args.count)
    thresholds = parse_thresholds(args.thresholds)

    device = resolve_device(args.device or config.get("device", "auto"))
    model = load_model(resolve_path(args.checkpoint), config, device)
    fit_channel_mode = str(config.get("fit_channel_mode", "mean"))
    image_norm = config.get("image_normalization", {})
    bargs = baseline_args()

    all_rows: list[dict[str, object]] = []
    for index, sample in enumerate(selected, start=1):
        sample_id = str(sample.get("sample_id") or Path(sample["image_out"]).stem)
        image_path = resolve_path(sample.get("image_out") or sample.get("single_fits") or "")
        mask_path = resolve_path(sample.get("mask_out") or "")

        raw = _read_input_image(image_path, fit_channel_mode)
        image = _normalize_image(raw, image_norm)
        target_heatmap = _read_mask_image(mask_path)
        if args.crop_size > 0:
            raw, (y0, x0) = center_crop(raw, args.crop_size)
            image, _ = center_crop(image, args.crop_size)
            target_heatmap, _ = center_crop(target_heatmap, args.crop_size)
            sample_id = f"{sample_id}_crop_y{y0}_x{x0}"
        target_yx = heatmap_to_centroids(target_heatmap, threshold=args.target_threshold, min_distance=args.min_distance)

        dao_yx = generate_mask(raw, "daofind_like", bargs).centroids_yx
        dao_metrics = detection_metrics(dao_yx, target_yx, radius_px=args.match_radius_px)
        all_rows.append({"sample_id": sample_id, "method": "daofind_like", **dao_metrics})

        if args.crop_size > 0:
            heatmap = direct_predict_crop(model, image, device)
        else:
            heatmap = predict_heatmap(
                model,
                image,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                fit_channel_mode=fit_channel_mode,
                device=device,
                batch_size=args.batch_size,
            )
        unet_predictions = {}
        for threshold in thresholds:
            pred_yx = heatmap_to_centroids(heatmap, threshold=threshold, min_distance=args.min_distance)
            metrics = detection_metrics(pred_yx, target_yx, radius_px=args.match_radius_px)
            method = f"unet_t{threshold:.2f}"
            all_rows.append({"sample_id": sample_id, "method": method, "threshold": threshold, **metrics})
            unet_predictions[method] = (pred_yx, metrics)

        best_method, (best_yx, best_metrics) = max(unet_predictions.items(), key=lambda item: item[1][1]["f1"])
        panels = [
            draw_points(raw, f"{sample_id} target={len(target_yx)}", target_yx),
            draw_points(raw, f"DAOFIND pred={len(dao_yx)} f1={dao_metrics['f1']:.3f}", target_yx, dao_yx),
            draw_points(raw, f"{best_method} pred={len(best_yx)} f1={best_metrics['f1']:.3f}", target_yx, best_yx),
        ]
        w = sum(panel.width for panel in panels)
        h = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        canvas.save(out_dir / f"{sample_id}_daofind_vs_unet.jpg", quality=92)
        np.save(out_dir / f"{sample_id}_unet_heatmap.npy", heatmap.astype(np.float16))
        print(f"[{index}/{len(selected)}] {sample_id}: dao_f1={dao_metrics['f1']:.3f}, best_unet={best_method} f1={best_metrics['f1']:.3f}")

    fieldnames = sorted({key for row in all_rows for key in row.keys()})
    with (out_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    (out_dir / "per_sample_metrics.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
