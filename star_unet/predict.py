from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from preprocessing.mask_generator import FITS_SUFFIXES
from .dataset import BITMAP_SUFFIXES, _normalize_image, _read_input_image
from .model import UNet
from .postprocess import heatmap_to_centroids, heatmap_to_uint8, normalize_for_display, write_prediction_csv
from .train import load_config, parse_features, resolve_device


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
IMAGE_SUFFIXES = FITS_SUFFIXES | BITMAP_SUFFIXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run U-Net star heatmap inference on FITS/images.")
    parser.add_argument("input", type=Path, help="Input FITS/image file or directory.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/star_unet/best.pt"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=Path("runs/star_unet/predictions"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--max-peaks", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of tiles to run per model forward pass.")
    parser.add_argument("--fit-channel-mode", default=None, choices=("mean", "first", "max", "luma"))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def input_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    return sorted(file for file in path.glob(pattern) if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES)


def load_model(checkpoint_path: Path, config: dict[str, Any], device: torch.device) -> UNet:
    features = parse_features(config.get("features", [32, 64]))
    model = UNet(in_channels=1, out_channels=1, features=features).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_heatmap(
    model: UNet,
    image: np.ndarray,
    tile_size: int,
    tile_overlap: int,
    fit_channel_mode: str,
    device: torch.device,
    batch_size: int = 1,
) -> np.ndarray:
    del fit_channel_mode
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    if tile_size <= 0 or max(h, w) <= tile_size:
        tensor = torch.from_numpy(image[None, None, :, :].astype(np.float32)).to(device)
        logits = model(tensor)
        return torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

    stride = max(1, int(tile_size) - max(0, int(tile_overlap)))
    y_starts = list(range(0, max(h - tile_size, 0) + 1, stride))
    x_starts = list(range(0, max(w - tile_size, 0) + 1, stride))
    if not y_starts or y_starts[-1] != max(h - tile_size, 0):
        y_starts.append(max(h - tile_size, 0))
    if not x_starts or x_starts[-1] != max(w - tile_size, 0):
        x_starts.append(max(w - tile_size, 0))

    acc = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    tiles: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(h, y0 + tile_size)
            x1 = min(w, x0 + tile_size)
            tile = image[y0:y1, x0:x1]
            tiles.append(tile.astype(np.float32))
            boxes.append((y0, y1, x0, x1))

    batch_size = max(1, int(batch_size))
    for start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[start : start + batch_size]
        batch_boxes = boxes[start : start + batch_size]
        tensor = torch.from_numpy(np.stack(batch_tiles, axis=0)[:, None, :, :]).to(device)
        logits = model(tensor)
        preds = torch.sigmoid(logits)[:, 0].detach().cpu().numpy().astype(np.float32)
        for pred, (y0, y1, x0, x1) in zip(preds, batch_boxes):
            acc[y0:y1, x0:x1] += pred
            weight[y0:y1, x0:x1] += 1.0
    return acc / np.maximum(weight, 1.0)


@torch.no_grad()
def predict_one(
    model: UNet,
    image_path: Path,
    config: dict[str, Any],
    fit_channel_mode: str,
    device: torch.device,
    tile_size: int = 1024,
    tile_overlap: int = 128,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    raw = _read_input_image(image_path, fit_channel_mode)
    image = _normalize_image(raw, config.get("image_normalization", {}))
    heatmap = predict_heatmap(model, image, tile_size, tile_overlap, fit_channel_mode, device, batch_size=batch_size)
    return raw, heatmap


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config.get("device", "auto"))
    fit_channel_mode = args.fit_channel_mode or config.get("fit_channel_mode", "mean")
    model = load_model(args.checkpoint, config, device)
    files = input_files(args.input, args.recursive)
    if not files:
        raise FileNotFoundError(f"no input images found: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in files:
        raw, heatmap = predict_one(
            model,
            image_path,
            config,
            fit_channel_mode,
            device,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            batch_size=args.batch_size,
        )
        centroids = heatmap_to_centroids(
            heatmap,
            threshold=args.threshold,
            min_distance=args.min_distance,
            max_peaks=args.max_peaks,
        )
        stem = image_path.stem
        heatmap_path = args.output / f"{stem}_heatmap.png"
        csv_path = args.output / f"{stem}_centroids.csv"
        display_path = args.output / f"{stem}_input_preview.png"
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(heatmap_to_uint8(heatmap)).save(heatmap_path)
        Image.fromarray(normalize_for_display(raw)).save(display_path)
        write_prediction_csv(csv_path, centroids)
        row = {
            "input": str(image_path),
            "heatmap": str(heatmap_path),
            "centroids": str(csv_path),
            "preview": str(display_path),
            "threshold": float(args.threshold),
            "count": int(len(centroids)),
        }
        rows.append(row)
        print(f"[predict] {image_path.name}: {len(centroids)} peaks -> {heatmap_path.name}")

    (args.output / "predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
