from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "tool"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from daofind_opt import daofind_like_detect, read_fits_image, resolve_manifest_path, subtract_local_background  # noqa: E402


def progress(index: int, total: int, text: str) -> None:
    width = 24
    done = int(round(width * index / max(total, 1)))
    bar = "#" * done + "-" * (width - done)
    print(f"[{bar}] {index}/{total} ({100.0 * index / max(total, 1):5.1f}%) {text}", flush=True)


def robust_sigma(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 1e-6:
        sigma = float(np.std(values))
    if not np.isfinite(sigma) or sigma <= 1e-6:
        sigma = 1.0
    return sigma


def make_global_radius_map(image: np.ndarray, boundary_sigma: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    finite = image[np.isfinite(image)]
    center = float(np.median(finite)) if finite.size else 0.0
    noise = robust_sigma(finite)
    threshold = center + float(boundary_sigma) * noise
    above = np.isfinite(image) & (image >= threshold)
    labels, count = ndi.label(above, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel(), minlength=count + 1).astype(np.int64)
    return labels, areas, center, threshold


def estimate_detected_radius_from_global_components(labels: np.ndarray, areas: np.ndarray, x: float, y: float) -> float:
    h, w = labels.shape
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return 2.0
    label = int(labels[yi, xi])
    if label <= 0:
        y0 = max(0, yi - 1)
        y1 = min(h, yi + 2)
        x0 = max(0, xi - 1)
        x1 = min(w, xi + 2)
        local_labels = labels[y0:y1, x0:x1]
        local_labels = local_labels[local_labels > 0]
        if local_labels.size == 0:
            return 2.0
        label = int(np.bincount(local_labels.ravel()).argmax())
    if label >= len(areas):
        return 2.0
    area = int(areas[label])
    if area <= 0:
        return 2.0
    return float(max(np.sqrt(float(area) / np.pi), 2.0))


def paint_gaussian(
    target: np.ndarray,
    weight: np.ndarray,
    x: float,
    y: float,
    radius: float,
    amp: float,
    w_amp: float,
) -> None:
    h, ww = target.shape
    radius = float(max(radius, 2.0))
    sigma = max(radius / 1.7, 0.75)
    paint_radius = int(np.ceil(radius))
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    y0 = max(0, yi - paint_radius)
    y1 = min(h, yi + paint_radius + 1)
    x0 = max(0, xi - paint_radius)
    x1 = min(ww, xi + paint_radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    g = np.exp(-((xx - float(x)) ** 2 + (yy - float(y)) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], float(amp) * g)
    weight[y0:y1, x0:x1] = np.maximum(weight[y0:y1, x0:x1], float(w_amp) * g)


def erase_ignore(ignore: np.ndarray, x: float, y: float, sigma: float) -> None:
    h, ww = ignore.shape
    radius = int(np.ceil(max(4.0 * sigma, 5.0)))
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    y0 = max(0, yi - radius)
    y1 = min(h, yi + radius + 1)
    x0 = max(0, xi - radius)
    x1 = min(ww, xi + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - float(x)) ** 2 + (yy - float(y)) ** 2 <= float(radius * radius)
    ignore[y0:y1, x0:x1][mask] = 0.0


def build_sample(row: pd.Series, args: argparse.Namespace, out: Path) -> dict[str, object]:
    data_model = (ROOT / args.data_model).resolve()
    image, _ = read_fits_image(resolve_manifest_path(data_model, row["single_fits"]))
    sample_id = str(row["sample_id"])
    residual = subtract_local_background(image, "local_mean", 25).astype(np.float32)
    h, w = image.shape
    target = np.zeros((h, w), dtype=np.float32)
    weight = np.full((h, w), float(args.background_weight), dtype=np.float32)
    valid = np.ones((h, w), dtype=np.float32)

    gaia = pd.read_csv((ROOT / args.gaia_dir / f"{sample_id}_gaia_true_stars_g20_pixel.csv").resolve())
    mag = pd.to_numeric(gaia["g_mag"], errors="coerce")
    gaia = gaia[np.isfinite(mag)].copy()
    mag = pd.to_numeric(gaia["g_mag"], errors="coerce").to_numpy()
    xy = gaia[["x", "y"]].to_numpy(dtype=np.float32)
    in_frame = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    gaia = gaia[in_frame].copy()
    mag = mag[in_frame]
    xy = xy[in_frame]

    detections = daofind_like_detect(
        image,
        sigma=float(args.baseline_sigma),
        fwhm=float(args.baseline_fwhm),
        background_mode="local_mean",
        filtsize=25,
        max_peaks=20000,
        exclude_border=8,
    )
    det_xy = np.asarray([[det.x, det.y] for det in detections], dtype=np.float32)
    det_tree = cKDTree(det_xy) if len(det_xy) else None
    gaia_tree = cKDTree(xy) if len(xy) else None
    radius_labels, radius_areas, radius_center, radius_threshold = make_global_radius_map(
        image,
        boundary_sigma=float(args.radius_boundary_sigma),
    )

    detected = np.zeros(len(gaia), dtype=bool)
    if det_tree is not None and len(gaia):
        dist, _ = det_tree.query(xy, distance_upper_bound=float(args.detect_match_radius))
        detected = np.isfinite(dist)

    stats = {
        "sample_id": sample_id,
        "gaia_le13": int(np.sum(mag <= 13.0)),
        "gaia_13_14": int(np.sum((mag > 13.0) & (mag <= 14.0))),
        "gaia_14_15": int(np.sum((mag > 14.0) & (mag <= 15.0))),
        "baseline_detections": int(len(detections)),
    }

    for idx, star in gaia.iterrows():
        x = float(star["x"])
        y = float(star["y"])
        gmag = float(star["g_mag"])
        local_index = gaia.index.get_loc(idx)
        is_detected = bool(detected[local_index])
        if is_detected and det_tree is not None:
            _, det_index = det_tree.query([[x, y]], k=1)
            det = detections[int(det_index[0])]
            radius = estimate_detected_radius_from_global_components(radius_labels, radius_areas, det.x, det.y)
        else:
            radius = float(args.undetected_radius)
        if gmag <= 13.0:
            paint_gaussian(target, weight, x, y, radius, 1.0, args.le13_detected_weight if is_detected else args.le13_missed_weight)
        elif gmag <= 14.0:
            paint_gaussian(target, weight, x, y, radius, 1.0, args.g13_14_detected_weight if is_detected else args.g13_14_missed_weight)
        elif gmag <= 15.0:
            erase_ignore(valid, x, y, max(radius / 2.0, 1.0))

    if gaia_tree is not None and len(det_xy):
        dist, _ = gaia_tree.query(det_xy, distance_upper_bound=float(args.ignore_non_gaia_match_radius))
        non_gaia = ~np.isfinite(dist)
        bright_det = np.asarray([det.snr >= float(args.ignore_non_gaia_snr) for det in detections], dtype=bool)
        ignore_xy = det_xy[non_gaia & bright_det]
        for x, y in ignore_xy:
            erase_ignore(valid, float(x), float(y), 2.0)
        stats["ignored_non_gaia_peaks"] = int(len(ignore_xy))
    else:
        stats["ignored_non_gaia_peaks"] = 0

    weight *= valid
    image_center = float(np.median(image[np.isfinite(image)]))
    image_noise = float(1.4826 * np.median(np.abs(image[np.isfinite(image)] - image_center)))
    image_norm = ((image.astype(np.float32) - image_center) / max(image_noise, 1e-6)).astype(np.float32)
    image_norm = np.clip(image_norm, -8.0, 20.0)

    np.savez_compressed(
        out / f"{sample_id}_softmask.npz",
        image=image_norm.astype(np.float32),
        target=target.astype(np.float32),
        weight=weight.astype(np.float32),
        valid=valid.astype(np.float32),
    )
    stats["target_pixels"] = int(np.count_nonzero(target > 0.05))
    stats["valid_pixels"] = int(np.count_nonzero(valid > 0))
    stats["radius_center"] = float(radius_center)
    stats["radius_threshold"] = float(radius_threshold)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Gaia soft-mask targets for the 08 U-Net experiment.")
    parser.add_argument("--data-model", type=Path, default=Path("data/data_model"))
    parser.add_argument("--selected-samples", type=Path, default=Path("data/data_20/selected_samples.csv"))
    parser.add_argument("--gaia-dir", type=Path, default=Path("data/data_gaia/gaia_annotations_right_fixed"))
    parser.add_argument("--output", type=Path, default=Path("data/data_20/unet_softmask"))
    parser.add_argument("--baseline-sigma", type=float, default=5.0)
    parser.add_argument("--baseline-fwhm", type=float, default=3.0)
    parser.add_argument("--detect-match-radius", type=float, default=4.0)
    parser.add_argument("--ignore-non-gaia-match-radius", type=float, default=5.0)
    parser.add_argument("--ignore-non-gaia-snr", type=float, default=8.0)
    parser.add_argument("--psf-radius", type=int, default=25)
    parser.add_argument("--undetected-radius", type=float, default=2.0)
    parser.add_argument("--radius-boundary-sigma", type=float, default=1.5)
    parser.add_argument("--background-weight", type=float, default=2.0)
    parser.add_argument("--le13-detected-weight", type=float, default=4.0)
    parser.add_argument("--le13-missed-weight", type=float, default=10.0)
    parser.add_argument("--g13-14-detected-weight", type=float, default=9.0)
    parser.add_argument("--g13-14-missed-weight", type=float, default=1.2)
    args = parser.parse_args()

    selected = pd.read_csv((ROOT / args.selected_samples).resolve())
    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for i, row in selected.iterrows():
        progress(i + 1, len(selected), str(row["sample_id"]))
        rows.append(build_sample(row, args, out))
    pd.DataFrame(rows).to_csv(out / "prepare_summary.csv", index=False)
    config = vars(args).copy()
    config["output"] = str(out)
    (out / "prepare_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
