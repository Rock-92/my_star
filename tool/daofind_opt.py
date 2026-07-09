from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi
from astropy.io import fits
from PIL import Image
from scipy.spatial import cKDTree


@dataclass
class Detection:
    y: float
    x: float
    peak: float
    flux: float
    snr: float


def odd_at_least(value: int, minimum: int = 3) -> int:
    value = max(int(value), int(minimum))
    return value if value % 2 == 1 else value + 1


def robust_center_noise(image: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(image[np.isfinite(image)], dtype=np.float32)
    if finite.size == 0:
        return 0.0, 1.0
    center = float(np.median(finite))
    noise = float(1.4826 * np.median(np.abs(finite - center)))
    if not np.isfinite(noise) or noise <= 0:
        q25, q75 = np.percentile(finite, [25.0, 75.0])
        noise = float((q75 - q25) / 1.349)
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.std(finite))
    if not np.isfinite(noise) or noise <= 0:
        noise = 1.0
    return center, noise


def subtract_local_background(image: np.ndarray, mode: str = "local_mean", size: int = 25) -> np.ndarray:
    mode = str(mode).lower()
    size = odd_at_least(size)
    image = np.asarray(image, dtype=np.float32)
    if mode in {"local_mean", "mean"}:
        return image - ndi.uniform_filter(image, size=size, output=np.float32)
    if mode in {"local_median", "median"}:
        return image - ndi.median_filter(image, size=size)
    if mode == "global_mean":
        return image - float(np.mean(image))
    if mode == "global_median":
        return image - float(np.median(image))
    if mode in {"none", "off"}:
        return image.copy()
    raise ValueError(f"unsupported background mode: {mode}")


def read_fits_image(path: Path, channel_mode: str = "mean") -> tuple[np.ndarray, fits.Header]:
    data, header = fits.getdata(path, header=True)
    image = np.asarray(data, dtype=np.float32)
    if image.ndim == 2:
        return image, header
    if image.ndim == 3:
        if channel_mode == "first":
            return image[0].astype(np.float32), header
        if channel_mode == "max":
            return image.max(axis=0).astype(np.float32), header
        if channel_mode == "luma" and image.shape[0] == 3:
            weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
            return np.tensordot(weights, image, axes=(0, 0)).astype(np.float32), header
        if channel_mode in {"mean", "luma"}:
            return image.mean(axis=0, dtype=np.float32), header
    raise ValueError(f"cannot convert FITS shape {image.shape} to grayscale: {path}")


def resolve_manifest_path(data_model: Path, path_text: str | Path) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    if path.is_absolute():
        return path
    single_star_root = data_model.resolve().parents[1]
    candidate = single_star_root / path
    if candidate.exists():
        return candidate
    return data_model / path


def read_mask_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def heatmap_to_centroids(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    min_distance: float = 3.0,
    exclude_border: int = 0,
    refine_radius: int = 2,
) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    window = max(3, int(round(float(min_distance) * 2.0 + 1.0)))
    if window % 2 == 0:
        window += 1
    local_max = heatmap == ndi.maximum_filter(heatmap, size=window)
    candidates = np.argwhere(local_max & (heatmap >= float(threshold)))
    if exclude_border > 0 and len(candidates):
        h, w = heatmap.shape
        y = candidates[:, 0]
        x = candidates[:, 1]
        keep = (y >= exclude_border) & (y < h - exclude_border) & (x >= exclude_border) & (x < w - exclude_border)
        candidates = candidates[keep]
    if len(candidates) == 0:
        return np.empty((0, 2), dtype=np.float32)
    scores = heatmap[candidates[:, 0], candidates[:, 1]]
    candidates = candidates[np.argsort(-scores)]
    selected: list[tuple[float, float]] = []
    min_sep2 = float(min_distance) ** 2
    h, w = heatmap.shape
    for y, x in candidates:
        if any((float(y) - cy) ** 2 + (float(x) - cx) ** 2 < min_sep2 for cy, cx in selected):
            continue
        y0 = max(0, int(y) - refine_radius)
        y1 = min(h, int(y) + refine_radius + 1)
        x0 = max(0, int(x) - refine_radius)
        x1 = min(w, int(x) + refine_radius + 1)
        patch = heatmap[y0:y1, x0:x1].astype(np.float64)
        weights = np.clip(patch - float(threshold), 0.0, None)
        total = float(np.sum(weights))
        if total > 0.0:
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
            cy = float(np.sum(yy * weights) / total + 0.5)
            cx = float(np.sum(xx * weights) / total + 0.5)
            selected.append((cy, cx))
        else:
            selected.append((float(y) + 0.5, float(x) + 0.5))
    return np.asarray(selected, dtype=np.float32).reshape((-1, 2))


def transform_stack_yx_to_single_yx(stack_yx: np.ndarray, row: dict[str, str]) -> np.ndarray:
    centroids = np.asarray(stack_yx, dtype=np.float32).reshape((-1, 2))
    if not len(centroids):
        return centroids
    matrix = np.asarray(
        [
            [float(row.get("label_transform_a", 1.0) or 1.0), float(row.get("label_transform_b", 0.0) or 0.0)],
            [float(row.get("label_transform_c", 0.0) or 0.0), float(row.get("label_transform_d", 1.0) or 1.0)],
        ],
        dtype=np.float32,
    )
    shift_xy = np.asarray(
        [
            float(row.get("label_shift_x_px", 0.0) or 0.0),
            float(row.get("label_shift_y_px", 0.0) or 0.0),
        ],
        dtype=np.float32,
    )
    xy = centroids[:, [1, 0]]
    transformed_xy = xy @ matrix.T + shift_xy
    return transformed_xy[:, [1, 0]].astype(np.float32)


def daofind_like_detect_from_response(
    response: np.ndarray,
    residual: np.ndarray,
    sigma: float = 5.0,
    fwhm: float = 3.0,
    max_peaks: int | None = 5000,
    min_separation: float | None = None,
    exclude_border: int = 8,
) -> list[Detection]:
    response = np.asarray(response, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    center, noise = robust_center_noise(response)
    threshold = center + float(sigma) * noise
    peak_window = odd_at_least(int(round(float(fwhm) * 2.0 + 1.0)))
    local_max = response == ndi.maximum_filter(response, size=peak_window)
    candidates = np.argwhere(local_max & (response > threshold))
    if exclude_border > 0 and len(candidates):
        h, w = response.shape
        y = candidates[:, 0]
        x = candidates[:, 1]
        keep = (y >= exclude_border) & (y < h - exclude_border) & (x >= exclude_border) & (x < w - exclude_border)
        candidates = candidates[keep]
    if len(candidates):
        scores = response[candidates[:, 0], candidates[:, 1]]
        candidates = candidates[np.argsort(-scores)]
    min_sep = max(float(fwhm), 1.0) if min_separation is None else float(min_separation)
    min_sep2 = min_sep * min_sep
    patch_radius = int(np.ceil(max(float(fwhm) * 1.5, 2.0)))
    detections: list[Detection] = []
    for y, x in candidates:
        if max_peaks is not None and len(detections) >= max_peaks:
            break
        if any((float(y) + 0.5 - det.y) ** 2 + (float(x) + 0.5 - det.x) ** 2 < min_sep2 for det in detections):
            continue
        y0 = max(0, int(y) - patch_radius)
        y1 = min(residual.shape[0], int(y) + patch_radius + 1)
        x0 = max(0, int(x) - patch_radius)
        x1 = min(residual.shape[1], int(x) + patch_radius + 1)
        patch = residual[y0:y1, x0:x1].astype(np.float64)
        weights = np.clip(patch, 0.0, None)
        total = float(np.sum(weights))
        if total > 0.0:
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
            cy = float(np.sum(yy * weights) / total + 0.5)
            cx = float(np.sum(xx * weights) / total + 0.5)
        else:
            cy = float(y) + 0.5
            cx = float(x) + 0.5
        peak = float(response[y, x])
        detections.append(Detection(y=cy, x=cx, peak=peak, flux=total, snr=(peak - center) / noise))
    return detections


def fixed_gaussian_response(residual: np.ndarray, fwhm: float = 3.0) -> np.ndarray:
    gaussian_sigma = max(float(fwhm) / 2.3548, 0.5)
    return ndi.gaussian_filter(np.asarray(residual, dtype=np.float32), sigma=gaussian_sigma)


def daofind_like_detect(
    image: np.ndarray,
    sigma: float = 5.0,
    fwhm: float = 3.0,
    background_mode: str = "local_mean",
    filtsize: int = 25,
    max_peaks: int | None = 5000,
    min_separation: float | None = None,
    exclude_border: int = 8,
) -> list[Detection]:
    residual = subtract_local_background(image, background_mode, filtsize)
    response = fixed_gaussian_response(residual, fwhm)
    return daofind_like_detect_from_response(response, residual, sigma, fwhm, max_peaks, min_separation, exclude_border)


def match_centroids(pred_yx: np.ndarray, target_yx: np.ndarray, radius_px: float) -> list[tuple[int, int, float]]:
    pred_yx = np.asarray(pred_yx, dtype=np.float32).reshape((-1, 2))
    target_yx = np.asarray(target_yx, dtype=np.float32).reshape((-1, 2))
    if len(pred_yx) == 0 or len(target_yx) == 0:
        return []
    pred_xy = np.column_stack((pred_yx[:, 1], pred_yx[:, 0]))
    target_xy = np.column_stack((target_yx[:, 1], target_yx[:, 0]))
    tree = cKDTree(target_xy)
    candidates: list[tuple[float, int, int]] = []
    for pred_index, xy in enumerate(pred_xy):
        for target_index in tree.query_ball_point(xy, r=float(radius_px)):
            distance = float(np.linalg.norm(xy - target_xy[target_index]))
            candidates.append((distance, pred_index, target_index))
    candidates.sort(key=lambda item: item[0])
    used_pred: set[int] = set()
    used_target: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for distance, pred_index, target_index in candidates:
        if pred_index in used_pred or target_index in used_target:
            continue
        used_pred.add(pred_index)
        used_target.add(target_index)
        matches.append((pred_index, target_index, distance))
    return matches


def detection_metrics(pred_yx: np.ndarray, target_yx: np.ndarray, radius_px: float) -> dict[str, Any]:
    matches = match_centroids(pred_yx, target_yx, radius_px)
    pred_count = int(len(pred_yx))
    target_count = int(len(target_yx))
    matched_count = int(len(matches))
    false_count = max(0, pred_count - matched_count)
    missed_count = max(0, target_count - matched_count)
    precision = matched_count / pred_count if pred_count else 0.0
    recall = matched_count / target_count if target_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pred_count": pred_count,
        "target_count": target_count,
        "matched_count": matched_count,
        "false_count": false_count,
        "missed_count": missed_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def photometry_at_centroids(
    image: np.ndarray,
    centroids_yx: np.ndarray,
    radius: int = 4,
    background_mode: str = "local_mean",
    background_size: int = 51,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    residual = subtract_local_background(image, background_mode, background_size)
    fluxes = np.zeros(len(centroids_yx), dtype=np.float32)
    peaks = np.zeros(len(centroids_yx), dtype=np.float32)
    h, w = image.shape
    for idx, (cy, cx) in enumerate(np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2))):
        y = int(round(float(cy) - 0.5))
        x = int(round(float(cx) - 0.5))
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        keep = (yy - cy) ** 2 + (xx - cx) ** 2 <= float(radius) ** 2
        patch = np.clip(residual[y0:y1, x0:x1], 0.0, None)
        if np.any(keep):
            fluxes[idx] = float(np.sum(patch[keep]))
            peaks[idx] = float(np.max(patch[keep]))
    return fluxes, peaks


def normalize_log_flux_by_group(fluxes: np.ndarray, low_pct: float = 10.0, high_pct: float = 95.0) -> np.ndarray:
    flux = np.asarray(fluxes, dtype=np.float32)
    valid = np.isfinite(flux) & (flux > 0.0)
    out = np.zeros_like(flux, dtype=np.float32)
    if not np.any(valid):
        return out
    log_flux = np.log10(np.maximum(flux[valid], 1e-6))
    lo, hi = np.percentile(log_flux, [low_pct, high_pct])
    if not np.isfinite(hi - lo) or hi <= lo:
        out[valid] = 1.0
    else:
        out[valid] = np.clip((log_flux - lo) / (hi - lo), 0.0, 1.0)
    return out


def component_sigmas_from_heatmap(
    heatmap: np.ndarray,
    centroids_yx: np.ndarray,
    threshold: float = 0.05,
    edge_value: float = 0.05,
    min_sigma: float = 0.8,
    max_sigma: float = 3.0,
) -> np.ndarray:
    mask = np.asarray(heatmap, dtype=np.float32) > float(threshold)
    labels, _ = ndi.label(mask)
    centroids = np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2))
    if len(centroids) == 0:
        return np.empty((0,), dtype=np.float32)
    areas = np.bincount(labels.ravel())
    h, w = labels.shape
    edge_value = min(max(float(edge_value), 1e-4), 0.95)
    sigma_factor = math.sqrt(-2.0 * math.log(edge_value))
    sigmas = np.empty(len(centroids), dtype=np.float32)
    fallback_area = float(np.count_nonzero(mask)) / max(len(centroids), 1)
    for idx, (cy, cx) in enumerate(centroids):
        y = int(np.clip(round(float(cy) - 0.5), 0, h - 1))
        x = int(np.clip(round(float(cx) - 0.5), 0, w - 1))
        label = int(labels[y, x])
        if label == 0:
            y0 = max(0, y - 2)
            y1 = min(h, y + 3)
            x0 = max(0, x - 2)
            x1 = min(w, x + 3)
            nearby = labels[y0:y1, x0:x1]
            nearby = nearby[nearby > 0]
            label = int(nearby[0]) if len(nearby) else 0
        area = float(areas[label]) if label > 0 else fallback_area
        equivalent_radius = math.sqrt(max(area, 1.0) / math.pi)
        sigma = equivalent_radius / max(sigma_factor, 1e-6)
        sigmas[idx] = float(np.clip(sigma, float(min_sigma), float(max_sigma)))
    return sigmas


def gaussian_patch(shape: tuple[int, int], cy: float, cx: float, sigma: float) -> tuple[slice, slice, np.ndarray]:
    radius = max(1, int(math.ceil(float(sigma) * 3.0)))
    h, w = shape
    y0 = max(0, int(math.floor(cy - radius)))
    y1 = min(h, int(math.ceil(cy + radius)) + 1)
    x0 = max(0, int(math.floor(cx - radius)))
    x1 = min(w, int(math.ceil(cx + radius)) + 1)
    if y0 >= y1 or x0 >= x1:
        return slice(0, 0), slice(0, 0), np.zeros((0, 0), dtype=np.float32)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    values = np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / max(float(sigma) ** 2, 1e-6)).astype(np.float32)
    return slice(y0, y1), slice(x0, x1), values
