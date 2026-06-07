from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree


def heatmap_to_centroids(
    heatmap: np.ndarray,
    threshold: float = 0.3,
    min_distance: float = 3.0,
    max_peaks: int | None = None,
    exclude_border: int = 0,
    refine_radius: int = 2,
) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got shape {heatmap.shape}")
    if heatmap.size == 0:
        return np.empty((0, 2), dtype=np.float32)

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
    if not len(candidates):
        return np.empty((0, 2), dtype=np.float32)

    scores = heatmap[candidates[:, 0], candidates[:, 1]]
    candidates = candidates[np.argsort(-scores)]

    selected: list[tuple[float, float]] = []
    min_sep2 = float(min_distance) ** 2
    h, w = heatmap.shape
    for y, x in candidates:
        if max_peaks is not None and len(selected) >= max_peaks:
            break
        cy0 = float(y)
        cx0 = float(x)
        if any((cy0 - cy) ** 2 + (cx0 - cx) ** 2 < min_sep2 for cy, cx in selected):
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
            cy = float(np.sum(yy * weights) / total)
            cx = float(np.sum(xx * weights) / total)
            selected.append((cy + 0.5, cx + 0.5))
        else:
            selected.append((cy0 + 0.5, cx0 + 0.5))

    return np.asarray(selected, dtype=np.float32).reshape((-1, 2))


def match_centroids(
    pred_yx: np.ndarray,
    target_yx: np.ndarray,
    radius_px: float = 4.0,
) -> list[tuple[int, int, float]]:
    pred_yx = np.asarray(pred_yx, dtype=np.float32).reshape((-1, 2))
    target_yx = np.asarray(target_yx, dtype=np.float32).reshape((-1, 2))
    if len(pred_yx) == 0 or len(target_yx) == 0:
        return []

    pred_xy = np.column_stack((pred_yx[:, 1], pred_yx[:, 0]))
    target_xy = np.column_stack((target_yx[:, 1], target_yx[:, 0]))
    tree = cKDTree(target_xy)

    candidates: list[tuple[float, int, int]] = []
    for pred_index, xy in enumerate(pred_xy):
        nearby = tree.query_ball_point(xy, r=float(radius_px))
        for target_index in nearby:
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


def detection_metrics(pred_yx: np.ndarray, target_yx: np.ndarray, radius_px: float = 4.0) -> dict[str, Any]:
    matches = match_centroids(pred_yx, target_yx, radius_px=radius_px)
    pred_count = int(len(pred_yx))
    target_count = int(len(target_yx))
    matched_count = int(len(matches))
    false_count = max(0, pred_count - matched_count)
    missed_count = max(0, target_count - matched_count)
    precision = matched_count / pred_count if pred_count else 0.0
    recall = matched_count / target_count if target_count else 0.0
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    distances = np.asarray([match[2] for match in matches], dtype=np.float32)
    return {
        "pred_count": pred_count,
        "target_count": target_count,
        "matched_count": matched_count,
        "false_count": false_count,
        "missed_count": missed_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_rate": false_count / pred_count if pred_count else 0.0,
        "match_distance_px_mean": float(np.mean(distances)) if len(distances) else None,
        "match_distance_px_p95": float(np.percentile(distances, 95)) if len(distances) else None,
    }


def heatmap_to_uint8(heatmap: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def normalize_for_display(image: np.ndarray, lower_percentile: float = 0.5, upper_percentile: float = 99.8) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [lower_percentile, upper_percentile])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    display = (np.nan_to_num(image, nan=float(low)) - float(low)) / (float(high) - float(low))
    return heatmap_to_uint8(display)


def write_prediction_csv(path: Path, centroids_yx: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["index,y,x"]
    for index, (y, x) in enumerate(np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2))):
        rows.append(f"{index},{float(y):.4f},{float(x):.4f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_triplet(
    path: Path,
    image_display: np.ndarray,
    target_heatmap: np.ndarray,
    pred_heatmap: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.asarray(image_display, dtype=np.uint8)
    target_u8 = heatmap_to_uint8(target_heatmap)
    pred_u8 = heatmap_to_uint8(pred_heatmap)
    target_color = cv2.applyColorMap(target_u8, cv2.COLORMAP_MAGMA)
    pred_color = cv2.applyColorMap(pred_u8, cv2.COLORMAP_VIRIDIS)
    image_color = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    triplet = np.concatenate([image_color, target_color, pred_color], axis=1)
    ok, encoded = cv2.imencode(".jpg", triplet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise OSError(f"failed to encode triplet preview: {path}")
    encoded.tofile(str(path))
