from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from astropy.io import fits
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / "tool"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from daofind_opt import (  # noqa: E402
    Detection as DaoDetection,
    daofind_like_detect,
    fixed_gaussian_response,
    match_centroids,
    read_fits_image,
    robust_center_noise,
    subtract_local_background,
)


@dataclass
class DetectedSource:
    y: float
    x: float
    peak: float
    flux: float
    snr: float
    area: float = 0.0
    method: str = ""


def resolve_path(data_model: Path, path_text: str | Path) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    if path.is_absolute():
        return path
    single_star_root = data_model.resolve().parents[1]
    candidate = single_star_root / path
    if candidate.exists():
        return candidate
    return data_model / path


def load_manifest(data_model: Path) -> pd.DataFrame:
    return pd.read_csv(data_model / "manifest.csv")


def selected_samples(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def robust_uint16(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint16)
    lo = float(np.percentile(finite, 0.1))
    hi = float(np.percentile(finite, 99.98))
    if not np.isfinite(hi) or hi <= lo:
        hi = float(np.max(finite))
    shifted = np.clip(image - lo, 0.0, max(hi - lo, 1.0))
    return np.clip(np.rint(shifted), 0, 65535).astype(np.uint16)


def tile_starts(length: int, patch_size: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, patch_size))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def gaussian_add(mask: np.ndarray, cy: float, cx: float, sigma: float, peak: float = 1.0) -> None:
    sigma = float(np.clip(sigma, 0.5, 6.0))
    radius = max(2, int(math.ceil(3.0 * sigma)))
    h, w = mask.shape
    y0 = max(0, int(math.floor(cy - radius)))
    y1 = min(h, int(math.ceil(cy + radius)) + 1)
    x0 = max(0, int(math.floor(cx - radius)))
    x1 = min(w, int(math.ceil(cx + radius)) + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    values = np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / max(sigma * sigma, 1e-6)).astype(np.float32)
    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], float(peak) * values)


def estimate_stack_source(
    aligned_stack: np.ndarray,
    y: float,
    x: float,
    verify_radius: int = 6,
    min_snr: float = 4.0,
    min_sigma: float = 0.75,
    max_sigma: float = 4.0,
) -> tuple[bool, float, float, float, float, float]:
    residual = np.asarray(aligned_stack, dtype=np.float32)
    h, w = residual.shape
    iy = int(round(float(y)))
    ix = int(round(float(x)))
    y0 = max(0, iy - int(verify_radius))
    y1 = min(h, iy + int(verify_radius) + 1)
    x0 = max(0, ix - int(verify_radius))
    x1 = min(w, ix + int(verify_radius) + 1)
    if y0 >= y1 or x0 >= x1:
        return False, float(y), float(x), 1.4, 0.0, 0.0
    patch = residual[y0:y1, x0:x1].astype(np.float32)
    center, noise = robust_center_noise(patch)
    pos = np.clip(patch - float(center), 0.0, None)
    peak_index = np.unravel_index(int(np.argmax(pos)), pos.shape)
    peak = float(pos[peak_index])
    snr = peak / max(float(noise), 1e-6)
    if snr < float(min_snr):
        return False, float(y), float(x), 1.4, peak, snr
    weights = pos.copy()
    weights[weights < max(0.20 * peak, 1e-6)] = 0.0
    total = float(weights.sum())
    if total <= 0:
        cy = float(y0 + peak_index[0] + 0.5)
        cx = float(x0 + peak_index[1] + 0.5)
        return True, cy, cx, 1.4, peak, snr
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    cy = float(((yy + 0.5) * weights).sum() / total)
    cx = float(((xx + 0.5) * weights).sum() / total)
    var = float(((((yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2) * weights).sum()) / total)
    sigma = float(np.sqrt(max(var / 2.0, min_sigma * min_sigma)))
    sigma = float(np.clip(sigma, min_sigma, max_sigma))
    return True, cy, cx, sigma, peak, snr


def sep_like_sources(image: np.ndarray, sigma: float = 5.0, minarea: int = 3, deblend: bool = True, max_sources: int = 10000) -> list[DetectedSource]:
    import sep

    arr = np.asarray(image, dtype=np.float32)
    bkg = sep.Background(arr)
    residual = arr - bkg.back()
    thresh = float(sigma)
    objects = sep.extract(
        residual,
        thresh,
        err=bkg.globalrms,
        minarea=int(minarea),
        deblend_nthresh=32 if deblend else 1,
        deblend_cont=0.005 if deblend else 1.0,
    )
    sources: list[DetectedSource] = []
    for obj in objects[: int(max_sources)]:
        y = float(obj["y"])
        x = float(obj["x"])
        flux = float(obj["flux"])
        peak = float(obj["peak"])
        snr = peak / max(float(bkg.globalrms), 1e-6)
        sources.append(DetectedSource(y=y, x=x, peak=peak, flux=flux, snr=snr, area=float(obj["npix"]), method="sextractor_sep"))
    return sources


def dao_to_sources(detections: list[DaoDetection], method: str) -> list[DetectedSource]:
    return [
        DetectedSource(y=float(d.y), x=float(d.x), peak=float(d.peak), flux=float(d.flux), snr=float(d.snr), method=method)
        for d in detections
    ]


def connected_sources(prob: np.ndarray, image_for_centroid: np.ndarray | None, threshold: float = 0.4, min_area: int = 1) -> list[DetectedSource]:
    binary = np.asarray(prob >= float(threshold), dtype=np.uint8)
    labels, num = ndi.label(binary)
    objects = ndi.find_objects(labels)
    sources: list[DetectedSource] = []
    for label, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        local = labels[slc] == label
        ys_local, xs_local = np.where(local)
        if len(ys_local) == 0:
            continue
        ys = ys_local + slc[0].start
        xs = xs_local + slc[1].start
        if len(ys) < int(min_area):
            continue
        weights = prob[ys, xs].astype(np.float64)
        total = float(weights.sum())
        if total > 0:
            cy = float((ys * weights).sum() / total + 0.5)
            cx = float((xs * weights).sum() / total + 0.5)
        else:
            cy = float(ys.mean() + 0.5)
            cx = float(xs.mean() + 0.5)
        flux = float(total)
        if image_for_centroid is not None:
            arr = np.asarray(image_for_centroid, dtype=np.float32)
            h, w = arr.shape
            iy = int(round(cy - 0.5))
            ix = int(round(cx - 0.5))
            y0 = max(0, iy - 4)
            y1 = min(h, iy + 5)
            x0 = max(0, ix - 4)
            x1 = min(w, ix + 5)
            if y0 < y1 and x0 < x1:
                flux = float(np.clip(arr[y0:y1, x0:x1] - np.median(arr[y0:y1, x0:x1]), 0.0, None).sum())
        sources.append(
            DetectedSource(
                y=cy,
                x=cx,
                peak=float(prob[ys, xs].max()),
                flux=flux,
                snr=float(prob[ys, xs].max()),
                area=float(len(ys)),
                method="cdn_connected",
            )
        )
    return sources


def aperture_flux(image: np.ndarray, y: float, x: float, radius: int = 4) -> float:
    arr = np.asarray(image, dtype=np.float32)
    residual = subtract_local_background(arr, "local_mean", 51)
    h, w = arr.shape
    iy = int(round(float(y) - 0.5))
    ix = int(round(float(x) - 0.5))
    y0 = max(0, iy - radius)
    y1 = min(h, iy + radius + 1)
    x0 = max(0, ix - radius)
    x1 = min(w, ix + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    keep = (yy + y0 + 0.5 - y) ** 2 + (xx + x0 + 0.5 - x) ** 2 <= float(radius) ** 2
    patch = np.clip(residual[y0:y1, x0:x1], 0.0, None)
    return float(patch[keep].sum()) if np.any(keep) else 0.0


def sources_to_dataframe(sources: list[DetectedSource]) -> pd.DataFrame:
    return pd.DataFrame([source.__dict__ for source in sources], columns=["y", "x", "peak", "flux", "snr", "area", "method"])


def target_dataframe(path: Path, mag_limit: float | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "g_mag" in df.columns and mag_limit is not None:
        mag = pd.to_numeric(df["g_mag"], errors="coerce")
        df = df[(mag.isna()) | (mag <= float(mag_limit))].copy()
    return df


def radius_by_mag(mag: np.ndarray | pd.Series, bright_radius: float = 3.0, faint_radius: float = 1.5, faint_mag: float = 13.5) -> np.ndarray:
    m = pd.to_numeric(pd.Series(mag), errors="coerce").to_numpy(np.float32)
    valid = np.isfinite(m)
    radii = np.full(m.shape, float(bright_radius), dtype=np.float32)
    if np.any(valid):
        t = np.clip((m[valid] - 8.0) / max(float(faint_mag) - 8.0, 1e-6), 0.0, 1.0)
        radii[valid] = float(bright_radius) * (1.0 - t) + float(faint_radius) * t
    return radii


def greedy_match_variable_radius(pred: pd.DataFrame, target: pd.DataFrame, radius: np.ndarray | float) -> list[tuple[int, int, float]]:
    if pred.empty or target.empty:
        return []
    pred_yx = pred[["y", "x"]].to_numpy(np.float32)
    target_yx = target[["y", "x"]].to_numpy(np.float32)
    pred_xy = pred_yx[:, [1, 0]]
    target_xy = target_yx[:, [1, 0]]
    if np.isscalar(radius):
        radii = np.full(len(target), float(radius), dtype=np.float32)
    else:
        radii = np.asarray(radius, dtype=np.float32).reshape((-1,))
    tree = cKDTree(target_xy)
    candidates: list[tuple[float, int, int]] = []
    for pi, xy in enumerate(pred_xy):
        max_r = float(np.max(radii)) if len(radii) else 0.0
        for ti in tree.query_ball_point(xy, r=max_r):
            dist = float(np.linalg.norm(xy - target_xy[ti]))
            if dist <= float(radii[ti]):
                candidates.append((dist, pi, ti))
    candidates.sort(key=lambda item: item[0])
    used_p: set[int] = set()
    used_t: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for dist, pi, ti in candidates:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matches.append((pi, ti, dist))
    return matches


def detection_metrics(pred: pd.DataFrame, target: pd.DataFrame, radius: np.ndarray | float) -> dict[str, float | int]:
    matches = greedy_match_variable_radius(pred, target, radius)
    pred_count = int(len(pred))
    target_count = int(len(target))
    matched_count = int(len(matches))
    precision = matched_count / pred_count if pred_count else 0.0
    recall = matched_count / target_count if target_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pred_count": pred_count,
        "target_count": target_count,
        "matched_count": matched_count,
        "false_count": int(pred_count - matched_count),
        "missed_count": int(target_count - matched_count),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
