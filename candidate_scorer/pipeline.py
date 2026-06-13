from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree

from preprocessing.mask_generator import estimate_mesh_background, generate_mask, subtract_background
from star_unet.evaluate import baseline_args


@dataclass
class CandidateSet:
    centroids_yx: np.ndarray
    source_mask: np.ndarray
    response: np.ndarray


def resolve_data_path(data_root: Path, path_text: str | Path) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    if path.is_absolute():
        return path
    repo_path = Path(__file__).resolve().parents[1] / path
    if repo_path.exists():
        return repo_path
    parts = list(path.parts)
    if "data_model" in parts:
        suffix = parts[parts.index("data_model") + 1 :]
        return data_root.joinpath(*suffix)
    return data_root / path


def parse_candidate_methods(text: str) -> list[tuple[str, float]]:
    methods: list[tuple[str, float]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        name, value = item.split(":", 1)
        name = name.strip().lower()
        aliases = {
            "dao": "daofind_like",
            "daofind": "daofind_like",
            "sextractor": "sextractor_like",
            "sex": "sextractor_like",
            "tetra3": "tetra3_like",
            "log": "multiscale_log",
            "alog": "adaptive_multiscale_log",
            "adaptive_log": "adaptive_multiscale_log",
        }
        method = aliases.get(name, name)
        if method not in {
            "daofind_like",
            "sextractor_like",
            "tetra3_like",
            "multiscale_log",
            "adaptive_multiscale_log",
        }:
            raise ValueError(f"unsupported candidate method: {name}")
        methods.append((method, float(value)))
    if not methods:
        raise ValueError("at least one candidate method is required")
    return methods


def _method_args(sigma: float) -> SimpleNamespace:
    args = SimpleNamespace(**vars(baseline_args()))
    args.sigma = float(sigma)
    args.background_mode = "local_mean"
    return args


def _matched_response(raw: np.ndarray, centroids_yx: np.ndarray) -> np.ndarray:
    residual = subtract_background(raw, "local_mean", 25)
    matched = ndi.gaussian_filter(residual, sigma=max(3.0 / 2.3548, 0.5))
    center = float(np.median(matched))
    noise = float(1.4826 * np.median(np.abs(matched - center)))
    noise = noise if np.isfinite(noise) and noise > 1e-6 else 1.0
    h, w = raw.shape
    values = []
    for cy, cx in np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2)):
        y = int(np.clip(round(float(cy) - 0.5), 0, h - 1))
        x = int(np.clip(round(float(cx) - 0.5), 0, w - 1))
        values.append((float(matched[y, x]) - center) / noise)
    return np.asarray(values, dtype=np.float32)


def _multiscale_log_candidates(raw: np.ndarray, threshold_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    residual = subtract_background(raw, "local_mean", 25)
    responses = [
        -ndi.gaussian_laplace(residual, sigma=scale) * (scale**2)
        for scale in (0.9, 1.3, 1.8, 2.4)
    ]
    response = np.max(np.stack(responses, axis=0), axis=0)
    center = float(np.median(response))
    noise = float(1.4826 * np.median(np.abs(response - center)))
    noise = noise if np.isfinite(noise) and noise > 1e-6 else 1.0
    local_max = response == ndi.maximum_filter(response, size=5)
    peaks = np.argwhere(local_max & (response > center + float(threshold_sigma) * noise))
    if len(peaks):
        h, w = raw.shape
        keep = (
            (peaks[:, 0] >= 3) & (peaks[:, 0] < h - 3)
            & (peaks[:, 1] >= 3) & (peaks[:, 1] < w - 3)
        )
        peaks = peaks[keep]
    centroids = peaks.astype(np.float32) + 0.5
    values = response[peaks[:, 0], peaks[:, 1]] if len(peaks) else np.empty((0,), dtype=np.float32)
    normalized = ((values - center) / noise).astype(np.float32)
    return centroids.reshape((-1, 2)), normalized


def _adaptive_multiscale_log_candidates(
    raw: np.ndarray,
    threshold_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    residual = subtract_background(raw, "local_mean", 25)
    responses = [
        -ndi.gaussian_laplace(residual, sigma=scale) * (scale**2)
        for scale in (0.9, 1.3, 1.8, 2.4)
    ]
    response = np.max(np.stack(responses, axis=0), axis=0).astype(np.float32)
    background, noise = estimate_mesh_background(response, mesh_size=64, filter_size=3)
    score = (response - background) / np.maximum(noise, 1e-6)
    local_max = score == ndi.maximum_filter(score, size=5)
    peaks = np.argwhere(local_max & (score > float(threshold_sigma)))
    if len(peaks):
        h, w = raw.shape
        keep = (
            (peaks[:, 0] >= 3) & (peaks[:, 0] < h - 3)
            & (peaks[:, 1] >= 3) & (peaks[:, 1] < w - 3)
        )
        peaks = peaks[keep]
    centroids = peaks.astype(np.float32) + 0.5
    values = score[peaks[:, 0], peaks[:, 1]] if len(peaks) else np.empty((0,), dtype=np.float32)
    return centroids.reshape((-1, 2)), values.astype(np.float32)


def deduplicate_candidates(
    centroids_yx: np.ndarray,
    source_ids: np.ndarray,
    response: np.ndarray,
    radius_px: float,
) -> CandidateSet:
    centroids = np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2))
    source_ids = np.asarray(source_ids, dtype=np.int32)
    response = np.asarray(response, dtype=np.float32)
    if not len(centroids):
        return CandidateSet(centroids, np.empty((0,), dtype=np.int64), response)

    order = np.argsort(-response)
    tree = cKDTree(np.column_stack((centroids[:, 1], centroids[:, 0])))
    consumed = np.zeros(len(centroids), dtype=bool)
    out_yx: list[np.ndarray] = []
    out_sources: list[int] = []
    out_response: list[float] = []
    for index in order:
        if consumed[index]:
            continue
        neighbors = tree.query_ball_point(
            [float(centroids[index, 1]), float(centroids[index, 0])],
            r=float(radius_px),
        )
        neighbors = [item for item in neighbors if not consumed[item]]
        source_mask = 0
        for neighbor in neighbors:
            source_mask |= 1 << int(source_ids[neighbor])
            consumed[neighbor] = True
        out_yx.append(centroids[index])
        out_sources.append(source_mask)
        out_response.append(float(response[index]))
    return CandidateSet(
        np.asarray(out_yx, dtype=np.float32).reshape((-1, 2)),
        np.asarray(out_sources, dtype=np.int64),
        np.asarray(out_response, dtype=np.float32),
    )


def generate_candidates(raw: np.ndarray, methods_text: str, dedup_radius_px: float = 2.5) -> CandidateSet:
    methods = parse_candidate_methods(methods_text)
    centroid_parts = []
    source_parts = []
    response_parts = []
    for source_id, (method, sigma) in enumerate(methods):
        if method == "multiscale_log":
            centroids, response = _multiscale_log_candidates(raw, sigma)
        elif method == "adaptive_multiscale_log":
            centroids, response = _adaptive_multiscale_log_candidates(raw, sigma)
        else:
            result = generate_mask(raw, method, _method_args(sigma))
            centroids = np.asarray(result.centroids_yx, dtype=np.float32).reshape((-1, 2))
            response = _matched_response(raw, centroids)
        centroid_parts.append(centroids)
        source_parts.append(np.full(len(centroids), source_id, dtype=np.int32))
        response_parts.append(response)
    if not centroid_parts:
        return CandidateSet(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )
    return deduplicate_candidates(
        np.concatenate(centroid_parts, axis=0),
        np.concatenate(source_parts, axis=0),
        np.concatenate(response_parts, axis=0),
        radius_px=dedup_radius_px,
    )


def normalize_feature(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, [0.5, 99.8])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip(
        (np.nan_to_num(image, nan=float(low)) - float(low)) / (float(high) - float(low)),
        0.0,
        1.0,
    ).astype(np.float32)


def make_center_aware_features(raw: np.ndarray, normalized: np.ndarray) -> np.ndarray:
    residual = subtract_background(raw, "local_mean", 25)
    matched = ndi.gaussian_filter(residual, sigma=max(3.0 / 2.3548, 0.5))
    return np.stack(
        [normalized, normalize_feature(residual), normalize_feature(matched)],
        axis=0,
    ).astype(np.float32)


def extract_patches(feature_image: np.ndarray, centroids_yx: np.ndarray, patch_size: int) -> np.ndarray:
    radius = int(patch_size) // 2
    padded = np.pad(feature_image, ((0, 0), (radius, radius), (radius, radius)), mode="reflect")
    patches = []
    for cy, cx in np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2)):
        y = int(round(float(cy) - 0.5)) + radius
        x = int(round(float(cx) - 0.5)) + radius
        patches.append(padded[:, y - radius : y + radius + 1, x - radius : x + radius + 1])
    if not patches:
        return np.empty((0, feature_image.shape[0], patch_size, patch_size), dtype=np.float32)
    return np.asarray(patches, dtype=np.float32)


def add_center_channels(patches: np.ndarray) -> np.ndarray:
    if len(patches) == 0:
        h = patches.shape[-2]
        w = patches.shape[-1]
        return np.empty((0, patches.shape[1] + 3, h, w), dtype=np.float32)
    h, w = patches.shape[-2:]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    y = (yy - (h - 1) * 0.5) / max((h - 1) * 0.5, 1.0)
    x = (xx - (w - 1) * 0.5) / max((w - 1) * 0.5, 1.0)
    gaussian = np.exp(-(x * x + y * y) / (2.0 * 0.18**2)).astype(np.float32)
    extra = np.stack([gaussian, x, y], axis=0)
    extra = np.broadcast_to(extra[None], (len(patches), 3, h, w))
    return np.concatenate([patches, extra], axis=1).astype(np.float32)


def score_nms(
    centroids_yx: np.ndarray,
    scores: np.ndarray,
    radius_px: float,
) -> np.ndarray:
    centroids = np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2))
    scores = np.asarray(scores, dtype=np.float32)
    if not len(centroids):
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(-scores)
    tree = cKDTree(np.column_stack((centroids[:, 1], centroids[:, 0])))
    suppressed = np.zeros(len(centroids), dtype=bool)
    keep = []
    for index in order:
        if suppressed[index]:
            continue
        keep.append(int(index))
        neighbors = tree.query_ball_point(
            [float(centroids[index, 1]), float(centroids[index, 0])],
            r=float(radius_px),
        )
        suppressed[neighbors] = True
        suppressed[index] = False
    return np.asarray(keep, dtype=np.int64)
