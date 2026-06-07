from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image
import scipy.ndimage as ndi
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from preprocessing.mask_generator import FITS_SUFFIXES, generate_mask, read_gray_image  # noqa: E402


@dataclass
class GroupInfo:
    group: str
    sub_dir: str
    stack_dir: str
    stack_fits: str
    stack_mask: str
    coord_val_holdout: bool
    single_count: int
    train_count: int
    val_count: int
    val_files: list[str]
    detected_stars: int | None = None
    mask_pixels: int | None = None
    skipped_misaligned_count: int = 0


@dataclass
class SampleRow:
    sample_id: str
    split: str
    group: str
    single_fits: str
    stack_fits: str
    image_out: str
    mask_out: str
    stack_mask: str
    label_shift_x_px: float
    label_shift_y_px: float
    label_alignment_method: str
    label_alignment_matches: int
    label_alignment_residual_px: float | None
    label_transform_a: float
    label_transform_b: float
    label_transform_c: float
    label_transform_d: float
    split_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build U-Net model data from S30Pro single-frame FITS and stack-derived masks.",
    )
    parser.add_argument("--root", type=Path, default=Path("data/data_S30Pro"))
    parser.add_argument("--output", type=Path, default=Path("data/data_model"))
    parser.add_argument("--single-pattern", default="*.fit")
    parser.add_argument("--stack-pattern", default="Stacked_*.fit")
    parser.add_argument("--coord-val-fraction", type=float, default=0.10)
    parser.add_argument("--frame-val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing output directory and append new samples after the current manifest.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-missing-stack", action="store_true")

    parser.add_argument("--fit-channel-mode", default="mean", choices=("mean", "first", "max", "luma"))
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--filtsize", type=int, default=25)
    parser.add_argument("--background-mode", default="local_mean")
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--max-area", type=int, default=100)
    parser.add_argument("--max-axis-ratio", type=float, default=None)
    parser.add_argument("--no-binary-open", action="store_true")
    parser.add_argument("--heatmap-radius-scale", type=float, default=2.0)
    parser.add_argument("--heatmap-edge-value", type=float, default=0.05)
    parser.add_argument("--min-heatmap-radius", type=float, default=1.0)
    parser.add_argument(
        "--no-align-single-labels",
        action="store_true",
        help="Disable single-frame DAOFind alignment and use only FITS WCS RA/DEC shift.",
    )
    parser.add_argument("--align-top-stack", type=int, default=500)
    parser.add_argument("--align-top-single", type=int, default=500)
    parser.add_argument("--align-max-shift-px", type=float, default=400.0)
    parser.add_argument("--align-bin-px", type=float, default=4.0)
    parser.add_argument("--align-candidates", type=int, default=12)
    parser.add_argument("--align-match-radius-px", type=float, default=16.0)
    parser.add_argument("--align-min-matches", type=int, default=8)
    parser.add_argument("--align-max-residual-px", type=float, default=5.0)
    parser.add_argument("--align-allow-scale", action="store_true")
    parser.add_argument("--fwhm", type=float, default=3.0, help="DAOFind-like Gaussian FWHM in pixels.")
    parser.add_argument("--peak-window", type=int, default=None)
    parser.add_argument("--mask-radius", type=float, default=None)
    parser.add_argument("--radius-mode", default="gaussian", choices=("constant", "linear", "sqrt", "log", "gaussian"))
    parser.add_argument("--min-mask-radius", type=float, default=None)
    parser.add_argument("--max-mask-radius", type=float, default=None)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--min-separation", type=float, default=None)
    parser.add_argument("--max-peaks", type=int, default=1200)
    parser.add_argument("--exclude-border", type=int, default=None)
    return parser.parse_args()


def natural_key(text: str) -> list[Any]:
    parts = re.split(r"(\d+)", text)
    return [int(part) if part.isdigit() else part.casefold() for part in parts]


def path_key(path: Path) -> list[Any]:
    return natural_key(path.name)


def stable_rng(seed: int, group_name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{group_name}".encode("utf-8")).digest()
    group_seed = int.from_bytes(digest[:8], "big")
    return random.Random(group_seed)


def safe_name(text: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip())
    safe = safe.strip("._-")
    return safe or "group"


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def fits_image_shape(path: Path) -> tuple[int, int]:
    header = fits.getheader(path)
    return int(header["NAXIS2"]), int(header["NAXIS1"])


def header_image_shape(header: fits.Header) -> tuple[int, int]:
    return int(header["NAXIS2"]), int(header["NAXIS1"])


def stack_to_single_label_shift_xy(stack_header: fits.Header, single_header: fits.Header) -> np.ndarray:
    wcs = WCS(stack_header, naxis=2)
    stack_ra = float(stack_header.get("RA", stack_header.get("CRVAL1")))
    stack_dec = float(stack_header.get("DEC", stack_header.get("CRVAL2")))
    single_ra = float(single_header.get("RA", stack_ra))
    single_dec = float(single_header.get("DEC", stack_dec))

    stack_xy = np.asarray(wcs.all_world2pix([[stack_ra, stack_dec]], 0)[0], dtype=np.float32)
    single_xy_on_stack = np.asarray(wcs.all_world2pix([[single_ra, single_dec]], 0)[0], dtype=np.float32)
    delta_xy = single_xy_on_stack - stack_xy
    return -delta_xy


def component_radii_from_mask(
    mask: np.ndarray,
    centroids_yx: np.ndarray,
    radius_scale: float,
    min_radius: float,
) -> np.ndarray:
    labels, _ = ndi.label(np.asarray(mask, dtype=bool))
    if len(centroids_yx) == 0:
        return np.empty((0,), dtype=np.float32)

    areas = np.bincount(labels.ravel())
    height, width = labels.shape
    radii = np.empty(len(centroids_yx), dtype=np.float32)

    for index, (cy, cx) in enumerate(centroids_yx):
        y = int(round(float(cy)))
        x = int(round(float(cx)))
        y = min(max(y, 0), height - 1)
        x = min(max(x, 0), width - 1)
        label = int(labels[y, x])
        if label == 0:
            y0 = max(0, y - 2)
            y1 = min(height, y + 3)
            x0 = max(0, x - 2)
            x1 = min(width, x + 3)
            nearby = labels[y0:y1, x0:x1]
            nearby = nearby[nearby > 0]
            label = int(nearby[0]) if len(nearby) else 0
        area = float(areas[label]) if label > 0 else float(np.count_nonzero(mask)) / max(len(centroids_yx), 1)
        equivalent_radius = np.sqrt(max(area, 1.0) / np.pi)
        radii[index] = max(float(min_radius), equivalent_radius * float(radius_scale))
    return radii


def gaussian_heatmap(
    shape: tuple[int, int],
    centroids_yx: np.ndarray,
    radii: np.ndarray,
    shift_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
    transform_matrix: np.ndarray | None = None,
    edge_value: float = 0.05,
) -> np.ndarray:
    height, width = shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    if len(centroids_yx) == 0:
        return heatmap

    shift_x = float(shift_xy[0])
    shift_y = float(shift_xy[1])
    matrix = np.eye(2, dtype=np.float32) if transform_matrix is None else np.asarray(transform_matrix, dtype=np.float32)
    edge_value = min(max(float(edge_value), 1e-4), 0.95)
    sigma_factor = np.sqrt(-2.0 * np.log(edge_value))

    for (cy, cx), radius in zip(centroids_yx, radii):
        radius = max(float(radius), 0.5)
        transformed = np.asarray([[float(cx), float(cy)]], dtype=np.float32) @ matrix.T
        center_x = float(transformed[0, 0]) + shift_x
        center_y = float(transformed[0, 1]) + shift_y
        y0 = max(0, int(np.floor(center_y - radius)))
        y1 = min(height, int(np.ceil(center_y + radius)) + 1)
        x0 = max(0, int(np.floor(center_x - radius)))
        x1 = min(width, int(np.ceil(center_x + radius)) + 1)
        if y0 >= y1 or x0 >= x1:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance2 = (yy - center_y) ** 2 + (xx - center_x) ** 2
        keep = distance2 <= radius * radius
        sigma = radius / sigma_factor
        values = np.exp(-0.5 * distance2 / max(sigma * sigma, 1e-6)).astype(np.float32)
        patch = heatmap[y0:y1, x0:x1]
        np.maximum(patch, values * keep, out=patch)
    return heatmap


def identity_transform() -> np.ndarray:
    return np.eye(2, dtype=np.float32)


def centroid_peak_scores(image: np.ndarray, centroids_yx: np.ndarray, radius: int = 2) -> np.ndarray:
    scores = np.zeros(len(centroids_yx), dtype=np.float32)
    height, width = image.shape[:2]
    for index, (cy, cx) in enumerate(centroids_yx):
        y = int(round(float(cy)))
        x = int(round(float(cx)))
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        if y0 < y1 and x0 < x1:
            scores[index] = float(np.nanmax(image[y0:y1, x0:x1]))
    return scores


def strongest_xy(image: np.ndarray, centroids_yx: np.ndarray, top_count: int) -> np.ndarray:
    if len(centroids_yx) == 0:
        return np.empty((0, 2), dtype=np.float32)
    scores = centroid_peak_scores(image, centroids_yx)
    order = np.argsort(scores)[::-1]
    if top_count > 0:
        order = order[:top_count]
    selected = centroids_yx[order]
    return np.column_stack((selected[:, 1], selected[:, 0])).astype(np.float32)


def coarse_translation_candidates(
    stack_xy: np.ndarray,
    single_xy: np.ndarray,
    wcs_shift_xy: np.ndarray,
    max_shift_px: float,
    bin_px: float,
    candidate_count: int,
) -> list[np.ndarray]:
    candidates = [np.asarray(wcs_shift_xy, dtype=np.float32)]
    if len(stack_xy) == 0 or len(single_xy) == 0:
        return candidates

    deltas = single_xy[:, None, :] - stack_xy[None, :, :]
    residual = deltas - np.asarray(wcs_shift_xy, dtype=np.float32)
    keep = (np.abs(residual[:, :, 0]) <= max_shift_px) & (np.abs(residual[:, :, 1]) <= max_shift_px)
    kept = deltas[keep]
    if len(kept) == 0:
        return candidates

    bins = np.rint(kept / max(float(bin_px), 1e-6)).astype(np.int32)
    unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    seen = {tuple(np.rint(candidates[0] / max(float(bin_px), 1e-6)).astype(np.int32).tolist())}
    for index in order:
        candidate = unique_bins[index].astype(np.float32) * float(bin_px)
        key = tuple(unique_bins[index].tolist())
        if key in seen:
            continue
        candidates.append(candidate)
        seen.add(key)
        if len(candidates) >= max(1, int(candidate_count)):
            break
    return candidates


def transform_xy(xy: np.ndarray, matrix: np.ndarray, shift_xy: np.ndarray) -> np.ndarray:
    return xy @ np.asarray(matrix, dtype=np.float32).T + np.asarray(shift_xy, dtype=np.float32)


def nearest_unique_pairs(source_xy: np.ndarray, target_xy: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(source_xy) == 0 or len(target_xy) == 0:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty, np.empty((0,), dtype=np.float32)

    tree = cKDTree(target_xy)
    distances, target_indices = tree.query(source_xy, distance_upper_bound=float(radius))
    valid = np.isfinite(distances) & (target_indices < len(target_xy))
    source_indices = np.flatnonzero(valid).astype(np.int32)
    target_indices = target_indices[valid].astype(np.int32)
    distances = distances[valid].astype(np.float32)
    order = np.argsort(distances)

    used_sources: set[int] = set()
    used_targets: set[int] = set()
    kept_source: list[int] = []
    kept_target: list[int] = []
    kept_dist: list[float] = []
    for idx in order:
        source_index = int(source_indices[idx])
        target_index = int(target_indices[idx])
        if source_index in used_sources or target_index in used_targets:
            continue
        used_sources.add(source_index)
        used_targets.add(target_index)
        kept_source.append(source_index)
        kept_target.append(target_index)
        kept_dist.append(float(distances[idx]))
    return (
        np.asarray(kept_source, dtype=np.int32),
        np.asarray(kept_target, dtype=np.int32),
        np.asarray(kept_dist, dtype=np.float32),
    )


def fit_similarity_transform(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    allow_scale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if len(source_xy) < 2:
        raise ValueError("at least two matched points are required for similarity transform")
    source = np.asarray(source_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    if allow_scale:
        denom = float(np.sum(source_centered * source_centered))
        scale = float(np.sum(singular_values) / denom) if denom > 1e-8 else 1.0
    else:
        scale = 1.0
    matrix = (scale * rotation).astype(np.float32)
    shift_xy = (target_mean - source_mean @ matrix.T).astype(np.float32)
    return matrix, shift_xy


def robust_similarity_alignment(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    initial_shift_xy: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    min_matches = int(args.align_min_matches)
    match_radius = float(args.align_match_radius_px)
    max_residual = float(args.align_max_residual_px)
    best: dict[str, Any] = {
        "method": "wcs_shift",
        "matrix": identity_transform(),
        "shift_xy": np.asarray(initial_shift_xy, dtype=np.float32),
        "matches": 0,
        "residual_px": None,
    }
    candidates = coarse_translation_candidates(
        source_xy,
        target_xy,
        np.asarray(initial_shift_xy, dtype=np.float32),
        float(args.align_max_shift_px),
        float(args.align_bin_px),
        int(args.align_candidates),
    )

    for candidate in candidates:
        matrix = identity_transform()
        shift_xy = np.asarray(candidate, dtype=np.float32)
        matched_source = np.empty((0,), dtype=np.int32)
        matched_target = np.empty((0,), dtype=np.int32)
        distances = np.empty((0,), dtype=np.float32)

        for _ in range(3):
            predicted_xy = transform_xy(source_xy, matrix, shift_xy)
            matched_source, matched_target, distances = nearest_unique_pairs(predicted_xy, target_xy, match_radius)
            if len(matched_source) < min_matches:
                break
            matrix, shift_xy = fit_similarity_transform(
                source_xy[matched_source],
                target_xy[matched_target],
                allow_scale=bool(args.align_allow_scale),
            )

        if len(matched_source) < min_matches:
            continue

        predicted_xy = transform_xy(source_xy[matched_source], matrix, shift_xy)
        residuals = np.linalg.norm(predicted_xy - target_xy[matched_target], axis=1)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        threshold = min(match_radius, max(max_residual, median + 3.0 * 1.4826 * mad))
        keep = residuals <= threshold
        if int(np.sum(keep)) >= min_matches:
            matrix, shift_xy = fit_similarity_transform(
                source_xy[matched_source][keep],
                target_xy[matched_target][keep],
                allow_scale=bool(args.align_allow_scale),
            )
            predicted_xy = transform_xy(source_xy[matched_source][keep], matrix, shift_xy)
            residuals = np.linalg.norm(predicted_xy - target_xy[matched_target][keep], axis=1)
            match_count = int(np.sum(keep))
        else:
            match_count = int(len(matched_source))

        residual_mean = float(np.mean(residuals)) if len(residuals) else None
        current = {
            "method": "daofind_similarity",
            "matrix": matrix,
            "shift_xy": shift_xy,
            "matches": match_count,
            "residual_px": residual_mean,
        }
        best_residual = best["residual_px"] if best["residual_px"] is not None else 1e9
        current_residual = residual_mean if residual_mean is not None else 1e9
        if (match_count, -current_residual) > (int(best["matches"]), -float(best_residual)):
            best = current
    return best


def write_heatmap(path: Path, heatmap: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    Image.fromarray((data * 255.0 + 0.5).astype(np.uint8)).save(path)


def discover_groups(root: Path, args: argparse.Namespace) -> list[tuple[str, Path, Path, Path, list[Path]]]:
    groups = []
    for sub_dir in sorted((path for path in root.iterdir() if path.is_dir() and path.name.endswith("_sub")), key=path_key):
        group_name = sub_dir.name[:-4]
        stack_dir = root / group_name
        if not stack_dir.exists():
            message = f"missing stack folder for {sub_dir}: expected {stack_dir}"
            if args.skip_missing_stack:
                print(f"[skip] {message}")
                continue
            raise FileNotFoundError(message)

        stack_files = sorted(
            (path for path in stack_dir.glob(args.stack_pattern) if path.suffix.lower() in FITS_SUFFIXES),
            key=path_key,
        )
        if not stack_files:
            stack_files = sorted(
                (path for path in stack_dir.iterdir() if path.is_file() and path.suffix.lower() in FITS_SUFFIXES),
                key=path_key,
            )
        if not stack_files:
            raise FileNotFoundError(f"no stacked FITS found in {stack_dir}")
        if len(stack_files) > 1:
            print(f"[warn] {stack_dir} has {len(stack_files)} stack FITS; using {stack_files[0].name}")

        single_files = sorted(
            (path for path in sub_dir.glob(args.single_pattern) if path.suffix.lower() in FITS_SUFFIXES),
            key=path_key,
        )
        if not single_files:
            print(f"[skip] no single FITS found in {sub_dir}")
            continue
        groups.append((group_name, stack_dir, sub_dir, stack_files[0], single_files))
    return groups


def validate_split_fractions(args: argparse.Namespace) -> None:
    for name in ("coord_val_fraction", "frame_val_fraction"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1], got {value}")


def choose_val_plan(
    groups: list[tuple[str, Path, Path, Path, list[Path]]],
    args: argparse.Namespace,
) -> tuple[set[str], dict[str, set[Path]], dict[str, str]]:
    coord_val_count = int(len(groups) * float(args.coord_val_fraction))
    group_names = [group[0] for group in groups]
    coord_rng = stable_rng(int(args.seed), "__coord_val_holdout__")
    coord_val_groups = set(coord_rng.sample(group_names, coord_val_count)) if coord_val_count else set()

    val_files_by_group: dict[str, set[Path]] = {}
    split_reason_by_group: dict[str, str] = {}
    for group_name, _, _, _, single_files in groups:
        if group_name in coord_val_groups:
            val_files_by_group[group_name] = set(single_files)
            split_reason_by_group[group_name] = "coord_holdout"
            continue

        frame_val_count = int(len(single_files) * float(args.frame_val_fraction))
        if frame_val_count:
            frame_rng = stable_rng(int(args.seed), group_name)
            val_files_by_group[group_name] = set(frame_rng.sample(single_files, frame_val_count))
        else:
            val_files_by_group[group_name] = set()
        split_reason_by_group[group_name] = "frame_holdout"
    return coord_val_groups, val_files_by_group, split_reason_by_group


def existing_generated_files(output: Path) -> list[Path]:
    checks = [
        output / "manifest.csv",
        output / "summary.json",
    ]
    for folder in (
        output / "stack_masks",
        output / "train" / "images",
        output / "train" / "masks",
        output / "val" / "images",
        output / "val" / "masks",
    ):
        if folder.exists():
            checks.extend(path for path in folder.iterdir() if path.is_file())
    return [path for path in checks if path.exists()]


def assert_output_ready(output: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    conflicts = existing_generated_files(output)
    if conflicts and not overwrite:
        preview = ", ".join(str(path) for path in conflicts[:5])
        if len(conflicts) > 5:
            preview += f", ... ({len(conflicts)} files total)"
        raise FileExistsError(f"{output} already has generated files: {preview}; use --overwrite or choose a new --output")


def make_dirs(output: Path) -> None:
    for split in ("train", "val"):
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "masks").mkdir(parents=True, exist_ok=True)
    (output / "stack_masks").mkdir(parents=True, exist_ok=True)


def build_stack_targets(stack_path: Path, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    image = read_gray_image(stack_path, fit_channel_mode=args.fit_channel_mode)
    result = generate_mask(image, "tetra3_like", args)
    radii = component_radii_from_mask(
        result.mask,
        result.centroids_yx,
        radius_scale=args.heatmap_radius_scale,
        min_radius=args.min_heatmap_radius,
    )
    heatmap = gaussian_heatmap(
        result.mask.shape,
        result.centroids_yx,
        radii,
        edge_value=args.heatmap_edge_value,
    )
    return heatmap, result.centroids_yx, radii, int(len(result.centroids_yx)), int(np.count_nonzero(result.mask))


def write_manifest_csv(path: Path, rows: list[SampleRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SampleRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_manifest_csv(path: Path) -> list[SampleRow]:
    if not path.exists():
        return []
    rows: list[SampleRow] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                SampleRow(
                    sample_id=str(raw["sample_id"]),
                    split=str(raw["split"]),
                    group=str(raw["group"]),
                    single_fits=str(raw["single_fits"]),
                    stack_fits=str(raw["stack_fits"]),
                    image_out=str(raw["image_out"]),
                    mask_out=str(raw["mask_out"]),
                    stack_mask=str(raw["stack_mask"]),
                    label_shift_x_px=float(raw["label_shift_x_px"]),
                    label_shift_y_px=float(raw["label_shift_y_px"]),
                    label_alignment_method=str(raw.get("label_alignment_method", "wcs_shift")),
                    label_alignment_matches=int(raw.get("label_alignment_matches", 0) or 0),
                    label_alignment_residual_px=(
                        float(raw["label_alignment_residual_px"])
                        if raw.get("label_alignment_residual_px") not in (None, "")
                        else None
                    ),
                    label_transform_a=float(raw.get("label_transform_a", 1.0) or 1.0),
                    label_transform_b=float(raw.get("label_transform_b", 0.0) or 0.0),
                    label_transform_c=float(raw.get("label_transform_c", 0.0) or 0.0),
                    label_transform_d=float(raw.get("label_transform_d", 1.0) or 1.0),
                    split_reason=str(raw["split_reason"]),
                )
            )
    return rows


def sample_number(sample_id: str) -> int:
    match = re.fullmatch(r"sample_(\d+)", sample_id)
    return int(match.group(1)) if match else 0


def next_sample_index(output: Path, rows: list[SampleRow]) -> int:
    max_index = max((sample_number(row.sample_id) for row in rows), default=0)
    for folder in (
        output / "train" / "images",
        output / "train" / "masks",
        output / "val" / "images",
        output / "val" / "masks",
    ):
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if path.is_file():
                max_index = max(max_index, sample_number(path.stem))
    return max_index + 1


def manifest_path_key(value: str) -> str:
    return value.replace("\\", "/").casefold()


def copy_sample(single_path: Path, image_out: Path, heatmap: np.ndarray, mask_out: Path, overwrite: bool) -> None:
    if not overwrite and (image_out.exists() or mask_out.exists()):
        raise FileExistsError(f"sample output already exists: {image_out} or {mask_out}")
    image_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(single_path, image_out)
    write_heatmap(mask_out, heatmap)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output = args.output.resolve()
    groups = discover_groups(root, args)
    if not groups:
        raise FileNotFoundError(f"no *_sub groups found under {root}")
    validate_split_fractions(args)
    coord_val_groups, val_files_by_group, split_reason_by_group = choose_val_plan(groups, args)

    assert_output_ready(output, args.overwrite, args.dry_run)
    if not args.dry_run:
        make_dirs(output)

    existing_rows = read_manifest_csv(output / "manifest.csv") if args.overwrite else []
    existing_single_keys = {manifest_path_key(row.single_fits) for row in existing_rows}
    rows: list[SampleRow] = list(existing_rows)
    group_infos: list[GroupInfo] = []
    sample_index = next_sample_index(output, existing_rows) if args.overwrite else 1
    new_sample_count = 0

    for group_index, (group_name, stack_dir, sub_dir, stack_path, single_files) in enumerate(groups, start=1):
        val_files = val_files_by_group[group_name]
        group_split_reason = split_reason_by_group[group_name]
        coord_val_holdout = group_name in coord_val_groups
        split_counts = {"train": 0, "val": 0}
        skipped_existing_count = 0
        skipped_misaligned_count = 0
        pending_single_files: list[Path] = []
        for single_path in single_files:
            single_key = manifest_path_key(relative_or_absolute(single_path))
            if single_key in existing_single_keys:
                skipped_existing_count += 1
            else:
                pending_single_files.append(single_path)

        stack_mask_name = f"stack_{group_index:03d}_{safe_name(group_name)}_tetra3.png"
        stack_mask_out = output / "stack_masks" / stack_mask_name
        stack_image: np.ndarray | None = None
        stack_heatmap: np.ndarray | None = None
        stack_centroids_yx: np.ndarray | None = None
        stack_radii: np.ndarray | None = None
        stack_align_xy: np.ndarray | None = None
        detected_stars: int | None = None
        mask_pixels: int | None = None

        stack_header = fits.getheader(stack_path)
        if not args.dry_run and pending_single_files:
            stack_image = read_gray_image(stack_path, fit_channel_mode=args.fit_channel_mode)
            stack_heatmap, stack_centroids_yx, stack_radii, detected_stars, mask_pixels = build_stack_targets(stack_path, args)
            stack_align_xy = strongest_xy(stack_image, stack_centroids_yx, int(args.align_top_stack))
            write_heatmap(stack_mask_out, stack_heatmap)

        stack_shape = header_image_shape(stack_header)
        for single_path in pending_single_files:
            single_header = fits.getheader(single_path)
            single_shape = header_image_shape(single_header)
            if single_shape != stack_shape:
                raise ValueError(
                    f"shape mismatch in group {group_name}: {single_path.name} has {single_shape}, "
                    f"but stack has {stack_shape}"
                )
            label_shift_xy = stack_to_single_label_shift_xy(stack_header, single_header)
            label_transform = identity_transform()
            alignment_method = "wcs_shift"
            alignment_matches = 0
            alignment_residual: float | None = None

            is_val = single_path in val_files
            split = "val" if is_val else "train"
            split_reason = group_split_reason if is_val else "train"
            sample_id = f"sample_{sample_index:06d}"
            image_out = output / split / "images" / f"{sample_id}{single_path.suffix.lower()}"
            mask_out = output / split / "masks" / f"{sample_id}.png"

            if not args.dry_run:
                assert stack_centroids_yx is not None
                assert stack_radii is not None
                if not args.no_align_single_labels:
                    assert stack_align_xy is not None
                    single_image = read_gray_image(single_path, fit_channel_mode=args.fit_channel_mode)
                    single_result = generate_mask(single_image, "daofind_like", args)
                    single_align_xy = strongest_xy(single_image, single_result.centroids_yx, int(args.align_top_single))
                    alignment = robust_similarity_alignment(stack_align_xy, single_align_xy, label_shift_xy, args)
                    label_transform = np.asarray(alignment["matrix"], dtype=np.float32)
                    label_shift_xy = np.asarray(alignment["shift_xy"], dtype=np.float32)
                    alignment_method = str(alignment["method"])
                    alignment_matches = int(alignment["matches"])
                    alignment_residual = (
                        float(alignment["residual_px"]) if alignment["residual_px"] is not None else None
                    )
                    is_misaligned = (
                        alignment_method != "daofind_similarity"
                        or alignment_matches < int(args.align_min_matches)
                        or alignment_residual is None
                        or alignment_residual > float(args.align_max_residual_px)
                    )
                    if is_misaligned:
                        skipped_misaligned_count += 1
                        residual_text = "none" if alignment_residual is None else f"{alignment_residual:.3f}"
                        print(
                            f"[skip] {group_name}/{single_path.name}: single-frame DAOFind did not match stack mask; "
                            f"method={alignment_method}, matches={alignment_matches}, "
                            f"residual_px={residual_text}, "
                            f"min_matches={int(args.align_min_matches)}, "
                            f"max_residual_px={float(args.align_max_residual_px):.3f}"
                        )
                        continue
                sample_heatmap = gaussian_heatmap(
                    single_shape,
                    stack_centroids_yx,
                    stack_radii,
                    shift_xy=label_shift_xy,
                    transform_matrix=label_transform,
                    edge_value=args.heatmap_edge_value,
                )
                copy_sample(single_path, image_out, sample_heatmap, mask_out, args.overwrite)

            split_counts[split] += 1
            rows.append(
                SampleRow(
                    sample_id=sample_id,
                    split=split,
                    group=group_name,
                    single_fits=relative_or_absolute(single_path),
                    stack_fits=relative_or_absolute(stack_path),
                    image_out=relative_or_absolute(image_out),
                    mask_out=relative_or_absolute(mask_out),
                    stack_mask=relative_or_absolute(stack_mask_out),
                    label_shift_x_px=round(float(label_shift_xy[0]), 4),
                    label_shift_y_px=round(float(label_shift_xy[1]), 4),
                    label_alignment_method=alignment_method,
                    label_alignment_matches=alignment_matches,
                    label_alignment_residual_px=(
                        round(float(alignment_residual), 4) if alignment_residual is not None else None
                    ),
                    label_transform_a=round(float(label_transform[0, 0]), 8),
                    label_transform_b=round(float(label_transform[0, 1]), 8),
                    label_transform_c=round(float(label_transform[1, 0]), 8),
                    label_transform_d=round(float(label_transform[1, 1]), 8),
                    split_reason=split_reason,
                )
            )
            existing_single_keys.add(manifest_path_key(relative_or_absolute(single_path)))
            sample_index += 1
            new_sample_count += 1

        group_infos.append(
            GroupInfo(
                group=group_name,
                sub_dir=relative_or_absolute(sub_dir),
                stack_dir=relative_or_absolute(stack_dir),
                stack_fits=relative_or_absolute(stack_path),
                stack_mask=relative_or_absolute(stack_mask_out),
                coord_val_holdout=coord_val_holdout,
                single_count=len(single_files),
                train_count=split_counts["train"],
                val_count=split_counts["val"],
                val_files=[relative_or_absolute(path) for path in sorted(val_files, key=path_key)],
                detected_stars=detected_stars,
                mask_pixels=mask_pixels,
                skipped_misaligned_count=skipped_misaligned_count,
            )
        )
        print(
            f"[group] {group_name}: singles={len(single_files)}, train={split_counts['train']}, "
            f"val={split_counts['val']}, skipped_existing={skipped_existing_count}, "
            f"skipped_misaligned={skipped_misaligned_count}, "
            f"coord_holdout={coord_val_holdout}, stack={stack_path.name}"
        )

    summary = {
        "root": relative_or_absolute(root),
        "output": relative_or_absolute(output),
        "sample_count": len(rows),
        "existing_sample_count": len(existing_rows),
        "new_sample_count": new_sample_count,
        "skipped_misaligned_count": sum(group.skipped_misaligned_count for group in group_infos),
        "train_count": sum(1 for row in rows if row.split == "train"),
        "val_count": sum(1 for row in rows if row.split == "val"),
        "group_count": len(group_infos),
        "coord_val_fraction": float(args.coord_val_fraction),
        "frame_val_fraction": float(args.frame_val_fraction),
        "coord_val_group_count": len(coord_val_groups),
        "coord_val_groups": sorted(coord_val_groups, key=natural_key),
        "seed": int(args.seed),
        "mask_method": "tetra3_like_gaussian_heatmap",
        "fit_channel_mode": args.fit_channel_mode,
        "label_alignment": (
            "stack tetra3 centroids are aligned to each single frame with DAOFind single-frame detections "
            "and a robust similarity transform; frames with insufficient or high-residual matches are skipped"
            if not args.no_align_single_labels
            else "stack tetra3 centroids shifted into each single-frame coordinate system from FITS RA/DEC and stack WCS"
        ),
        "label_alignment_params": {
            "enabled": not bool(args.no_align_single_labels),
            "top_stack": int(args.align_top_stack),
            "top_single": int(args.align_top_single),
            "max_shift_px": float(args.align_max_shift_px),
            "bin_px": float(args.align_bin_px),
            "candidates": int(args.align_candidates),
            "match_radius_px": float(args.align_match_radius_px),
            "min_matches": int(args.align_min_matches),
            "max_residual_px": float(args.align_max_residual_px),
            "allow_scale": bool(args.align_allow_scale),
        },
        "heatmap": {
            "radius_scale": float(args.heatmap_radius_scale),
            "edge_value": float(args.heatmap_edge_value),
            "min_radius": float(args.min_heatmap_radius),
        },
        "groups": [asdict(group) for group in group_infos],
    }

    if not args.dry_run:
        write_manifest_csv(output / "manifest.csv", rows)
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    summary = build_dataset(args)
    action = "would write" if args.dry_run else "wrote"
    print(
        f"[done] kept {summary['existing_sample_count']} existing samples, "
        f"{action} {summary['new_sample_count']} new samples: "
        f"total={summary['sample_count']}, "
        f"train={summary['train_count']}, val={summary['val_count']}, groups={summary['group_count']}"
    )
    print(f"[done] output: {summary['output']}")


if __name__ == "__main__":
    main()
