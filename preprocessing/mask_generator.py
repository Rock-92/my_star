from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import scipy.ndimage as ndi


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FITS_BLOCK_BYTES = 2880
FITS_SUFFIXES = {".fit", ".fits", ".fts"}
METHODS = ("tetra3_like", "sextractor_like", "daofind_like")
DEFAULT_SIGMA_BY_METHOD = {
    "tetra3_like": 2.5,
    "sextractor_like": 2.5,
    "daofind_like": 5.0,
}


@dataclass
class MaskResult:
    method: str
    mask: np.ndarray
    centroids_yx: np.ndarray
    debug: dict[str, Any]


def parse_fits_value(value: str) -> Any:
    value = value.strip()
    if value.startswith("'") and value.endswith("'"):
        return value.strip("'").strip()
    if value in {"T", "F"}:
        return value == "T"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_fits_header(path: str | Path) -> tuple[dict[str, Any], int]:
    raw = Path(path).read_bytes()
    cards = []
    for offset in range(0, len(raw), 80):
        card = raw[offset : offset + 80].decode("ascii", errors="replace")
        cards.append(card)
        if card.startswith("END"):
            break
    else:
        raise ValueError(f"missing FITS END card: {path}")

    header: dict[str, Any] = {}
    for card in cards:
        if "=" not in card[:10]:
            continue
        key = card[:8].strip()
        value = card[10:80].split("/", 1)[0].strip()
        header[key] = parse_fits_value(value)

    header_bytes = len(cards) * 80
    data_offset = ((header_bytes + FITS_BLOCK_BYTES - 1) // FITS_BLOCK_BYTES) * FITS_BLOCK_BYTES
    return header, data_offset


def read_fits_image(path: str | Path, channel_mode: str = "mean") -> np.ndarray:
    path = Path(path)
    raw = path.read_bytes()
    header, data_offset = parse_fits_header(path)

    bitpix = int(header.get("BITPIX", 0))
    dtype_by_bitpix = {
        8: ">u1",
        16: ">i2",
        32: ">i4",
        -32: ">f4",
        -64: ">f8",
    }
    if bitpix not in dtype_by_bitpix:
        raise ValueError(f"unsupported FITS BITPIX={bitpix}: {path}")

    naxis = int(header.get("NAXIS", 0))
    if naxis < 2:
        raise ValueError(f"FITS image must have at least two axes: {path}")
    axes = [int(header[f"NAXIS{index}"]) for index in range(1, naxis + 1)]
    count = int(np.prod(axes))

    dtype = np.dtype(dtype_by_bitpix[bitpix])
    data = np.frombuffer(raw[data_offset : data_offset + count * dtype.itemsize], dtype=dtype)
    if data.size != count:
        raise ValueError(f"FITS data is shorter than header dimensions require: {path}")

    image = data.astype(np.float32)
    image = image * float(header.get("BSCALE", 1.0)) + float(header.get("BZERO", 0.0))
    image = image.reshape(tuple(reversed(axes)))

    if image.ndim == 2:
        return image
    if image.ndim == 3:
        if channel_mode == "first":
            return image[0]
        if channel_mode == "max":
            return image.max(axis=0)
        if channel_mode == "luma" and image.shape[0] == 3:
            weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
            return np.tensordot(weights, image, axes=(0, 0)).astype(np.float32)
        if channel_mode in {"mean", "luma"}:
            return image.mean(axis=0, dtype=np.float32)
    raise ValueError(f"cannot convert FITS image shape {image.shape} to grayscale")


def read_gray_image(path: str | Path, fit_channel_mode: str = "mean") -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in FITS_SUFFIXES:
        return read_fits_image(path, channel_mode=fit_channel_mode)
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def odd_at_least(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


def as_gray_float(image: Any) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
    raise ValueError(f"image must be 2D grayscale or RGB, got shape {image.shape}")


def robust_noise(image: np.ndarray) -> tuple[float, float]:
    center = float(np.median(image))
    noise = float(1.4826 * np.median(np.abs(image - center)))
    if not np.isfinite(noise) or noise <= 0:
        q25, q75 = np.percentile(image, [25.0, 75.0])
        noise = float((q75 - q25) / 1.349)
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.std(image))
    if not np.isfinite(noise) or noise <= 0:
        noise = 1.0
    return center, noise


def subtract_background(image: np.ndarray, mode: str | None, size: int) -> np.ndarray:
    if mode is None or str(mode).lower() in {"none", "off"}:
        return image.copy()
    mode = str(mode).lower()
    size = odd_at_least(size)
    if mode in {"local_mean", "mean"}:
        return image - ndi.uniform_filter(image, size=size, output=np.float32)
    if mode in {"local_median", "median"}:
        return image - ndi.median_filter(image, size=size)
    if mode == "global_mean":
        return image - float(np.mean(image))
    if mode == "global_median":
        return image - float(np.median(image))
    raise ValueError(f"unsupported background mode: {mode}")


def estimate_mesh_background(
    image: np.ndarray,
    mesh_size: int = 64,
    filter_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    mesh_size = max(8, int(mesh_size))
    height, width = image.shape
    grid_h = int(math.ceil(height / mesh_size))
    grid_w = int(math.ceil(width / mesh_size))
    background_grid = np.empty((grid_h, grid_w), dtype=np.float32)
    noise_grid = np.empty((grid_h, grid_w), dtype=np.float32)

    for gy in range(grid_h):
        y0 = gy * mesh_size
        y1 = min(y0 + mesh_size, height)
        for gx in range(grid_w):
            x0 = gx * mesh_size
            x1 = min(x0 + mesh_size, width)
            tile = image[y0:y1, x0:x1]
            background, noise = robust_noise(tile)
            background_grid[gy, gx] = background
            noise_grid[gy, gx] = noise

    filter_size = odd_at_least(filter_size, minimum=1)
    if filter_size > 1:
        background_grid = ndi.median_filter(background_grid, size=filter_size)
        noise_grid = ndi.median_filter(noise_grid, size=filter_size)

    zoom = (height / background_grid.shape[0], width / background_grid.shape[1])
    background_map = ndi.zoom(background_grid, zoom, order=1)[:height, :width]
    noise_map = ndi.zoom(noise_grid, zoom, order=1)[:height, :width]
    noise_floor = float(np.median(noise_grid))
    noise_map = np.maximum(noise_map, max(noise_floor * 0.1, 1e-6))
    return background_map.astype(np.float32), noise_map.astype(np.float32)


def region_axis_ratio(y: np.ndarray, x: np.ndarray, weights: np.ndarray, cy: float, cx: float) -> float:
    dy = y - (cy - 0.5)
    dx = x - (cx - 0.5)
    total = float(np.sum(weights))
    if total <= 0:
        return 1.0
    m2_xx = float(max(0.0, np.sum(dx * dx * weights) / total))
    m2_yy = float(max(0.0, np.sum(dy * dy * weights) / total))
    m2_xy = float(np.sum(dx * dy * weights) / total)
    common = math.sqrt((m2_xx - m2_yy) ** 2 + 4.0 * m2_xy**2)
    major = math.sqrt(max(0.0, 2.0 * (m2_xx + m2_yy + common)))
    minor = math.sqrt(max(0.0, 2.0 * (m2_xx + m2_yy - common)))
    return major / max(minor, 1e-9)


def connected_region_mask(
    binary: np.ndarray,
    weight_image: np.ndarray,
    min_area: int = 5,
    max_area: int | None = 100,
    max_axis_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    labels, raw_regions = ndi.label(binary)
    objects = ndi.find_objects(labels)
    kept_mask = np.zeros(binary.shape, dtype=bool)
    rows: list[tuple[float, float, float, int, float]] = []

    for label_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        region_mask = labels[slices] == label_id
        area = int(np.count_nonzero(region_mask))
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        values = weight_image[slices][region_mask].astype(np.float64)
        weights = np.clip(values, 0.0, None)
        total = float(np.sum(weights))
        if total <= 0 or not np.isfinite(total):
            weights = np.ones_like(values, dtype=np.float64)
            total = float(weights.size)

        yy, xx = np.nonzero(region_mask)
        y = yy.astype(np.float64) + slices[0].start
        x = xx.astype(np.float64) + slices[1].start
        cy = float(np.sum(y * weights) / total + 0.5)
        cx = float(np.sum(x * weights) / total + 0.5)
        axis_ratio = region_axis_ratio(y, x, weights, cy, cx)
        if max_axis_ratio is not None and axis_ratio > max_axis_ratio:
            continue

        target = kept_mask[slices]
        target[region_mask] = True
        rows.append((float(np.sum(values)), cy, cx, area, axis_ratio))

    if rows:
        detections = np.asarray(rows, dtype=np.float32)
        detections = detections[np.argsort(-detections[:, 0])]
        centroids = detections[:, 1:3]
    else:
        detections = np.empty((0, 5), dtype=np.float32)
        centroids = np.empty((0, 2), dtype=np.float32)

    debug = {
        "raw_regions": int(raw_regions),
        "kept_regions": int(len(centroids)),
        "area_min": int(np.min(detections[:, 3])) if len(detections) else 0,
        "area_max": int(np.max(detections[:, 3])) if len(detections) else 0,
    }
    return kept_mask, centroids, debug


def draw_disks(shape: tuple[int, int], centroids_yx: np.ndarray, radius: float | np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if np.isscalar(radius):
        radii = np.full(len(centroids_yx), max(float(radius), 0.5), dtype=np.float32)
    else:
        radii = np.asarray(radius, dtype=np.float32)
        if len(radii) != len(centroids_yx):
            raise ValueError("radius array length must match centroids")
        radii = np.maximum(radii, 0.5)

    for (cy, cx), disk_radius in zip(centroids_yx, radii):
        radius_i = int(math.ceil(float(disk_radius)))
        radius2 = float(disk_radius) * float(disk_radius)
        y0 = max(0, int(math.floor(cy)) - radius_i)
        y1 = min(shape[0], int(math.floor(cy)) + radius_i + 1)
        x0 = max(0, int(math.floor(cx)) - radius_i)
        x1 = min(shape[1], int(math.floor(cx)) + radius_i + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2 <= radius2
    return mask


def peak_scaled_radii(
    peaks: np.ndarray,
    center: float,
    threshold: float,
    gaussian_sigma: float,
    mode: str = "gaussian",
    base_radius: float | None = None,
    min_radius: float | None = None,
    max_radius: float | None = None,
    radius_scale: float = 1.0,
) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=np.float32)
    if base_radius is None:
        base_radius = max(float(gaussian_sigma) * 1.5, 1.5)
    if min_radius is None:
        min_radius = float(base_radius) if base_radius is not None else max(float(gaussian_sigma) * 0.75, 1.0)
    if max_radius is None:
        max_radius = max(float(gaussian_sigma) * 4.0, float(base_radius), 4.0)

    signal = np.maximum(peaks - float(center), 0.0)
    threshold_signal = max(float(threshold) - float(center), 1e-6)
    ratio = np.maximum(signal / threshold_signal, 1.0)

    mode = mode.lower()
    if mode == "constant":
        radii = np.full_like(ratio, float(base_radius), dtype=np.float32)
    elif mode == "linear":
        radii = float(min_radius) + float(radius_scale) * (ratio - 1.0)
    elif mode == "sqrt":
        radii = float(min_radius) + float(radius_scale) * np.sqrt(ratio - 1.0)
    elif mode == "log":
        radii = float(min_radius) + float(radius_scale) * np.log(ratio)
    elif mode == "gaussian":
        radii = float(radius_scale) * float(gaussian_sigma) * np.sqrt(2.0 * np.log(ratio))
        radii = np.maximum(radii, float(min_radius))
    else:
        raise ValueError(f"unsupported radius mode: {mode}")

    return np.clip(radii, float(min_radius), float(max_radius)).astype(np.float32)


def tetra3_like_mask(
    image: Any,
    sigma: float = 2.5,
    filtsize: int = 25,
    background_mode: str = "local_mean",
    min_area: int = 5,
    max_area: int | None = 100,
    max_axis_ratio: float | None = None,
    binary_open: bool = True,
) -> MaskResult:
    gray = as_gray_float(image)
    residual = subtract_background(gray, background_mode, filtsize)
    center, noise = robust_noise(residual)
    threshold = center + float(sigma) * noise
    binary = residual > threshold
    if binary_open:
        binary = ndi.binary_opening(binary)
    mask, centroids, region_debug = connected_region_mask(
        binary,
        residual,
        min_area=min_area,
        max_area=max_area,
        max_axis_ratio=max_axis_ratio,
    )
    debug = {
        "threshold": float(threshold),
        "center": float(center),
        "noise": float(noise),
        **region_debug,
    }
    return MaskResult("tetra3_like", mask, centroids, debug)


def sextractor_like_mask(
    image: Any,
    sigma: float = 1.5,
    mesh_size: int = 64,
    mesh_filter_size: int = 3,
    filter_sigma: float = 1.0,
    min_area: int = 5,
    max_area: int | None = 300,
    max_axis_ratio: float | None = None,
    deblend: bool = False,
) -> MaskResult:
    gray = as_gray_float(image)
    background, noise_map = estimate_mesh_background(gray, mesh_size=mesh_size, filter_size=mesh_filter_size)
    residual = gray - background
    filtered = ndi.gaussian_filter(residual, sigma=max(float(filter_sigma), 0.0)) if filter_sigma > 0 else residual
    threshold_map = float(sigma) * noise_map
    binary = filtered > threshold_map
    mask, centroids, region_debug = connected_region_mask(
        binary,
        residual,
        min_area=min_area,
        max_area=max_area,
        max_axis_ratio=max_axis_ratio,
    )
    debug = {
        "sigma": float(sigma),
        "mesh_size": int(mesh_size),
        "mesh_filter_size": int(mesh_filter_size),
        "filter_sigma": float(filter_sigma),
        "noise_median": float(np.median(noise_map)),
        "background_median": float(np.median(background)),
        "deblend": bool(deblend),
        **region_debug,
    }
    return MaskResult("sextractor_like", mask, centroids, debug)


def daofind_like_mask(
    image: Any,
    sigma: float = 5.0,
    fwhm: float = 3.0,
    background_mode: str = "local_median",
    filtsize: int = 25,
    peak_window: int | None = None,
    mask_radius: float | None = None,
    radius_mode: str = "gaussian",
    min_mask_radius: float | None = None,
    max_mask_radius: float | None = None,
    radius_scale: float = 1.0,
    min_separation: float | None = None,
    max_peaks: int | None = None,
    exclude_border: int | None = None,
) -> MaskResult:
    gray = as_gray_float(image)
    residual = subtract_background(gray, background_mode, filtsize)
    gaussian_sigma = max(float(fwhm) / 2.3548, 0.5)
    matched = ndi.gaussian_filter(residual, sigma=gaussian_sigma)
    center, noise = robust_noise(matched)
    threshold = center + float(sigma) * noise

    if peak_window is None:
        peak_window = odd_at_least(int(round(fwhm * 2.0 + 1.0)))
    else:
        peak_window = odd_at_least(peak_window)
    local_max = matched == ndi.maximum_filter(matched, size=peak_window)
    candidates = np.argwhere(local_max & (matched > threshold))

    if exclude_border is None:
        exclude_border = int(math.ceil(max(fwhm, 1.0)))
    if exclude_border > 0 and len(candidates):
        h, w = gray.shape
        y = candidates[:, 0]
        x = candidates[:, 1]
        keep = (y >= exclude_border) & (y < h - exclude_border) & (x >= exclude_border) & (x < w - exclude_border)
        candidates = candidates[keep]

    if len(candidates):
        scores = matched[candidates[:, 0], candidates[:, 1]]
        order = np.argsort(-scores)
        candidates = candidates[order]
    if min_separation is None:
        min_separation = max(float(fwhm), 1.0)
    min_sep2 = float(min_separation) ** 2

    centroids: list[tuple[float, float]] = []
    peak_values: list[float] = []
    patch_radius = int(math.ceil(max(float(fwhm) * 1.5, 2.0)))
    for y, x in candidates:
        if max_peaks is not None and len(centroids) >= max_peaks:
            break
        if any((float(y) + 0.5 - cy) ** 2 + (float(x) + 0.5 - cx) ** 2 < min_sep2 for cy, cx in centroids):
            continue

        peak_values.append(float(matched[y, x]))
        y0 = max(0, int(y) - patch_radius)
        y1 = min(residual.shape[0], int(y) + patch_radius + 1)
        x0 = max(0, int(x) - patch_radius)
        x1 = min(residual.shape[1], int(x) + patch_radius + 1)
        patch = residual[y0:y1, x0:x1].astype(np.float64)
        weights = np.clip(patch, 0.0, None)
        total = float(np.sum(weights))
        if total <= 0:
            centroids.append((float(y) + 0.5, float(x) + 0.5))
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        cy = float(np.sum(yy * weights) / total + 0.5)
        cx = float(np.sum(xx * weights) / total + 0.5)
        centroids.append((cy, cx))

    centroids_array = np.asarray(centroids, dtype=np.float32).reshape((-1, 2))
    peak_array = np.asarray(peak_values, dtype=np.float32)
    radii = peak_scaled_radii(
        peak_array,
        center=float(center),
        threshold=float(threshold),
        gaussian_sigma=float(gaussian_sigma),
        mode=radius_mode,
        base_radius=mask_radius,
        min_radius=min_mask_radius,
        max_radius=max_mask_radius,
        radius_scale=radius_scale,
    )
    mask = draw_disks(gray.shape, centroids_array, radius=radii)
    debug = {
        "threshold": float(threshold),
        "center": float(center),
        "noise": float(noise),
        "fwhm": float(fwhm),
        "gaussian_sigma": float(gaussian_sigma),
        "peak_window": int(peak_window),
        "raw_peaks": int(len(candidates)),
        "kept_regions": int(len(centroids_array)),
        "radius_mode": radius_mode,
        "mask_radius_min": float(np.min(radii)) if len(radii) else 0.0,
        "mask_radius_max": float(np.max(radii)) if len(radii) else 0.0,
        "mask_radius_mean": float(np.mean(radii)) if len(radii) else 0.0,
        "peak_min": float(np.min(peak_array)) if len(peak_array) else 0.0,
        "peak_max": float(np.max(peak_array)) if len(peak_array) else 0.0,
    }
    return MaskResult("daofind_like", mask, centroids_array, debug)


def generate_mask(image: Any, method: str, args: argparse.Namespace) -> MaskResult:
    sigma = DEFAULT_SIGMA_BY_METHOD[method] if args.sigma is None else args.sigma
    if method == "tetra3_like":
        return tetra3_like_mask(
            image,
            sigma=sigma,
            filtsize=args.filtsize,
            background_mode=args.background_mode,
            min_area=args.min_area,
            max_area=args.max_area,
            max_axis_ratio=args.max_axis_ratio,
            binary_open=not args.no_binary_open,
        )
    if method == "sextractor_like":
        return sextractor_like_mask(
            image,
            sigma=sigma,
            mesh_size=args.mesh_size,
            mesh_filter_size=args.mesh_filter_size,
            filter_sigma=args.filter_sigma,
            min_area=args.min_area,
            max_area=args.max_area,
            max_axis_ratio=args.max_axis_ratio,
        )
    if method == "daofind_like":
        return daofind_like_mask(
            image,
            sigma=sigma,
            fwhm=args.fwhm,
            background_mode=args.background_mode,
            filtsize=args.filtsize,
            peak_window=args.peak_window,
            mask_radius=args.mask_radius,
            radius_mode=args.radius_mode,
            min_mask_radius=args.min_mask_radius,
            max_mask_radius=args.max_mask_radius,
            radius_scale=args.radius_scale,
            min_separation=args.min_separation,
            max_peaks=args.max_peaks,
            exclude_border=args.exclude_border,
        )
    raise ValueError(f"unknown method: {method}")


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_to_uint8(mask)).save(path)


def json_ready(result: MaskResult, source: Path, output: Path) -> dict[str, Any]:
    return {
        "source": str(source),
        "output": str(output),
        "method": result.method,
        "star_count": int(len(result.centroids_yx)),
        "mask_pixels": int(np.count_nonzero(result.mask)),
        "centroids_yx": result.centroids_yx.astype(float).tolist(),
        "debug": result.debug,
    }


def input_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    return sorted(
        file
        for file in path.glob(pattern)
        if file.is_file() and file.suffix.lower() in FITS_SUFFIXES
    )


def output_path_for(source: Path, input_root: Path, output: Path, method: str, method_count: int) -> Path:
    if input_root.is_file() and method_count == 1 and output.suffix:
        return output
    if input_root.is_file():
        return output / f"{source.stem}_{method}.png"
    relative = source.relative_to(input_root)
    return output / relative.with_name(f"{relative.stem}_{method}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate star masks from FITS images with tetra3-like, SExtractor-like, or DAOFind-like methods.",
    )
    parser.add_argument("input", type=Path, help="Input FITS file or directory.")
    parser.add_argument("output", type=Path, help="Output PNG mask path or output directory.")
    parser.add_argument(
        "--method",
        choices=METHODS + ("all",),
        default="tetra3_like",
        help="Mask generation method.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively process FITS files in a directory.")
    parser.add_argument("--fit-channel-mode", default="mean", choices=("mean", "first", "max", "luma"))
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Detection threshold in robust noise sigmas. If omitted, each method uses its own default.",
    )
    parser.add_argument("--filtsize", type=int, default=25, help="Local background filter size.")
    parser.add_argument("--background-mode", default="local_mean")
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--max-area", type=int, default=100)
    parser.add_argument("--max-axis-ratio", type=float, default=None)
    parser.add_argument("--no-binary-open", action="store_true")

    parser.add_argument("--mesh-size", type=int, default=64, help="SExtractor-like background mesh size.")
    parser.add_argument("--mesh-filter-size", type=int, default=3)
    parser.add_argument("--filter-sigma", type=float, default=1.0, help="SExtractor-like detection filter sigma.")

    parser.add_argument("--fwhm", type=float, default=3.0, help="DAOFind-like Gaussian FWHM in pixels.")
    parser.add_argument("--peak-window", type=int, default=None)
    parser.add_argument("--mask-radius", type=float, default=None, help="DAOFind-like constant/base output disk radius.")
    parser.add_argument(
        "--radius-mode",
        choices=("gaussian", "linear", "sqrt", "log", "constant"),
        default="gaussian",
        help="DAOFind-like mapping from peak strength to output disk radius.",
    )
    parser.add_argument("--min-mask-radius", type=float, default=None)
    parser.add_argument("--max-mask-radius", type=float, default=None)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--min-separation", type=float, default=None)
    parser.add_argument("--max-peaks", type=int, default=None)
    parser.add_argument("--exclude-border", type=int, default=None)

    parser.add_argument("--write-json", action="store_true", help="Write sidecar JSON metadata next to each mask.")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    sources = input_files(args.input, recursive=args.recursive)
    if not sources:
        raise FileNotFoundError(f"no FITS files found: {args.input}")

    methods = list(METHODS) if args.method == "all" else [args.method]
    summaries = []
    for source in sources:
        image = read_gray_image(source, fit_channel_mode=args.fit_channel_mode)
        for method in methods:
            result = generate_mask(image, method, args)
            output = output_path_for(source, args.input, args.output, method, len(methods))
            write_mask(output, result.mask)
            summary = json_ready(result, source, output)
            summaries.append({k: v for k, v in summary.items() if k != "centroids_yx"})
            if args.write_json:
                output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"{source.name} [{method}] -> {output.name}: "
                f"{len(result.centroids_yx)} stars, mask_pixels={int(np.count_nonzero(result.mask))}"
            )

    if len(summaries) > 1:
        print(json.dumps({"processed": len(summaries), "results": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
