from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[4]
FITS_BLOCK_BYTES = 2880
FITS_SUFFIXES = {".fit", ".fits", ".fts"}


def resolve_data_path(data_root: Path, path_text: str | Path) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    if path.is_absolute():
        return path
    parts = list(path.parts)
    if data_root.name in parts:
        return data_root.joinpath(*parts[parts.index(data_root.name) + 1 :])
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return data_root / path


def load_manifest_rows(data_root: Path, split_reason: str, count: int = 0) -> list[dict[str, str]]:
    manifest = data_root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"missing manifest: {manifest}")
    rows: list[dict[str, str]] = []
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("split_reason") == split_reason:
                rows.append(row)
    if count and count > 0:
        rows = rows[: int(count)]
    if not rows:
        raise RuntimeError(f"no rows found for split_reason={split_reason!r} in {manifest}")
    return rows


def load_split_rows(split_csv: str | Path, split_name: str, count: int = 0) -> list[dict[str, str]]:
    split_path = Path(split_csv)
    if not split_path.exists():
        raise FileNotFoundError(f"missing split csv: {split_path}")
    rows: list[dict[str, str]] = []
    with split_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("split11") == split_name:
                rows.append(row)
    if count and count > 0:
        rows = rows[: int(count)]
    if not rows:
        raise RuntimeError(f"no rows found for split11={split_name!r} in {split_path}")
    return rows


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


def parse_fits_header(path: Path) -> tuple[dict[str, Any], int]:
    raw = path.read_bytes()
    cards: list[str] = []
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


def read_fits_image(path: Path, channel_mode: str = "mean") -> np.ndarray:
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
    axes = [int(header[f"NAXIS{axis_index}"]) for axis_index in range(1, naxis + 1)]
    count = int(np.prod(axes))
    dtype = np.dtype(dtype_by_bitpix[bitpix])
    data = np.frombuffer(raw[data_offset : data_offset + count * dtype.itemsize], dtype=dtype)
    if data.size != count:
        raise ValueError(f"FITS data is shorter than header dimensions require: {path}")
    image = data.astype(np.float32)
    image = image * float(header.get("BSCALE", 1.0)) + float(header.get("BZERO", 0.0))
    image = image.reshape(tuple(reversed(axes)))
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3:
        mode = str(channel_mode).lower()
        if mode == "first":
            return image[0].astype(np.float32)
        if mode == "max":
            return image.max(axis=0).astype(np.float32)
        if mode == "luma" and image.shape[0] == 3:
            weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
            return np.tensordot(weights, image, axes=(0, 0)).astype(np.float32)
        if mode in {"mean", "luma"}:
            return image.mean(axis=0, dtype=np.float32)
    raise ValueError(f"cannot convert FITS image shape {image.shape} to grayscale: {path}")


def read_gray_image(path: Path, fit_channel_mode: str = "mean") -> np.ndarray:
    if path.suffix.lower() in FITS_SUFFIXES:
        return read_fits_image(path, channel_mode=fit_channel_mode)
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def read_mask_image(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    max_value = float(np.max(mask)) if mask.size else 0.0
    if max_value > 1.0:
        mask = mask / 255.0
    return np.clip(np.nan_to_num(mask, nan=0.0), 0.0, 1.0).astype(np.float32)


def normalize_image(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    mode = str(config.get("mode", "percentile")).lower()
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    if mode in {"none", "raw"}:
        fill = float(np.nanmedian(finite))
        return np.nan_to_num(image, nan=fill).astype(np.float32)
    if mode == "minmax":
        low = float(np.min(finite))
        high = float(np.max(finite))
    elif mode == "percentile":
        low_p = float(config.get("lower_percentile", 0.5))
        high_p = float(config.get("upper_percentile", 99.8))
        low, high = np.percentile(finite, [low_p, high_p])
        low = float(low)
        high = float(high)
    else:
        raise ValueError(f"unsupported image normalization mode: {mode}")
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (np.nan_to_num(image, nan=low) - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def heatmap_to_centroids(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    min_distance: float = 3.0,
    max_peaks: int | None = None,
) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2 or heatmap.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    window = max(3, int(round(float(min_distance) * 2.0 + 1.0)))
    if window % 2 == 0:
        window += 1
    local_max = heatmap == ndi.maximum_filter(heatmap, size=window)
    candidates = np.argwhere(local_max & (heatmap >= float(threshold)))
    if not len(candidates):
        return np.empty((0, 2), dtype=np.float32)
    scores = heatmap[candidates[:, 0], candidates[:, 1]]
    candidates = candidates[np.argsort(-scores)]
    selected: list[tuple[float, float]] = []
    min_sep2 = float(min_distance) ** 2
    for y, x in candidates:
        cy = float(y) + 0.5
        cx = float(x) + 0.5
        if any((cy - py) ** 2 + (cx - px) ** 2 < min_sep2 for py, px in selected):
            continue
        selected.append((cy, cx))
        if max_peaks is not None and len(selected) >= max_peaks:
            break
    return np.asarray(selected, dtype=np.float32).reshape((-1, 2))


def centroids_to_delta(shape: tuple[int, int], centroids_yx: np.ndarray) -> np.ndarray:
    height, width = shape
    delta = np.zeros((height, width), dtype=np.float32)
    for y, x in np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2)):
        iy = int(round(float(y) - 0.5))
        ix = int(round(float(x) - 0.5))
        if 0 <= iy < height and 0 <= ix < width:
            delta[iy, ix] = 1.0
    return delta


def deepsource_target_from_delta(
    delta: np.ndarray,
    mode: str = "deepsource",
    gaussian_sigma: float = 1.2,
    triangle_radius: float = 6.0,
    background_level: float = 0.05,
    alpha: float = 0.75,
) -> np.ndarray:
    mode = str(mode).lower()
    delta = np.asarray(delta, dtype=np.float32)
    if mode == "delta":
        return delta
    if mode == "gaussian":
        target = ndi.gaussian_filter(delta, sigma=float(gaussian_sigma))
        max_value = float(np.max(target))
        return (target / max_value).astype(np.float32) if max_value > 0 else target.astype(np.float32)
    if mode != "deepsource":
        raise ValueError(f"unsupported target mode: {mode}")

    if not np.any(delta > 0):
        return np.zeros_like(delta, dtype=np.float32)
    distance = ndi.distance_transform_edt(delta <= 0)
    triangle = np.clip(1.0 - distance / max(float(triangle_radius), 1e-6), 0.0, 1.0)
    smooth = ndi.gaussian_filter(triangle.astype(np.float32), sigma=float(gaussian_sigma))
    max_value = float(np.max(smooth))
    if max_value > 0:
        smooth = smooth / max_value
    # DeepSource weights source demand by intensity^alpha and keeps a faint
    # background term. Here source intensities are unavailable, so delta peaks
    # have unit intensity and the background term prevents an all-zero bias.
    target = (smooth + float(background_level) * delta) ** float(alpha)
    return np.clip(target, 0.0, 1.0).astype(np.float32)


def transform_stack_centroids_to_single(centroids_yx: np.ndarray, row: dict[str, str]) -> np.ndarray:
    centroids = np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2))
    if centroids.size == 0:
        return centroids
    matrix = np.asarray(
        [
            [float(row.get("label_transform_a") or 1.0), float(row.get("label_transform_b") or 0.0)],
            [float(row.get("label_transform_c") or 0.0), float(row.get("label_transform_d") or 1.0)],
        ],
        dtype=np.float32,
    )
    shift = np.asarray(
        [float(row.get("label_shift_x_px") or 0.0), float(row.get("label_shift_y_px") or 0.0)],
        dtype=np.float32,
    )
    xy = centroids[:, [1, 0]]
    transformed_xy = xy @ matrix.T + shift[None, :]
    return transformed_xy[:, [1, 0]].astype(np.float32)


def estimate_stack_photometry(
    stack_image: np.ndarray,
    centroids_yx: np.ndarray,
    aperture_radius: float = 5.0,
    annulus_inner: float = 8.0,
    annulus_outer: float = 14.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(stack_image, dtype=np.float32)
    centroids = np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2))
    fluxes = np.zeros(len(centroids), dtype=np.float32)
    fwhms = np.full(len(centroids), 3.0, dtype=np.float32)
    mags = np.full(len(centroids), np.inf, dtype=np.float32)
    if image.ndim != 2 or not len(centroids):
        return fluxes, fwhms, mags

    height, width = image.shape
    outer = max(float(annulus_outer), float(aperture_radius) + 1.0)
    radius_px = max(2, int(np.ceil(outer)))
    aperture2 = float(aperture_radius) ** 2
    annulus_inner2 = float(annulus_inner) ** 2
    annulus_outer2 = outer**2
    for idx, (cy, cx) in enumerate(centroids):
        y0 = max(0, int(np.floor(float(cy))) - radius_px)
        y1 = min(height, int(np.floor(float(cy))) + radius_px + 1)
        x0 = max(0, int(np.floor(float(cx))) - radius_px)
        x1 = min(width, int(np.floor(float(cx))) + radius_px + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        patch = image[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        rr2 = (yy + 0.5 - float(cy)) ** 2 + (xx + 0.5 - float(cx)) ** 2
        annulus = (rr2 >= annulus_inner2) & (rr2 <= annulus_outer2)
        finite_patch = patch[np.isfinite(patch)]
        if not finite_patch.size:
            continue
        if np.any(annulus):
            finite_annulus = patch[annulus & np.isfinite(patch)]
            background = float(np.median(finite_annulus)) if finite_annulus.size else float(np.median(finite_patch))
        else:
            background = float(np.median(finite_patch))
        aperture = rr2 <= aperture2
        signal = np.maximum(np.nan_to_num(patch, nan=background) - background, 0.0)
        weights = signal * aperture
        flux = float(np.sum(weights))
        if flux <= 0.0 or not np.isfinite(flux):
            continue
        sigma = float(np.sqrt(max(np.sum(weights * rr2) / (2.0 * flux), 1e-6)))
        fluxes[idx] = flux
        fwhms[idx] = np.float32(np.clip(2.355 * sigma, 1.5, 12.0))
        mags[idx] = np.float32(-2.5 * np.log10(max(flux, 1e-6)))
    return fluxes, fwhms, mags


def target_weights_from_photometry(
    fluxes: np.ndarray,
    fwhms: np.ndarray,
    min_amplitude: float = 0.25,
    max_amplitude: float = 1.0,
    min_radius: float = 2.0,
    max_radius: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flux = np.asarray(fluxes, dtype=np.float32)
    fwhm = np.asarray(fwhms, dtype=np.float32)
    valid = np.isfinite(flux) & (flux > 0.0)
    brightness = np.zeros_like(flux, dtype=np.float32)
    mags = np.full_like(flux, np.inf, dtype=np.float32)
    if np.any(valid):
        log_flux = np.log10(np.maximum(flux[valid], 1e-6))
        low, high = np.percentile(log_flux, [10.0, 95.0])
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        brightness[valid] = np.clip((log_flux - low) / (high - low), 0.0, 1.0)
        mags[valid] = -2.5 * log_flux
    amp = float(min_amplitude) + (float(max_amplitude) - float(min_amplitude)) * brightness
    size_norm = np.clip((fwhm - 1.5) / max(float(max_radius) - 1.5, 1e-6), 0.0, 1.0)
    radius_brightness = float(min_radius) + (float(max_radius) - float(min_radius)) * brightness
    radius_size = float(min_radius) + (float(max_radius) - float(min_radius)) * size_norm
    radius = 0.65 * radius_brightness + 0.35 * radius_size
    radius = np.clip(radius, float(min_radius), float(max_radius))
    amp[~valid] = float(min_amplitude)
    radius[~valid] = float(min_radius)
    return amp.astype(np.float32), radius.astype(np.float32), mags.astype(np.float32)


def variable_gaussian_target(
    shape: tuple[int, int],
    centroids_yx: np.ndarray,
    amplitudes: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    height, width = shape
    target = np.zeros((height, width), dtype=np.float32)
    for (cy, cx), amplitude, radius in zip(
        np.asarray(centroids_yx, dtype=np.float32).reshape((-1, 2)),
        np.asarray(amplitudes, dtype=np.float32).reshape((-1,)),
        np.asarray(radii, dtype=np.float32).reshape((-1,)),
    ):
        if not (np.isfinite(cy) and np.isfinite(cx) and np.isfinite(radius)):
            continue
        if cx < 0 or cy < 0 or cx >= width or cy >= height:
            continue
        radius = float(np.clip(radius, 1.0, 32.0))
        extent = max(2, int(np.ceil(radius * 2.0)))
        y0 = max(0, int(np.floor(cy)) - extent)
        y1 = min(height, int(np.floor(cy)) + extent + 1)
        x0 = max(0, int(np.floor(cx)) - extent)
        x1 = min(width, int(np.floor(cx)) + extent + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        rr2 = (yy + 0.5 - float(cy)) ** 2 + (xx + 0.5 - float(cx)) ** 2
        sigma = radius / np.sqrt(-2.0 * np.log(0.05))
        blob = float(amplitude) * np.exp(-0.5 * rr2 / max(sigma**2, 1e-6))
        target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], blob.astype(np.float32))
    return np.clip(target, 0.0, 1.0).astype(np.float32)


class DeepSourceStarDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split_reason: str,
        count: int = 0,
        split_csv: str | Path | None = None,
        split_name: str | None = None,
        crop_size: int = 512,
        crops_per_image: int = 4,
        fit_channel_mode: str = "mean",
        image_normalization: dict[str, Any] | None = None,
        mask_threshold: float = 0.5,
        min_distance: float = 3.0,
        target_mode: str = "deepsource",
        gaussian_sigma: float = 1.2,
        triangle_radius: float = 6.0,
        background_level: float = 0.05,
        alpha: float = 0.75,
        target_weighting: str = "none",
        target_min_amplitude: float = 0.25,
        target_max_amplitude: float = 1.0,
        target_min_radius: float = 2.0,
        target_max_radius: float = 8.0,
        target_aperture_radius: float = 5.0,
        target_annulus_inner: float = 8.0,
        target_annulus_outer: float = 14.0,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root)
        if split_csv is not None:
            self.rows = load_split_rows(split_csv, split_name or split_reason, count=count)
        else:
            self.rows = load_manifest_rows(self.data_root, split_reason=split_reason, count=count)
        self.crop_size = int(crop_size)
        self.crops_per_image = max(1, int(crops_per_image))
        self.fit_channel_mode = fit_channel_mode
        self.image_normalization = image_normalization or {
            "mode": "percentile",
            "lower_percentile": 0.5,
            "upper_percentile": 99.8,
        }
        self.mask_threshold = float(mask_threshold)
        self.min_distance = float(min_distance)
        self.target_mode = target_mode
        self.gaussian_sigma = float(gaussian_sigma)
        self.triangle_radius = float(triangle_radius)
        self.background_level = float(background_level)
        self.alpha = float(alpha)
        self.target_weighting = str(target_weighting).lower()
        self.target_min_amplitude = float(target_min_amplitude)
        self.target_max_amplitude = float(target_max_amplitude)
        self.target_min_radius = float(target_min_radius)
        self.target_max_radius = float(target_max_radius)
        self.target_aperture_radius = float(target_aperture_radius)
        self.target_annulus_inner = float(target_annulus_inner)
        self.target_annulus_outer = float(target_annulus_outer)
        self.seed = int(seed)
        self._missing_stack_warning_printed = False

    def __len__(self) -> int:
        return len(self.rows) * self.crops_per_image

    def _load_pair(self, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        image_path = resolve_data_path(self.data_root, row.get("image_out") or row.get("single_fits") or "")
        mask_path = resolve_data_path(self.data_root, row.get("mask_out") or "")
        raw = read_gray_image(image_path, self.fit_channel_mode)
        image = normalize_image(raw, self.image_normalization)
        if self.target_weighting == "stack_photometry":
            weighted = self._load_stack_photometry_target(row, image.shape)
            if weighted is not None:
                return image.astype(np.float32), weighted.astype(np.float32)
        mask = read_mask_image(mask_path)
        centroids = heatmap_to_centroids(mask, threshold=self.mask_threshold, min_distance=self.min_distance)
        delta = centroids_to_delta(image.shape, centroids)
        target = deepsource_target_from_delta(
            delta,
            mode=self.target_mode,
            gaussian_sigma=self.gaussian_sigma,
            triangle_radius=self.triangle_radius,
            background_level=self.background_level,
            alpha=self.alpha,
        )
        return image.astype(np.float32), target.astype(np.float32)

    def _load_stack_photometry_target(self, row: dict[str, str], image_shape: tuple[int, int]) -> np.ndarray | None:
        stack_mask_text = row.get("stack_mask") or ""
        if not stack_mask_text:
            return None
        stack_mask_path = resolve_data_path(self.data_root, stack_mask_text)
        if not stack_mask_path.exists():
            return None
        stack_mask = read_mask_image(stack_mask_path)
        stack_centroids = heatmap_to_centroids(
            stack_mask,
            threshold=self.mask_threshold,
            min_distance=self.min_distance,
        )
        if not len(stack_centroids):
            return np.zeros(image_shape, dtype=np.float32)

        stack_path = resolve_data_path(self.data_root, row.get("stack_fits") or "")
        if stack_path.exists():
            stack_image = read_gray_image(stack_path, self.fit_channel_mode)
            fluxes, fwhms, _ = estimate_stack_photometry(
                stack_image,
                stack_centroids,
                aperture_radius=self.target_aperture_radius,
                annulus_inner=self.target_annulus_inner,
                annulus_outer=self.target_annulus_outer,
            )
        else:
            if not self._missing_stack_warning_printed:
                print(
                    f"[warn] stack_fits not found, using stack_mask-only target sizing: {stack_path}",
                    flush=True,
                )
                self._missing_stack_warning_printed = True
            fluxes = stack_mask[
                np.clip(np.round(stack_centroids[:, 0] - 0.5).astype(int), 0, stack_mask.shape[0] - 1),
                np.clip(np.round(stack_centroids[:, 1] - 0.5).astype(int), 0, stack_mask.shape[1] - 1),
            ].astype(np.float32)
            fwhms = np.full(len(stack_centroids), self.triangle_radius, dtype=np.float32)

        single_centroids = transform_stack_centroids_to_single(stack_centroids, row)
        keep = (
            (single_centroids[:, 0] >= 0.0)
            & (single_centroids[:, 0] < image_shape[0])
            & (single_centroids[:, 1] >= 0.0)
            & (single_centroids[:, 1] < image_shape[1])
        )
        if not np.any(keep):
            return np.zeros(image_shape, dtype=np.float32)
        amplitudes, radii, _ = target_weights_from_photometry(
            fluxes[keep],
            fwhms[keep],
            min_amplitude=self.target_min_amplitude,
            max_amplitude=self.target_max_amplitude,
            min_radius=self.target_min_radius,
            max_radius=self.target_max_radius,
        )
        return variable_gaussian_target(image_shape, single_centroids[keep], amplitudes, radii)

    def _crop(self, image: np.ndarray, target: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.crop_size <= 0:
            return image, target
        crop = int(self.crop_size)
        height, width = image.shape
        pad_h = max(0, crop - height)
        pad_w = max(0, crop - width)
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=float(np.median(image)))
            target = np.pad(target, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
            height, width = image.shape
        rng = np.random.default_rng(self.seed + int(index) * 1009)
        # Half of the crops are centered on a positive target when possible.
        positive = np.argwhere(target > 0.5)
        if len(positive) and index % 2 == 0:
            y, x = positive[int(rng.integers(0, len(positive)))]
            y0 = int(np.clip(y - crop // 2, 0, max(0, height - crop)))
            x0 = int(np.clip(x - crop // 2, 0, max(0, width - crop)))
        else:
            y0 = int(rng.integers(0, height - crop + 1)) if height > crop else 0
            x0 = int(rng.integers(0, width - crop + 1)) if width > crop else 0
        return image[y0 : y0 + crop, x0 : x0 + crop], target[y0 : y0 + crop, x0 : x0 + crop]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index // self.crops_per_image]
        image, target = self._load_pair(row)
        image, target = self._crop(image, target, index)
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "target": torch.from_numpy(target[None].astype(np.float32)),
            "sample_id": row["sample_id"],
        }


class PrecomputedCropDataset(Dataset):
    def __init__(self, crop_data_dir: str | Path, split_name: str) -> None:
        self.path = Path(crop_data_dir) / f"{split_name}.npz"
        if not self.path.exists():
            raise FileNotFoundError(f"missing precomputed crop file: {self.path}")
        with np.load(self.path, allow_pickle=False) as data:
            self.images = data["images"].astype(np.float32)
            self.targets = data["targets"].astype(np.float32)
            self.sample_ids = data["sample_ids"].astype(str)
        if self.images.shape != self.targets.shape:
            raise ValueError(f"image/target shape mismatch in {self.path}: {self.images.shape} vs {self.targets.shape}")
        if self.images.ndim != 4 or self.images.shape[1] != 1:
            raise ValueError(f"expected images shaped [N, 1, H, W], got {self.images.shape}")

    @property
    def rows(self) -> list[dict[str, str]]:
        return [{"sample_id": sample_id} for sample_id in sorted(set(self.sample_ids.tolist()))]

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "image": torch.from_numpy(self.images[index]),
            "target": torch.from_numpy(self.targets[index]),
            "sample_id": str(self.sample_ids[index]),
        }
