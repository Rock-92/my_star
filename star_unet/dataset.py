from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.mask_generator import FITS_SUFFIXES, read_gray_image


BITMAP_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGE_SUFFIXES = BITMAP_SUFFIXES | FITS_SUFFIXES
MASK_SUFFIXES = BITMAP_SUFFIXES


def _read_bitmap_gray(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _read_input_image(path: Path, fit_channel_mode: str) -> np.ndarray:
    if path.suffix.lower() in FITS_SUFFIXES:
        return read_gray_image(path, fit_channel_mode=fit_channel_mode).astype(np.float32)
    return _read_bitmap_gray(path).astype(np.float32)


def _read_mask_image(path: Path) -> np.ndarray:
    mask = _read_bitmap_gray(path)
    mask_float = mask.astype(np.float32)
    if np.issubdtype(mask.dtype, np.integer):
        dtype_max = float(np.iinfo(mask.dtype).max)
        if dtype_max > 0:
            mask_float /= dtype_max
    elif float(np.nanmax(mask_float)) > 1.0:
        mask_float /= 255.0
    return np.clip(np.nan_to_num(mask_float, nan=0.0), 0.0, 1.0)


def _normalize_image(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    mode = str(config.get("mode", "percentile")).lower()
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    if mode in {"none", "raw"}:
        return np.nan_to_num(image, nan=float(np.nanmedian(finite))).astype(np.float32)

    if mode == "minmax":
        low = float(np.min(finite))
        high = float(np.max(finite))
    elif mode == "percentile":
        low_percentile = float(config.get("lower_percentile", 0.5))
        high_percentile = float(config.get("upper_percentile", 99.8))
        low, high = np.percentile(finite, [low_percentile, high_percentile])
        low = float(low)
        high = float(high)
    else:
        raise ValueError(f"unsupported image normalization mode: {mode}")

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (np.nan_to_num(image, nan=low) - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _nested_config(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    return bool(config.get(key, default))


def _config_float(config: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(config.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _rotation_augmentation_angle(config: dict[str, Any]) -> float:
    if not config or not _config_bool(config, "enabled", False):
        return 0.0

    rotation_cfg = _nested_config(config, "rotation")
    rotation_enabled = _config_bool(rotation_cfg, "enabled", False)
    rotation_prob = _config_float(rotation_cfg, "prob", 1.0)
    rotation_degrees = _config_float(rotation_cfg, "degrees", 0.0)
    if rotation_enabled and rotation_degrees > 0.0 and np.random.random() < rotation_prob:
        return float(np.random.uniform(-rotation_degrees, rotation_degrees))
    return 0.0


def _rotate_pair(
    image: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    angle = _rotation_augmentation_angle(config)
    if abs(angle) < 1e-6:
        return image, mask

    height, width = image.shape
    center = ((width - 1) * 0.5, (height - 1) * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0).astype(np.float32)
    background = float(np.median(image))

    augmented_image = cv2.warpAffine(
        image.astype(np.float32),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )
    augmented_mask = cv2.warpAffine(
        mask.astype(np.float32),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return augmented_image.astype(np.float32), np.clip(augmented_mask, 0.0, 1.0).astype(np.float32)


def _crop_pair(
    image: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if not config or not _config_bool(config, "enabled", False):
        return image, mask

    crop_h = int(config.get("height", 1024))
    crop_w = int(config.get("width", 1024))
    if crop_h <= 0 or crop_w <= 0:
        return image, mask

    height, width = image.shape
    pad_h = max(0, crop_h - height)
    pad_w = max(0, crop_w - width)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=float(np.median(image)))
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
        height, width = image.shape

    mode = str(config.get("mode", "random")).lower()
    if mode == "center":
        y0 = max(0, (height - crop_h) // 2)
        x0 = max(0, (width - crop_w) // 2)
    elif mode == "random":
        y0 = int(np.random.randint(0, height - crop_h + 1)) if height > crop_h else 0
        x0 = int(np.random.randint(0, width - crop_w + 1)) if width > crop_w else 0
    else:
        raise ValueError(f"unsupported crop mode: {mode}")

    return image[y0 : y0 + crop_h, x0 : x0 + crop_w], mask[y0 : y0 + crop_h, x0 : x0 + crop_w]


class StarMapDataset(Dataset):
    """Dataset for image/mask segmentation pairs.

    Expected layout:

        root/
          images/ sample_0001.png
          masks/  sample_0001.png
    """

    def __init__(
        self,
        root: str | Path,
        augmentation: dict[str, Any] | None = None,
        fit_channel_mode: str = "mean",
        image_normalization: dict[str, Any] | None = None,
        crop: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.augmentation = augmentation or {}
        self.fit_channel_mode = fit_channel_mode
        self.image_normalization = image_normalization or {}
        self.crop = crop or {}

        if not self.image_dir.exists():
            raise FileNotFoundError(f"missing image directory: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"missing mask directory: {self.mask_dir}")

        self.samples: list[tuple[Path, Path]] = []
        for image_path in sorted(self.image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            mask_path = self._matching_file(self.mask_dir, image_path.stem)
            self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"no samples found under {self.image_dir}")

    @staticmethod
    def _matching_file(directory: Path, stem: str) -> Path:
        for suffix in MASK_SUFFIXES:
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"missing mask for sample {stem!r} under {directory}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.samples[index]

        image = _normalize_image(
            _read_input_image(image_path, self.fit_channel_mode),
            self.image_normalization,
        )
        mask = _read_mask_image(mask_path)
        image, mask = _crop_pair(image, mask, self.crop)
        image, mask = _rotate_pair(image, mask, self.augmentation)

        if image.shape != mask.shape:
            raise ValueError(f"shape mismatch for {image_path.name}: image={image.shape}, mask={mask.shape}")

        return {
            "image": torch.from_numpy(image[None, :, :].astype(np.float32)),
            "mask": torch.from_numpy(mask[None, :, :].astype(np.float32)),
            "name": image_path.name,
        }
