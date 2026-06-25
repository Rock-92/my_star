from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from PIL import Image

# 训练集比例
DEFAULT_TRAIN_RATIO = 1.0
# 默认亮星
DEFAULT_RADIUS_BRIGHT_MAG = 6.0
# 暗星最大像素半径
DEFAULT_UNDETECTED_MASK_MAX_RADIUS_PX = 1.0


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tetra3 import Tetra3  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class GaiaCatalog:
    source_id: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    mag: np.ndarray


@dataclass
class AdaptiveStarMask:
    pixels: np.ndarray
    center: np.ndarray
    radius_px: float
    peak_snr: float


@dataclass
class MaskRadiusModel:
    fitted: bool
    slope: float | None
    intercept: float | None
    source_count: int

    def predict(self, mag: float, args: argparse.Namespace) -> float:
        prior = prior_mask_radius(mag, args)
        if not self.fitted or self.slope is None or self.intercept is None:
            return prior
        radius2 = self.slope * float(mag) + self.intercept
        if not np.isfinite(radius2):
            return prior
        radius2 = float(np.clip(radius2, args.mask_min_radius_px**2, args.mask_radius_px**2))
        return float(math.sqrt(radius2))

    def as_dict(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "slope": self.slope,
            "intercept": self.intercept,
            "source_count": self.source_count,
            "fit_target": "radius_px_squared",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve star images with tetra3, project a local Gaia catalogue into each image, "
            "and generate centroid CSV + segmentation mask labels."
        )
    )
    parser.add_argument("--image-dir", type=Path, default=None, help="Legacy mode: label images already in this folder.")
    parser.add_argument("--raw-image-dir", type=Path, default=Path("dataset/raw_images"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Random fraction of raw images assigned to train.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sample-prefix", default="sample")
    parser.add_argument("--sample-digits", type=int, default=4)
    parser.add_argument("--output-image-ext", default=".png", help="Output image suffix, default .png.")
    parser.add_argument("--gaia-csv", type=Path, default=Path("preprocessing/gaia_data/gaia16.csv"))
    parser.add_argument("--centroid-dir", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--mag-limit", type=float, default=16.0)
    parser.add_argument("--tetra3-db", default="tyc_main3-40")
    parser.add_argument("--fov-estimate", type=float, default=None, help="Optional horizontal FOV estimate in degrees.")
    parser.add_argument("--fov-max-error", type=float, default=None, help="Optional allowed FOV error in degrees.")
    parser.add_argument("--solve-timeout-ms", type=float, default=15000.0)
    parser.add_argument("--pattern-checking-stars", type=int, default=12)
    parser.add_argument("--match-radius", type=float, default=0.01)
    parser.add_argument(
        "--radius-bright-mag",
        type=float,
        default=DEFAULT_RADIUS_BRIGHT_MAG,
        help="Magnitude treated as the bright end when scaling radii.",
    )
    parser.add_argument(
        "--mask-min-radius-px",
        type=float,
        default=0.5,
        help="Smallest fallback disk radius for the faintest catalogue stars.",
    )
    parser.add_argument(
        "--mask-radius-px",
        type=float,
        default=2.0,
        help="Largest fallback disk radius for bright catalogue stars.",
    )
    parser.add_argument(
        "--undetected-mask-max-radius-px",
        type=float,
        default=DEFAULT_UNDETECTED_MASK_MAX_RADIUS_PX,
        help="Upper radius cap for catalogue fallback stars that local detection cannot see.",
    )
    parser.add_argument(
        "--mask-radius-fit-min-stars",
        type=int,
        default=3,
        help="Minimum adaptive detections needed to fit magnitude-to-mask-radius relation.",
    )
    parser.add_argument(
        "--adaptive-window-min-radius-px",
        type=float,
        default=4.0,
        help="Local search half-window radius for faint stars.",
    )
    parser.add_argument(
        "--adaptive-window-radius-px",
        type=float,
        default=8.0,
        help="Local search half-window radius for bright stars.",
    )
    parser.add_argument("--adaptive-threshold-sigma", type=float, default=2.5)
    parser.add_argument("--adaptive-peak-fraction", type=float, default=0.25)
    parser.add_argument(
        "--adaptive-min-radius-px",
        type=float,
        default=1.0,
        help="Maximum connected-component radius retained for faint stars.",
    )
    parser.add_argument(
        "--adaptive-max-radius-px",
        type=float,
        default=6.0,
        help="Maximum connected-component radius retained for bright stars.",
    )
    parser.add_argument(
        "--snap-min-radius-px",
        type=float,
        default=1.5,
        help="Peak-to-catalogue snap radius for faint stars.",
    )
    parser.add_argument(
        "--snap-radius-px",
        type=float,
        default=4.0,
        help="Peak-to-catalogue snap radius for bright stars.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("hybrid", "adaptive", "fixed"),
        default="hybrid",
        help="hybrid uses adaptive local foreground when visible, otherwise a fixed disk.",
    )
    parser.add_argument("--no-snap-centroid", action="store_true", help="Keep centroid CSV at Gaia-projected positions.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def load_gaia_catalog(path: Path, mag_limit: float) -> GaiaCatalog:
    if not path.exists():
        raise FileNotFoundError(f"Gaia catalogue not found: {path}")

    source_ids: list[str] = []
    ras: list[float] = []
    decs: list[float] = []
    mags: list[float] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Gaia CSV has no header: {path}")
        fields = {name.lower(): name for name in reader.fieldnames}
        ra_field = first_present(fields, ("ra", "raj2000", "ra_icrs"))
        dec_field = first_present(fields, ("dec", "dej2000", "de_icrs"))
        mag_field = first_present(fields, ("phot_g_mean_mag", "gmag", "mag", "magnitude"))
        id_field = first_present(fields, ("source_id", "sourceid", "id"), required=False)
        if not ra_field or not dec_field or not mag_field:
            raise ValueError(
                "Gaia CSV must contain RA, Dec and magnitude columns. "
                f"Found columns: {reader.fieldnames}"
            )

        for index, row in enumerate(reader):
            try:
                mag = float(row[mag_field])
                if mag > mag_limit:
                    continue
                ra = float(row[ra_field]) % 360.0
                dec = float(row[dec_field])
            except (TypeError, ValueError):
                continue
            ras.append(ra)
            decs.append(dec)
            mags.append(mag)
            source_ids.append(row[id_field] if id_field else str(index))

    if not ras:
        raise RuntimeError(f"No Gaia stars with magnitude <= {mag_limit} found in {path}")

    catalog = GaiaCatalog(
        source_id=np.asarray(source_ids, dtype=object),
        ra=np.asarray(ras, dtype=np.float64),
        dec=np.asarray(decs, dtype=np.float64),
        mag=np.asarray(mags, dtype=np.float32),
    )
    log(
        f"Loaded Gaia catalogue: {len(catalog.ra)} stars <= {mag_limit:.2f}; "
        f"catalogue max kept mag={float(np.max(catalog.mag)):.2f}"
    )
    if float(np.max(catalog.mag)) + 0.05 < mag_limit:
        log(
            f"WARNING: this Gaia file only reaches G~{float(np.max(catalog.mag)):.2f}, "
            f"so it cannot label all stars down to G={mag_limit:.2f}."
        )
    return catalog


def first_present(fields: dict[str, str], names: tuple[str, ...], required: bool = True) -> str | None:
    for name in names:
        if name.lower() in fields:
            return fields[name.lower()]
    if required:
        return None
    return None


def image_paths(image_dir: Path, limit: int | None) -> list[Path]:
    paths = [path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]
    return paths[:limit] if limit is not None else paths


def resolve_raw_image_dir(path: Path) -> Path:
    if path.exists():
        return path
    # Accept the singular spelling mentioned in conversation.
    sibling = path.parent / "raw_image"
    if path.name == "raw_images" and sibling.exists():
        return sibling
    raise FileNotFoundError(f"raw image directory not found: {path}")


def split_raw_images(paths: list[Path], train_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    shuffled = list(paths)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    train_count = int(round(len(shuffled) * train_ratio))
    train_count = min(len(shuffled), max(0, train_count))
    return shuffled[:train_count], shuffled[train_count:]


def sample_name(index: int, args: argparse.Namespace) -> str:
    return f"{args.sample_prefix}_{index:0{args.sample_digits}d}"


def normalized_output_suffix(args: argparse.Namespace) -> str:
    suffix = args.output_image_ext.strip()
    if not suffix:
        suffix = ".png"
    if not suffix.startswith("."):
        suffix = "." + suffix
    return suffix.lower()


def read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        return image
    try:
        return np.asarray(Image.open(path).convert("L"))
    except Exception as exc:
        raise FileNotFoundError(f"Could not read image: {path}") from exc


def solve_image(t3: Tetra3, path: Path, args: argparse.Namespace) -> dict[str, Any]:
    image = Image.open(path)
    return t3.solve_from_image(
        image,
        fov_estimate=args.fov_estimate,
        fov_max_error=args.fov_max_error,
        pattern_checking_stars=args.pattern_checking_stars,
        match_radius=args.match_radius,
        solve_timeout=args.solve_timeout_ms,
        return_matches=True,
    )


def build_wcs_from_solution(solution: dict[str, Any], image_shape: tuple[int, int]) -> WCS:
    matched_stars = np.asarray(solution.get("matched_stars", []), dtype=float)
    matched_centroids = np.asarray(solution.get("matched_centroids", []), dtype=float)
    if matched_stars.ndim != 2 or matched_centroids.ndim != 2 or len(matched_stars) < 4:
        raise RuntimeError("tetra3 solution has too few matched stars to fit WCS")

    # tetra3 centroids are y,x; astropy fit_wcs_from_points expects x,y.
    x = matched_centroids[:, 1]
    y = matched_centroids[:, 0]
    sky = SkyCoord(ra=matched_stars[:, 0] * u.deg, dec=matched_stars[:, 1] * u.deg, frame="icrs")
    wcs = fit_wcs_from_points((x, y), sky, projection="TAN")
    wcs.array_shape = image_shape
    return wcs


def angular_prefilter(catalog: GaiaCatalog, ra0: float, dec0: float, radius_deg: float) -> np.ndarray:
    dec_margin = radius_deg
    dec_keep = (catalog.dec >= dec0 - dec_margin) & (catalog.dec <= dec0 + dec_margin)
    cos_dec = max(0.1, abs(math.cos(math.radians(dec0))))
    ra_margin = min(180.0, radius_deg / cos_dec)
    dra = ((catalog.ra - ra0 + 180.0) % 360.0) - 180.0
    ra_keep = np.abs(dra) <= ra_margin
    return dec_keep & ra_keep


def project_gaia_to_image(
    catalog: GaiaCatalog,
    wcs: WCS,
    solution: dict[str, Any],
    image_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    height, width = image_shape
    fov = float(solution["FOV"])
    diagonal_fov = fov * math.sqrt(width * width + height * height) / width
    radius_deg = diagonal_fov / 2.0 + 1.0
    pre = angular_prefilter(catalog, float(solution["RA"]), float(solution["Dec"]), radius_deg)
    if not np.any(pre):
        return empty_projection()

    candidate_indices = np.flatnonzero(pre)
    world = np.column_stack((catalog.ra[candidate_indices], catalog.dec[candidate_indices]))
    xy = wcs.all_world2pix(world, 0)
    x = xy[:, 0]
    y = xy[:, 1]
    keep = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x < width) & (y >= 0.0) & (y < height)
    selected = candidate_indices[keep]
    return {
        "source_id": catalog.source_id[selected],
        "ra": catalog.ra[selected],
        "dec": catalog.dec[selected],
        "mag": catalog.mag[selected],
        "x": x[keep].astype(np.float32),
        "y": y[keep].astype(np.float32),
    }


def empty_projection() -> dict[str, np.ndarray]:
    return {
        "source_id": np.asarray([], dtype=object),
        "ra": np.asarray([], dtype=np.float64),
        "dec": np.asarray([], dtype=np.float64),
        "mag": np.asarray([], dtype=np.float32),
        "x": np.asarray([], dtype=np.float32),
        "y": np.asarray([], dtype=np.float32),
    }


def magnitude_scale(mag: float, args: argparse.Namespace) -> float:
    """Return 1 for bright stars and 0 near the selected magnitude limit."""

    faint_mag = float(args.mag_limit)
    bright_mag = min(float(args.radius_bright_mag), faint_mag - 1e-3)
    scale = (faint_mag - float(mag)) / max(faint_mag - bright_mag, 1e-6)
    return float(np.clip(scale, 0.0, 1.0))


def scaled_radius(mag: float, args: argparse.Namespace, faint: float, bright: float) -> float:
    scale = magnitude_scale(mag, args)
    return float(faint + scale * (bright - faint))


def adaptive_window_radius(mag: float, args: argparse.Namespace) -> int:
    radius = scaled_radius(
        mag,
        args,
        float(args.adaptive_window_min_radius_px),
        float(args.adaptive_window_radius_px),
    )
    return max(1, int(math.ceil(radius)))


def snap_radius(mag: float, args: argparse.Namespace) -> float:
    return scaled_radius(mag, args, float(args.snap_min_radius_px), float(args.snap_radius_px))


def adaptive_component_radius(mag: float, args: argparse.Namespace) -> float:
    return scaled_radius(mag, args, float(args.adaptive_min_radius_px), float(args.adaptive_max_radius_px))


def prior_mask_radius(mag: float, args: argparse.Namespace) -> float:
    return scaled_radius(mag, args, float(args.mask_min_radius_px), float(args.mask_radius_px))


def cap_undetected_mask_radius(radius: float, args: argparse.Namespace) -> float:
    cap = float(args.undetected_mask_max_radius_px)
    if not np.isfinite(cap) or cap <= 0:
        return float(radius)
    return float(min(float(radius), cap))


def fit_mask_radius_model(
    mags: list[float],
    radii: list[float],
    args: argparse.Namespace,
) -> MaskRadiusModel:
    finite = [
        (float(mag), float(radius))
        for mag, radius in zip(mags, radii)
        if np.isfinite(mag) and np.isfinite(radius) and radius > 0
    ]
    if len(finite) < int(args.mask_radius_fit_min_stars):
        return MaskRadiusModel(fitted=False, slope=None, intercept=None, source_count=len(finite))

    x = np.asarray([item[0] for item in finite], dtype=np.float64)
    y = np.asarray([item[1] ** 2 for item in finite], dtype=np.float64)
    if len(np.unique(np.round(x, 3))) < 2:
        return MaskRadiusModel(fitted=False, slope=None, intercept=None, source_count=len(finite))

    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope > 0:
        return MaskRadiusModel(fitted=False, slope=None, intercept=None, source_count=len(finite))
    return MaskRadiusModel(
        fitted=True,
        slope=float(slope),
        intercept=float(intercept),
        source_count=len(finite),
    )


def generate_mask_and_centroids(
    image: np.ndarray,
    projection: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mask = np.zeros(image.shape, dtype=np.uint8)
    adjusted_xy = np.column_stack((projection["x"], projection["y"])).astype(np.float32)
    mask_methods = np.asarray(["none"] * len(adjusted_xy), dtype=object)
    mask_radii = np.zeros(len(adjusted_xy), dtype=np.float32)
    adaptive_results: list[AdaptiveStarMask | None] = [None] * len(adjusted_xy)

    for idx, (x, y, mag) in enumerate(zip(projection["x"], projection["y"], projection["mag"])):
        if args.mask_mode == "fixed":
            continue

        adaptive = adaptive_star_mask(image, float(x), float(y), float(mag), args)
        if adaptive is not None:
            adaptive_results[idx] = adaptive
            mask[adaptive.pixels[:, 0], adaptive.pixels[:, 1]] = 255
            if not args.no_snap_centroid:
                adjusted_xy[idx] = adaptive.center
            mask_methods[idx] = "adaptive"
            mask_radii[idx] = adaptive.radius_px

    detected_mags = [
        float(mag)
        for mag, result in zip(projection["mag"], adaptive_results)
        if result is not None
    ]
    detected_radii = [float(result.radius_px) for result in adaptive_results if result is not None]
    radius_model = fit_mask_radius_model(detected_mags, detected_radii, args)

    for idx, (x, y, mag) in enumerate(zip(projection["x"], projection["y"], projection["mag"])):
        if mask_methods[idx] == "adaptive":
            continue
        if args.mask_mode not in {"hybrid", "fixed"}:
            continue

        radius = cap_undetected_mask_radius(radius_model.predict(float(mag), args), args)
        draw_disk(mask, float(x), float(y), radius, 255)
        mask_radii[idx] = radius
        mask_methods[idx] = "fixed" if args.mask_mode == "fixed" else "fallback"

    radius_info = radius_model.as_dict()
    if detected_radii:
        radius_info.update(
            {
                "detected_radius_min": float(np.min(detected_radii)),
                "detected_radius_max": float(np.max(detected_radii)),
                "detected_radius_mean": float(np.mean(detected_radii)),
            }
        )
    return mask, adjusted_xy, mask_methods, mask_radii, radius_info


def draw_disk(mask: np.ndarray, x: float, y: float, radius: float, value: int) -> None:
    height, width = mask.shape
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or xi >= width or yi < 0 or yi >= height:
        return
    if radius < 1.0:
        mask[yi, xi] = int(value)
        return
    cv2.circle(mask, (xi, yi), max(1, int(round(radius))), int(value), thickness=-1)


def adaptive_star_mask(
    image: np.ndarray,
    x: float,
    y: float,
    mag: float,
    args: argparse.Namespace,
) -> AdaptiveStarMask | None:
    height, width = image.shape
    xi = int(round(x))
    yi = int(round(y))
    radius = adaptive_window_radius(mag, args)
    x0 = max(0, xi - radius)
    x1 = min(width, xi + radius + 1)
    y0 = max(0, yi - radius)
    y1 = min(height, yi + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    patch = image[y0:y1, x0:x1].astype(np.float32)
    background = float(np.median(patch))
    mad = float(np.median(np.abs(patch - background)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(patch))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    peak_flat = int(np.argmax(patch))
    py, px = np.unravel_index(peak_flat, patch.shape)
    peak = float(patch[py, px])
    peak_global = np.asarray([x0 + px, y0 + py], dtype=np.float32)
    if np.linalg.norm(peak_global - np.asarray([x, y], dtype=np.float32)) > snap_radius(mag, args):
        return None
    if peak <= background + args.adaptive_threshold_sigma * sigma:
        return None

    threshold = max(
        background + args.adaptive_threshold_sigma * sigma,
        background + args.adaptive_peak_fraction * (peak - background),
    )
    binary = (patch >= threshold).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    label = int(labels[py, px])
    if label <= 0 or num_labels <= label:
        return None

    local_yx = np.argwhere(labels == label)
    if local_yx.size == 0:
        return None

    global_y = local_yx[:, 0] + y0
    global_x = local_yx[:, 1] + x0
    max_r2 = adaptive_component_radius(mag, args) ** 2
    near = (global_x - peak_global[0]) ** 2 + (global_y - peak_global[1]) ** 2 <= max_r2
    if not np.any(near):
        return None
    global_y = global_y[near]
    global_x = global_x[near]

    weights = image[global_y, global_x].astype(np.float32) - background
    weights = np.clip(weights, 0.0, None)
    if float(np.sum(weights)) > 0:
        cx = float(np.sum(global_x * weights) / np.sum(weights))
        cy = float(np.sum(global_y * weights) / np.sum(weights))
    else:
        cx = float(peak_global[0])
        cy = float(peak_global[1])
    pixels = np.column_stack((global_y, global_x)).astype(np.int32)
    radius_px = float(math.sqrt(len(pixels) / math.pi))
    peak_snr = float((peak - background) / max(sigma, 1e-6))
    return AdaptiveStarMask(
        pixels=pixels,
        center=np.asarray([cx, cy], dtype=np.float32),
        radius_px=radius_px,
        peak_snr=peak_snr,
    )


def write_centroids(
    path: Path,
    projection: dict[str, np.ndarray],
    xy: np.ndarray,
    mask_methods: np.ndarray,
    mask_radii: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "source_id", "ra", "dec", "phot_g_mean_mag", "mask_method", "mask_radius_px"])
        for i in range(len(xy)):
            writer.writerow(
                [
                    float(xy[i, 0]),
                    float(xy[i, 1]),
                    projection["source_id"][i],
                    float(projection["ra"][i]),
                    float(projection["dec"][i]),
                    float(projection["mag"][i]),
                    mask_methods[i],
                    float(mask_radii[i]),
                ]
            )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items() if k != "visual"}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_sample_image(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == output_path.resolve():
        return
    if output_path.suffix.lower() == source_path.suffix.lower():
        shutil.copy2(source_path, output_path)
        return
    image = Image.open(source_path)
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.convert("RGB").save(output_path, quality=95)
    else:
        image.save(output_path)


def process_sample(
    source_path: Path,
    output_image_path: Path,
    centroid_path: Path,
    mask_path: Path,
    metadata_path: Path,
    t3: Tetra3,
    catalog: GaiaCatalog,
    args: argparse.Namespace,
    split: str | None = None,
) -> dict[str, Any]:
    metadata_dir = metadata_path.parent
    mask_dir = mask_path.parent

    if (
        not args.overwrite
        and output_image_path.exists()
        and centroid_path.exists()
        and mask_path.exists()
        and metadata_path.exists()
    ):
        log(f"skip existing: {output_image_path.name}")
        return {
            "source_image": str(source_path),
            "image": str(output_image_path),
            "split": split,
            "status": "skipped",
        }

    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "source_image": str(source_path),
        "image": str(output_image_path),
        "split": split,
        "status": "started",
    }

    try:
        image = read_gray(source_path)
        solution = solve_image(t3, source_path, args)
        metadata["tetra3"] = json_ready(solution)
        if solution.get("RA") is None:
            metadata["status"] = "solve_failed"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            log(f"solve failed: {source_path.name}")
            return metadata

        wcs = build_wcs_from_solution(solution, image.shape)
        projection = project_gaia_to_image(catalog, wcs, solution, image.shape)
        mask, xy, mask_methods, mask_radii, radius_model_info = generate_mask_and_centroids(image, projection, args)

        save_sample_image(source_path, output_image_path)
        write_centroids(centroid_path, projection, xy, mask_methods, mask_radii)
        mask_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(mask_path)

        metadata.update(
            {
                "status": "ok",
                "image_shape": [int(image.shape[0]), int(image.shape[1])],
                "output_image_path": str(output_image_path),
                "centroid_path": str(centroid_path),
                "mask_path": str(mask_path),
                "gaia_mag_limit": args.mag_limit,
                "gaia_projected_count": int(len(xy)),
                "mask_pixels": int(np.count_nonzero(mask)),
                "mask_methods": {
                    method: int(np.sum(mask_methods == method))
                    for method in sorted(set(mask_methods.tolist()))
                },
                "mask_radius_model": radius_model_info,
                "mask_radius_px": {
                    "min": float(np.min(mask_radii)) if len(mask_radii) else 0.0,
                    "max": float(np.max(mask_radii)) if len(mask_radii) else 0.0,
                    "mean": float(np.mean(mask_radii)) if len(mask_radii) else 0.0,
                },
                "undetected_mask_max_radius_px": float(args.undetected_mask_max_radius_px),
            }
        )
        metadata_path.write_text(json.dumps(json_ready(metadata), indent=2), encoding="utf-8")
        log(
            f"ok: {source_path.name} -> {output_image_path.name}, "
            f"{len(xy)} stars, mask pixels={int(np.count_nonzero(mask))}"
        )
        return metadata
    except Exception as exc:
        metadata["status"] = "error"
        metadata["error"] = repr(exc)
        metadata_path.write_text(json.dumps(json_ready(metadata), indent=2), encoding="utf-8")
        log(f"error: {source_path.name}: {exc}")
        return metadata


def process_image(path: Path, t3: Tetra3, catalog: GaiaCatalog, args: argparse.Namespace) -> dict[str, Any]:
    centroid_dir = args.centroid_dir or path.parent.parent / "centroids"
    mask_dir = args.mask_dir or path.parent.parent / "masks"
    metadata_dir = args.metadata_dir or path.parent.parent / "metadata"
    centroid_path = centroid_dir / f"{path.stem}.csv"
    mask_path = mask_dir / f"{path.stem}.png"
    metadata_path = metadata_dir / f"{path.stem}.json"
    return process_sample(
        source_path=path,
        output_image_path=path,
        centroid_path=centroid_path,
        mask_path=mask_path,
        metadata_path=metadata_path,
        t3=t3,
        catalog=catalog,
        args=args,
        split=None,
    )


def write_split_manifest(dataset_dir: Path, rows: list[dict[str, Any]]) -> None:
    path = dataset_dir / "preprocessing_manifest.csv"
    columns = [
        "split",
        "sample_id",
        "status",
        "source_image",
        "image",
        "centroid_path",
        "mask_path",
        "RA",
        "Dec",
        "Roll",
        "FOV",
        "Matches",
        "RMSE",
        "gaia_projected_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            tetra3 = row.get("tetra3", {}) if isinstance(row.get("tetra3"), dict) else {}
            writer.writerow(
                {
                    "split": row.get("split", ""),
                    "sample_id": Path(str(row.get("image", ""))).stem,
                    "status": row.get("status", ""),
                    "source_image": row.get("source_image", ""),
                    "image": row.get("image", row.get("output_image_path", "")),
                    "centroid_path": row.get("centroid_path", ""),
                    "mask_path": row.get("mask_path", ""),
                    "RA": tetra3.get("RA", ""),
                    "Dec": tetra3.get("Dec", ""),
                    "Roll": tetra3.get("Roll", ""),
                    "FOV": tetra3.get("FOV", ""),
                    "Matches": tetra3.get("Matches", ""),
                    "RMSE": tetra3.get("RMSE", ""),
                    "gaia_projected_count": row.get("gaia_projected_count", ""),
                }
            )
    log(f"Wrote manifest: {path}")


def run_existing_image_dir(args: argparse.Namespace, t3: Tetra3, catalog: GaiaCatalog) -> list[dict[str, Any]]:
    if args.image_dir is None or not args.image_dir.exists():
        raise FileNotFoundError(f"image directory not found: {args.image_dir}")
    paths = image_paths(args.image_dir, args.limit)
    if not paths:
        raise RuntimeError(f"No images found under {args.image_dir}")

    results: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        log(f"[{index}/{len(paths)}] {path.name}")
        results.append(process_image(path, t3, catalog, args))
    return results


def run_raw_split(args: argparse.Namespace, t3: Tetra3, catalog: GaiaCatalog) -> list[dict[str, Any]]:
    raw_dir = resolve_raw_image_dir(args.raw_image_dir)
    paths = image_paths(raw_dir, args.limit)
    if not paths:
        raise RuntimeError(f"No raw images found under {raw_dir}")

    train_paths, val_paths = split_raw_images(paths, args.train_ratio, args.random_seed)
    log(
        f"Raw split from {raw_dir}: {len(train_paths)} train candidate(s), "
        f"{len(val_paths)} val candidate(s), train_ratio={args.train_ratio:.3f}"
    )

    output_suffix = normalized_output_suffix(args)
    results: list[dict[str, Any]] = []
    split_jobs = (("train", train_paths), ("val", val_paths))
    for split, split_paths in split_jobs:
        split_dir = args.dataset_dir / split
        image_dir = split_dir / "images"
        centroid_dir = split_dir / "centroids"
        mask_dir = split_dir / "masks"
        metadata_dir = split_dir / "metadata"
        for directory in (image_dir, centroid_dir, mask_dir, metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)

        successful_in_split = 0
        for index, source_path in enumerate(split_paths, start=1):
            stem = sample_name(index, args)
            output_image_path = image_dir / f"{stem}{output_suffix}"
            centroid_path = centroid_dir / f"{stem}.csv"
            mask_path = mask_dir / f"{stem}.png"
            metadata_path = metadata_dir / f"{stem}.json"
            log(f"[{split} {index}/{len(split_paths)}] {source_path.name} -> {output_image_path.name}")
            result = process_sample(
                source_path=source_path,
                output_image_path=output_image_path,
                centroid_path=centroid_path,
                mask_path=mask_path,
                metadata_path=metadata_path,
                t3=t3,
                catalog=catalog,
                args=args,
                split=split,
            )
            results.append(result)
            successful_in_split += int(result.get("status") == "ok")
        log(f"{split}: {successful_in_split}/{len(split_paths)} solved and labeled")
    write_split_manifest(args.dataset_dir, results)
    return results


def main() -> None:
    args = parse_args()
    catalog = load_gaia_catalog(args.gaia_csv, args.mag_limit)

    log(f"Loading tetra3 database: {args.tetra3_db}")
    t3 = Tetra3(load_database=args.tetra3_db)

    if args.image_dir is not None:
        results = run_existing_image_dir(args, t3, catalog)
    else:
        results = run_raw_split(args, t3, catalog)

    ok = sum(1 for result in results if result.get("status") == "ok")
    skipped = sum(1 for result in results if result.get("status") == "skipped")
    failed = len(results) - ok - skipped
    log(f"Done. ok={ok}, skipped={skipped}, failed={failed}, total={len(results)}")


if __name__ == "__main__":
    main()
