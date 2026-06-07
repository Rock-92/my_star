from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.mask_generator import METHODS, generate_mask, read_gray_image  # noqa: E402


DEFAULT_GAIA_CSV = Path("preprocessing/gaia_data/gaia_g12p0_ra0p0_360p0_decm90p0_90p0_step20p0x20p0_pm.csv")


@dataclass
class GaiaCatalog:
    source_id: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    mag: np.ndarray
    pmra: np.ndarray
    pmdec: np.ndarray
    ref_epoch: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FITS star extraction masks against an offline Gaia catalogue.",
    )
    parser.add_argument("--stack-root", type=Path, default=Path("data/data_S30Pro"))
    parser.add_argument("--gaia-csv", type=Path, default=DEFAULT_GAIA_CSV)
    parser.add_argument("--out-json", type=Path, default=Path("preprocessing/gaia_data/gaia_extraction_eval.json"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--match-radius-px", type=float, default=3.0, help="Match radius for faint Gaia stars.")
    parser.add_argument("--match-radius-bright-px", type=float, default=4.0, help="Match radius for bright Gaia stars.")
    parser.add_argument("--match-radius-bright-mag", type=float, default=6.0)
    parser.add_argument("--match-radius-faint-mag", type=float, default=12.0)
    parser.add_argument("--match-radius-mode", choices=("constant", "linear", "sqrt"), default="linear")
    parser.add_argument("--gaia-margin-deg", type=float, default=1.0)
    parser.add_argument("--mag-bin-size", type=float, default=0.5)
    parser.add_argument("--min-bin-stars", type=int, default=5)

    parser.add_argument("--fit-channel-mode", default="mean", choices=("mean", "first", "max", "luma"))
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--filtsize", type=int, default=25)
    parser.add_argument("--background-mode", default="local_mean")
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--max-area", type=int, default=100)
    parser.add_argument("--max-axis-ratio", type=float, default=None)
    parser.add_argument("--no-binary-open", action="store_true")
    parser.add_argument("--mesh-size", type=int, default=64)
    parser.add_argument("--mesh-filter-size", type=int, default=3)
    parser.add_argument("--filter-sigma", type=float, default=1.0)
    parser.add_argument("--fwhm", type=float, default=3.0)
    parser.add_argument("--peak-window", type=int, default=None)
    parser.add_argument("--mask-radius", type=float, default=None)
    parser.add_argument("--radius-mode", choices=("gaussian", "linear", "sqrt", "log", "constant"), default="gaussian")
    parser.add_argument("--min-mask-radius", type=float, default=None)
    parser.add_argument("--max-mask-radius", type=float, default=None)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--min-separation", type=float, default=None)
    parser.add_argument("--max-peaks", type=int, default=None)
    parser.add_argument("--exclude-border", type=int, default=None)
    return parser.parse_args()


def load_gaia_catalog(path: Path) -> GaiaCatalog:
    usecols = ["source_id", "ra", "dec", "pmra", "pmdec", "ref_epoch", "phot_g_mean_mag"]
    data = pd.read_csv(path, usecols=usecols)
    return GaiaCatalog(
        source_id=data["source_id"].to_numpy(),
        ra=data["ra"].to_numpy(dtype=np.float64),
        dec=data["dec"].to_numpy(dtype=np.float64),
        mag=data["phot_g_mean_mag"].to_numpy(dtype=np.float32),
        pmra=data["pmra"].fillna(0.0).to_numpy(dtype=np.float64),
        pmdec=data["pmdec"].fillna(0.0).to_numpy(dtype=np.float64),
        ref_epoch=data["ref_epoch"].fillna(2016.0).to_numpy(dtype=np.float64),
    )


def stacked_fits_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("*/*.fit")
        if path.is_file() and not path.parent.name.endswith("_sub") and path.name.startswith("Stacked_")
    )


def year_fraction_from_name(path: Path) -> float | None:
    match = re.search(r"_(\d{8})-\d{6}", path.name)
    if not match:
        return None
    text = match.group(1)
    year = int(text[:4])
    month = int(text[4:6])
    day = int(text[6:8])
    month_days = [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    doy = sum(month_days[: month - 1]) + day
    return year + (doy - 0.5) / sum(month_days)


def proper_motion_corrected(catalog: GaiaCatalog, indices: np.ndarray, epoch: float | None) -> tuple[np.ndarray, np.ndarray]:
    ra = catalog.ra[indices].astype(np.float64, copy=True)
    dec = catalog.dec[indices].astype(np.float64, copy=True)
    if epoch is None:
        return ra, dec
    dt = float(epoch) - catalog.ref_epoch[indices]
    cos_dec = np.cos(np.deg2rad(dec))
    cos_dec = np.where(np.abs(cos_dec) < 1e-6, 1e-6, cos_dec)
    ra += catalog.pmra[indices] * dt / (3.6e6 * cos_dec)
    dec += catalog.pmdec[indices] * dt / 3.6e6
    return ra, dec


def wcs_from_fits(path: Path) -> tuple[WCS, fits.Header, tuple[int, int]]:
    header = fits.getheader(path)
    width = int(header["NAXIS1"])
    height = int(header["NAXIS2"])
    wcs = WCS(header, naxis=2)
    return wcs, header, (height, width)


def angular_prefilter(catalog: GaiaCatalog, header: fits.Header, image_shape: tuple[int, int], margin_deg: float) -> np.ndarray:
    height, width = image_shape
    ra0 = float(header.get("CRVAL1", header.get("RA", 0.0)))
    dec0 = float(header.get("CRVAL2", header.get("DEC", 0.0)))
    cd11 = abs(float(header.get("CD1_1", header.get("CDELT1", 0.001))))
    cd12 = abs(float(header.get("CD1_2", 0.0)))
    cd21 = abs(float(header.get("CD2_1", 0.0)))
    cd22 = abs(float(header.get("CD2_2", header.get("CDELT2", 0.001))))
    scale_x = math.hypot(cd11, cd21)
    scale_y = math.hypot(cd12, cd22)
    radius_deg = math.hypot(width * scale_x, height * scale_y) * 0.5 + float(margin_deg)

    dec_keep = (catalog.dec >= dec0 - radius_deg) & (catalog.dec <= dec0 + radius_deg)
    cos_dec = max(0.1, abs(math.cos(math.radians(dec0))))
    ra_margin = min(180.0, radius_deg / cos_dec)
    dra = ((catalog.ra - ra0 + 180.0) % 360.0) - 180.0
    ra_keep = np.abs(dra) <= ra_margin
    return dec_keep & ra_keep


def project_gaia(
    catalog: GaiaCatalog,
    indices: np.ndarray,
    wcs: WCS,
    image_shape: tuple[int, int],
    epoch: float | None,
) -> dict[str, np.ndarray]:
    height, width = image_shape
    ra, dec = proper_motion_corrected(catalog, indices, epoch)
    world = np.column_stack((ra, dec))
    xy = wcs.all_world2pix(world, 0)
    x = xy[:, 0]
    y = xy[:, 1]
    keep = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x < width) & (y >= 0.0) & (y < height)
    selected = indices[keep]
    return {
        "source_id": catalog.source_id[selected],
        "x": x[keep].astype(np.float32),
        "y": y[keep].astype(np.float32),
        "mag": catalog.mag[selected],
    }


def match_extractions(
    centroids_yx: np.ndarray,
    gaia_projection: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[tuple[int, int, float, float]]:
    if len(centroids_yx) == 0 or len(gaia_projection["x"]) == 0:
        return []

    extracted_xy = np.column_stack((centroids_yx[:, 1], centroids_yx[:, 0])).astype(np.float32)
    gaia_xy = np.column_stack((gaia_projection["x"], gaia_projection["y"])).astype(np.float32)
    radii = gaia_match_radii(gaia_projection["mag"], args)
    max_radius = float(np.max(radii)) if len(radii) else float(args.match_radius_px)
    tree = cKDTree(gaia_xy)

    candidates = []
    for extracted_index, xy in enumerate(extracted_xy):
        nearby = tree.query_ball_point(xy, r=max_radius)
        for gaia_index in nearby:
            distance = float(np.linalg.norm(xy - gaia_xy[gaia_index]))
            radius = float(radii[gaia_index])
            if distance <= radius:
                candidates.append((distance / max(radius, 1e-6), distance, extracted_index, gaia_index, radius))
    candidates.sort(key=lambda item: (item[0], item[1]))

    used_extracted: set[int] = set()
    used_gaia: set[int] = set()
    matches: list[tuple[int, int, float, float]] = []
    for _, distance, extracted_index, gaia_index, radius in candidates:
        if extracted_index in used_extracted or gaia_index in used_gaia:
            continue
        used_extracted.add(extracted_index)
        used_gaia.add(gaia_index)
        matches.append((extracted_index, gaia_index, distance, radius))
    return matches


def gaia_match_radii(mags: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    faint_radius = float(args.match_radius_px)
    bright_radius = float(args.match_radius_bright_px)
    if args.match_radius_mode == "constant":
        return np.full(len(mags), faint_radius, dtype=np.float32)

    bright_mag = float(args.match_radius_bright_mag)
    faint_mag = float(args.match_radius_faint_mag)
    if abs(faint_mag - bright_mag) < 1e-6:
        return np.full(len(mags), faint_radius, dtype=np.float32)

    # Smaller magnitude means brighter. t=1 at the bright end, t=0 at the faint end.
    t = (faint_mag - np.asarray(mags, dtype=np.float32)) / (faint_mag - bright_mag)
    t = np.clip(t, 0.0, 1.0)
    if args.match_radius_mode == "sqrt":
        t = np.sqrt(t)
    return (faint_radius + t * (bright_radius - faint_radius)).astype(np.float32)


def completeness_bins(
    gaia_mags: np.ndarray,
    matched_gaia_indices: np.ndarray,
    bin_size: float,
    min_bin_stars: int,
) -> list[dict[str, Any]]:
    if len(gaia_mags) == 0:
        return []
    min_mag = math.floor(float(np.nanmin(gaia_mags)) / bin_size) * bin_size
    max_mag = math.ceil(float(np.nanmax(gaia_mags)) / bin_size) * bin_size
    matched_mask = np.zeros(len(gaia_mags), dtype=bool)
    matched_mask[matched_gaia_indices] = True
    rows = []
    edge = min_mag
    while edge < max_mag + 1e-9:
        next_edge = edge + bin_size
        in_bin = (gaia_mags >= edge) & (gaia_mags < next_edge)
        total = int(np.sum(in_bin))
        matched = int(np.sum(in_bin & matched_mask))
        if total >= min_bin_stars:
            rows.append(
                {
                    "mag_min": round(edge, 3),
                    "mag_max": round(next_edge, 3),
                    "gaia_count": total,
                    "matched_count": matched,
                    "completeness": matched / total if total else 0.0,
                }
            )
        edge = next_edge
    return rows


def limiting_magnitude_from_bins(rows: list[dict[str, Any]], threshold: float) -> float | None:
    valid = [row["mag_max"] for row in rows if row["completeness"] >= threshold]
    return float(max(valid)) if valid else None


def evaluate_one_method(
    image: np.ndarray,
    method: str,
    args: argparse.Namespace,
    gaia_projection: dict[str, np.ndarray],
) -> dict[str, Any]:
    result = generate_mask(image, method, args)
    matches = match_extractions(result.centroids_yx, gaia_projection, args)
    matched_gaia_indices = np.asarray([match[1] for match in matches], dtype=np.int64)
    matched_distances = np.asarray([match[2] for match in matches], dtype=np.float32)
    matched_radii = np.asarray([match[3] for match in matches], dtype=np.float32)
    matched_mags = gaia_projection["mag"][matched_gaia_indices] if len(matches) else np.empty((0,), dtype=np.float32)

    extracted_count = int(len(result.centroids_yx))
    matched_count = int(len(matches))
    false_count = max(0, extracted_count - matched_count)
    gaia_count = int(len(gaia_projection["x"]))
    bins = completeness_bins(gaia_projection["mag"], matched_gaia_indices, args.mag_bin_size, args.min_bin_stars)

    return {
        "method": method,
        "extracted_count": extracted_count,
        "gaia_count": gaia_count,
        "matched_count": matched_count,
        "false_count": false_count,
        "false_extraction_rate": false_count / extracted_count if extracted_count else 0.0,
        "gaia_recall": matched_count / gaia_count if gaia_count else 0.0,
        "faintest_matched_g_mag": float(np.max(matched_mags)) if len(matched_mags) else None,
        "brightest_unmatched_gaia_g_mag": brightest_unmatched_mag(gaia_projection["mag"], matched_gaia_indices),
        "complete_50_mag": limiting_magnitude_from_bins(bins, 0.5),
        "complete_80_mag": limiting_magnitude_from_bins(bins, 0.8),
        "match_distance_px_mean": float(np.mean(matched_distances)) if len(matched_distances) else None,
        "match_distance_px_p95": float(np.percentile(matched_distances, 95)) if len(matched_distances) else None,
        "match_radius_px_mean": float(np.mean(matched_radii)) if len(matched_radii) else None,
        "match_radius_px_min": float(np.min(matched_radii)) if len(matched_radii) else None,
        "match_radius_px_max": float(np.max(matched_radii)) if len(matched_radii) else None,
        "mask_pixels": int(np.count_nonzero(result.mask)),
        "debug": result.debug,
        "magnitude_bins": bins,
    }


def brightest_unmatched_mag(gaia_mags: np.ndarray, matched_gaia_indices: np.ndarray) -> float | None:
    if len(gaia_mags) == 0:
        return None
    matched = np.zeros(len(gaia_mags), dtype=bool)
    matched[matched_gaia_indices] = True
    unmatched = gaia_mags[~matched]
    return float(np.min(unmatched)) if len(unmatched) else None


def aggregate_by_method(image_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        method_rows = [
            method_result
            for image_result in image_results
            for method_result in image_result["methods"]
            if method_result["method"] == method
        ]
        if not method_rows:
            continue
        extracted = sum(row["extracted_count"] for row in method_rows)
        matched = sum(row["matched_count"] for row in method_rows)
        gaia = sum(row["gaia_count"] for row in method_rows)
        false = sum(row["false_count"] for row in method_rows)
        faintest = [row["faintest_matched_g_mag"] for row in method_rows if row["faintest_matched_g_mag"] is not None]
        rows.append(
            {
                "method": method,
                "images": len(method_rows),
                "extracted_count": extracted,
                "gaia_count": gaia,
                "matched_count": matched,
                "false_count": false,
                "false_extraction_rate": false / extracted if extracted else 0.0,
                "gaia_recall": matched / gaia if gaia else 0.0,
                "faintest_matched_g_mag": max(faintest) if faintest else None,
            }
        )
    return rows


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    catalog = load_gaia_catalog(args.gaia_csv)
    stack_paths = stacked_fits_paths(args.stack_root)
    if not stack_paths:
        raise FileNotFoundError(f"no stacked FITS found under {args.stack_root}")

    image_results = []
    for path in stack_paths:
        print(f"[image] {path}")
        image = read_gray_image(path, fit_channel_mode=args.fit_channel_mode)
        wcs, header, image_shape = wcs_from_fits(path)
        epoch = year_fraction_from_name(path)
        pre = angular_prefilter(catalog, header, image_shape, args.gaia_margin_deg)
        gaia_projection = project_gaia(catalog, np.flatnonzero(pre), wcs, image_shape, epoch)
        print(f"  Gaia projected: {len(gaia_projection['x'])}")

        method_rows = []
        for method in args.methods:
            row = evaluate_one_method(image, method, args, gaia_projection)
            method_rows.append(row)
            print(
                f"  {method}: extracted={row['extracted_count']} matched={row['matched_count']} "
                f"false={row['false_extraction_rate']:.3f} recall={row['gaia_recall']:.3f} "
                f"faintestG={row['faintest_matched_g_mag']}"
            )

        image_results.append(
            {
                "image": str(path),
                "object": str(header.get("OBJECT", "")).strip(),
                "image_shape": [int(image_shape[0]), int(image_shape[1])],
                "epoch": epoch,
                "gaia_projected_count": int(len(gaia_projection["x"])),
                "methods": method_rows,
            }
        )

    output = {
        "config": {
            "stack_root": str(args.stack_root),
            "gaia_csv": str(args.gaia_csv),
            "methods": args.methods,
            "match_radius": {
                "mode": args.match_radius_mode,
                "faint_px": float(args.match_radius_px),
                "bright_px": float(args.match_radius_bright_px),
                "bright_mag": float(args.match_radius_bright_mag),
                "faint_mag": float(args.match_radius_faint_mag),
            },
            "mag_bin_size": float(args.mag_bin_size),
            "min_bin_stars": int(args.min_bin_stars),
            "mask_params": {
                "sigma": args.sigma,
                "filtsize": args.filtsize,
                "background_mode": args.background_mode,
                "min_area": args.min_area,
                "max_area": args.max_area,
                "fwhm": args.fwhm,
                "radius_mode": args.radius_mode,
            },
        },
        "summary_by_method": aggregate_by_method(image_results),
        "images": image_results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out_json}")
    print(json.dumps(output["summary_by_method"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
