from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.evaluate_mask_methods_gaia import (  # noqa: E402
    DEFAULT_GAIA_CSV,
    angular_prefilter,
    evaluate_one_method,
    load_gaia_catalog,
    limiting_magnitude_from_bins,
    match_extractions,
    project_gaia,
    year_fraction_from_name,
)
from preprocessing.mask_generator import METHODS, generate_mask, read_gray_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate selected S30Pro single frames against Gaia.")
    parser.add_argument("--root", type=Path, default=Path("data/data_S30Pro"))
    parser.add_argument("--gaia-csv", type=Path, default=Path("preprocessing/gaia_data/gaia_s30pro_4fields_g16_local.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("preprocessing/gaia_data/gaia_single_frame_eval_g16.json"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--samples-per-set", type=int, default=3)
    parser.add_argument("--match-radius-px", type=float, default=3.0)
    parser.add_argument("--match-radius-bright-px", type=float, default=4.0)
    parser.add_argument("--match-radius-bright-mag", type=float, default=6.0)
    parser.add_argument("--match-radius-faint-mag", type=float, default=12.0)
    parser.add_argument("--match-radius-mode", choices=("constant", "linear", "sqrt"), default="linear")
    parser.add_argument("--gaia-margin-deg", type=float, default=1.0)
    parser.add_argument("--mag-bin-size", type=float, default=0.5)
    parser.add_argument("--min-bin-stars", type=int, default=5)
    parser.add_argument("--no-align-translation", action="store_true")
    parser.add_argument("--align-method", choices=METHODS, default="daofind_like")
    parser.add_argument("--align-max-shift-px", type=float, default=400.0)
    parser.add_argument("--align-bin-px", type=float, default=4.0)
    parser.add_argument("--align-top-detections", type=int, default=300)
    parser.add_argument("--align-top-gaia", type=int, default=500)
    parser.add_argument("--align-refine-radius-px", type=float, default=12.0)
    parser.add_argument("--align-candidates", type=int, default=12)

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


def selected_single_frames(root: Path, samples_per_set: int) -> list[tuple[str, Path, Path]]:
    rows = []
    for stack_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.endswith("_sub")):
        stack_files = sorted(stack_dir.glob("Stacked_*.fit"))
        sub_dir = root / f"{stack_dir.name}_sub"
        if not stack_files or not sub_dir.exists():
            continue
        singles = sorted(
            path
            for path in sub_dir.glob("*.fit")
            if "_20260602-" in path.name and "_20.0s_IRCUT_" in path.name
        )
        if not singles:
            singles = sorted(sub_dir.glob("*.fit"))
        if not singles:
            continue

        count = min(max(1, samples_per_set), len(singles))
        if count == 1:
            indices = [len(singles) // 2]
        else:
            indices = sorted({round(i * (len(singles) - 1) / (count - 1)) for i in range(count)})
        for index in indices:
            rows.append((stack_dir.name, stack_files[0], singles[int(index)]))
    return rows


def shifted_wcs_for_single(stack_path: Path, single_path: Path) -> tuple[WCS, fits.Header, tuple[int, int], dict[str, float]]:
    stack_header = fits.getheader(stack_path)
    single_header = fits.getheader(single_path)
    height = int(stack_header["NAXIS2"])
    width = int(stack_header["NAXIS1"])

    wcs = WCS(stack_header, naxis=2)
    stack_ra = float(stack_header.get("RA", stack_header.get("CRVAL1")))
    stack_dec = float(stack_header.get("DEC", stack_header.get("CRVAL2")))
    single_ra = float(single_header.get("RA", stack_ra))
    single_dec = float(single_header.get("DEC", stack_dec))

    stack_xy = np.asarray(wcs.all_world2pix([[stack_ra, stack_dec]], 0)[0], dtype=float)
    single_xy_on_stack = np.asarray(wcs.all_world2pix([[single_ra, single_dec]], 0)[0], dtype=float)
    delta = single_xy_on_stack - stack_xy

    shifted_header = stack_header.copy()
    shifted_header["CRPIX1"] = float(stack_header["CRPIX1"]) - float(delta[0])
    shifted_header["CRPIX2"] = float(stack_header["CRPIX2"]) - float(delta[1])
    shifted_wcs = WCS(shifted_header, naxis=2)
    shift_info = {
        "delta_x_px": float(delta[0]),
        "delta_y_px": float(delta[1]),
        "stack_ra": stack_ra,
        "stack_dec": stack_dec,
        "single_ra": single_ra,
        "single_dec": single_dec,
    }
    return shifted_wcs, shifted_header, (height, width), shift_info


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


def shifted_gaia_projection(gaia_projection: dict[str, np.ndarray], shift_xy: np.ndarray) -> dict[str, np.ndarray]:
    shifted = {key: value.copy() for key, value in gaia_projection.items()}
    shifted["x"] = shifted["x"] + np.float32(shift_xy[0])
    shifted["y"] = shifted["y"] + np.float32(shift_xy[1])
    return shifted


def brightest_gaia_xy(gaia_projection: dict[str, np.ndarray], top_count: int) -> np.ndarray:
    x = gaia_projection["x"]
    y = gaia_projection["y"]
    mag = gaia_projection["mag"]
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(mag)
    indices = np.flatnonzero(finite)
    if len(indices) == 0:
        return np.empty((0, 2), dtype=np.float32)
    indices = indices[np.argsort(mag[indices])]
    if top_count > 0:
        indices = indices[:top_count]
    return np.column_stack((x[indices], y[indices])).astype(np.float32)


def strongest_detection_xy(image: np.ndarray, centroids_yx: np.ndarray, top_count: int) -> np.ndarray:
    if len(centroids_yx) == 0:
        return np.empty((0, 2), dtype=np.float32)
    scores = centroid_peak_scores(image, centroids_yx)
    indices = np.argsort(scores)[::-1]
    if top_count > 0:
        indices = indices[:top_count]
    selected = centroids_yx[indices]
    return np.column_stack((selected[:, 1], selected[:, 0])).astype(np.float32)


def coarse_translation_candidates(
    detection_xy: np.ndarray,
    gaia_xy: np.ndarray,
    max_shift_px: float,
    bin_px: float,
    candidate_count: int,
) -> list[np.ndarray]:
    if len(detection_xy) == 0 or len(gaia_xy) == 0:
        return [np.asarray([0.0, 0.0], dtype=np.float32)]

    deltas = detection_xy[:, None, :] - gaia_xy[None, :, :]
    keep = (np.abs(deltas[:, :, 0]) <= max_shift_px) & (np.abs(deltas[:, :, 1]) <= max_shift_px)
    kept = deltas[keep]
    if len(kept) == 0:
        return [np.asarray([0.0, 0.0], dtype=np.float32)]

    bins = np.rint(kept / max(bin_px, 1e-6)).astype(np.int32)
    unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][: max(1, candidate_count)]
    return [(unique_bins[index].astype(np.float32) * float(bin_px)) for index in order]


def refine_translation(
    initial_shift_xy: np.ndarray,
    centroids_yx: np.ndarray,
    gaia_projection: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> np.ndarray:
    if len(centroids_yx) == 0 or len(gaia_projection["x"]) == 0:
        return initial_shift_xy.astype(np.float32)

    detection_xy = np.column_stack((centroids_yx[:, 1], centroids_yx[:, 0])).astype(np.float32)
    gaia_xy = np.column_stack((gaia_projection["x"], gaia_projection["y"])).astype(np.float32)
    shift = initial_shift_xy.astype(np.float32)

    for radius in (float(args.align_refine_radius_px), max(float(args.match_radius_bright_px) * 1.5, 5.0)):
        shifted_gaia_xy = gaia_xy + shift
        tree = cKDTree(shifted_gaia_xy)
        distances, indices = tree.query(detection_xy, distance_upper_bound=radius)
        valid = np.isfinite(distances) & (indices < len(gaia_xy))
        if np.sum(valid) < 3:
            continue
        residuals = detection_xy[valid] - gaia_xy[indices[valid]]
        median = np.median(residuals, axis=0)
        scatter = np.linalg.norm(residuals - median, axis=1)
        mad = np.median(np.abs(scatter - np.median(scatter)))
        keep = scatter <= max(radius, 3.0 * 1.4826 * mad)
        if np.sum(keep) >= 3:
            shift = np.median(residuals[keep], axis=0).astype(np.float32)
        else:
            shift = median.astype(np.float32)
    return shift


def estimate_translation_alignment(
    image: np.ndarray,
    centroids_yx: np.ndarray,
    gaia_projection: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    detection_xy = strongest_detection_xy(image, centroids_yx, args.align_top_detections)
    gaia_xy = brightest_gaia_xy(gaia_projection, args.align_top_gaia)
    candidates = coarse_translation_candidates(
        detection_xy,
        gaia_xy,
        float(args.align_max_shift_px),
        float(args.align_bin_px),
        int(args.align_candidates),
    )

    best: dict[str, Any] | None = None
    best_projection = gaia_projection
    for candidate in candidates:
        shift_xy = refine_translation(candidate, centroids_yx, gaia_projection, args)
        shifted_projection = shifted_gaia_projection(gaia_projection, shift_xy)
        matches = match_extractions(centroids_yx, shifted_projection, args)
        distances = np.asarray([match[2] for match in matches], dtype=np.float32)
        score = int(len(matches))
        row = {
            "shift_x_px": float(shift_xy[0]),
            "shift_y_px": float(shift_xy[1]),
            "matched_count": score,
            "match_distance_px_mean": float(np.mean(distances)) if len(distances) else None,
            "match_distance_px_p95": float(np.percentile(distances, 95)) if len(distances) else None,
        }
        if best is None or (score, -(row["match_distance_px_mean"] or 1e9)) > (
            best["matched_count"],
            -(best["match_distance_px_mean"] or 1e9),
        ):
            best = row
            best_projection = shifted_projection

    if best is None:
        best = {
            "shift_x_px": 0.0,
            "shift_y_px": 0.0,
            "matched_count": 0,
            "match_distance_px_mean": None,
            "match_distance_px_p95": None,
        }
    best["align_method"] = args.align_method
    best["candidate_count"] = int(len(candidates))
    return best, best_projection


def aggregate_magnitude_bins(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[float, float], dict[str, Any]] = {}
    for row in method_rows:
        for bin_row in row["magnitude_bins"]:
            key = (float(bin_row["mag_min"]), float(bin_row["mag_max"]))
            target = totals.setdefault(
                key,
                {
                    "mag_min": bin_row["mag_min"],
                    "mag_max": bin_row["mag_max"],
                    "gaia_count": 0,
                    "matched_count": 0,
                },
            )
            target["gaia_count"] += int(bin_row["gaia_count"])
            target["matched_count"] += int(bin_row["matched_count"])

    rows = []
    for key in sorted(totals):
        row = totals[key]
        row["completeness"] = row["matched_count"] / row["gaia_count"] if row["gaia_count"] else 0.0
        rows.append(row)
    return rows


def aggregate_by_method(frame_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        method_rows = [
            method_result
            for frame in frame_results
            for method_result in frame["methods"]
            if method_result["method"] == method
        ]
        if not method_rows:
            continue
        extracted = sum(row["extracted_count"] for row in method_rows)
        matched = sum(row["matched_count"] for row in method_rows)
        gaia = sum(row["gaia_count"] for row in method_rows)
        false = sum(row["false_count"] for row in method_rows)
        faintest = [row["faintest_matched_g_mag"] for row in method_rows if row["faintest_matched_g_mag"] is not None]
        bins = aggregate_magnitude_bins(method_rows)
        rows.append(
            {
                "method": method,
                "frames": len(method_rows),
                "extracted_count": extracted,
                "gaia_count": gaia,
                "matched_count": matched,
                "false_count": false,
                "false_extraction_rate": false / extracted if extracted else 0.0,
                "gaia_recall": matched / gaia if gaia else 0.0,
                "faintest_matched_g_mag": max(faintest) if faintest else None,
                "complete_50_mag": limiting_magnitude_from_bins(bins, 0.5),
                "complete_80_mag": limiting_magnitude_from_bins(bins, 0.8),
                "magnitude_bins": bins,
            }
        )
    return rows


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    catalog = load_gaia_catalog(args.gaia_csv)
    rows = selected_single_frames(args.root, args.samples_per_set)
    if not rows:
        raise FileNotFoundError(f"no single frames found under {args.root}")

    frame_results = []
    for set_name, stack_path, single_path in rows:
        print(f"[frame] {set_name}: {single_path.name}")
        image = read_gray_image(single_path, fit_channel_mode=args.fit_channel_mode)
        wcs, header, image_shape, shift_info = shifted_wcs_for_single(stack_path, single_path)
        epoch = year_fraction_from_name(single_path)
        pre = angular_prefilter(catalog, header, image_shape, args.gaia_margin_deg)
        gaia_projection = project_gaia(catalog, np.flatnonzero(pre), wcs, image_shape, epoch)
        alignment = None
        if not args.no_align_translation:
            align_result = generate_mask(image, args.align_method, args)
            alignment, gaia_projection = estimate_translation_alignment(image, align_result.centroids_yx, gaia_projection, args)

        message = f"  Gaia projected: {len(gaia_projection['x'])}, header shift=({shift_info['delta_x_px']:.2f}, {shift_info['delta_y_px']:.2f}) px"
        if alignment is not None:
            message += (
                f", refined shift=({alignment['shift_x_px']:.2f}, {alignment['shift_y_px']:.2f}) px"
                f", align matches={alignment['matched_count']}"
            )
        print(message)

        method_rows = []
        for method in args.methods:
            row = evaluate_one_method(image, method, args, gaia_projection)
            method_rows.append(row)
            print(
                f"  {method}: extracted={row['extracted_count']} matched={row['matched_count']} "
                f"false={row['false_extraction_rate']:.3f} recall={row['gaia_recall']:.3f} "
                f"faintestG={row['faintest_matched_g_mag']}"
            )

        frame_results.append(
            {
                "set": set_name,
                "stack_image": str(stack_path),
                "single_frame": str(single_path),
                "image_shape": [int(image_shape[0]), int(image_shape[1])],
                "epoch": epoch,
                "shift_info": shift_info,
                "translation_alignment": alignment,
                "gaia_projected_count": int(len(gaia_projection["x"])),
                "methods": method_rows,
            }
        )

    output = {
        "config": {
            "root": str(args.root),
            "gaia_csv": str(args.gaia_csv),
            "samples_per_set": int(args.samples_per_set),
            "methods": args.methods,
            "match_radius": {
                "mode": args.match_radius_mode,
                "faint_px": float(args.match_radius_px),
                "bright_px": float(args.match_radius_bright_px),
                "bright_mag": float(args.match_radius_bright_mag),
                "faint_mag": float(args.match_radius_faint_mag),
            },
            "wcs_note": "Single frames have no full WCS; evaluation uses corresponding stacked WCS shifted by single-frame RA/Dec.",
            "translation_alignment": None
            if args.no_align_translation
            else {
                "align_method": args.align_method,
                "max_shift_px": float(args.align_max_shift_px),
                "bin_px": float(args.align_bin_px),
                "top_detections": int(args.align_top_detections),
                "top_gaia": int(args.align_top_gaia),
                "refine_radius_px": float(args.align_refine_radius_px),
            },
        },
        "summary_by_method": aggregate_by_method(frame_results),
        "frames": frame_results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out_json}")
    print(json.dumps(output["summary_by_method"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
