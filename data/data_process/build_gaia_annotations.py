from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "tool"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from daofind_opt import daofind_like_detect, read_csv_rows, read_fits_image, resolve_manifest_path, write_csv  # noqa: E402


def align_stack_to_single(stack: np.ndarray, row: dict[str, str], output_shape: tuple[int, int]) -> np.ndarray:
    matrix_xy = np.asarray(
        [
            [float(row.get("label_transform_a", 1.0) or 1.0), float(row.get("label_transform_b", 0.0) or 0.0)],
            [float(row.get("label_transform_c", 0.0) or 0.0), float(row.get("label_transform_d", 1.0) or 1.0)],
        ],
        dtype=np.float64,
    )
    shift_xy = np.asarray(
        [
            float(row.get("label_shift_x_px", 0.0) or 0.0),
            float(row.get("label_shift_y_px", 0.0) or 0.0),
        ],
        dtype=np.float64,
    )
    inv_xy = np.linalg.inv(matrix_xy)
    affine_yx = np.asarray([[inv_xy[1, 1], inv_xy[1, 0]], [inv_xy[0, 1], inv_xy[0, 0]]], dtype=np.float64)
    offset_xy = -shift_xy @ inv_xy.T
    offset_yx = np.asarray([offset_xy[1], offset_xy[0]], dtype=np.float64)
    return ndi.affine_transform(
        stack.astype(np.float32),
        matrix=affine_yx,
        offset=offset_yx,
        output_shape=output_shape,
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)


def tangent_arcsec(ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    ra0 = np.deg2rad(float(ra0_deg))
    dec0 = np.deg2rad(float(dec0_deg))
    dra = ra - ra0
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(dra)
    xi = np.cos(dec) * np.sin(dra) / np.maximum(cosc, 1e-12)
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(dra)) / np.maximum(cosc, 1e-12)
    return np.column_stack((np.rad2deg(xi) * 3600.0, np.rad2deg(eta) * 3600.0)).astype(np.float64)


def similarity_from_pairs(source_xy: np.ndarray, target_xy: np.ndarray, allow_scale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    s_mean = source.mean(axis=0)
    t_mean = target.mean(axis=0)
    s0 = source - s_mean
    t0 = target - t_mean
    u, singular_values, vt = np.linalg.svd(s0.T @ t0)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    scale = float(np.sum(singular_values) / max(float(np.sum(s0 * s0)), 1e-9)) if allow_scale else 1.0
    matrix = scale * rotation
    shift = t_mean - s_mean @ matrix.T
    return matrix, shift


def unique_matches(source_xy: np.ndarray, target_xy: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(source_xy) == 0 or len(target_xy) == 0:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty, np.empty((0,), dtype=np.float64)
    tree = cKDTree(target_xy)
    distances, target_indices = tree.query(source_xy, distance_upper_bound=float(radius))
    valid = np.isfinite(distances) & (target_indices < len(target_xy))
    source_indices = np.flatnonzero(valid).astype(np.int32)
    target_indices = target_indices[valid].astype(np.int32)
    distances = distances[valid].astype(np.float64)
    order = np.argsort(distances)
    used_s: set[int] = set()
    used_t: set[int] = set()
    kept_s: list[int] = []
    kept_t: list[int] = []
    kept_d: list[float] = []
    for idx in order:
        s = int(source_indices[idx])
        t = int(target_indices[idx])
        if s in used_s or t in used_t:
            continue
        used_s.add(s)
        used_t.add(t)
        kept_s.append(s)
        kept_t.append(t)
        kept_d.append(float(distances[idx]))
    return np.asarray(kept_s, dtype=np.int32), np.asarray(kept_t, dtype=np.int32), np.asarray(kept_d, dtype=np.float64)


def coarse_shift_candidates(source_xy: np.ndarray, target_xy: np.ndarray, bin_px: float, top_k: int) -> list[np.ndarray]:
    shifts = (target_xy[None, :, :] - source_xy[:, None, :]).reshape((-1, 2))
    bins = np.round(shifts / float(bin_px)).astype(np.int32)
    unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
    order = np.argsort(-counts)[: int(top_k)]
    candidates = []
    for idx in order:
        mask = np.all(bins == unique_bins[idx], axis=1)
        candidates.append(np.median(shifts[mask], axis=0))
    return candidates


def detect_image_stars(image: np.ndarray, sigmas: list[float], fwhm: float, max_peaks: int) -> np.ndarray:
    for sigma in sigmas:
        detections = daofind_like_detect(
            image,
            sigma=sigma,
            fwhm=fwhm,
            background_mode="local_mean",
            filtsize=25,
            max_peaks=max_peaks,
            exclude_border=8,
        )
        if len(detections) >= 50:
            break
    rows = [(det.x, det.y, det.peak, det.snr, det.flux) for det in detections]
    if not rows:
        return np.empty((0, 5), dtype=np.float64)
    rows.sort(key=lambda item: item[2], reverse=True)
    return np.asarray(rows, dtype=np.float64)


def fit_gaia_to_image(
    gaia_xy0: np.ndarray,
    gaia_mag: np.ndarray,
    image_xy: np.ndarray,
    width: int,
    height: int,
    arcsec_per_px: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    center_xy = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
    bright = np.argsort(gaia_mag)[: int(args.fit_gaia_top)]
    gaia_plane = gaia_xy0[bright]
    gaia_mag_bright = gaia_mag[bright]
    image_fit = image_xy[: int(args.fit_image_top), :2]
    base_scale = 1.0 / float(arcsec_per_px)
    parities = [
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[-1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[1.0, 0.0], [0.0, -1.0]]),
        np.asarray([[-1.0, 0.0], [0.0, -1.0]]),
    ]
    best: dict[str, Any] | None = None
    angles = np.arange(0.0, 360.0, float(args.angle_step_deg))
    for parity_index, parity in enumerate(parities):
        for angle in angles:
            theta = np.deg2rad(float(angle))
            rot = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
            init_matrix = base_scale * (rot @ parity)
            pred = gaia_plane @ init_matrix.T + center_xy
            in_frame = (
                (pred[:, 0] >= -200)
                & (pred[:, 0] < width + 200)
                & (pred[:, 1] >= -200)
                & (pred[:, 1] < height + 200)
            )
            if np.count_nonzero(in_frame) < 10:
                continue
            src0 = pred[in_frame]
            mag0 = gaia_mag_bright[in_frame]
            src_keep = np.argsort(mag0)[: min(len(src0), int(args.shift_gaia_top))]
            src_for_shift = src0[src_keep]
            tgt_for_shift = image_fit[: int(args.shift_image_top)]
            for shift in coarse_shift_candidates(src_for_shift, tgt_for_shift, args.shift_bin_px, args.shift_candidates):
                matrix = init_matrix.copy()
                trans = center_xy + shift
                source_plane = gaia_plane[in_frame]
                target_pred = source_plane @ matrix.T + trans
                matched_s = matched_t = None
                distances = np.empty((0,), dtype=np.float64)
                for _ in range(4):
                    matched_s, matched_t, distances = unique_matches(target_pred, image_fit, args.match_radius_px)
                    if len(matched_s) < int(args.min_matches):
                        break
                    matrix, trans = similarity_from_pairs(source_plane[matched_s], image_fit[matched_t], allow_scale=True)
                    target_pred = source_plane @ matrix.T + trans
                if matched_s is None or len(matched_s) < int(args.min_matches):
                    continue
                pred_matched = source_plane[matched_s] @ matrix.T + trans
                residuals = np.linalg.norm(pred_matched - image_fit[matched_t], axis=1)
                med = float(np.median(residuals))
                mad = float(np.median(np.abs(residuals - med)))
                keep = residuals <= min(float(args.match_radius_px), med + 3.0 * 1.4826 * mad)
                if np.count_nonzero(keep) >= int(args.min_matches):
                    matrix, trans = similarity_from_pairs(source_plane[matched_s][keep], image_fit[matched_t][keep], allow_scale=True)
                    pred_matched = source_plane[matched_s][keep] @ matrix.T + trans
                    residuals = np.linalg.norm(pred_matched - image_fit[matched_t][keep], axis=1)
                score = (len(residuals), -float(np.median(residuals)) if len(residuals) else -1e9)
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "matrix": matrix,
                        "shift": trans,
                        "matches": int(len(residuals)),
                        "median_residual_px": float(np.median(residuals)) if len(residuals) else None,
                        "mean_residual_px": float(np.mean(residuals)) if len(residuals) else None,
                        "angle_deg": float(angle),
                        "parity_index": int(parity_index),
                    }
    if best is None:
        raise RuntimeError("failed to fit Gaia-to-image transform")
    return best


def read_manifest_row(data_model: Path, sample_id: str) -> dict[str, str]:
    rows = [row for row in read_csv_rows(data_model / "manifest.csv") if row.get("sample_id") == sample_id]
    if not rows:
        raise RuntimeError(f"sample not found: {sample_id}")
    return rows[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Gaia-based true star annotations for a single and its aligned stack.")
    parser.add_argument("--data-model", type=Path, default=Path("data/data_model"))
    parser.add_argument("--gaia-csv", type=Path, default=Path("data/data_gaia/gaia_catalog_right_fixed/sample_000001_gaia_dr3_g20p0_right_fixed.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/data_gaia/gaia_annotations"))
    parser.add_argument("--sample-id", default="sample_000001")
    parser.add_argument("--mag-limit", type=float, default=20.0)
    parser.add_argument("--fit-gaia-mag-limit", type=float, default=15.0)
    parser.add_argument("--fit-gaia-top", type=int, default=1200)
    parser.add_argument("--fit-image-top", type=int, default=1200)
    parser.add_argument("--shift-gaia-top", type=int, default=120)
    parser.add_argument("--shift-image-top", type=int, default=120)
    parser.add_argument("--angle-step-deg", type=float, default=5.0)
    parser.add_argument("--shift-bin-px", type=float, default=32.0)
    parser.add_argument("--shift-candidates", type=int, default=8)
    parser.add_argument("--match-radius-px", type=float, default=12.0)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--detect-fwhm", type=float, default=3.0)
    parser.add_argument("--detect-max-peaks", type=int, default=3000)
    parser.add_argument("--annotation-margin-px", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_model = args.data_model.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    row = read_manifest_row(data_model, args.sample_id)
    single, single_header = read_fits_image(resolve_manifest_path(data_model, row["image_out"]))
    stack, _ = read_fits_image(resolve_manifest_path(data_model, row["stack_fits"]))
    aligned_stack = align_stack_to_single(stack, row, single.shape)
    h, w = single.shape

    ra0 = float(single_header.get("RA", single_header.get("CRVAL1")))
    dec0 = float(single_header.get("DEC", single_header.get("CRVAL2")))
    arcsec_per_px = 206.265 * float(single_header["XPIXSZ"]) / float(single_header["FOCALLEN"])

    gaia = pd.read_csv(args.gaia_csv)
    gaia = gaia[np.isfinite(gaia["ra"]) & np.isfinite(gaia["dec"]) & np.isfinite(gaia["phot_g_mean_mag"])]
    gaia = gaia[gaia["phot_g_mean_mag"] <= float(args.mag_limit)].copy()
    plane = tangent_arcsec(gaia["ra"].to_numpy(), gaia["dec"].to_numpy(), ra0, dec0)

    fit_mask = gaia["phot_g_mean_mag"].to_numpy() <= float(args.fit_gaia_mag_limit)
    fit_plane = plane[fit_mask]
    fit_mag = gaia["phot_g_mean_mag"].to_numpy()[fit_mask]
    if len(fit_plane) < int(args.min_matches):
        fit_plane = plane
        fit_mag = gaia["phot_g_mean_mag"].to_numpy()

    stack_detections = detect_image_stars(aligned_stack, [8.0, 7.0, 6.0, 5.0, 4.5], args.detect_fwhm, args.detect_max_peaks)
    if len(stack_detections) < int(args.min_matches):
        raise RuntimeError(f"not enough stack detections for fitting: {len(stack_detections)}")
    solution = fit_gaia_to_image(fit_plane, fit_mag, stack_detections, w, h, arcsec_per_px, args)

    xy = plane @ solution["matrix"].T + solution["shift"]
    inside = (
        (xy[:, 0] >= -float(args.annotation_margin_px))
        & (xy[:, 0] < w + float(args.annotation_margin_px))
        & (xy[:, 1] >= -float(args.annotation_margin_px))
        & (xy[:, 1] < h + float(args.annotation_margin_px))
    )
    ann = gaia.loc[inside].copy()
    ann["x"] = xy[inside, 0]
    ann["y"] = xy[inside, 1]
    ann["single_x"] = ann["x"]
    ann["single_y"] = ann["y"]
    ann["stack_aligned_x"] = ann["x"]
    ann["stack_aligned_y"] = ann["y"]
    stack_to_single_matrix = np.asarray(
        [
            [float(row.get("label_transform_a", 1.0) or 1.0), float(row.get("label_transform_b", 0.0) or 0.0)],
            [float(row.get("label_transform_c", 0.0) or 0.0), float(row.get("label_transform_d", 1.0) or 1.0)],
        ],
        dtype=np.float64,
    )
    stack_to_single_shift = np.asarray(
        [float(row.get("label_shift_x_px", 0.0) or 0.0), float(row.get("label_shift_y_px", 0.0) or 0.0)],
        dtype=np.float64,
    )
    single_xy = ann[["single_x", "single_y"]].to_numpy(dtype=np.float64)
    raw_stack_xy = (single_xy - stack_to_single_shift) @ np.linalg.inv(stack_to_single_matrix).T
    ann["raw_stack_x"] = raw_stack_xy[:, 0]
    ann["raw_stack_y"] = raw_stack_xy[:, 1]
    ann["g_mag"] = ann["phot_g_mean_mag"]
    ann = ann.sort_values("g_mag")

    cols = [
        "source_id",
        "ra",
        "dec",
        "x",
        "y",
        "single_x",
        "single_y",
        "stack_aligned_x",
        "stack_aligned_y",
        "raw_stack_x",
        "raw_stack_y",
        "g_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "parallax",
        "pmra",
        "pmdec",
        "ruwe",
    ]
    cols = [col for col in cols if col in ann.columns]
    common_path = out / f"{args.sample_id}_gaia_true_stars_g20_pixel.csv"
    single_path = out / f"{args.sample_id}_single_gaia_true_stars_g20.csv"
    stack_path = out / f"{args.sample_id}_aligned_stack_gaia_true_stars_g20.csv"
    raw_stack_path = out / f"{args.sample_id}_raw_stack_gaia_true_stars_g20.csv"
    ann[cols].to_csv(common_path, index=False)
    ann[cols].rename(columns={"x": "single_x", "y": "single_y"}).to_csv(single_path, index=False)
    ann[cols].rename(columns={"x": "stack_x", "y": "stack_y"}).to_csv(stack_path, index=False)
    ann[cols].rename(columns={"raw_stack_x": "stack_x", "raw_stack_y": "stack_y"}).to_csv(raw_stack_path, index=False)

    meta = {
        "sample_id": args.sample_id,
        "gaia_csv": str(args.gaia_csv.resolve()),
        "source_count_input": int(len(gaia)),
        "annotation_count_in_frame": int(len(ann)),
        "center_ra_deg": ra0,
        "center_dec_deg": dec0,
        "arcsec_per_px_header": arcsec_per_px,
        "fit_detection_count": int(len(stack_detections)),
        "fit_solution": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in solution.items()
            if key != "score"
        },
        "outputs": {
            "common": str(common_path),
            "single": str(single_path),
            "aligned_stack": str(stack_path),
            "raw_stack": str(raw_stack_path),
        },
    }
    (out / f"{args.sample_id}_gaia_annotation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
