from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "tool"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from daofind_opt import read_fits_image, resolve_manifest_path  # noqa: E402


def tangent_arcsec(ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    ra0 = math.radians(float(ra0_deg))
    dec0 = math.radians(float(dec0_deg))
    dra = ra - ra0
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(dra)
    xi = np.cos(dec) * np.sin(dra) / np.maximum(cosc, 1e-12)
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(dra)) / np.maximum(cosc, 1e-12)
    return np.column_stack((np.rad2deg(xi) * 3600.0, np.rad2deg(eta) * 3600.0))


def inverse_tangent_arcsec(xi_eta_arcsec: np.ndarray, ra0_deg: float, dec0_deg: float) -> np.ndarray:
    xi = np.deg2rad(np.asarray(xi_eta_arcsec[:, 0], dtype=np.float64) / 3600.0)
    eta = np.deg2rad(np.asarray(xi_eta_arcsec[:, 1], dtype=np.float64) / 3600.0)
    ra0 = math.radians(float(ra0_deg))
    dec0 = math.radians(float(dec0_deg))
    denom = np.cos(dec0) - eta * np.sin(dec0)
    ra = ra0 + np.arctan2(xi, denom)
    dec = np.arctan2(
        np.sin(dec0) + eta * np.cos(dec0),
        np.sqrt(denom * denom + xi * xi),
    )
    return np.column_stack((np.rad2deg(ra) % 360.0, np.rad2deg(dec)))


def image_xy_to_radec(xy: np.ndarray, meta: dict) -> np.ndarray:
    matrix = np.asarray(meta["fit_solution"]["matrix"], dtype=np.float64)
    shift = np.asarray(meta["fit_solution"]["shift"], dtype=np.float64)
    plane = (np.asarray(xy, dtype=np.float64) - shift) @ np.linalg.inv(matrix).T
    return inverse_tangent_arcsec(plane, float(meta["center_ra_deg"]), float(meta["center_dec_deg"]))


def project_catalog(catalog: pd.DataFrame, meta: dict) -> np.ndarray:
    matrix = np.asarray(meta["fit_solution"]["matrix"], dtype=np.float64)
    shift = np.asarray(meta["fit_solution"]["shift"], dtype=np.float64)
    plane = tangent_arcsec(catalog["ra"].to_numpy(), catalog["dec"].to_numpy(), meta["center_ra_deg"], meta["center_dec_deg"])
    return plane @ matrix.T + shift


def query_strip(ra_min: float, ra_max: float, dec_min: float, dec_max: float, mag_limit: float, tile_deg: float, retries: int, retry_sleep: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    ra_edges = np.arange(ra_min, ra_max, tile_deg)
    dec_edges = np.arange(dec_min, dec_max, tile_deg)
    if len(ra_edges) == 0 or ra_edges[-1] < ra_max:
        ra_edges = np.append(ra_edges, ra_max)
    if len(dec_edges) == 0 or dec_edges[-1] < dec_max:
        dec_edges = np.append(dec_edges, dec_max)
    total = max((len(ra_edges) - 1) * (len(dec_edges) - 1), 1)
    index = 0
    for i in range(len(ra_edges) - 1):
        for j in range(len(dec_edges) - 1):
            index += 1
            r0, r1 = float(ra_edges[i]), float(ra_edges[i + 1])
            d0, d1 = float(dec_edges[j]), float(dec_edges[j + 1])
            query = f"""
            SELECT
                source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error,
                ruwe, astrometric_excess_noise, radial_velocity
            FROM gaiadr3.gaia_source
            WHERE
                ra >= {r0:.10f} AND ra < {r1:.10f}
                AND dec >= {d0:.10f} AND dec < {d1:.10f}
                AND phot_g_mean_mag <= {float(mag_limit):.4f}
            """
            print(f"    [tile {index}/{total}] ra=({r0:.5f},{r1:.5f}) dec=({d0:.5f},{d1:.5f})", flush=True)
            table = None
            last_error: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    table = Gaia.launch_job(query, dump_to_file=False).get_results()
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"      attempt {attempt}/{retries} failed: {exc}", flush=True)
                    if attempt < retries:
                        time.sleep(float(retry_sleep))
            if table is None:
                raise RuntimeError("Gaia query failed") from last_error
            print(f"      rows={len(table)}", flush=True)
            if len(table):
                frames.append(table.to_pandas())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("source_id")


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplement right-edge Gaia sources, then project with the existing fixed astrometric solution.")
    parser.add_argument("--data-model", type=Path, default=Path("data/data_model"))
    parser.add_argument("--selected-samples", type=Path, default=Path("data/data_20/selected_samples.csv"))
    parser.add_argument("--old-annotation-dir", type=Path, default=Path("data/data_gaia/gaia_annotations"))
    parser.add_argument("--catalog-output", type=Path, default=Path("data/data_gaia/gaia_catalog_right_fixed"))
    parser.add_argument("--annotation-output", type=Path, default=Path("data/data_gaia/gaia_annotations_right_fixed"))
    parser.add_argument("--mag-limit", type=float, default=20.0)
    parser.add_argument("--overlap-px", type=float, default=100.0)
    parser.add_argument("--margin-deg", type=float, default=0.08)
    parser.add_argument("--tile-deg", type=float, default=0.45)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    data_model = (ROOT / args.data_model).resolve()
    selected = pd.read_csv((ROOT / args.selected_samples).resolve())
    old_ann_dir = (ROOT / args.old_annotation_dir).resolve()
    catalog_out = (ROOT / args.catalog_output).resolve()
    ann_out = (ROOT / args.annotation_output).resolve()
    catalog_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)
    Gaia.ROW_LIMIT = -1

    summary: list[dict[str, object]] = []
    for index, row in selected.iterrows():
        sample_id = str(row["sample_id"])
        print(f"[{index + 1}/{len(selected)}] {sample_id}", flush=True)
        image, _ = read_fits_image(resolve_manifest_path(data_model, row["single_fits"]))
        h, w = image.shape
        meta_path = old_ann_dir / f"{sample_id}_gaia_annotation_meta.json"
        old_ann_path = old_ann_dir / f"{sample_id}_gaia_true_stars_g20_pixel.csv"
        if not meta_path.exists() or not old_ann_path.exists():
            print("  skip missing old annotation/meta", flush=True)
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        old_catalog_path = Path(meta["gaia_csv"])
        old_catalog = pd.read_csv(old_catalog_path)
        old_catalog["source_id"] = old_catalog["source_id"].astype(str)
        old_ann = pd.read_csv(old_ann_path)
        old_x_max = float(old_ann["x"].max())
        x0 = max(0.0, old_x_max - float(args.overlap_px))
        merged_path = catalog_out / f"{sample_id}_gaia_dr3_g{str(args.mag_limit).replace('.', 'p')}_right_fixed.csv"
        strip_path = catalog_out / f"{sample_id}_right_strip_g{str(args.mag_limit).replace('.', 'p')}.csv"

        if merged_path.exists() and args.skip_existing:
            merged = pd.read_csv(merged_path)
        else:
            corners = np.asarray([[x0, 0.0], [w - 1.0, 0.0], [x0, h - 1.0], [w - 1.0, h - 1.0]], dtype=np.float64)
            radec = image_xy_to_radec(corners, meta)
            ra_values = radec[:, 0]
            dec_values = radec[:, 1]
            if np.ptp(ra_values) > 180:
                raise RuntimeError("RA wrap-around is not handled")
            dec_center = float(np.mean(dec_values))
            cos_dec = max(math.cos(math.radians(dec_center)), 0.1)
            ra_min = float(np.min(ra_values) - float(args.margin_deg) / cos_dec)
            ra_max = float(np.max(ra_values) + float(args.margin_deg) / cos_dec)
            dec_min = float(np.min(dec_values) - float(args.margin_deg))
            dec_max = float(np.max(dec_values) + float(args.margin_deg))
            print(f"  old_x_max={old_x_max:.1f}; strip x=({x0:.1f},{w - 1})", flush=True)
            supplement = query_strip(
                ra_min,
                ra_max,
                dec_min,
                dec_max,
                float(args.mag_limit),
                float(args.tile_deg),
                int(args.retries),
                float(args.retry_sleep),
            )
            if len(supplement):
                supplement["source_id"] = supplement["source_id"].astype(str)
                supplement.to_csv(strip_path, index=False)
                merged = pd.concat([old_catalog, supplement], ignore_index=True).drop_duplicates("source_id")
            else:
                merged = old_catalog.copy()
            merged.to_csv(merged_path, index=False)

        xy = project_catalog(merged, meta)
        inside = (xy[:, 0] >= -2.0) & (xy[:, 0] < w + 2.0) & (xy[:, 1] >= -2.0) & (xy[:, 1] < h + 2.0)
        ann = merged.loc[inside].copy()
        ann["x"] = xy[inside, 0]
        ann["y"] = xy[inside, 1]
        ann["single_x"] = ann["x"]
        ann["single_y"] = ann["y"]
        ann["stack_aligned_x"] = ann["x"]
        ann["stack_aligned_y"] = ann["y"]
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
            "g_mag",
            "phot_bp_mean_mag",
            "phot_rp_mean_mag",
            "parallax",
            "pmra",
            "pmdec",
            "ruwe",
        ]
        cols = [col for col in cols if col in ann.columns]
        ann[cols].to_csv(ann_out / f"{sample_id}_gaia_true_stars_g20_pixel.csv", index=False)
        ann[cols].to_csv(ann_out / f"{sample_id}_single_gaia_true_stars_g20.csv", index=False)
        meta_new = dict(meta)
        meta_new["gaia_csv"] = str(merged_path)
        meta_new["annotation_count_in_frame"] = int(len(ann))
        meta_new["fixed_solution_from"] = str(meta_path)
        (ann_out / f"{sample_id}_gaia_annotation_meta.json").write_text(json.dumps(meta_new, indent=2), encoding="utf-8")
        new_x_max = float(ann["x"].max()) if len(ann) else float("nan")
        print(f"  annotation {len(old_ann)} -> {len(ann)}; x_max {old_x_max:.1f} -> {new_x_max:.1f}", flush=True)
        summary.append(
            {
                "sample_id": sample_id,
                "old_count": int(len(old_ann)),
                "new_count": int(len(ann)),
                "old_x_max": old_x_max,
                "new_x_max": new_x_max,
                "right_gap_px": float(w - new_x_max),
            }
        )
    pd.DataFrame(summary).to_csv(ann_out / "right_supplement_summary.csv", index=False)


if __name__ == "__main__":
    main()
