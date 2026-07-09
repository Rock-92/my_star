from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table, unique, vstack
from astroquery.gaia import Gaia

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "tool"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from daofind_opt import read_fits_image, resolve_manifest_path  # noqa: E402


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
    solution = meta["fit_solution"]
    matrix = np.asarray(solution["matrix"], dtype=np.float64)
    shift = np.asarray(solution["shift"], dtype=np.float64)
    plane = (np.asarray(xy, dtype=np.float64) - shift) @ np.linalg.inv(matrix).T
    return inverse_tangent_arcsec(plane, float(meta["center_ra_deg"]), float(meta["center_dec_deg"]))


def query_gaia_box(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    mag_limit: float,
    tile_deg: float,
    retries: int,
    retry_sleep: float,
) -> tuple[Table, str]:
    parts: list[Table] = []
    queries: list[str] = []
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
            result = None
            last_error: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    result = Gaia.launch_job(query, dump_to_file=False).get_results()
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"      attempt {attempt}/{retries} failed: {exc}", flush=True)
                    if attempt < retries:
                        time.sleep(float(retry_sleep))
            if result is None:
                raise RuntimeError("Gaia strip tile failed") from last_error
            print(f"      rows={len(result)}", flush=True)
            if len(result):
                parts.append(result)
            queries.append(query)
    if not parts:
        return Table(), "\n\n".join(queries)
    return unique(vstack(parts), keys="source_id"), "\n\n".join(queries)


def table_from_csv(path: Path) -> Table:
    table = Table.read(path, format="csv")
    if "source_id" in table.colnames:
        table["source_id"] = table["source_id"].astype(str)
    return table


def normalize_source_id(table: Table) -> Table:
    if "source_id" in table.colnames:
        table = table.copy()
        table["source_id"] = table["source_id"].astype(str)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplement only the missing right-edge Gaia strip and rebuild annotations.")
    parser.add_argument("--data-model", type=Path, default=Path("data/data_model"))
    parser.add_argument("--selected-samples", type=Path, default=Path("data/data_20/selected_samples.csv"))
    parser.add_argument("--old-catalog-dir", type=Path, default=Path("data/data_gaia/gaia_catalog"))
    parser.add_argument("--old-annotation-dir", type=Path, default=Path("data/data_gaia/gaia_annotations"))
    parser.add_argument("--catalog-output", type=Path, default=Path("data/data_gaia/gaia_catalog_right_supplement"))
    parser.add_argument("--annotation-output", type=Path, default=Path("data/data_gaia/gaia_annotations_right_supplement"))
    parser.add_argument("--mag-limit", type=float, default=20.0)
    parser.add_argument("--fit-gaia-mag-limit", type=float, default=15.0)
    parser.add_argument("--overlap-px", type=float, default=80.0)
    parser.add_argument("--margin-deg", type=float, default=0.08)
    parser.add_argument("--tile-deg", type=float, default=0.45)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    data_model = (ROOT / args.data_model).resolve()
    selected = pd.read_csv((ROOT / args.selected_samples).resolve())
    old_catalog_dir = (ROOT / args.old_catalog_dir).resolve()
    old_ann_dir = (ROOT / args.old_annotation_dir).resolve()
    catalog_out = (ROOT / args.catalog_output).resolve()
    ann_out = (ROOT / args.annotation_output).resolve()
    catalog_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)
    Gaia.ROW_LIMIT = -1

    for index, row in selected.iterrows():
        sample_id = str(row["sample_id"])
        print(f"[{index + 1}/{len(selected)}] {sample_id}", flush=True)
        image, _ = read_fits_image(resolve_manifest_path(data_model, row["single_fits"]))
        h, w = image.shape
        old_ann_path = old_ann_dir / f"{sample_id}_gaia_true_stars_g20_pixel.csv"
        old_meta_path = old_ann_dir / f"{sample_id}_gaia_annotation_meta.json"
        old_catalog_path = old_catalog_dir / f"{sample_id}_gaia_dr3_g{str(args.mag_limit).replace('.', 'p')}.csv"
        if not old_ann_path.exists() or not old_meta_path.exists() or not old_catalog_path.exists():
            print("  skip: missing old catalog/annotation/meta", flush=True)
            continue
        old_ann = pd.read_csv(old_ann_path)
        old_x_max = float(old_ann["x"].max())
        x0 = max(0.0, old_x_max - float(args.overlap_px))
        if x0 >= w - 8:
            print(f"  skip: old annotation already reaches x={old_x_max:.1f}", flush=True)
            continue
        meta = json.loads(old_meta_path.read_text(encoding="utf-8"))
        strip_corners = np.asarray(
            [
                [x0, 0.0],
                [w - 1.0, 0.0],
                [x0, h - 1.0],
                [w - 1.0, h - 1.0],
            ],
            dtype=np.float64,
        )
        radec = image_xy_to_radec(strip_corners, meta)
        ra_values = radec[:, 0]
        dec_values = radec[:, 1]
        if np.ptp(ra_values) > 180:
            raise RuntimeError("RA wrap-around is not handled for this field")
        dec_center = float(np.mean(dec_values))
        cos_dec = max(math.cos(math.radians(dec_center)), 0.1)
        ra_min = float(np.min(ra_values) - float(args.margin_deg) / cos_dec)
        ra_max = float(np.max(ra_values) + float(args.margin_deg) / cos_dec)
        dec_min = float(np.min(dec_values) - float(args.margin_deg))
        dec_max = float(np.max(dec_values) + float(args.margin_deg))
        merged_path = catalog_out / f"{sample_id}_gaia_dr3_g{str(args.mag_limit).replace('.', 'p')}_right_supp.csv"
        supp_path = catalog_out / f"{sample_id}_right_strip_only_g{str(args.mag_limit).replace('.', 'p')}.csv"
        if merged_path.exists() and args.skip_existing:
            print(f"  reuse merged catalog: {merged_path}", flush=True)
        else:
            print(
                f"  old_x_max={old_x_max:.1f}, strip x=({x0:.1f},{w - 1}) "
                f"ra=({ra_min:.6f},{ra_max:.6f}) dec=({dec_min:.6f},{dec_max:.6f})",
                flush=True,
            )
            supp, query_text = query_gaia_box(
                ra_min,
                ra_max,
                dec_min,
                dec_max,
                float(args.mag_limit),
                float(args.tile_deg),
                int(args.retries),
                float(args.retry_sleep),
            )
            supp = normalize_source_id(supp)
            if len(supp):
                supp.write(supp_path, format="csv", overwrite=True)
                merged = unique(vstack([table_from_csv(old_catalog_path), supp]), keys="source_id")
            else:
                merged = table_from_csv(old_catalog_path)
            merged.write(merged_path, format="csv", overwrite=True)
            meta_out = {
                "sample_id": sample_id,
                "old_catalog_count": int(len(table_from_csv(old_catalog_path))),
                "supplement_count_raw": int(len(supp)),
                "merged_count": int(len(merged)),
                "old_annotation_count": int(len(old_ann)),
                "old_x_max": old_x_max,
                "strip_x0": x0,
                "strip_x1": float(w - 1),
                "query_ra_min": ra_min,
                "query_ra_max": ra_max,
                "query_dec_min": dec_min,
                "query_dec_max": dec_max,
                "query": query_text,
            }
            (catalog_out / f"{sample_id}_right_supp_meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
            print(f"  merged rows={len(merged)} -> {merged_path}", flush=True)

        cmd = [
            sys.executable,
            str(ROOT / "data/data_process/build_gaia_annotations.py"),
            "--data-model",
            str(data_model),
            "--sample-id",
            sample_id,
            "--gaia-csv",
            str(merged_path),
            "--output",
            str(ann_out),
            "--mag-limit",
            str(float(args.mag_limit)),
            "--fit-gaia-mag-limit",
            str(float(args.fit_gaia_mag_limit)),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
