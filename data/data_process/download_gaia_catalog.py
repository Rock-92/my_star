from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import vstack, unique
from astroquery.gaia import Gaia


def read_manifest_row(data_model: Path, sample_id: str) -> dict[str, str]:
    with (data_model / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["sample_id"] == sample_id:
                return row
    raise RuntimeError(f"sample_id not found in manifest: {sample_id}")


def resolve_manifest_path(data_model: Path, path_text: str) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    if path.is_absolute():
        return path
    single_star_root = data_model.resolve().parents[1]
    candidate = single_star_root / path
    if candidate.exists():
        return candidate
    return data_model / path


def estimate_radius_deg(header: fits.Header, margin: float) -> tuple[float, float]:
    width = int(header["NAXIS1"])
    height = int(header["NAXIS2"])
    focal_mm = float(header["FOCALLEN"])
    pix_um = float(header.get("XPIXSZ", header.get("YPIXSZ")))
    arcsec_per_px = 206.265 * pix_um / focal_mm
    radius_deg = 0.5 * float(np.hypot(width, height)) * arcsec_per_px / 3600.0
    return radius_deg * float(margin), arcsec_per_px


def estimate_box_deg(header: fits.Header, margin: float) -> tuple[float, float, float]:
    width = int(header["NAXIS1"])
    height = int(header["NAXIS2"])
    focal_mm = float(header["FOCALLEN"])
    pix_um = float(header.get("XPIXSZ", header.get("YPIXSZ")))
    arcsec_per_px = 206.265 * pix_um / focal_mm
    width_deg = width * arcsec_per_px / 3600.0 * float(margin)
    height_deg = height * arcsec_per_px / 3600.0 * float(margin)
    return width_deg, height_deg, arcsec_per_px


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Gaia DR3 sources for the current FITS field.")
    parser.add_argument("--data-model", type=Path, default=Path("data/data_model"))
    parser.add_argument("--sample-id", default="sample_000001")
    parser.add_argument("--output", type=Path, default=Path("data/data_gaia/gaia_catalog"))
    parser.add_argument("--mag-limit", type=float, default=20.0)
    parser.add_argument("--radius-margin", type=float, default=1.03)
    parser.add_argument("--mode", choices=("circle", "tiles"), default="tiles")
    parser.add_argument("--tile-deg", type=float, default=0.45)
    parser.add_argument("--row-limit", type=int, default=-1, help="Astroquery Gaia row limit; -1 means unlimited.")
    parser.add_argument("--tile-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_model = args.data_model.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    row = read_manifest_row(data_model, args.sample_id)
    image_path = resolve_manifest_path(data_model, row["image_out"])
    header = fits.getheader(image_path)
    ra = float(header.get("RA", header.get("CRVAL1")))
    dec = float(header.get("DEC", header.get("CRVAL2")))
    radius_deg, arcsec_per_px = estimate_radius_deg(header, args.radius_margin)
    width_deg, height_deg, arcsec_per_px = estimate_box_deg(header, args.radius_margin)
    center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    Gaia.ROW_LIMIT = int(args.row_limit)
    query = f"""
    SELECT
        source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
        parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error,
        ruwe, astrometric_excess_noise, radial_velocity
    FROM gaiadr3.gaia_source
    WHERE
        CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra:.10f}, {dec:.10f}, {radius_deg:.10f})
        ) = 1
        AND phot_g_mean_mag <= {float(args.mag_limit):.4f}
    """

    print(f"sample_id={args.sample_id}")
    print(f"image={image_path}")
    print(f"center_ra_dec_deg={ra:.8f},{dec:.8f}")
    print(f"arcsec_per_px={arcsec_per_px:.6f}")
    print(f"query_radius_deg={radius_deg:.6f}")
    print(f"query_box_deg={width_deg:.6f} x {height_deg:.6f}")
    print(f"mag_limit=G<={float(args.mag_limit):.2f}")
    if args.mode == "circle":
        job = Gaia.launch_job_async(query, dump_to_file=False)
        table = job.get_results()
        query_text = query
    else:
        dec_min = dec - height_deg / 2.0
        dec_max = dec + height_deg / 2.0
        ra_half_width = width_deg / (2.0 * max(np.cos(np.deg2rad(dec)), 0.1))
        ra_min = ra - ra_half_width
        ra_max = ra + ra_half_width
        tile_deg = float(args.tile_deg)
        ra_edges = np.arange(ra_min, ra_max, tile_deg)
        dec_edges = np.arange(dec_min, dec_max, tile_deg)
        if len(ra_edges) == 0 or ra_edges[-1] < ra_max:
            ra_edges = np.append(ra_edges, ra_max)
        if len(dec_edges) == 0 or dec_edges[-1] < dec_max:
            dec_edges = np.append(dec_edges, dec_max)
        parts = []
        tile_queries = []
        total_tiles = max((len(ra_edges) - 1) * (len(dec_edges) - 1), 1)
        tile_index = 0
        for i in range(len(ra_edges) - 1):
            for j in range(len(dec_edges) - 1):
                tile_index += 1
                r0, r1 = float(ra_edges[i]), float(ra_edges[i + 1])
                d0, d1 = float(dec_edges[j]), float(dec_edges[j + 1])
                tile_query = f"""
                SELECT
                    source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                    parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error,
                    ruwe, astrometric_excess_noise, radial_velocity
                FROM gaiadr3.gaia_source
                WHERE
                    ra >= {r0:.10f} AND ra < {r1:.10f}
                    AND dec >= {d0:.10f} AND dec < {d1:.10f}
                    AND phot_g_mean_mag <= {float(args.mag_limit):.4f}
                """
                print(f"[tile {tile_index}/{total_tiles}] ra=({r0:.5f},{r1:.5f}) dec=({d0:.5f},{d1:.5f})", flush=True)
                tile_table = None
                last_error: Exception | None = None
                for attempt in range(1, int(args.tile_retries) + 1):
                    try:
                        job = Gaia.launch_job(tile_query, dump_to_file=False)
                        tile_table = job.get_results()
                        break
                    except Exception as exc:
                        last_error = exc
                        print(f"  attempt {attempt}/{args.tile_retries} failed: {exc}", flush=True)
                        if attempt < int(args.tile_retries):
                            time.sleep(float(args.retry_sleep))
                if tile_table is None:
                    raise RuntimeError(f"tile {tile_index}/{total_tiles} failed after retries") from last_error
                print(f"  rows={len(tile_table)}", flush=True)
                if len(tile_table):
                    parts.append(tile_table)
                tile_queries.append(tile_query)
        if not parts:
            raise RuntimeError("Gaia returned no sources")
        table = unique(vstack(parts), keys="source_id")
        query_text = "\n\n".join(tile_queries)

    csv_path = out / f"{args.sample_id}_gaia_dr3_g{str(args.mag_limit).replace('.', 'p')}.csv"
    vot_path = out / f"{args.sample_id}_gaia_dr3_g{str(args.mag_limit).replace('.', 'p')}.vot"
    table.write(csv_path, format="csv", overwrite=True)
    table.write(vot_path, format="votable", overwrite=True)

    meta = {
        "sample_id": args.sample_id,
        "image": str(image_path),
        "gaia_table": "gaiadr3.gaia_source",
        "center_ra_deg": ra,
        "center_dec_deg": dec,
        "radius_deg": radius_deg,
        "box_width_deg": width_deg,
        "box_height_deg": height_deg,
        "mode": args.mode,
        "tile_deg": float(args.tile_deg),
        "mag_limit_g": float(args.mag_limit),
        "source_count": int(len(table)),
        "arcsec_per_px": arcsec_per_px,
        "naxis1": int(header["NAXIS1"]),
        "naxis2": int(header["NAXIS2"]),
        "focal_mm": float(header["FOCALLEN"]),
        "pixel_um": float(header.get("XPIXSZ", header.get("YPIXSZ"))),
        "csv": str(csv_path),
        "votable": str(vot_path),
        "query": query_text,
    }
    (out / f"{args.sample_id}_gaia_dr3_g{str(args.mag_limit).replace('.', 'p')}_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
