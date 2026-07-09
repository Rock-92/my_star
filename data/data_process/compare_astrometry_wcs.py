from __future__ import annotations

import json
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


def tangent_inverse_arcsec(xi_eta_arcsec: np.ndarray, ra0_deg: float, dec0_deg: float) -> SkyCoord:
    xi = np.deg2rad(np.asarray(xi_eta_arcsec[:, 0], dtype=float) / 3600.0)
    eta = np.deg2rad(np.asarray(xi_eta_arcsec[:, 1], dtype=float) / 3600.0)
    ra0 = np.deg2rad(float(ra0_deg))
    dec0 = np.deg2rad(float(dec0_deg))
    denom = np.cos(dec0) - eta * np.sin(dec0)
    ra = ra0 + np.arctan2(xi, denom)
    dec = np.arctan2(
        np.sin(dec0) + eta * np.cos(dec0),
        np.sqrt(xi * xi + denom * denom),
    )
    return SkyCoord(ra=np.rad2deg(ra) * u.deg, dec=np.rad2deg(dec) * u.deg, frame="icrs")


def main() -> None:
    wcs_path = Path(r"C:\Users\Lenovo\Downloads\wcs (1).fits")
    ours_meta = Path(r"E:\Code\my_star\data\data_gaia\gaia_annotations_right_fixed\sample_000001_gaia_annotation_meta.json")

    hdr = fits.getheader(wcs_path)
    wcs = WCS(hdr)
    cd = np.array([[hdr["CD1_1"], hdr["CD1_2"]], [hdr["CD2_1"], hdr["CD2_2"]]], dtype=float)
    astrometry_scale = float(np.sqrt(abs(np.linalg.det(cd))) * 3600.0)
    points = np.array(
        [
            [1080.0, 1920.0],
            [0.0, 0.0],
            [2160.0, 0.0],
            [0.0, 3840.0],
            [2160.0, 3840.0],
            [float(hdr["CRPIX1"]), float(hdr["CRPIX2"])],
        ],
        dtype=float,
    )
    sky = wcs.pixel_to_world(points[:, 0], points[:, 1])
    screenshot_center = SkyCoord(236.646 * u.deg, 24.411 * u.deg)

    meta = json.loads(ours_meta.read_text(encoding="utf-8"))
    matrix = np.array(meta["fit_solution"]["matrix"], dtype=float)
    shift = np.array(meta["fit_solution"]["shift"], dtype=float)
    ours_scale = float(1.0 / np.sqrt(abs(np.linalg.det(matrix))))
    ours_plane = (points - shift) @ np.linalg.inv(matrix).T
    ours_sky = tangent_inverse_arcsec(ours_plane, meta["center_ra_deg"], meta["center_dec_deg"])

    print("Astrometry.net WCS")
    print(f"  IMAGEW x IMAGEH: {hdr['IMAGEW']} x {hdr['IMAGEH']}")
    print(f"  CRPIX: ({hdr['CRPIX1']:.6f}, {hdr['CRPIX2']:.6f})")
    print(f"  CRVAL: ({hdr['CRVAL1']:.9f}, {hdr['CRVAL2']:.9f})")
    print(f"  pixel scale: {astrometry_scale:.6f} arcsec/px")
    for name, coord in zip(["center", "corner00", "corner10", "corner01", "corner11", "crpix"], sky):
        print(f"  {name:8s}: RA={coord.ra.deg:.9f}, Dec={coord.dec.deg:.9f}")
    print(f"  center vs screenshot center separation: {screenshot_center.separation(sky[0]).arcsec:.3f} arcsec")

    print("\nOur Gaia fit for sample_000001")
    print(f"  header center RA/Dec used for tangent plane: ({meta['center_ra_deg']:.9f}, {meta['center_dec_deg']:.9f})")
    print(f"  header initial scale: {meta['arcsec_per_px_header']:.6f} arcsec/px")
    print(f"  fitted scale: {ours_scale:.6f} arcsec/px")
    print(f"  matches: {meta['fit_solution']['matches']}")
    print(f"  median residual: {meta['fit_solution']['median_residual_px']:.6f} px")
    print(f"  mean residual: {meta['fit_solution']['mean_residual_px']:.6f} px")
    print(f"  matrix: {matrix.tolist()}")
    print(f"  shift: {meta['fit_solution']['shift']}")
    for name, coord in zip(["center", "corner00", "corner10", "corner01", "corner11", "crpix"], ours_sky):
        print(f"  ours {name:8s}: RA={coord.ra.deg:.9f}, Dec={coord.dec.deg:.9f}")
    print(f"  our center vs astrometry center separation: {ours_sky[0].separation(sky[0]).arcsec:.3f} arcsec")


if __name__ == "__main__":
    main()
