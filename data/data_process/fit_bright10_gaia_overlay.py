from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "tool", ROOT / "data" / "data_process"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from daofind_opt import read_fits_image, resolve_manifest_path  # noqa: E402
from build_gaia_annotations import detect_image_stars, fit_gaia_to_image, tangent_arcsec  # noqa: E402


def robust_show(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    lo, hi = np.percentile(finite, [1.0, 99.85])
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Gaia-to-single using only the 10 brightest image detections.")
    parser.add_argument("--sample-id", default="sample_000747")
    parser.add_argument("--data-model", type=Path, default=Path("data/data_model"))
    parser.add_argument("--selected-samples", type=Path, default=Path("data/data_20/selected_samples.csv"))
    parser.add_argument("--gaia-csv", type=Path, default=Path("data/data_gaia/gaia_catalog_right_fixed/sample_000747_gaia_dr3_g20p0_right_fixed.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/data_20/gaia_overlay_check"))
    parser.add_argument("--circle-radius", type=float, default=24.0)
    parser.add_argument("--mag-limit", type=float, default=13.0)
    args = parser.parse_args()

    data_model = (ROOT / args.data_model).resolve()
    selected = pd.read_csv((ROOT / args.selected_samples).resolve())
    row = selected[selected["sample_id"] == args.sample_id].iloc[0]
    single_path = resolve_manifest_path(data_model, row["single_fits"])
    image, header = read_fits_image(single_path)
    h, w = image.shape

    ra0 = float(header.get("RA", header.get("CRVAL1")))
    dec0 = float(header.get("DEC", header.get("CRVAL2")))
    header_arcsec_per_px = 206.265 * float(header["XPIXSZ"]) / float(header["FOCALLEN"])

    gaia = pd.read_csv((ROOT / args.gaia_csv).resolve())
    gaia = gaia[np.isfinite(gaia["ra"]) & np.isfinite(gaia["dec"]) & np.isfinite(gaia["phot_g_mean_mag"])].copy()
    plane = tangent_arcsec(gaia["ra"].to_numpy(), gaia["dec"].to_numpy(), ra0, dec0)

    image_detections = detect_image_stars(image, [10.0, 8.0, 6.0, 5.0], fwhm=3.0, max_peaks=500)
    image_top10 = image_detections[:10].copy()
    if len(image_top10) < 4:
        raise RuntimeError(f"not enough image detections: {len(image_top10)}")

    fit_mask = gaia["phot_g_mean_mag"].to_numpy() <= 15.0
    fit_plane = plane[fit_mask]
    fit_mag = gaia["phot_g_mean_mag"].to_numpy()[fit_mask]

    fit_args = SimpleNamespace(
        fit_gaia_top=1200,
        fit_image_top=10,
        shift_gaia_top=60,
        shift_image_top=10,
        angle_step_deg=2.0,
        shift_bin_px=32.0,
        shift_candidates=12,
        match_radius_px=18.0,
        min_matches=4,
    )
    solution = fit_gaia_to_image(fit_plane, fit_mag, image_top10, w, h, header_arcsec_per_px, fit_args)

    xy = plane @ np.asarray(solution["matrix"], dtype=np.float64).T + np.asarray(solution["shift"], dtype=np.float64)
    ann = gaia.copy()
    ann["x"] = xy[:, 0]
    ann["y"] = xy[:, 1]
    inside = (ann["x"] >= 0) & (ann["x"] < w) & (ann["y"] >= 0) & (ann["y"] < h)
    ann = ann[inside].copy()
    stars = ann[ann["phot_g_mean_mag"] <= float(args.mag_limit)].copy().sort_values("phot_g_mean_mag")

    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(image_top10, columns=["x", "y", "peak", "snr", "flux"]).to_csv(out / f"{args.sample_id}_bright10_image_detections.csv", index=False)
    stars.to_csv(out / f"{args.sample_id}_bright10_fit_gaia_g13_pixels.csv", index=False)

    show = robust_show(image)
    fig, ax = plt.subplots(figsize=(10, 16), constrained_layout=True)
    ax.imshow(show, cmap="gray", origin="upper")
    first = True
    for _, star in stars.iterrows():
        ax.add_patch(
            Circle(
                (float(star["x"]), float(star["y"])),
                radius=float(args.circle_radius),
                fill=False,
                edgecolor="red",
                linewidth=1.2,
                alpha=0.95,
                label=f"Bright10-fit Gaia G<={args.mag_limit:g}" if first else None,
            )
        )
        first = False
    ax.scatter(image_top10[:, 0], image_top10[:, 1], c="lime", s=22, marker="+", linewidths=1.4, label="image brightest 10")
    ax.set_title(
        f"{args.sample_id}: Gaia G<={args.mag_limit:g} circles fitted by image brightest 10\n"
        f"fit matches={solution['matches']}, median residual={solution['median_residual_px']:.3f}px"
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    out_png = out / f"{args.sample_id}_bright10_fit_gaia_g13_radius{int(args.circle_radius)}_overlay_full.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    meta = {
        "sample_id": args.sample_id,
        "single_path": str(single_path),
        "header_arcsec_per_px": header_arcsec_per_px,
        "fit_solution": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in solution.items()
            if key != "score"
        },
        "gaia_g13_count_inside": int(len(stars)),
        "output_png": str(out_png),
    }
    pd.Series(meta, dtype="object").to_json(out / f"{args.sample_id}_bright10_fit_meta.json", force_ascii=False, indent=2)
    print(meta)


if __name__ == "__main__":
    main()
