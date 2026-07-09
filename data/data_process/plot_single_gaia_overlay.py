from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "tool"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from daofind_opt import read_fits_image, resolve_manifest_path  # noqa: E402


def robust_show(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    lo, hi = np.percentile(finite, [1.0, 99.85])
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def add_circles(ax, stars: pd.DataFrame, radius: float, color: str, label: str | None = None, linewidth: float = 1.0) -> None:
    first = True
    for _, star in stars.iterrows():
        ax.add_patch(
            Circle(
                (float(star["x"]), float(star["y"])),
                radius=radius,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                alpha=0.95,
                label=label if first and label else None,
            )
        )
        first = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default="sample_000747")
    parser.add_argument("--data-model", type=Path, default=Path("data/data_model"))
    parser.add_argument("--selected-samples", type=Path, default=Path("data/data_20/selected_samples.csv"))
    parser.add_argument("--gaia-dir", type=Path, default=Path("data/data_gaia/gaia_annotations_right_fixed"))
    parser.add_argument("--output", type=Path, default=Path("data/data_20/gaia_overlay_check"))
    parser.add_argument("--mag-limit", type=float, default=12.0)
    parser.add_argument("--max-stars", type=int, default=160)
    parser.add_argument("--circle-radius", type=float, default=8.0)
    parser.add_argument("--crop-half", type=int, default=70)
    parser.add_argument("--num-crops", type=int, default=6)
    parser.add_argument("--color-bins", action="store_true", help="Draw G<=13, 13-14, and 14-15 stars in different colors.")
    args = parser.parse_args()

    data_model = (ROOT / args.data_model).resolve()
    selected = pd.read_csv((ROOT / args.selected_samples).resolve())
    row = selected[selected["sample_id"] == args.sample_id].iloc[0]
    single_path = resolve_manifest_path(data_model, row["single_fits"])
    image, _ = read_fits_image(single_path)
    show = robust_show(image)

    gaia_path = (ROOT / args.gaia_dir / f"{args.sample_id}_gaia_true_stars_g20_pixel.csv").resolve()
    gaia = pd.read_csv(gaia_path)
    stars = gaia[pd.to_numeric(gaia["g_mag"], errors="coerce") <= float(args.mag_limit)].copy()
    stars = stars.sort_values("g_mag").head(int(args.max_stars)).reset_index(drop=True)

    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 16), constrained_layout=True)
    ax.imshow(show, cmap="gray", origin="upper")
    if args.color_bins:
        mag = pd.to_numeric(stars["g_mag"], errors="coerce")
        add_circles(ax, stars[mag <= 13.0], args.circle_radius, "red", "Gaia G<=13", linewidth=1.4)
        add_circles(ax, stars[(mag > 13.0) & (mag <= 14.0)], args.circle_radius, "yellow", "Gaia 13<G<=14", linewidth=1.2)
        add_circles(ax, stars[(mag > 14.0) & (mag <= 15.0)], args.circle_radius, "cyan", "Gaia 14<G<=15", linewidth=1.0)
    else:
        add_circles(ax, stars, args.circle_radius, "red", f"Gaia G<={args.mag_limit:g}")
    ax.set_title(f"{args.sample_id}: single FITS with Gaia bright-star circles, n={len(stars)}")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.legend(loc="upper right")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    full_path = out / f"{args.sample_id}_gaia_bright_overlay_full.png"
    fig.savefig(full_path, dpi=180)
    plt.close(fig)

    crop_rows = stars.iloc[np.linspace(0, max(len(stars) - 1, 0), min(args.num_crops, len(stars)), dtype=int)]
    crop_paths = []
    for i, (_, star) in enumerate(crop_rows.iterrows(), start=1):
        x = float(star["x"])
        y = float(star["y"])
        half = int(args.crop_half)
        x0 = max(0, int(round(x)) - half)
        x1 = min(image.shape[1], int(round(x)) + half + 1)
        y0 = max(0, int(round(y)) - half)
        y1 = min(image.shape[0], int(round(y)) + half + 1)
        local = stars[(stars["x"] >= x0) & (stars["x"] < x1) & (stars["y"] >= y0) & (stars["y"] < y1)]
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        ax.imshow(show[y0:y1, x0:x1], cmap="gray", origin="upper", extent=[x0, x1, y1, y0])
        if args.color_bins:
            mag_local = pd.to_numeric(local["g_mag"], errors="coerce")
            add_circles(ax, local[mag_local <= 13.0], args.circle_radius, "red", "G<=13", linewidth=1.4)
            add_circles(ax, local[(mag_local > 13.0) & (mag_local <= 14.0)], args.circle_radius, "yellow", "13<G<=14", linewidth=1.2)
            add_circles(ax, local[(mag_local > 14.0) & (mag_local <= 15.0)], args.circle_radius, "cyan", "14<G<=15", linewidth=1.0)
        else:
            add_circles(ax, local, args.circle_radius, "red", "Gaia")
        ax.scatter([x], [y], c="cyan", s=18, marker="+", linewidths=1.5, label="selected")
        ax.set_title(f"{args.sample_id} crop {i}: G={float(star['g_mag']):.2f}, x={x:.2f}, y={y:.2f}")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.legend(loc="upper right")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        crop_path = out / f"{args.sample_id}_gaia_bright_overlay_crop_{i:02d}.png"
        fig.savefig(crop_path, dpi=180)
        plt.close(fig)
        crop_paths.append(crop_path)

    summary = pd.DataFrame(
        [{"kind": "full", "path": str(full_path), "sample_id": args.sample_id, "n_stars": len(stars)}]
        + [{"kind": "crop", "path": str(path), "sample_id": args.sample_id, "n_stars": len(stars)} for path in crop_paths]
    )
    summary.to_csv(out / f"{args.sample_id}_gaia_overlay_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
