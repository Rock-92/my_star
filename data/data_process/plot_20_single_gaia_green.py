from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "tool"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from daofind_opt import read_fits_image, resolve_manifest_path  # noqa: E402


def robust_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    lo, hi = np.percentile(finite, [1.0, 99.85])
    scaled = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def draw_overlay(image: np.ndarray, stars: pd.DataFrame, radius: int, width: int = 3) -> Image.Image:
    rgb = Image.fromarray(robust_u8(image), mode="L").convert("RGB")
    draw = ImageDraw.Draw(rgb)
    for _, star in stars.iterrows():
        x = float(star["x"])
        y = float(star["y"])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 255, 0), width=width)
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Gaia G<=13.5 green circles on the selected 20 single frames.")
    parser.add_argument("--data-model", type=Path, default=Path("data/data_model"))
    parser.add_argument("--selected-samples", type=Path, default=Path("data/data_20/selected_samples.csv"))
    parser.add_argument("--gaia-dir", type=Path, default=Path("data/data_gaia/gaia_annotations_right_fixed"))
    parser.add_argument("--output", type=Path, default=Path("data/data_20/gaia_green_preview"))
    parser.add_argument("--mag-limit", type=float, default=13.5)
    parser.add_argument("--radius", type=int, default=24)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--thumb-width", type=int, default=360)
    args = parser.parse_args()

    data_model = (ROOT / args.data_model).resolve()
    selected = pd.read_csv((ROOT / args.selected_samples).resolve())
    gaia_dir = (ROOT / args.gaia_dir).resolve()
    out = (ROOT / args.output).resolve()
    full_dir = out / "full"
    preview_dir = out / "preview"
    full_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    thumbs: list[Image.Image] = []
    rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        sample_id = str(row["sample_id"])
        single_path = resolve_manifest_path(data_model, row["single_fits"])
        gaia_path = gaia_dir / f"{sample_id}_gaia_true_stars_g20_pixel.csv"
        if not gaia_path.exists():
            print(f"[skip] {sample_id}: missing {gaia_path}")
            continue
        image, _ = read_fits_image(single_path)
        gaia = pd.read_csv(gaia_path)
        mag = pd.to_numeric(gaia["g_mag"], errors="coerce")
        stars = gaia[mag <= float(args.mag_limit)].copy()

        overlay = draw_overlay(image, stars, radius=int(args.radius), width=int(args.line_width))
        full_path = full_dir / f"{sample_id}_single_gaia_g{str(args.mag_limit).replace('.', 'p')}_green.png"
        overlay.save(full_path)

        thumb = overlay.copy()
        new_h = int(round(thumb.height * int(args.thumb_width) / thumb.width))
        thumb = thumb.resize((int(args.thumb_width), new_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(thumb)
        draw.rectangle((0, 0, thumb.width, 28), fill=(0, 0, 0))
        draw.text((6, 6), f"{sample_id}  G<={args.mag_limit:g}  n={len(stars)}", fill=(0, 255, 0))
        preview_path = preview_dir / f"{sample_id}_preview.png"
        thumb.save(preview_path)
        thumbs.append(thumb)
        rows.append(
            {
                "sample_id": sample_id,
                "group": row.get("group", ""),
                "feature_split": row.get("feature_split", ""),
                "gaia_count": int(len(stars)),
                "full_path": str(full_path),
                "preview_path": str(preview_path),
            }
        )
        print(f"[done] {sample_id}: {len(stars)} stars -> {full_path}")

    if thumbs:
        cols = 4
        rows_n = int(np.ceil(len(thumbs) / cols))
        cell_w = max(t.width for t in thumbs)
        cell_h = max(t.height for t in thumbs)
        sheet = Image.new("RGB", (cols * cell_w, rows_n * cell_h), (20, 20, 20))
        for idx, thumb in enumerate(thumbs):
            x = (idx % cols) * cell_w
            y = (idx // cols) * cell_h
            sheet.paste(thumb, (x, y))
        sheet_path = out / f"selected20_gaia_g{str(args.mag_limit).replace('.', 'p')}_green_contact_sheet.png"
        sheet.save(sheet_path)
        print(f"[done] contact sheet -> {sheet_path}")

    pd.DataFrame(rows).to_csv(out / "overlay_summary.csv", index=False)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
