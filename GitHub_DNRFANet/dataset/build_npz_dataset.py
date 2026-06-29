import argparse
import csv
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


IMAGE_EXTS = {".fits", ".fit", ".fts", ".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MASK_EXTS = {".fits", ".fit", ".fts", ".npy", ".npz", ".png", ".tif", ".tiff"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_model(input_arg: str) -> Path:
    given = Path(input_arg)
    candidates = [
        given,
        Path.cwd() / input_arg,
        project_root() / input_arg,
        project_root().parent / input_arg,
        project_root().parent / "single_star_test" / "data" / input_arg,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Cannot find data directory. Tried:\n" + "\n".join(str(c) for c in candidates))


def read_manifest(input_dir: Path):
    manifest_path = input_dir / "manifest.csv"
    if not manifest_path.exists():
        return {}
    rows = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            image_out = row.get("image_out", "").replace("\\", "/")
            mask_out = row.get("mask_out", "").replace("\\", "/")
            if not image_out:
                continue
            image_name = Path(image_out).stem
            rows[image_name] = {
                "split": row.get("split", ""),
                "mask": input_dir.parent / mask_out if mask_out else None,
            }
    return rows


def collect_split_images(input_dir: Path):
    items = []
    for split in ("train", "val", "test"):
        image_dir = input_dir / split / "images"
        if not image_dir.exists():
            continue
        for path in sorted(image_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                items.append((split, path))
    if items:
        return items
    return [("train", p) for p in sorted(input_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "mask" not in p.stem.lower()]


def find_mask(image_path: Path, input_dir: Path, manifest_rows: dict):
    from_manifest = manifest_rows.get(image_path.stem, {}).get("mask")
    if from_manifest and from_manifest.exists():
        return from_manifest

    candidates = []
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        mask_parts = parts[:idx] + ["masks"] + parts[idx + 1:]
        candidates.append(Path(*mask_parts).with_suffix(".png"))
        candidates.append(Path(*mask_parts).with_suffix(image_path.suffix))

    candidates.extend([
        image_path.parent.parent / "masks" / f"{image_path.stem}.png",
        image_path.parent.parent / "masks" / f"{image_path.stem}{image_path.suffix}",
        input_dir / "masks" / f"{image_path.stem}.png",
        input_dir / "stack_masks" / f"{image_path.stem}.png",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_array(path: Path, npz_key: Optional[str] = None):
    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fts"}:
        from astropy.io import fits
        if hasattr(fits, "getdata"):
            arr = fits.getdata(path)
        else:
            with fits.open(path) as hdul:
                arr = hdul[0].data
    elif suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        arr = data[npz_key] if npz_key and npz_key in data else data[data.files[0]]
    else:
        from PIL import Image
        arr = np.array(Image.open(path))

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape} from {path}")
    return arr


def hierarchy_uint8(image):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.clip(image, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
    out = np.empty((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    out[..., 0] = (image >> 8) & 0xFF
    out[..., 1] = (image >> 4) & 0xFF
    out[..., 2] = image & 0xFF
    return out


def normalize_mask(mask):
    mask = np.nan_to_num(mask.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mask.max() > 1.0:
        mask /= 255.0
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def pad_to_patch(arr, patch_size: int):
    h, w = arr.shape[:2]
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    pads = [(0, pad_h), (0, pad_w)]
    if arr.ndim == 3:
        pads.append((0, 0))
    return np.pad(arr, pads, mode="constant")


def sample_coords(h: int, w: int, patch_size: int, count: int, rng: random.Random):
    max_y = max(0, h - patch_size)
    max_x = max(0, w - patch_size)
    return [
        (rng.randint(0, max_y) if max_y else 0, rng.randint(0, max_x) if max_x else 0)
        for _ in range(count)
    ]


def main():
    parser = argparse.ArgumentParser(description="Build compressed NPZ patches from data_model.")
    parser.add_argument("--input", default="data_model", help="Input data directory. Cloud default: ./data_model")
    parser.add_argument("--output", default="dataset/npz_patches", help="Output directory.")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-image", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--preprocess", choices=["hierarchy", "raw"], default="hierarchy")
    parser.add_argument("--require-mask", action="store_true")
    args = parser.parse_args()

    input_dir = resolve_data_model(args.input)
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_root() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest_rows = read_manifest(input_dir)
    image_items = collect_split_images(input_dir)
    split_lists = {"train": [], "val": [], "test": []}
    manifest = []
    skipped = []

    for split, image_path in tqdm(image_items, desc="Cropping"):
        try:
            image = load_array(image_path)
        except Exception as exc:
            skipped.append({"source": str(image_path), "reason": f"image load failed: {exc}"})
            continue

        mask_path = find_mask(image_path, input_dir, manifest_rows)
        mask = None
        if mask_path is not None:
            try:
                mask = normalize_mask(load_array(mask_path))
            except Exception as exc:
                skipped.append({"source": str(image_path), "reason": f"mask load failed: {exc}"})
                if args.require_mask:
                    continue
        elif args.require_mask:
            skipped.append({"source": str(image_path), "reason": "no matched mask"})
            continue

        image = pad_to_patch(image, args.patch_size)
        if mask is not None:
            mask = pad_to_patch(mask, args.patch_size)
            if mask.shape[:2] != image.shape[:2]:
                skipped.append({"source": str(image_path), "reason": f"mask shape {mask.shape} != image shape {image.shape}"})
                continue

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx, (y, x) in enumerate(sample_coords(image.shape[0], image.shape[1], args.patch_size, args.patches_per_image, rng)):
            patch_2d = image[y:y + args.patch_size, x:x + args.patch_size]
            image_patch = hierarchy_uint8(patch_2d) if args.preprocess == "hierarchy" else patch_2d
            payload = {
                "image": image_patch,
                "source": np.array(str(image_path)),
                "y": np.array(y, dtype=np.int32),
                "x": np.array(x, dtype=np.int32),
            }
            if mask is not None:
                payload["mask"] = mask[y:y + args.patch_size, x:x + args.patch_size]
            out_path = split_dir / f"{image_path.stem}_{idx:02d}_y{y}_x{x}.npz"
            np.savez_compressed(out_path, **payload)
            split_lists.setdefault(split, []).append(str(out_path))
            manifest.append({"path": str(out_path), "split": split, "source": str(image_path), "mask": str(mask_path) if mask_path else "", "y": y, "x": x})

    for split, paths in split_lists.items():
        (output_dir / f"{split}.txt").write_text("\n".join(paths), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "skipped.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Source images: {len(image_items)}")
    print(f"NPZ patches: {len(manifest)}")
    print(f"Train/val/test: {len(split_lists.get('train', []))}/{len(split_lists.get('val', []))}/{len(split_lists.get('test', []))}")
    print(f"Skipped: {len(skipped)}")
    if skipped:
        print("First skipped examples:")
        for item in skipped[:10]:
            print(f"  - {item['source']}: {item['reason']}")
    if image_items and not manifest:
        raise RuntimeError("Found source images, but generated 0 patches. Check skipped.json or the messages above.")


if __name__ == "__main__":
    main()
