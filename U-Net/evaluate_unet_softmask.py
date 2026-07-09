from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / "tool"
UNET_DIR = ROOT / "U-Net"
for path in (TOOL_DIR, UNET_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from daofind_opt import read_fits_image, resolve_manifest_path  # noqa: E402
from train_unet_softmask import UNet256  # noqa: E402


def infer_full(model: torch.nn.Module, image: np.ndarray, patch_size: int, stride: int, device: torch.device) -> np.ndarray:
    h, w = image.shape
    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    ys = list(range(0, max(h - patch_size, 0) + 1, stride))
    xs = list(range(0, max(w - patch_size, 0) + 1, stride))
    if not ys or ys[-1] != h - patch_size:
        ys.append(h - patch_size)
    if not xs or xs[-1] != w - patch_size:
        xs.append(w - patch_size)
    window_1d = np.hanning(patch_size).astype(np.float32)
    window_1d = np.maximum(window_1d, 0.05)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    model.eval()
    with torch.no_grad():
        total = len(ys) * len(xs)
        done = 0
        for y0 in ys:
            for x0 in xs:
                done += 1
                patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
                inp = torch.from_numpy(patch[None, None].astype(np.float32)).to(device)
                prob = torch.sigmoid(model(inp))[0, 0].detach().cpu().numpy().astype(np.float32)
                prob_sum[y0 : y0 + patch_size, x0 : x0 + patch_size] += prob * window
                weight_sum[y0 : y0 + patch_size, x0 : x0 + patch_size] += window
                if done % 20 == 0 or done == total:
                    print(f"\r  infer patches {done}/{total}", end="", flush=True)
        print(flush=True)
    return prob_sum / np.maximum(weight_sum, 1e-6)


def _line_valley(local_prob: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    dist = float(np.linalg.norm(a - b))
    samples = max(int(np.ceil(dist * 2.0)) + 1, 5)
    ys = np.linspace(float(a[0]), float(b[0]), samples)
    xs = np.linspace(float(a[1]), float(b[1]), samples)
    values = ndi.map_coordinates(local_prob, [ys, xs], order=1, mode="nearest")
    if len(values) > 4:
        values = values[1:-1]
    return float(np.min(values))


def _component_peaks(
    local_prob: np.ndarray,
    local_label: np.ndarray,
    threshold: float,
    min_peak_distance: float = 3.0,
    valley_drop: float = 0.08,
    valley_ratio: float = 0.90,
) -> list[tuple[float, float, float]]:
    local_max = local_label & (local_prob >= ndi.maximum_filter(local_prob, size=3, mode="nearest") - 1e-8)
    coords = np.argwhere(local_max)
    if len(coords) == 0:
        coords = np.argwhere(local_label)
        if len(coords) == 0:
            return []
        scores = local_prob[coords[:, 0], coords[:, 1]]
        best = coords[int(np.argmax(scores))]
        return [(float(np.max(scores)), float(best[0]), float(best[1]))]

    scores = local_prob[coords[:, 0], coords[:, 1]]
    order = np.argsort(-scores)
    accepted: list[tuple[float, np.ndarray]] = []
    for idx in order:
        coord = coords[int(idx)].astype(np.float64)
        score = float(scores[int(idx)])
        if score < float(threshold):
            continue
        if not accepted:
            accepted.append((score, coord))
            continue
        nearest_score, nearest_coord = min(accepted, key=lambda item: float(np.linalg.norm(coord - item[1])))
        distance = float(np.linalg.norm(coord - nearest_coord))
        if distance < float(min_peak_distance):
            continue
        valley = _line_valley(local_prob, coord, nearest_coord)
        weak_peak = min(score, nearest_score)
        if (weak_peak - valley) >= float(valley_drop) and valley <= weak_peak * float(valley_ratio):
            accepted.append((score, coord))
    return [(score, float(coord[0]), float(coord[1])) for score, coord in accepted]


def _component_centroid(local_prob: np.ndarray, local_label: np.ndarray, threshold: float) -> tuple[float, float, float]:
    weights = np.where(local_label, np.clip(local_prob - float(threshold), 1e-4, None), 0.0)
    total = float(weights.sum())
    yy, xx = np.mgrid[0 : local_label.shape[0], 0 : local_label.shape[1]].astype(np.float64)
    if total > 0:
        cy = float((yy * weights).sum() / total)
        cx = float((xx * weights).sum() / total)
    else:
        coords = np.argwhere(local_label)
        cy = float(coords[:, 0].mean())
        cx = float(coords[:, 1].mean())
    score = float(local_prob[local_label].max())
    return score, cy, cx


def mask_connected_centroids(prob: np.ndarray, threshold: float, min_area: int, exclude_border: int) -> np.ndarray:
    mask = np.asarray(prob, dtype=np.float32) >= float(threshold)
    if exclude_border > 0:
        mask[:exclude_border, :] = False
        mask[-exclude_border:, :] = False
        mask[:, :exclude_border] = False
        mask[:, -exclude_border:] = False
    labels, nlab = ndi.label(mask)
    if nlab == 0:
        return np.empty((0, 2), dtype=np.float32)
    objects = ndi.find_objects(labels)
    points: list[tuple[float, float, float]] = []
    for label_index, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        local_label = labels[slc] == label_index
        area = int(np.count_nonzero(local_label))
        if area < int(min_area):
            continue
        y0 = slc[0].start
        x0 = slc[1].start
        local_prob = prob[slc].astype(np.float64)
        score, py, px = _component_centroid(local_prob, local_label, float(threshold))
        points.append((score, float(y0 + py), float(x0 + px)))
    points.sort(reverse=True)
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray([(cy, cx) for _, cy, cx in points], dtype=np.float32)


def unique_match(pred_xy: np.ndarray, target_xy: np.ndarray, radius: float) -> tuple[int, np.ndarray]:
    pred_xy = np.asarray(pred_xy, dtype=np.float32).reshape((-1, 2))
    target_xy = np.asarray(target_xy, dtype=np.float32).reshape((-1, 2))
    if len(pred_xy) == 0 or len(target_xy) == 0:
        return 0, np.zeros(len(target_xy), dtype=bool)
    tree = cKDTree(target_xy)
    pairs: list[tuple[float, int, int]] = []
    for pi, xy in enumerate(pred_xy):
        for ti in tree.query_ball_point(xy, r=float(radius)):
            pairs.append((float(np.linalg.norm(xy - target_xy[ti])), pi, ti))
    pairs.sort()
    used_p: set[int] = set()
    used_t: set[int] = set()
    matched_target = np.zeros(len(target_xy), dtype=bool)
    for _, pi, ti in pairs:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matched_target[ti] = True
    return len(used_p), matched_target


def robust_u8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    lo, hi = np.percentile(finite, [1.0, 99.85])
    return np.rint(np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def draw_overlay(base_image: np.ndarray, pred_xy: np.ndarray, target_df: pd.DataFrame, out: Path) -> None:
    im = Image.fromarray(robust_u8(base_image), mode="L").convert("RGB")
    draw = ImageDraw.Draw(im)
    for _, row in target_df.iterrows():
        x = float(row["x"])
        y = float(row["y"])
        color = (0, 255, 0) if float(row["g_mag"]) <= 13.0 else (0, 255, 255)
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), outline=color, width=2)
    for x, y in pred_xy:
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 0, 0), width=2)
    im.save(out, quality=95)


def evaluate_sample(sample_id: str, args: argparse.Namespace, model: torch.nn.Module, device: torch.device, out: Path) -> dict[str, object]:
    data = np.load((ROOT / args.data_dir / f"{sample_id}_softmask.npz").resolve())
    image_norm = data["image"].astype(np.float32)
    prob = infer_full(model, image_norm, int(args.patch_size), int(args.stride), device)
    pred_yx = mask_connected_centroids(prob, threshold=float(args.threshold), min_area=int(args.min_area), exclude_border=8)
    pred_xy = pred_yx[:, [1, 0]] if len(pred_yx) else np.empty((0, 2), dtype=np.float32)

    gaia = pd.read_csv((ROOT / args.gaia_dir / f"{sample_id}_gaia_true_stars_g20_pixel.csv").resolve())
    mag = pd.to_numeric(gaia["g_mag"], errors="coerce")
    target = gaia[mag <= 14.0].copy()
    target_xy = target[["x", "y"]].to_numpy(dtype=np.float32)
    matched, matched_target = unique_match(pred_xy, target_xy, float(args.match_radius))
    precision = matched / max(len(pred_xy), 1)
    recall = matched / max(len(target_xy), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    le13 = pd.to_numeric(target["g_mag"], errors="coerce").to_numpy() <= 13.0
    bin13 = (pd.to_numeric(target["g_mag"], errors="coerce").to_numpy() > 13.0) & (pd.to_numeric(target["g_mag"], errors="coerce").to_numpy() <= 14.0)
    result = {
        "sample_id": sample_id,
        "threshold": float(args.threshold),
        "pred_count": int(len(pred_xy)),
        "target_le14": int(len(target_xy)),
        "matched": int(matched),
        "false": int(len(pred_xy) - matched),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "target_le13": int(np.sum(le13)),
        "matched_le13": int(np.sum(matched_target & le13)),
        "recall_le13": float(np.sum(matched_target & le13) / max(np.sum(le13), 1)),
        "target_13_14": int(np.sum(bin13)),
        "matched_13_14": int(np.sum(matched_target & bin13)),
        "recall_13_14": float(np.sum(matched_target & bin13) / max(np.sum(bin13), 1)),
    }
    np.save(out / f"{sample_id}_prob.npy", prob.astype(np.float32))
    Image.fromarray(np.rint(np.clip(prob, 0, 1) * 255).astype(np.uint8), mode="L").save(out / f"{sample_id}_prob.png")
    pd.DataFrame({"x": pred_xy[:, 0] if len(pred_xy) else [], "y": pred_xy[:, 1] if len(pred_xy) else []}).to_csv(out / f"{sample_id}_pred_peaks.csv", index=False)

    selected = pd.read_csv((ROOT / args.selected_samples).resolve())
    row = selected[selected["sample_id"] == sample_id].iloc[0]
    raw, _ = read_fits_image(resolve_manifest_path((ROOT / args.data_model).resolve(), row["single_fits"]))
    draw_overlay(raw, pred_xy, target, out / f"{sample_id}_overlay_pred_red_gaia_green_cyan.jpg")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the 08 U-Net soft-mask detector.")
    parser.add_argument("--data-model", type=Path, default=Path("data/data_model"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/data_20/unet_softmask"))
    parser.add_argument("--gaia-dir", type=Path, default=Path("data/data_gaia/gaia_annotations_right_fixed"))
    parser.add_argument("--selected-samples", type=Path, default=Path("data/data_20/selected_samples.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("U-Net/runs/unet_softmask/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("U-Net/result/unet_softmask_eval"))
    parser.add_argument("--samples", nargs="+", default=["sample_000001", "sample_000747"])
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--match-radius", type=float, default=4.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load((ROOT / args.checkpoint).resolve(), map_location=device)
    base = int(ckpt.get("config", {}).get("base_channels", args.base_channels))
    model = UNet256(base=base).to(device)
    model.load_state_dict(ckpt["model"])
    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in args.samples:
        print(f"[eval] {sid}")
        rows.append(evaluate_sample(sid, args, model, device, out))
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
