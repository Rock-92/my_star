from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
PROJECT_TOOL_DIR = ROOT / "tool"
UNET_DIR = ROOT / "U-Net"
for path in (PROJECT_TOOL_DIR, UNET_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from classical_detectors import dao_like_sources  # noqa: E402
from daofind_opt import read_fits_image, resolve_manifest_path  # noqa: E402
from train_unet_softmask import UNet256  # noqa: E402


def progress(current: int, total: int, text: str, start: float) -> None:
    width = 24
    done = int(round(width * current / max(total, 1)))
    bar = "#" * done + "-" * (width - done)
    elapsed = time.time() - start
    eta = elapsed / max(current, 1) * max(total - current, 0)
    print(f"\r[{bar}] {current}/{total} ({100.0 * current / max(total, 1):5.1f}%) elapsed {elapsed:7.1f}s ETA {eta:7.1f}s {text}", end="", flush=True)
    if current >= total:
        print(flush=True)


def parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


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
        for y0 in ys:
            for x0 in xs:
                patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
                inp = torch.from_numpy(patch[None, None].astype(np.float32)).to(device)
                prob = torch.sigmoid(model(inp))[0, 0].detach().cpu().numpy().astype(np.float32)
                prob_sum[y0 : y0 + patch_size, x0 : x0 + patch_size] += prob * window
                weight_sum[y0 : y0 + patch_size, x0 : x0 + patch_size] += window
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

    points: list[tuple[float, float, float, int]] = []
    for label_id, slc in enumerate(ndi.find_objects(labels), start=1):
        if slc is None:
            continue
        local = labels[slc] == label_id
        area = int(np.count_nonzero(local))
        if area < int(min_area):
            continue
        y0 = slc[0].start
        x0 = slc[1].start
        local_prob = prob[slc].astype(np.float64)
        score, py, px = _component_centroid(local_prob, local, float(threshold))
        points.append((score, float(y0 + py), float(x0 + px), area))
    points.sort(reverse=True)
    return np.asarray([(cy, cx) for _, cy, cx, _ in points], dtype=np.float32)


def radius_by_mag(mag: np.ndarray, bright_radius: float, faint_radius: float, faint_mag: float) -> np.ndarray:
    mag = np.asarray(mag, dtype=np.float32)
    denom = max(float(faint_mag) - 10.0, 1e-6)
    t = np.clip((mag - 10.0) / denom, 0.0, 1.0)
    return float(bright_radius) * (1.0 - t) + float(faint_radius) * t


def match_predictions(pred_yx: np.ndarray, target: pd.DataFrame, radii: np.ndarray) -> tuple[int, np.ndarray]:
    pred_yx = np.asarray(pred_yx, dtype=np.float32).reshape((-1, 2))
    if len(pred_yx) == 0 or target.empty:
        return 0, np.zeros(len(target), dtype=bool)

    target_yx = target[["y", "x"]].to_numpy(dtype=np.float32)
    max_radius = float(np.max(radii)) if len(radii) else 0.0
    tree = cKDTree(target_yx)
    pairs: list[tuple[float, int, int]] = []
    for pi, yx in enumerate(pred_yx):
        for ti in tree.query_ball_point(yx, r=max_radius):
            dist = float(np.linalg.norm(yx - target_yx[ti]))
            if dist <= float(radii[ti]):
                pairs.append((dist, pi, ti))
    pairs.sort()

    used_pred: set[int] = set()
    used_target: set[int] = set()
    matched_target = np.zeros(len(target), dtype=bool)
    for _, pi, ti in pairs:
        if pi in used_pred or ti in used_target:
            continue
        used_pred.add(pi)
        used_target.add(ti)
        matched_target[ti] = True
    return len(used_pred), matched_target


def load_target(gaia_dir: Path, sample_id: str, mag_limit: float) -> pd.DataFrame:
    path = gaia_dir / f"{sample_id}_gaia_true_stars_g20_pixel.csv"
    gaia = pd.read_csv(path)
    gaia["g_mag"] = pd.to_numeric(gaia["g_mag"], errors="coerce")
    gaia = gaia[np.isfinite(gaia["x"]) & np.isfinite(gaia["y"]) & np.isfinite(gaia["g_mag"])].copy()
    return gaia[gaia["g_mag"] <= float(mag_limit)].copy().reset_index(drop=True)


def summarize(method: str, split: str, sample_id: str, pred_yx: np.ndarray, target: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, object], np.ndarray]:
    mag = target["g_mag"].to_numpy(dtype=np.float32)
    radii = radius_by_mag(mag, args.bright_radius, args.faint_radius, args.faint_mag)
    matched, matched_target = match_predictions(pred_yx, target, radii)
    pred_count = int(len(pred_yx))
    target_count = int(len(target))
    precision = matched / max(pred_count, 1)
    recall = matched / max(target_count, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    row = {
        "method": method,
        "split": split,
        "sample_id": sample_id,
        "pred_count": pred_count,
        "matched_count": int(matched),
        "false_count": int(pred_count - matched),
        "target_count": target_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return row, matched_target


def mag_bin_rows(method: str, split: str, sample_id: str, target: pd.DataFrame, matched_target: np.ndarray, bins: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    mag = target["g_mag"].to_numpy(dtype=np.float32)
    for lo, hi in zip(bins[:-1], bins[1:]):
        keep = (mag >= float(lo)) & (mag < float(hi))
        target_count = int(np.count_nonzero(keep))
        matched_count = int(np.count_nonzero(matched_target & keep))
        rows.append(
            {
                "method": method,
                "split": split,
                "sample_id": sample_id,
                "mag_lo": lo,
                "mag_hi": hi,
                "target_count": target_count,
                "matched_count": matched_count,
                "recall": matched_count / max(target_count, 1),
            }
        )
    return rows


def aggregate_overall(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    group_cols = ["method"]
    out = (
        df.groupby(group_cols, as_index=False)[["pred_count", "matched_count", "false_count", "target_count"]]
        .sum()
        .sort_values("method")
        .reset_index(drop=True)
    )
    out["precision"] = out["matched_count"] / out["pred_count"].clip(lower=1)
    out["recall"] = out["matched_count"] / out["target_count"].clip(lower=1)
    out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"]).replace(0, np.nan)
    return out.fillna(0.0)


def aggregate_bins(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = (
        df.groupby(["method", "mag_lo", "mag_hi"], as_index=False)[["target_count", "matched_count"]]
        .sum()
        .sort_values(["method", "mag_lo"])
        .reset_index(drop=True)
    )
    out["recall"] = out["matched_count"] / out["target_count"].clip(lower=1)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate U-Net soft-mask detections against Gaia and compare daofind-like sigma sweeps.")
    parser.add_argument("--data-model", type=Path, default=Path("data/data_model"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/data_20/unet_softmask"))
    parser.add_argument("--gaia-dir", type=Path, default=Path("data/data_gaia/gaia_annotations_right_fixed"))
    parser.add_argument("--selected-samples", type=Path, default=Path("data/data_20/selected_samples.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("U-Net/runs/unet_softmask_10ep/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("test/unet_vs_daofind_eval"))
    parser.add_argument("--samples", nargs="*", default=None, help="Optional sample ids. Default: all selected samples.")
    parser.add_argument("--splits", nargs="*", default=None, help="Optional split names from selected_samples.csv, e.g. train val.")
    parser.add_argument("--model-threshold", type=float, default=0.5)
    parser.add_argument("--model-min-area", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--daofind-sigmas", default="4.0,4.5,5.0")
    parser.add_argument("--fwhm", type=float, default=3.0)
    parser.add_argument("--filtsize", type=int, default=25)
    parser.add_argument("--max-peaks", type=int, default=20000)
    parser.add_argument("--mag-limit", type=float, default=14.0)
    parser.add_argument("--mag-bins", default="0,10,11,12,12.5,13,13.5,14")
    parser.add_argument("--bright-radius", type=float, default=4.0)
    parser.add_argument("--faint-radius", type=float, default=1.5)
    parser.add_argument("--faint-mag", type=float, default=13.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv((ROOT / args.selected_samples).resolve())
    if args.samples:
        selected = selected[selected["sample_id"].isin(args.samples)].copy()
    if args.splits:
        selected = selected[selected["split"].isin(args.splits)].copy()
    selected = selected.reset_index(drop=True)
    if selected.empty:
        raise SystemExit("No samples selected for evaluation.")

    device = torch.device(args.device)
    ckpt_path = (ROOT / args.checkpoint).resolve()
    ckpt = torch.load(ckpt_path, map_location=device)
    base = int(ckpt.get("config", {}).get("base_channels", args.base_channels))
    model = UNet256(base=base).to(device)
    model.load_state_dict(ckpt["model"])

    bins = parse_csv_floats(args.mag_bins)
    sigmas = parse_csv_floats(args.daofind_sigmas)
    sample_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    start = time.time()

    data_model = (ROOT / args.data_model).resolve()
    data_dir = (ROOT / args.data_dir).resolve()
    gaia_dir = (ROOT / args.gaia_dir).resolve()

    for index, row in selected.iterrows():
        sample_id = str(row["sample_id"])
        split = str(row.get("split", ""))
        progress(index + 1, len(selected), sample_id, start)

        target = load_target(gaia_dir, sample_id, args.mag_limit)
        single, _ = read_fits_image(resolve_manifest_path(data_model, row["single_fits"]))

        soft = np.load(data_dir / f"{sample_id}_softmask.npz")
        prob = infer_full(model, soft["image"].astype(np.float32), args.patch_size, args.stride, device)
        model_yx = mask_connected_centroids(prob, args.model_threshold, args.model_min_area, exclude_border=8)
        method = f"unet_mask_thr{args.model_threshold:g}"
        summary, matched_target = summarize(method, split, sample_id, model_yx, target, args)
        sample_rows.append(summary)
        bin_rows.extend(mag_bin_rows(method, split, sample_id, target, matched_target, bins))
        pd.DataFrame({"y": model_yx[:, 0] if len(model_yx) else [], "x": model_yx[:, 1] if len(model_yx) else []}).to_csv(out / f"{sample_id}_{method}_peaks.csv", index=False)

        for sigma in sigmas:
            sources = dao_like_sources(single, sigma=sigma, fwhm=args.fwhm, filtsize=args.filtsize, max_peaks=args.max_peaks)
            pred_yx = np.asarray([(src.y, src.x) for src in sources], dtype=np.float32).reshape((-1, 2))
            method = f"daofind_like_sigma{sigma:g}"
            summary, matched_target = summarize(method, split, sample_id, pred_yx, target, args)
            sample_rows.append(summary)
            bin_rows.extend(mag_bin_rows(method, split, sample_id, target, matched_target, bins))

    sample_df = pd.DataFrame(sample_rows)
    bin_df = pd.DataFrame(bin_rows)
    overall = aggregate_overall(sample_rows)
    by_mag = aggregate_bins(bin_rows)

    sample_df.to_csv(out / "per_sample_metrics.csv", index=False)
    bin_df.to_csv(out / "per_sample_mag_recall.csv", index=False)
    overall.to_csv(out / "overall_metrics.csv", index=False)
    by_mag.to_csv(out / "mag_bin_recall.csv", index=False)
    (out / "eval_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print("\n[overall]")
    print(overall.to_string(index=False))
    print("\n[mag_bin_recall]")
    print(by_mag.to_string(index=False))
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
