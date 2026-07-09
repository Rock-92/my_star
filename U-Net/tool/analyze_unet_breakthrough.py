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


def parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def progress(current: int, total: int, text: str, start: float) -> None:
    width = 24
    done = int(round(width * current / max(total, 1)))
    bar = "#" * done + "-" * (width - done)
    elapsed = time.time() - start
    eta = elapsed / max(current, 1) * max(total - current, 0)
    print(f"\r[{bar}] {current}/{total} ({100.0 * current / max(total, 1):5.1f}%) elapsed {elapsed:7.1f}s ETA {eta:7.1f}s {text}", end="", flush=True)
    if current >= total:
        print(flush=True)


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
    w1 = np.maximum(np.hanning(patch_size).astype(np.float32), 0.05)
    win = np.outer(w1, w1).astype(np.float32)
    model.eval()
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
                inp = torch.from_numpy(patch[None, None].astype(np.float32)).to(device)
                prob = torch.sigmoid(model(inp))[0, 0].detach().cpu().numpy().astype(np.float32)
                prob_sum[y0 : y0 + patch_size, x0 : x0 + patch_size] += prob * win
                weight_sum[y0 : y0 + patch_size, x0 : x0 + patch_size] += win
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


def mask_centroids(prob: np.ndarray, threshold: float, min_area: int = 2, exclude_border: int = 8) -> np.ndarray:
    mask = prob >= float(threshold)
    if exclude_border > 0:
        mask[:exclude_border, :] = False
        mask[-exclude_border:, :] = False
        mask[:, :exclude_border] = False
        mask[:, -exclude_border:] = False
    labels, nlab = ndi.label(mask)
    if nlab == 0:
        return np.empty((0, 2), dtype=np.float32)
    points: list[tuple[float, float, float]] = []
    for label_id, slc in enumerate(ndi.find_objects(labels), start=1):
        if slc is None:
            continue
        local = labels[slc] == label_id
        if int(np.count_nonzero(local)) < int(min_area):
            continue
        y0 = slc[0].start
        x0 = slc[1].start
        local_prob = prob[slc].astype(np.float64)
        score, py, px = _component_centroid(local_prob, local, float(threshold))
        points.append((score, float(y0 + py), float(x0 + px)))
    points.sort(reverse=True)
    return np.asarray([(cy, cx) for _, cy, cx in points], dtype=np.float32)


def radius_by_mag(mag: np.ndarray, bright_radius: float, faint_radius: float, faint_mag: float) -> np.ndarray:
    mag = np.asarray(mag, dtype=np.float32)
    t = np.clip((mag - 10.0) / max(float(faint_mag) - 10.0, 1e-6), 0.0, 1.0)
    return float(bright_radius) * (1.0 - t) + float(faint_radius) * t


def match_with_ids(pred_yx: np.ndarray, target: pd.DataFrame, radii: np.ndarray) -> tuple[int, np.ndarray]:
    pred_yx = np.asarray(pred_yx, dtype=np.float32).reshape((-1, 2))
    matched_target = np.zeros(len(target), dtype=bool)
    if len(pred_yx) == 0 or len(target) == 0:
        return 0, matched_target
    target_yx = target[["y", "x"]].to_numpy(dtype=np.float32)
    tree = cKDTree(target_yx)
    pairs: list[tuple[float, int, int]] = []
    max_radius = float(np.max(radii)) if len(radii) else 0.0
    for pi, yx in enumerate(pred_yx):
        for ti in tree.query_ball_point(yx, r=max_radius):
            dist = float(np.linalg.norm(yx - target_yx[ti]))
            if dist <= float(radii[ti]):
                pairs.append((dist, pi, ti))
    pairs.sort()
    used_p: set[int] = set()
    used_t: set[int] = set()
    for _, pi, ti in pairs:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matched_target[ti] = True
    return len(used_p), matched_target


def metrics(method: str, pred_yx: np.ndarray, target: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, object], np.ndarray]:
    mag = target["g_mag"].to_numpy(dtype=np.float32)
    radii = radius_by_mag(mag, args.bright_radius, args.faint_radius, args.faint_mag)
    matched, matched_target = match_with_ids(pred_yx, target, radii)
    pred = int(len(pred_yx))
    target_count = int(len(target))
    precision = matched / max(pred, 1)
    recall = matched / max(target_count, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {
        "method": method,
        "pred_count": pred,
        "matched_count": int(matched),
        "false_count": int(pred - matched),
        "target_count": target_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }, matched_target


def load_target(gaia_dir: Path, sample_id: str, mag_limit: float) -> pd.DataFrame:
    df = pd.read_csv(gaia_dir / f"{sample_id}_gaia_true_stars_g20_pixel.csv")
    df["g_mag"] = pd.to_numeric(df["g_mag"], errors="coerce")
    df = df[np.isfinite(df["x"]) & np.isfinite(df["y"]) & np.isfinite(df["g_mag"])].copy()
    df = df[df["g_mag"] <= float(mag_limit)].copy().reset_index(drop=True)
    df["target_index"] = np.arange(len(df), dtype=np.int32)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether U-Net is beyond a daofind threshold interpolation.")
    parser.add_argument("--data-model", type=Path, default=Path("data/data_model"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/data_20/unet_softmask"))
    parser.add_argument("--gaia-dir", type=Path, default=Path("data/data_gaia/gaia_annotations_right_fixed"))
    parser.add_argument("--selected-samples", type=Path, default=Path("data/data_20/selected_samples.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("U-Net/runs/unet_softmask_10ep/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("test/unet_breakthrough_analysis"))
    parser.add_argument("--model-thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--daofind-sigmas", default="3.5,3.75,4.0,4.25,4.5,4.75,5.0,5.25,5.5,6.0")
    parser.add_argument("--mag-limit", type=float, default=14.0)
    parser.add_argument("--bright-radius", type=float, default=4.0)
    parser.add_argument("--faint-radius", type=float, default=1.5)
    parser.add_argument("--faint-mag", type=float, default=13.5)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--fwhm", type=float, default=3.0)
    parser.add_argument("--filtsize", type=int, default=25)
    parser.add_argument("--max-peaks", type=int, default=30000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv((ROOT / args.selected_samples).resolve())
    data_model = (ROOT / args.data_model).resolve()
    data_dir = (ROOT / args.data_dir).resolve()
    gaia_dir = (ROOT / args.gaia_dir).resolve()

    device = torch.device(args.device)
    ckpt = torch.load((ROOT / args.checkpoint).resolve(), map_location=device)
    base = int(ckpt.get("config", {}).get("base_channels", 32))
    model = UNet256(base=base).to(device)
    model.load_state_dict(ckpt["model"])

    model_thresholds = parse_csv_floats(args.model_thresholds)
    sigmas = parse_csv_floats(args.daofind_sigmas)
    global_rows: dict[str, dict[str, object]] = {}
    global_matches: dict[str, list[np.ndarray]] = {}
    target_tables: list[pd.DataFrame] = []

    start = time.time()
    for i, row in selected.iterrows():
        sid = str(row["sample_id"])
        progress(i + 1, len(selected), sid, start)
        target = load_target(gaia_dir, sid, args.mag_limit)
        target_tables.append(target.assign(global_sample=sid))
        single, _ = read_fits_image(resolve_manifest_path(data_model, row["single_fits"]))

        soft = np.load(data_dir / f"{sid}_softmask.npz")
        prob_path = out / f"{sid}_prob.npy"
        if prob_path.exists():
            prob = np.load(prob_path).astype(np.float32)
        else:
            prob = infer_full(model, soft["image"].astype(np.float32), args.patch_size, args.stride, device)
            np.save(prob_path, prob.astype(np.float32))

        for thr in model_thresholds:
            method = f"unet_thr{thr:g}"
            pred = mask_centroids(prob, thr)
            row_metrics, matched_target = metrics(method, pred, target, args)
            agg = global_rows.setdefault(method, {"method": method, "pred_count": 0, "matched_count": 0, "false_count": 0, "target_count": 0})
            for key in ("pred_count", "matched_count", "false_count", "target_count"):
                agg[key] = int(agg[key]) + int(row_metrics[key])
            global_matches.setdefault(method, []).append(matched_target)

        for sigma in sigmas:
            method = f"daofind_sigma{sigma:g}"
            sources = dao_like_sources(single, sigma=sigma, fwhm=args.fwhm, filtsize=args.filtsize, max_peaks=args.max_peaks)
            pred = np.asarray([(src.y, src.x) for src in sources], dtype=np.float32).reshape((-1, 2))
            row_metrics, matched_target = metrics(method, pred, target, args)
            agg = global_rows.setdefault(method, {"method": method, "pred_count": 0, "matched_count": 0, "false_count": 0, "target_count": 0})
            for key in ("pred_count", "matched_count", "false_count", "target_count"):
                agg[key] = int(agg[key]) + int(row_metrics[key])
            global_matches.setdefault(method, []).append(matched_target)

    rows = []
    for method, agg in global_rows.items():
        pred = int(agg["pred_count"])
        matched = int(agg["matched_count"])
        target = int(agg["target_count"])
        precision = matched / max(pred, 1)
        recall = matched / max(target, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        rows.append({**agg, "precision": precision, "recall": recall, "f1": f1})
    pr = pd.DataFrame(rows).sort_values(["method"]).reset_index(drop=True)
    pr.to_csv(out / "pr_points.csv", index=False)

    all_targets = pd.concat(target_tables, ignore_index=True)
    unet_key = "unet_thr0.5"
    compare_keys = ["daofind_sigma4", "daofind_sigma4.5", "daofind_sigma5"]
    unique_rows = []
    if unet_key in global_matches:
        unet_match = np.concatenate(global_matches[unet_key])
        for key in compare_keys:
            if key not in global_matches:
                continue
            dao_match = np.concatenate(global_matches[key])
            unet_only = unet_match & ~dao_match
            dao_only = dao_match & ~unet_match
            both = unet_match & dao_match
            neither = ~unet_match & ~dao_match
            for name, mask in ((f"{unet_key}_only_vs_{key}", unet_only), (f"{key}_only_vs_{unet_key}", dao_only), (f"both_{unet_key}_{key}", both), (f"neither_{unet_key}_{key}", neither)):
                mags = all_targets.loc[mask, "g_mag"].to_numpy(dtype=np.float32)
                unique_rows.append(
                    {
                        "comparison": name,
                        "count": int(np.count_nonzero(mask)),
                        "mean_mag": float(np.mean(mags)) if len(mags) else np.nan,
                        "median_mag": float(np.median(mags)) if len(mags) else np.nan,
                        "count_13_13p5": int(np.count_nonzero((mags >= 13.0) & (mags < 13.5))) if len(mags) else 0,
                        "count_13p5_14": int(np.count_nonzero((mags >= 13.5) & (mags <= 14.0))) if len(mags) else 0,
                    }
                )
    unique_df = pd.DataFrame(unique_rows)
    unique_df.to_csv(out / "unique_target_matches.csv", index=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print("\n[PR points]")
    print(pr.to_string(index=False))
    print("\n[unique target matches]")
    print(unique_df.to_string(index=False))
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
