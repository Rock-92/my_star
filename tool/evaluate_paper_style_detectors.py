from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.stats import SigmaClip
from photutils.detection import DAOStarFinder
from tqdm import tqdm

from paper_repro_utils import (
    daofind_like_detect,
    dao_to_sources,
    detection_metrics,
    radius_by_mag,
    read_fits_image,
    resolve_path,
    selected_samples,
    sep_like_sources,
    sources_to_dataframe,
    subtract_local_background,
    target_dataframe,
    write_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-style target-level comparison for CDN-Net and classical detectors.")
    parser.add_argument("--data-model", type=Path, default=Path("data/data_model"))
    parser.add_argument("--selected-samples", type=Path, default=Path("data/data_20/selected_samples.csv"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/data_20"))
    parser.add_argument("--cdn-infer-dir", type=Path, default=Path("U-Net/result/unet_softmask_eval"))
    parser.add_argument("--output", type=Path, default=Path("tool/result/paper_style_comparison"))
    parser.add_argument("--sample-split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--mag-limit", type=float, default=13.5)
    parser.add_argument("--daofind-sigmas", default="4.0,4.5,5.0")
    parser.add_argument("--daostarfinder-sigmas", default="4.0,4.5,5.0")
    parser.add_argument("--sextractor-sigmas", default="4.0,4.5,5.0")
    parser.add_argument("--cdn-thresholds", default="0.3,0.4,0.5")
    parser.add_argument("--fwhm", type=float, default=3.0)
    parser.add_argument("--filtsize", type=int, default=25)
    parser.add_argument("--bright-radius", type=float, default=3.0)
    parser.add_argument("--faint-radius", type=float, default=1.5)
    parser.add_argument("--faint-mag", type=float, default=13.5)
    parser.add_argument("--mag-bins", default="0,10,11,12,12.5,13,13.5")
    parser.add_argument("--max-peaks", type=int, default=10000)
    parser.add_argument("--fit-channel-mode", choices=("mean", "first", "max", "luma"), default="mean")
    return parser.parse_args()


def daostarfinder_sources(image: np.ndarray, sigma: float, fwhm: float, filtsize: int, max_peaks: int) -> pd.DataFrame:
    residual = subtract_local_background(image, "local_mean", filtsize)
    center = float(np.median(residual[np.isfinite(residual)]))
    mad = float(np.median(np.abs(residual[np.isfinite(residual)] - center)))
    noise = max(1.4826 * mad, 1e-6)
    finder = DAOStarFinder(
        threshold=float(sigma) * noise,
        fwhm=float(fwhm),
        sigma_radius=1.5,
        sharplo=0.2,
        sharphi=1.0,
        roundlo=-1.0,
        roundhi=1.0,
        exclude_border=True,
    )
    table = finder(residual)
    rows = []
    if table is not None and len(table):
        table.sort("peak")
        for row in reversed(table[-int(max_peaks) :]):
            rows.append(
                {
                    "y": float(row["ycentroid"]),
                    "x": float(row["xcentroid"]),
                    "peak": float(row["peak"]),
                    "flux": float(row["flux"]),
                    "snr": float(row["peak"]) / noise,
                    "area": 0.0,
                    "method": "daostarfinder",
                }
            )
    return pd.DataFrame(rows, columns=["y", "x", "peak", "flux", "snr", "area", "method"])


def target_for_sample(dataset_dir: Path, sample_id: str, mag_limit: float) -> pd.DataFrame:
    target = target_dataframe(dataset_dir / "targets" / f"{sample_id}_targets.csv", mag_limit=mag_limit)
    if target.empty:
        return pd.DataFrame(columns=["y", "x", "g_mag", "source"])
    target["g_mag"] = pd.to_numeric(target.get("g_mag", np.nan), errors="coerce")
    return target


def eval_one(pred: pd.DataFrame, target: pd.DataFrame, args: argparse.Namespace) -> dict[str, float | int]:
    radii = radius_by_mag(target.get("g_mag", pd.Series([np.nan] * len(target))), args.bright_radius, args.faint_radius, args.faint_mag)
    return detection_metrics(pred, target, radii)


def by_mag_rows(method: str, sample_id: str, pred: pd.DataFrame, target: pd.DataFrame, args: argparse.Namespace) -> list[dict[str, object]]:
    bins = parse_csv_floats(args.mag_bins)
    out: list[dict[str, object]] = []
    mag = pd.to_numeric(target.get("g_mag", np.nan), errors="coerce")
    for lo, hi in zip(bins[:-1], bins[1:]):
        subset = target[(mag >= lo) & (mag < hi)].copy()
        if subset.empty:
            out.append({"sample_id": sample_id, "method": method, "mag_lo": lo, "mag_hi": hi, "target_count": 0, "matched_count": 0, "recall": np.nan})
            continue
        metrics = eval_one(pred, subset, args)
        out.append(
            {
                "sample_id": sample_id,
                "method": method,
                "mag_lo": lo,
                "mag_hi": hi,
                "target_count": int(metrics["target_count"]),
                "matched_count": int(metrics["matched_count"]),
                "recall": float(metrics["recall"]),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    data_model = (ROOT / args.data_model).resolve()
    selected = selected_samples((ROOT / args.selected_samples).resolve())
    if args.sample_split != "all":
        selected = selected[selected["feature_split"] == args.sample_split].copy()
    dataset_dir = (ROOT / args.dataset_dir).resolve()
    cdn_dir = (ROOT / args.cdn_infer_dir).resolve()
    output = (ROOT / args.output).resolve()
    det_dir = output / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)

    overall_rows: list[dict[str, object]] = []
    by_mag_all: list[dict[str, object]] = []

    for _, sample in tqdm(selected.iterrows(), total=len(selected), desc="evaluate detectors"):
        sample_id = str(sample["sample_id"])
        image, _ = read_fits_image(resolve_path(data_model, sample["single_fits"]), args.fit_channel_mode)
        target = target_for_sample(dataset_dir, sample_id, float(args.mag_limit))
        if target.empty:
            continue

        methods: list[tuple[str, pd.DataFrame]] = []

        for sigma in parse_csv_floats(args.daofind_sigmas):
            det = daofind_like_detect(
                image,
                sigma=sigma,
                fwhm=float(args.fwhm),
                background_mode="local_mean",
                filtsize=int(args.filtsize),
                max_peaks=int(args.max_peaks),
                min_separation=float(args.fwhm),
                exclude_border=8,
            )
            methods.append((f"daofind_like_sigma{sigma:g}", sources_to_dataframe(dao_to_sources(det, "daofind_like"))))

        for sigma in parse_csv_floats(args.daostarfinder_sigmas):
            methods.append((f"daostarfinder_sigma{sigma:g}", daostarfinder_sources(image, sigma, float(args.fwhm), int(args.filtsize), int(args.max_peaks))))

        for sigma in parse_csv_floats(args.sextractor_sigmas):
            methods.append((f"sextractor_sep_sigma{sigma:g}", sources_to_dataframe(sep_like_sources(image, sigma=sigma, minarea=3, deblend=True, max_sources=int(args.max_peaks)))))

        for threshold in parse_csv_floats(args.cdn_thresholds):
            det_path = cdn_dir / sample_id / f"detections_thr{threshold:.2f}.csv"
            if det_path.exists():
                methods.append((f"cdnnet_connected_thr{threshold:g}", pd.read_csv(det_path)))

        for method, pred in methods:
            if pred.empty:
                pred = pd.DataFrame(columns=["y", "x", "peak", "flux", "snr", "area", "method"])
            pred.to_csv(det_dir / f"{sample_id}_{method}.csv", index=False)
            metrics = eval_one(pred, target, args)
            overall_rows.append({"sample_id": sample_id, "feature_split": sample["feature_split"], "method": method, **metrics})
            by_mag_all.extend(by_mag_rows(method, sample_id, pred, target, args))

    overall = pd.DataFrame(overall_rows)
    by_mag = pd.DataFrame(by_mag_all)
    overall.to_csv(output / "per_sample_metrics.csv", index=False)
    by_mag.to_csv(output / "per_sample_by_mag.csv", index=False)
    if not overall.empty:
        grouped = overall.groupby("method", as_index=False).agg(
            pred_count=("pred_count", "sum"),
            target_count=("target_count", "sum"),
            matched_count=("matched_count", "sum"),
            false_count=("false_count", "sum"),
            missed_count=("missed_count", "sum"),
        )
        grouped["precision"] = grouped["matched_count"] / grouped["pred_count"].replace(0, np.nan)
        grouped["recall"] = grouped["matched_count"] / grouped["target_count"].replace(0, np.nan)
        grouped["f1"] = 2.0 * grouped["precision"] * grouped["recall"] / (grouped["precision"] + grouped["recall"])
        grouped.to_csv(output / "overall_metrics.csv", index=False)
        print(grouped.sort_values("f1", ascending=False).to_string(index=False))
    if not by_mag.empty:
        mag_group = by_mag.groupby(["method", "mag_lo", "mag_hi"], as_index=False).agg(target_count=("target_count", "sum"), matched_count=("matched_count", "sum"))
        mag_group["recall"] = mag_group["matched_count"] / mag_group["target_count"].replace(0, np.nan)
        mag_group.to_csv(output / "overall_by_mag.csv", index=False)
    (output / "settings.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[done] wrote {output}")


if __name__ == "__main__":
    main()
