from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
PROJECT_TOOL_DIR = ROOT / "tool"
UNET_DIR = ROOT / "U-Net"
UNET_PLUS_DIR = ROOT / "U-Net+"
UNET_TOOL_DIR = UNET_DIR / "tool"
for path in (PROJECT_TOOL_DIR, UNET_DIR, UNET_PLUS_DIR, UNET_TOOL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_unet_breakthrough import infer_full, mask_centroids, metrics  # noqa: E402
from classical_detectors import dao_like_sources  # noqa: E402
from daofind_opt import read_fits_image, resolve_manifest_path  # noqa: E402
from train_unet_softmask import UNet256  # noqa: E402
from train_unet_plus_softmask import MultiScaleUNetPlus  # noqa: E402


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


def load_target(gaia_dir: Path, sample_id: str, mag_limit: float) -> pd.DataFrame:
    df = pd.read_csv(gaia_dir / f"{sample_id}_gaia_true_stars_g20_pixel.csv")
    df["g_mag"] = pd.to_numeric(df["g_mag"], errors="coerce")
    df = df[np.isfinite(df["x"]) & np.isfinite(df["y"]) & np.isfinite(df["g_mag"])].copy()
    return df[df["g_mag"] <= float(mag_limit)].copy().reset_index(drop=True)


def aggregate_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = (
        df.groupby(["split_name", "family", "method", "param"], as_index=False)[
            ["pred_count", "matched_count", "false_count", "target_count"]
        ]
        .sum()
        .sort_values(["split_name", "family", "param"])
        .reset_index(drop=True)
    )
    out["precision"] = out["matched_count"] / out["pred_count"].clip(lower=1)
    out["recall"] = out["matched_count"] / out["target_count"].clip(lower=1)
    out["f1"] = 2.0 * out["precision"] * out["recall"] / (out["precision"] + out["recall"]).replace(0, np.nan)
    return out.fillna(0.0)


def plot_split(df: pd.DataFrame, split_name: str, out: Path) -> None:
    sub = df[df["split_name"] == split_name].copy()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    line_width = 0.55
    marker_width = line_width * 1.2

    for family, color, marker, label in (
        ("daofind", "red", "o", "daofind-like"),
        ("unet", "green", "s", "U-Net"),
        ("unet_plus", "blue", "^", "U-Net+"),
    ):
        part = sub[sub["family"] == family].sort_values("param")
        if part.empty:
            continue
        ax.plot(
            part["precision"],
            part["recall"],
            color=color,
            marker=marker,
            linewidth=line_width,
            markersize=2.2,
            markeredgewidth=marker_width,
            label=label,
        )
        for _, row in part.iterrows():
            txt = f"{row['param']:.2g}"
            ax.annotate(txt, (row["precision"], row["recall"]), textcoords="offset points", xytext=(3, 3), fontsize=5.5, color=color, alpha=0.85)

    ax.set_title(f"{split_name}: Gaia G<=15 precision-recall")
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out / f"pr_curve_{split_name}.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/eval PR curves for U-Net and daofind-like with Gaia G<=15 targets.")
    parser.add_argument("--data-model", type=Path, default=Path("data/data_model"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/data_20/unet_softmask"))
    parser.add_argument("--gaia-dir", type=Path, default=Path("data/data_gaia/gaia_annotations_right_fixed"))
    parser.add_argument("--selected-samples", type=Path, default=Path("data/data_20/selected_samples.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("U-Net/runs/unet_softmask_10ep/best.pt"))
    parser.add_argument("--unet-plus-checkpoint", type=Path, default=Path("U-Net+/runs/unet_plus_softmask_10ep/best.pt"))
    parser.add_argument("--no-unet-plus", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("test/unet_pr_train_eval_g15"))
    parser.add_argument("--model-thresholds", default="0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.32,0.35,0.38,0.40,0.42,0.45,0.48,0.50,0.52,0.55,0.58,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--daofind-sigmas", default="3.25,3.50,3.75,4.00,4.25,4.50,4.75,5.00,5.25,5.50,5.75,6.00,6.25,6.50")
    parser.add_argument("--mag-limit", type=float, default=15.0)
    parser.add_argument("--bright-radius", type=float, default=4.0)
    parser.add_argument("--faint-radius", type=float, default=1.5)
    parser.add_argument("--faint-mag", type=float, default=13.5)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--fwhm", type=float, default=3.0)
    parser.add_argument("--filtsize", type=int, default=25)
    parser.add_argument("--max-peaks", type=int, default=40000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = (ROOT / args.output).resolve()
    cache = out / "prob_cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv((ROOT / args.selected_samples).resolve())
    if "feature_split" not in selected.columns:
        raise SystemExit("selected_samples.csv needs a feature_split column for this train/eval plot.")
    selected = selected[selected["feature_split"].isin(["train", "val"])].copy().reset_index(drop=True)

    device = torch.device(args.device)
    ckpt = torch.load((ROOT / args.checkpoint).resolve(), map_location=device)
    base = int(ckpt.get("config", {}).get("base_channels", 32))
    model = UNet256(base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    plus_model: MultiScaleUNetPlus | None = None
    if not args.no_unet_plus:
        plus_ckpt = torch.load((ROOT / args.unet_plus_checkpoint).resolve(), map_location=device)
        plus_base = int(plus_ckpt.get("config", {}).get("base_channels", 32))
        plus_model = MultiScaleUNetPlus(base=plus_base).to(device)
        plus_model.load_state_dict(plus_ckpt["model"])
        plus_model.eval()

    data_model = (ROOT / args.data_model).resolve()
    data_dir = (ROOT / args.data_dir).resolve()
    gaia_dir = (ROOT / args.gaia_dir).resolve()
    model_thresholds = parse_csv_floats(args.model_thresholds)
    sigmas = parse_csv_floats(args.daofind_sigmas)
    rows: list[dict[str, object]] = []

    start = time.time()
    for i, row in selected.iterrows():
        sample_id = str(row["sample_id"])
        split_name = "train18" if str(row["feature_split"]) == "train" else "eval2"
        progress(i + 1, len(selected), f"{split_name} {sample_id}", start)

        target = load_target(gaia_dir, sample_id, args.mag_limit)
        single, _ = read_fits_image(resolve_manifest_path(data_model, row["single_fits"]))
        soft = np.load(data_dir / f"{sample_id}_softmask.npz")
        prob_path = cache / f"{sample_id}_unet_prob.npy"
        legacy_prob_path = cache / f"{sample_id}_prob.npy"
        if prob_path.exists():
            prob = np.load(prob_path).astype(np.float32)
        elif legacy_prob_path.exists():
            prob = np.load(legacy_prob_path).astype(np.float32)
            np.save(prob_path, prob.astype(np.float32))
        else:
            prob = infer_full(model, soft["image"].astype(np.float32), args.patch_size, args.stride, device)
            np.save(prob_path, prob.astype(np.float32))

        for thr in model_thresholds:
            pred = mask_centroids(prob, threshold=thr, min_area=2, exclude_border=8)
            met, _ = metrics(f"unet_thr{thr:g}", pred, target, args)
            met.update({"split_name": split_name, "family": "unet", "method": "U-Net", "param": float(thr)})
            rows.append(met)

        if plus_model is not None:
            plus_prob_path = cache / f"{sample_id}_unet_plus_prob.npy"
            if plus_prob_path.exists():
                plus_prob = np.load(plus_prob_path).astype(np.float32)
            else:
                plus_prob = infer_full(plus_model, soft["image"].astype(np.float32), args.patch_size, args.stride, device)
                np.save(plus_prob_path, plus_prob.astype(np.float32))
            for thr in model_thresholds:
                pred = mask_centroids(plus_prob, threshold=thr, min_area=2, exclude_border=8)
                met, _ = metrics(f"unet_plus_thr{thr:g}", pred, target, args)
                met.update({"split_name": split_name, "family": "unet_plus", "method": "U-Net+", "param": float(thr)})
                rows.append(met)

        for sigma in sigmas:
            sources = dao_like_sources(single, sigma=sigma, fwhm=args.fwhm, filtsize=args.filtsize, max_peaks=args.max_peaks)
            pred = np.asarray([(src.y, src.x) for src in sources], dtype=np.float32).reshape((-1, 2))
            met, _ = metrics(f"daofind_sigma{sigma:g}", pred, target, args)
            met.update({"split_name": split_name, "family": "daofind", "method": "daofind-like", "param": float(sigma)})
            rows.append(met)

    per_sample = pd.DataFrame(rows)
    overall = aggregate_rows(rows)
    per_sample.to_csv(out / "per_sample_pr_points.csv", index=False)
    overall.to_csv(out / "overall_pr_points.csv", index=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    plot_split(overall, "train18", out)
    plot_split(overall, "eval2", out)
    print("\n[overall]")
    print(overall.to_string(index=False))
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
