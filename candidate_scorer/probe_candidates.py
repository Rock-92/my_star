from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from candidate_scorer.evaluate import load_rows, select_rows
from candidate_scorer.pipeline import generate_candidates, resolve_data_path
from star_unet.dataset import _read_input_image, _read_mask_image
from star_unet.evaluate import resolve_path
from star_unet.postprocess import detection_metrics, heatmap_to_centroids
from star_unet.train import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure candidate recall ceilings on full frames.")
    parser.add_argument("--data-root", type=Path, default=Path("data/data_model"))
    parser.add_argument("--config", type=Path, default=Path("star_unet/config.json"))
    parser.add_argument("--split-reason", default="frame_holdout")
    parser.add_argument("--candidate-sets", default=(
        "dao2=daofind:2.0;"
        "dao2p5=daofind:2.5;"
        "dao_multi=daofind:2.0,daofind:2.5,daofind:3.0;"
        "log=log:2.5;"
        "mixed=daofind:2.0,daofind:2.5,sextractor:1.5,log:2.5"
    ))
    parser.add_argument("--dedup-radius-px", type=float, default=2.5)
    parser.add_argument("--match-radius-px", type=float, default=4.0)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument(
        "--candidate-budgets",
        default="0",
        help="Comma-separated per-frame top-response limits. 0 keeps all candidates.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/candidate_probe"))
    return parser.parse_args()


def parse_sets(text: str) -> list[tuple[str, str]]:
    sets = []
    for item in text.split(";"):
        if not item.strip():
            continue
        name, methods = item.split("=", 1)
        sets.append((name.strip(), methods.strip()))
    return sets


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    channel_mode = str(config.get("fit_channel_mode", "mean"))
    data_root = resolve_path(args.data_root)
    rows = select_rows(load_rows(data_root, args.split_reason), "", args.count)
    candidate_sets = parse_sets(args.candidate_sets)
    budgets = [int(value.strip()) for value in args.candidate_budgets.split(",") if value.strip()]
    totals = {
        (name, budget): {"pred_count": 0, "target_count": 0, "matched_count": 0, "seconds": 0.0}
        for name, _ in candidate_sets
        for budget in budgets
    }
    per_sample = []
    for index, sample in enumerate(rows, start=1):
        raw = _read_input_image(
            resolve_data_path(data_root, sample.get("image_out") or sample.get("single_fits") or ""),
            channel_mode,
        )
        targets = heatmap_to_centroids(
            _read_mask_image(resolve_data_path(data_root, sample.get("mask_out") or "")),
            threshold=args.target_threshold,
            min_distance=args.min_distance,
        )
        sample_rows = []
        for name, methods in candidate_sets:
            started = time.perf_counter()
            candidate_set = generate_candidates(raw, methods, args.dedup_radius_px)
            elapsed = time.perf_counter() - started
            response_order = np.argsort(-candidate_set.response)
            for budget in budgets:
                selected = response_order if budget <= 0 else response_order[:budget]
                candidates = candidate_set.centroids_yx[selected]
                metrics = detection_metrics(candidates, targets, radius_px=args.match_radius_px)
                sample_rows.append({
                    "name": name,
                    "methods": methods,
                    "candidate_budget": budget,
                    "seconds": elapsed,
                    **metrics,
                })
                for key in ("pred_count", "target_count", "matched_count"):
                    totals[(name, budget)][key] += int(metrics[key])
                totals[(name, budget)]["seconds"] += elapsed
        per_sample.append({"sample_id": sample["sample_id"], "results": sample_rows})
        print(f"[{index}/{len(rows)}] {sample['sample_id']}", flush=True)

    summary = []
    for name, methods in candidate_sets:
        for budget in budgets:
            row = totals[(name, budget)]
            recall = row["matched_count"] / max(row["target_count"], 1)
            precision = row["matched_count"] / max(row["pred_count"], 1)
            oracle_f1 = 2 * recall / max(1 + recall, 1e-12)
            summary.append({
                "name": name,
                "methods": methods,
                "candidate_budget": budget,
                "samples": len(rows),
                **row,
                "precision": precision,
                "recall": recall,
                "oracle_f1": oracle_f1,
                "mean_candidates": row["pred_count"] / max(len(rows), 1),
                "mean_seconds": row["seconds"] / max(len(rows), 1),
            })
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps({"summary": summary, "per_sample": per_sample}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
