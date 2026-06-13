from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(module: str, *args: object) -> None:
    command = [sys.executable, "-m", module, *[str(value) for value in args]]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage runner for the candidate scorer v2 cloud workflow.")
    parser.add_argument(
        "stage",
        choices=("probe", "build", "train", "hard", "retrain", "dev", "test"),
    )
    parser.add_argument("--candidate-methods", default="daofind:2.0,daofind:2.5,sextractor:1.5")
    parser.add_argument("--data-root", default="data_model")
    parser.add_argument("--data-dir", default="data/candidate_scorer_v2")
    parser.add_argument("--hard-data-dir", default="data/candidate_scorer_v2_hard")
    parser.add_argument("--run-dir", default="runs/candidate_scorer_v2_seed42")
    parser.add_argument("--hard-run-dir", default="runs/candidate_scorer_v2_hard_seed42")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoints", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "probe":
        run(
            "candidate_scorer.probe_candidates",
            "--data-root", args.data_root,
            "--split-reason", "frame_holdout",
            "--out-dir", "runs/candidate_probe_dev",
        )
    elif args.stage == "build":
        command = [
            "--candidate-methods", args.candidate_methods,
            "--data-root", args.data_root,
            "--crop-mode", "stratified",
            "--crops-per-image", 5,
            "--train-split-reason", "train",
            "--val-split-reason", "frame_holdout",
            "--out-dir", args.data_dir,
        ]
        if args.resume:
            command.append("--resume")
        run("candidate_scorer.build_dataset", *command)
    elif args.stage == "train":
        command = [
            "--data-dir", args.data_dir,
            "--data-root", args.data_root,
            "--out-dir", args.run_dir,
            "--epochs", args.epochs,
            "--batch-size", args.batch_size,
            "--eval-batch-size", args.eval_batch_size,
            "--num-workers", args.num_workers,
            "--device", args.device,
            "--seed", args.seed,
        ]
        if args.amp:
            command.append("--amp")
        run("candidate_scorer.train", *command)
    elif args.stage == "hard":
        checkpoint = args.checkpoint or f"{args.run_dir}/candidate_scorer_best.pt"
        command = [
            "--checkpoint", checkpoint,
            "--data-root", args.data_root,
            "--out-dir", args.hard_data_dir,
            "--batch-size", args.eval_batch_size,
            "--device", args.device,
        ]
        if args.resume:
            command.append("--resume")
        run("candidate_scorer.build_hard_dataset", *command)
    elif args.stage == "retrain":
        command = [
            "--data-dir", args.data_dir,
            "--data-root", args.data_root,
            "--hard-negative-data-dir", args.hard_data_dir,
            "--out-dir", args.hard_run_dir,
            "--epochs", args.epochs,
            "--batch-size", args.batch_size,
            "--eval-batch-size", args.eval_batch_size,
            "--num-workers", args.num_workers,
            "--device", args.device,
            "--seed", args.seed,
        ]
        if args.amp:
            command.append("--amp")
        run("candidate_scorer.train", *command)
    elif args.stage == "dev":
        checkpoints = args.checkpoints or f"{args.hard_run_dir}/candidate_scorer_best.pt"
        run(
            "candidate_scorer.evaluate",
            "--checkpoints", checkpoints,
            "--data-root", args.data_root,
            "--split-reason", "frame_holdout",
            "--batch-size", args.eval_batch_size,
            "--device", args.device,
            "--score-thresholds", "0.05:0.95:0.01",
            "--out-dir", f"{args.hard_run_dir}/eval_dev",
        )
    else:
        if args.fixed_threshold is None:
            raise ValueError("test stage requires --fixed-threshold selected on frame_holdout")
        checkpoints = args.checkpoints or f"{args.hard_run_dir}/candidate_scorer_best.pt"
        run(
            "candidate_scorer.evaluate",
            "--checkpoints", checkpoints,
            "--data-root", args.data_root,
            "--split-reason", "coord_holdout",
            "--fixed-threshold", args.fixed_threshold,
            "--batch-size", args.eval_batch_size,
            "--device", args.device,
            "--out-dir", f"{args.hard_run_dir}/eval_coord_holdout_locked",
        )


if __name__ == "__main__":
    main()
