from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candidate_scorer.model import CandidateScorer
from star_unet.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the candidate scorer on prebuilt patch data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/candidate_scorer_sigma2p5"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/candidate_scorer_sigma2p5"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to continue training from.")
    return parser.parse_args()


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["patches"].astype(np.float32), data["labels"].astype(np.float32)


def evaluate(
    model: nn.Module,
    patches: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
    device: torch.device,
    batch_size: int,
    criterion: nn.Module | None = None,
) -> tuple[float | None, list[dict[str, float]]]:
    scores = []
    total_loss = 0.0
    total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            batch = torch.from_numpy(patches[start : start + batch_size]).to(device)
            batch_labels = torch.from_numpy(labels[start : start + batch_size]).to(device)
            logits = model(batch)
            if criterion is not None:
                total_loss += float(criterion(logits, batch_labels).item()) * len(batch_labels)
                total += len(batch_labels)
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    scores_np = np.concatenate(scores)
    rows = []
    for threshold in thresholds:
        pred = scores_np >= threshold
        truth = labels > 0.5
        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = int(np.count_nonzero(~pred & truth))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({"score_threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1})
    val_loss = total_loss / max(total, 1) if criterion is not None else None
    return val_loss, rows


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir if args.data_dir.is_absolute() else REPO_ROOT / args.data_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y = load_npz(data_dir / "train_patches.npz")
    val_x, val_y = load_npz(data_dir / "val_patches.npz")
    device = resolve_device(args.device)
    input_channels = int(train_x.shape[1])
    model = CandidateScorer(input_channels=input_channels).to(device)
    if args.resume is not None:
        resume_path = args.resume if args.resume.is_absolute() else REPO_ROOT / args.resume
        checkpoint = torch.load(resume_path, map_location=device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"[resume] loaded {resume_path}")
    neg = int(np.sum(train_y <= 0.5))
    pos = int(np.sum(train_y > 0.5))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, neg / max(pos, 1))], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    thresholds = [float(item.strip()) for item in args.score_thresholds.split(",") if item.strip()]
    history = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for patches, labels in loader:
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(patches), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total += len(labels)
        val_loss, rows = evaluate(model, val_x, val_y, thresholds, device, args.batch_size, criterion=criterion)
        best = max(rows, key=lambda row: row["f1"])
        train_loss = total_loss / max(total, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best": best})
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best": best,
            "input_channels": input_channels,
        }
        torch.save(checkpoint, out_dir / "candidate_scorer_last.pt")
        torch.save(checkpoint, out_dir / "candidate_scorer.pt")
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            torch.save(checkpoint, out_dir / "candidate_scorer_best.pt")
            best_note = " saved_best"
        else:
            best_note = ""
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f}"
            f" val_loss={val_loss:.4f}"
            f" best_t={best['score_threshold']}"
            f" f1={best['f1']:.3f}"
            f" p={best['precision']:.3f}"
            f" r={best['recall']:.3f}"
            f"{best_note}"
        )

    _, final_rows = evaluate(model, val_x, val_y, thresholds, device, args.batch_size, criterion=criterion)
    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
