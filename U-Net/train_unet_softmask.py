from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]


def progress(current: int, total: int, text: str, start: float) -> None:
    width = 24
    done = int(round(width * current / max(total, 1)))
    bar = "#" * done + "-" * (width - done)
    elapsed = time.time() - start
    eta = elapsed / max(current, 1) * max(total - current, 0)
    print(f"\r[{bar}] {current}/{total} ({100.0 * current / max(total, 1):5.1f}%) elapsed {elapsed:7.1f}s ETA {eta:7.1f}s {text}", end="", flush=True)
    if current >= total:
        print(flush=True)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, first_kernel: int = 3) -> None:
        super().__init__()
        padding = int(first_kernel) // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, int(first_kernel), padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet256(nn.Module):
    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base, first_kernel=5)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.bottleneck = ConvBlock(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class SoftMaskPatchDataset(Dataset):
    def __init__(self, sample_ids: list[str], data_dir: Path, patch_size: int, patches_per_epoch: int, positive_prob: float, seed: int) -> None:
        self.sample_ids = sample_ids
        self.data_dir = data_dir
        self.patch_size = int(patch_size)
        self.patches_per_epoch = int(patches_per_epoch)
        self.positive_prob = float(positive_prob)
        self.rng = random.Random(int(seed))
        self.arrays: dict[str, dict[str, np.ndarray]] = {}
        self.positive_pixels: dict[str, np.ndarray] = {}
        for sid in sample_ids:
            data = np.load(data_dir / f"{sid}_softmask.npz")
            arrays = {key: data[key].astype(np.float32) for key in ("image", "target", "weight", "valid")}
            self.arrays[sid] = arrays
            yy, xx = np.where((arrays["target"] > 0.15) & (arrays["weight"] > 0))
            self.positive_pixels[sid] = np.column_stack((yy, xx)).astype(np.int32) if len(yy) else np.empty((0, 2), dtype=np.int32)

    def __len__(self) -> int:
        return self.patches_per_epoch

    def _sample_center(self, sid: str) -> tuple[int, int]:
        image = self.arrays[sid]["image"]
        h, w = image.shape
        half = self.patch_size // 2
        positives = self.positive_pixels[sid]
        if len(positives) and self.rng.random() < self.positive_prob:
            y, x = positives[self.rng.randrange(len(positives))]
            cy = int(np.clip(y + self.rng.randint(-32, 32), half, h - half - 1))
            cx = int(np.clip(x + self.rng.randint(-32, 32), half, w - half - 1))
        else:
            cy = self.rng.randint(half, h - half - 1)
            cx = self.rng.randint(half, w - half - 1)
        return cy, cx

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sid = self.sample_ids[self.rng.randrange(len(self.sample_ids))]
        arrays = self.arrays[sid]
        cy, cx = self._sample_center(sid)
        half = self.patch_size // 2
        y0 = cy - half
        x0 = cx - half
        y1 = y0 + self.patch_size
        x1 = x0 + self.patch_size
        item = {
            "image": torch.from_numpy(arrays["image"][y0:y1, x0:x1][None]),
            "target": torch.from_numpy(arrays["target"][y0:y1, x0:x1][None]),
            "weight": torch.from_numpy(arrays["weight"][y0:y1, x0:x1][None]),
        }
        return item


def weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted = loss * weight
    return weighted.sum() / torch.clamp(weight.sum(), min=1.0)


def weighted_bce_loss_parts(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted = loss * weight
    return weighted.sum(), torch.clamp(weight.sum(), min=1.0)


def make_split(selected: pd.DataFrame, val_samples: list[str]) -> tuple[list[str], list[str]]:
    all_samples = [str(sid) for sid in selected["sample_id"].tolist()]
    val = [sid for sid in val_samples if sid in all_samples]
    train = [sid for sid in all_samples if sid not in set(val)]
    return train, val


def tile_starts(length: int, patch_size: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, patch_size))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


@torch.no_grad()
def evaluate_val_loss(model: nn.Module, val_ids: list[str], data_dir: Path, patch_size: int, batch_size: int, device: torch.device) -> float:
    if not val_ids:
        return float("nan")
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    for sid in val_ids:
        data = np.load(data_dir / f"{sid}_softmask.npz")
        image = data["image"].astype(np.float32)
        target = data["target"].astype(np.float32)
        weight = data["weight"].astype(np.float32)
        ys = tile_starts(image.shape[0], patch_size)
        xs = tile_starts(image.shape[1], patch_size)
        batch_images: list[np.ndarray] = []
        batch_targets: list[np.ndarray] = []
        batch_weights: list[np.ndarray] = []
        for y0 in ys:
            for x0 in xs:
                y1 = y0 + patch_size
                x1 = x0 + patch_size
                batch_images.append(image[y0:y1, x0:x1][None])
                batch_targets.append(target[y0:y1, x0:x1][None])
                batch_weights.append(weight[y0:y1, x0:x1][None])
                if len(batch_images) >= batch_size:
                    img = torch.from_numpy(np.stack(batch_images)).to(device, non_blocking=True)
                    tgt = torch.from_numpy(np.stack(batch_targets)).to(device, non_blocking=True)
                    wgt = torch.from_numpy(np.stack(batch_weights)).to(device, non_blocking=True)
                    logits = model(img)
                    loss_sum, weight_sum = weighted_bce_loss_parts(logits, tgt, wgt)
                    total_loss += float(loss_sum.item())
                    total_weight += float(weight_sum.item())
                    batch_images.clear()
                    batch_targets.clear()
                    batch_weights.clear()
        if batch_images:
            img = torch.from_numpy(np.stack(batch_images)).to(device, non_blocking=True)
            tgt = torch.from_numpy(np.stack(batch_targets)).to(device, non_blocking=True)
            wgt = torch.from_numpy(np.stack(batch_weights)).to(device, non_blocking=True)
            logits = model(img)
            loss_sum, weight_sum = weighted_bce_loss_parts(logits, tgt, wgt)
            total_loss += float(loss_sum.item())
            total_weight += float(weight_sum.item())
    return total_loss / max(total_weight, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 08 U-Net soft-mask detector.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/data_20/unet_softmask"))
    parser.add_argument("--selected-samples", type=Path, default=Path("data/data_20/selected_samples.csv"))
    parser.add_argument("--output", type=Path, default=Path("U-Net/runs/unet_softmask"))
    parser.add_argument("--val-samples", nargs="+", default=["sample_000001", "sample_000747"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-epoch", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--positive-prob", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    selected = pd.read_csv((ROOT / args.selected_samples).resolve())
    train_ids, val_ids = make_split(selected, args.val_samples)
    if len(train_ids) != 18 or len(val_ids) != 2:
        print(f"[warn] split is train={len(train_ids)} val={len(val_ids)}")

    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_dir = (ROOT / args.data_dir).resolve()
    dataset = SoftMaskPatchDataset(train_ids, data_dir, args.patch_size, args.patches_per_epoch, args.positive_prob, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=args.device.startswith("cuda"))
    device = torch.device(args.device)
    model = UNet256(base=int(args.base_channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    config = vars(args).copy()
    config["train_ids"] = train_ids
    config["val_ids"] = val_ids
    (out / "train_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        start = time.time()
        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = weighted_bce_loss(logits, target, weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())
            count += 1
            progress(step, len(loader), f"epoch {epoch}/{args.epochs} loss={total / max(count, 1):.6f}", start)
        mean_loss = total / max(count, 1)
        val_loss = evaluate_val_loss(model, val_ids, data_dir, args.patch_size, args.val_batch_size, device)
        history.append({"epoch": epoch, "train_loss": mean_loss, "val_loss": val_loss})
        pd.DataFrame(history).to_csv(out / "loss_history.csv", index=False)
        ckpt = {"model": model.state_dict(), "config": config, "epoch": epoch, "train_loss": mean_loss, "val_loss": val_loss}
        torch.save(ckpt, out / "last.pt")
        score_loss = val_loss if np.isfinite(val_loss) else mean_loss
        if score_loss < best_loss:
            best_loss = score_loss
            torch.save(ckpt, out / "best.pt")
        print(f"[epoch {epoch}/{args.epochs}] train_loss={mean_loss:.6f} val_loss={val_loss:.6f}")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
