import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from model.dataloader_npz import NPZPatchDataset
from model.load_param_data import load_param
from model.model_factory import get_model
from model.utils import weights_init_xavier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def project_root():
    return Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def build_model(args, device):
    nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
    ARFAM, DNRFANet = get_model("DNRFANet")
    model = DNRFANet(
        num_classes=1,
        input_channels=args.in_channels,
        block=ARFAM,
        num_blocks=num_blocks,
        nb_filter=nb_filter,
        netdepth=args.netdepth,
        scale_method=args.scale_method,
        deep_supervision=str(args.deep_supervision),
    )
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {args.resume}")
    else:
        model.apply(weights_init_xavier)
    return model.to(device)


def siou_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    soft_iou = (intersection + smooth) / (union + smooth)
    gt_soft_iou = (target.sum(dim=(1, 2, 3)) + smooth) / ((target > 0).sum(dim=(1, 2, 3)) + smooth)
    soft_iou = soft_iou / gt_soft_iou
    return 1 - soft_iou.mean()


def sfl_loss(pred, target, gamma=2.0):
    pred = torch.sigmoid(pred)
    pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
    target = torch.clamp(target, 0.0, 1.0)
    pt = (1 - target) * torch.log(1 - pred) + target * torch.log(pred)
    weight = torch.pow(torch.abs(target - pred), gamma)
    return (-weight * pt).mean()


def smooth_iou_loss(pred, target, smooth=1e-12):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return (1 - (intersection + smooth) / (union + smooth)).mean()


def compute_loss(outputs, target, loss_func):
    preds = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    loss = 0.0
    for pred in preds:
        if loss_func == "SIoU":
            loss = loss + siou_loss(pred, target)
        elif loss_func == "SFL":
            loss = loss + sfl_loss(pred, target) / 100.0
        elif loss_func == "SIoU+SFL":
            loss = loss + siou_loss(pred, target) + sfl_loss(pred, target) / 100.0
        elif loss_func == "IoU":
            loss = loss + smooth_iou_loss(pred, target)
        else:
            raise ValueError(f"Unsupported loss: {loss_func}")
    return loss / len(preds)


def batch_iou(outputs, target, threshold):
    pred = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
    prob = torch.sigmoid(pred)
    pred_bin = prob > threshold
    target_bin = target > 0
    intersection = (pred_bin & target_bin).sum(dim=(1, 2, 3)).float()
    union = (pred_bin | target_bin).sum(dim=(1, 2, 3)).float()
    valid = union > 0
    iou = torch.ones_like(union)
    iou[valid] = intersection[valid] / union[valid]
    return iou.mean().item()


def update_binary_counts(outputs, target, threshold, counts):
    pred = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
    pred_bin = torch.sigmoid(pred) > threshold
    target_bin = target > 0
    counts["tp"] += (pred_bin & target_bin).sum().item()
    counts["fp"] += (pred_bin & ~target_bin).sum().item()
    counts["fn"] += (~pred_bin & target_bin).sum().item()


def precision_recall_f1(counts):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def run_epoch(model, loader, optimizer, device, args, train):
    model.train(train)
    total_loss = 0.0
    total_iou = 0.0
    total_items = 0
    counts = {"tp": 0, "fp": 0, "fn": 0}
    iterator = tqdm(loader, desc="train" if train else "val")

    max_batches = args.max_train_batches if train else args.max_val_batches
    for batch_idx, (images, masks) in enumerate(iterator):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = compute_loss(outputs, masks, args.loss_func)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_iou += batch_iou(outputs, masks, args.threshold) * batch_size
        if not train:
            update_binary_counts(outputs, masks, args.threshold, counts)
        total_items += batch_size
        postfix = {
            "loss": total_loss / max(1, total_items),
            "iou": total_iou / max(1, total_items),
        }
        if not train:
            precision, recall, f1 = precision_recall_f1(counts)
            postfix.update({"precision": precision, "recall": recall, "f1": f1})
        iterator.set_postfix(**postfix)

    metrics = {
        "loss": total_loss / max(1, total_items),
        "iou": total_iou / max(1, total_items),
    }
    if not train:
        precision, recall, f1 = precision_recall_f1(counts)
        metrics.update({"precision": precision, "recall": recall, "f1": f1})
    return metrics


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_iou, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_val_iou": best_val_iou,
            "args": vars(args),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train DNRFA-Net from NPZ patches.")
    parser.add_argument("--data-root", default="dataset/npz_patches")
    parser.add_argument("--train-list", default="")
    parser.add_argument("--val-list", default="")
    parser.add_argument("--save-dir", default="result/dnrfanet_npz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Debug only. 0 means no limit.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Debug only. 0 means no limit.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=["Adagrad", "Adam"], default="Adagrad")
    parser.add_argument("--loss-func", choices=["SIoU", "SFL", "SIoU+SFL", "IoU"], default="SIoU+SFL")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--channel-size", choices=["one", "two", "three", "four"], default="three")
    parser.add_argument("--backbone", choices=["resnet_10", "resnet_18", "resnet_34", "vgg_10"], default="resnet_18")
    parser.add_argument("--netdepth", type=int, choices=[3, 4, 5], default=4)
    parser.add_argument("--scale-method", choices=["deconv", "biinterp", "nearest"], default="deconv")
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--deep-supervision", type=str2bool, default=False)
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)
    train_list = resolve_path(args.train_list) if args.train_list else data_root / "train.txt"
    val_list = resolve_path(args.val_list) if args.val_list else data_root / "val.txt"
    save_dir = resolve_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train_set = NPZPatchDataset(train_list, require_mask=True)
    val_set = NPZPatchDataset(val_list, require_mask=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    model = build_model(args, device)
    if args.optimizer == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    print(f"Train list: {train_list} ({len(train_set)} samples)")
    print(f"Val list: {val_list} ({len(val_set)} samples)")
    print(f"Save dir: {save_dir}")
    print(f"Device: {device}")

    best_val_iou = -1.0
    log_path = save_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_iou",
                "val_loss",
                "val_iou",
                "val_precision",
                "val_recall",
                "val_f1",
            ],
        )
        writer.writeheader()

    (save_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args, train=False)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        with log_path.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=row.keys()).writerow(row)

        save_checkpoint(save_dir / "last.pth.tar", model, optimizer, scheduler, epoch, best_val_iou, args)
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            save_checkpoint(save_dir / "best.pth.tar", model, optimizer, scheduler, epoch, best_val_iou, args)
            print(f"Saved best checkpoint: val_iou={best_val_iou:.6f}")


if __name__ == "__main__":
    main()
