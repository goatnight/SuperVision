from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spinevision.checkpoint import save_checkpoint
from spinevision.data import SagittalDataset, scan_sagittal_slices
from spinevision.losses import DiceCELoss
from spinevision.metrics import dice_per_class, merge_metric_dicts
from spinevision.models import build_model, count_parameters

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train lightweight SwinUMamba-style spine segmentation.")
    parser.add_argument("--train-dir", required=True, help="Training data root: data/train/paXXX/img,label.")
    parser.add_argument("--val-dir", default=None, help="Optional validation data root. If omitted, split train-dir.")
    parser.add_argument("--out-dir", default="runs/swinumamba_lite", help="Checkpoint/output directory.")
    parser.add_argument("--model", default="lite", choices=["lite", "official"], help="lite is CPU deployable; official needs Swin-UMamba dependencies.")
    parser.add_argument("--num-classes", type=int, default=3, help="Output classes including background label 0.")
    parser.add_argument(
        "--label-mode",
        default="auto",
        choices=["auto", "binary", "three", "preserve"],
        help="auto: 2->binary, 3->old three-class recode, >=4->preserve raw labels.",
    )
    parser.add_argument("--image-size", type=int, default=320, help="Center pad/crop size. Use 256 for very small CPU deployment.")
    parser.add_argument("--base-channels", type=int, default=16, help="Lite model width. Use 8 or 12 for 2GB CPU.")
    parser.add_argument("--official-base-dim", type=int, default=24, help="Width for optional official SwinUMambaD.")
    parser.add_argument("--depth", type=int, default=1, help="Lite block depth per stage.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0, help="Keep 0 on low-memory Windows/CPU machines.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-threads", type=int, default=2, help="CPU thread cap for low-memory deployment checks.")
    parser.add_argument("--amp", action="store_true", help="CUDA mixed precision.")
    parser.add_argument("--augment", action="store_true", help="Enable light shift/rotate augmentation.")
    parser.add_argument("--shift", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--rotate", type=int, default=0)
    parser.add_argument("--elastic", type=float, default=0.0, help="Disabled by default; expensive on CPU.")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_records(records, val_split: float, seed: int):
    if not 0 < val_split < 1:
        raise ValueError("--val-split must be between 0 and 1 when --val-dir is not provided.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(records))
    val_count = max(1, int(len(indices) * val_split))
    val_indices = set(indices[:val_count].tolist())
    train_records = [rec for i, rec in enumerate(records) if i not in val_indices]
    val_records = [rec for i, rec in enumerate(records) if i in val_indices]
    return train_records, val_records


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp: bool, epoch: int):
    model.train()
    losses = []
    iterator = tqdm(loader, desc=f"epoch {epoch} train", leave=False) if tqdm else loader
    for batch in iterator:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(image)
            loss = criterion(logits, label)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.detach().cpu()))
        if tqdm:
            iterator.set_postfix(loss=np.mean(losses[-20:]))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes: int):
    model.eval()
    losses = []
    metric_rows = []
    iterator = tqdm(loader, desc="validate", leave=False) if tqdm else loader
    for batch in iterator:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(image)
        loss = criterion(logits, label)
        logits_for_pred = logits[0] if isinstance(logits, (list, tuple)) else logits
        pred = torch.argmax(logits_for_pred, dim=1)
        losses.append(float(loss.detach().cpu()))
        metric_rows.append(dice_per_class(pred.cpu(), label.cpu(), num_classes=num_classes))
    metrics = merge_metric_dicts(metric_rows)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def main():
    args = parse_args()
    if args.num_classes < 2:
        raise ValueError("--num-classes must be >= 2.")
    set_seed(args.seed)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.val_dir:
        train_records = scan_sagittal_slices(args.train_dir)
        val_records = scan_sagittal_slices(args.val_dir)
    else:
        records = scan_sagittal_slices(args.train_dir)
        train_records, val_records = split_records(records, args.val_split, args.seed)

    if not train_records or not val_records:
        raise RuntimeError(f"Empty split: train={len(train_records)}, val={len(val_records)}")

    train_ds = SagittalDataset(
        records=train_records,
        num_classes=args.num_classes,
        target_size=args.image_size,
        augment=args.augment,
        shift=tuple(args.shift),
        rotate=args.rotate,
        elastic=args.elastic,
        label_mode=args.label_mode,
        seed=args.seed,
        cache_cases=1 if device.type == "cpu" else 2,
    )
    val_ds = SagittalDataset(
        records=val_records,
        num_classes=args.num_classes,
        target_size=args.image_size,
        augment=False,
        label_mode=args.label_mode,
        cache_cases=1 if device.type == "cpu" else 2,
    )
    train_loader = make_loader(train_ds, args.batch_size, args.workers, shuffle=True, device=device)
    val_loader = make_loader(val_ds, batch_size=1, workers=args.workers, shuffle=False, device=device)

    model = build_model(
        model_name=args.model,
        num_classes=args.num_classes,
        in_channels=1,
        base_channels=args.base_channels,
        depth=args.depth,
        official_base_dim=args.official_base_dim,
    ).to(device)
    criterion = DiceCELoss(num_classes=args.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    config = vars(args).copy()
    config.update({"in_channels": 1, "train_slices": len(train_records), "val_slices": len(val_records)})
    best_dice = -1.0
    print(f"[Model] {args.model} params={count_parameters(model):,} device={device}", flush=True)
    print(f"[Data] train={len(train_ds)} val={len(val_ds)} image_size={args.image_size}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch)
        val_metrics = validate(model, val_loader, criterion, device, args.num_classes)
        scheduler.step()

        mean_dice = val_metrics["mean_dice"]
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} mean_dice={mean_dice:.4f}",
            flush=True,
        )

        save_checkpoint(out_dir / "last.pt", model, config, epoch=epoch, val_metrics=val_metrics)
        if mean_dice > best_dice:
            best_dice = mean_dice
            save_checkpoint(out_dir / "best.pt", model, config, epoch=epoch, val_metrics=val_metrics)
            print(f"[Checkpoint] saved best.pt mean_dice={best_dice:.4f}", flush=True)


if __name__ == "__main__":
    main()
