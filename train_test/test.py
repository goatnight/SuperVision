from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spinevision.checkpoint import load_model_from_checkpoint
from spinevision.data import SagittalDataset
from spinevision.metrics import dice_per_class, merge_metric_dicts

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained spine segmentation checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="runs/test")
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to checkpoint image_size.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--save-predictions", action="store_true", help="Save per-slice npz predictions.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    model, config, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    image_size = args.image_size or int(config["image_size"])
    num_classes = int(config["num_classes"])
    dataset = SagittalDataset(
        data_dir=args.data_dir,
        num_classes=num_classes,
        target_size=image_size,
        augment=False,
        label_mode=config.get("label_mode", "auto"),
        cache_cases=1,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    rows = []
    iterator = tqdm(loader, desc="test", leave=False) if tqdm else loader
    with torch.no_grad():
        for batch in iterator:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            logits = model(image)
            logits = logits[0] if isinstance(logits, (list, tuple)) else logits
            pred = torch.argmax(logits, dim=1)
            metrics = dice_per_class(pred.cpu(), label.cpu(), num_classes=num_classes)
            name = batch["filename"][0]
            row = {"filename": name, **metrics}
            rows.append(row)

            if args.save_predictions:
                np.savez_compressed(
                    pred_dir / f"{name}.npz",
                    pred=pred.squeeze(0).cpu().numpy().astype(np.uint8),
                    label=label.squeeze(0).cpu().numpy().astype(np.uint8),
                )

    summary = merge_metric_dicts([{k: v for k, v in row.items() if k != "filename"} for row in rows])
    metric_path = out_dir / "metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["filename", "mean_dice"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Checkpoint] epoch={checkpoint.get('epoch')} {args.checkpoint}", flush=True)
    print(f"[Metrics] {summary}", flush=True)
    print(f"[Saved] {metric_path}", flush=True)


if __name__ == "__main__":
    main()
