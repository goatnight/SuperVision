from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import file_sha256, load_config, resolve_device, save_json, set_seed
from data import CaseROIDataset, read_case_table
from metrics import binary_metrics, bootstrap_confidence_intervals, optimal_youden_threshold
from model import ROIAssistedCaseClassifier


def make_loader(dataset: CaseROIDataset, config: dict, shuffle: bool) -> DataLoader:
    workers = int(config["num_workers"])
    generator = torch.Generator().manual_seed(int(config["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
        drop_last=False,
    )


def autocast_context(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype in {"bfloat16", "float16"}
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
    threshold: float | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            with autocast_context(device, amp_dtype):
                outputs = model(images, masks)
                loss = criterion(outputs["logits"], targets)
            probs = torch.softmax(outputs["logits"].float(), dim=1)[:, 1].cpu().numpy()
            attention = outputs["attention"].float().cpu().numpy()
            gates = outputs["mean_gate"].float().cpu().numpy()
            losses.extend([float(loss)] * len(probs))
            labels.extend(targets.cpu().numpy().tolist())
            probabilities.extend(probs.tolist())
            for index, patient_id in enumerate(batch["patient_id"]):
                rows.append(
                    {
                        "patient_id": patient_id,
                        "center": batch["center"][index],
                        "label": int(targets[index].item()),
                        "probability": float(probs[index]),
                        "slice_attention": json.dumps(attention[index].tolist()),
                        "roi_gate": json.dumps(gates[index].tolist()),
                    }
                )
    y_true = np.asarray(labels)
    y_prob = np.asarray(probabilities)
    selected_threshold = optimal_youden_threshold(y_true, y_prob) if threshold is None else threshold
    metrics = binary_metrics(y_true, y_prob, selected_threshold)
    metrics["loss"] = float(np.mean(losses))
    predictions = pd.DataFrame(rows)
    predictions["prediction"] = (predictions["probability"] >= selected_threshold).astype(int)
    return metrics, predictions


def train_epoch(
    model: ROIAssistedCaseClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: str,
) -> float:
    model.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_dtype):
            outputs = model(images, masks)
            loss = criterion(outputs["logits"], targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ROI-assisted patient-level spondylolisthesis classifier")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device(args.device)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    table = read_case_table(config["label_csv"])
    train_dataset = CaseROIDataset(table, "train", config["cache_dir"], config["augmentation"])
    val_dataset = CaseROIDataset(table, "val", config["cache_dir"])
    test_dataset = CaseROIDataset(table, "test", config["cache_dir"])
    train_loader = make_loader(train_dataset, config, shuffle=True)
    val_loader = make_loader(val_dataset, config, shuffle=False)
    test_loader = make_loader(test_dataset, config, shuffle=False)

    model = ROIAssistedCaseClassifier(
        config["segmentation_project"],
        config["segmentation_checkpoint"],
        num_segmentation_classes=int(config["num_segmentation_classes"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        roi_dilation=int(config["roi_dilation"]),
    ).to(device)
    model.set_encoder_mode("frozen")
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    encoder_parameters = [parameter for parameter in model.parameters() if id(parameter) in encoder_ids]
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": float(config["encoder_learning_rate"])},
            {"params": head_parameters, "lr": float(config["learning_rate"])},
        ],
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))

    best_auc = -1.0
    epochs_without_improvement = 0
    history = []
    best_path = output_dir / "best.pt"
    started = time.time()
    for epoch in range(int(config["epochs"])):
        if epoch == int(config["freeze_encoder_epochs"]):
            model.set_encoder_mode(str(config["unfreeze_mode"]))
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config["amp_dtype"])
        val_metrics, val_predictions = evaluate(model, val_loader, device, config["amp_dtype"], threshold=None)
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "lr_encoder": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if val_metrics["auc"] > best_auc + 1e-5:
            best_auc = float(val_metrics["auc"])
            epochs_without_improvement = 0
            val_predictions.to_csv(output_dir / "best_val_predictions.csv", index=False)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "model_metadata": model.model_metadata(),
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                    "decision_threshold": val_metrics["threshold"],
                    "segmentation_checkpoint_sha256": file_sha256(config["segmentation_checkpoint"]),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        if epochs_without_improvement >= int(config["patience"]):
            print(f"Early stopping after {epoch + 1} epochs")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(checkpoint["decision_threshold"])
    val_metrics, val_predictions = evaluate(model, val_loader, device, config["amp_dtype"], threshold)
    test_metrics, test_predictions = evaluate(model, test_loader, device, config["amp_dtype"], threshold)
    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    test_ci = bootstrap_confidence_intervals(
        test_predictions["label"].to_numpy(),
        test_predictions["probability"].to_numpy(),
        threshold,
        int(config["bootstrap_samples"]),
        int(config["seed"]),
    )
    summary = {
        "status": "completed",
        "device": str(device),
        "duration_seconds": time.time() - started,
        "best_epoch": int(checkpoint["epoch"]),
        "decision_threshold_selected_on_val": threshold,
        "split_sizes": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
        "val": val_metrics,
        "test": test_metrics,
        "test_bootstrap_95_ci": test_ci,
        "checkpoint": str(best_path),
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
