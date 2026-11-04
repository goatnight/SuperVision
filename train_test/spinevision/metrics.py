from __future__ import annotations

import torch


@torch.no_grad()
def dice_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> dict[str, float]:
    scores: dict[str, float] = {}
    for cls in range(1, num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        denom = pred_mask.sum().float() + target_mask.sum().float()
        if denom.item() == 0:
            scores[f"dice_cls_{cls}"] = 1.0
        else:
            inter = (pred_mask & target_mask).sum().float()
            scores[f"dice_cls_{cls}"] = float(((2 * inter + eps) / (denom + eps)).item())
    if scores:
        scores["mean_dice"] = float(sum(scores.values()) / len(scores))
    else:
        scores["mean_dice"] = 0.0
    return scores


def merge_metric_dicts(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {"mean_dice": 0.0}
    keys = metrics[0].keys()
    return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}
