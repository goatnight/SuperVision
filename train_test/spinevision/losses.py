from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.include_background = include_background
        self.eps = float(eps)

    def forward(self, logits, target: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            weights = [1.0 / (2**i) for i in range(len(logits))]
            total = sum(weights)
            weights = [w / total for w in weights]
            return sum(w * self._single_loss(out, target) for w, out in zip(weights, logits))
        return self._single_loss(logits, target)

    def _single_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[-2:] != target.shape[-2:]:
            target_for_ce = F.interpolate(target[:, None].float(), size=logits.shape[-2:], mode="nearest").squeeze(1).long()
        else:
            target_for_ce = target

        ce = F.cross_entropy(logits, target_for_ce)
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target_for_ce, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if not self.include_background and self.num_classes > 1:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_oh, dims)
        denominator = torch.sum(probs + target_oh, dims)
        dice_loss = 1.0 - torch.mean((2.0 * intersection + self.eps) / (denominator + self.eps))
        return self.ce_weight * ce + self.dice_weight * dice_loss
