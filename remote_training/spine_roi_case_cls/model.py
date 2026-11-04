from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_segmentation_encoder(project_root: str | Path, checkpoint_path: str | Path):
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from spinevision.checkpoint import load_model_from_checkpoint

    segmentation_model, config, _ = load_model_from_checkpoint(
        checkpoint_path, torch.device("cpu")
    )
    required = ("enc1", "enc2", "enc3", "enc4")
    if not all(hasattr(segmentation_model, name) for name in required):
        raise TypeError("The segmentation checkpoint does not expose enc1..enc4")
    encoder = SegmentationEncoder(
        segmentation_model.enc1,
        segmentation_model.enc2,
        segmentation_model.enc3,
        segmentation_model.enc4,
    )
    out_channels = int(config.get("base_channels", 16)) * 6
    return encoder, out_channels, config


class SegmentationEncoder(nn.Module):
    """Reuses the encoder tensors from the trained SwinUMambaLite segmenter."""

    def __init__(self, enc1: nn.Module, enc2: nn.Module, enc3: nn.Module, enc4: nn.Module):
        super().__init__()
        self.enc1 = enc1
        self.enc2 = enc2
        self.enc3 = enc3
        self.enc4 = enc4

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.enc1(inputs)
        features = self.enc2(features)
        features = self.enc3(features)
        return self.enc4(features)


class ROIAssistedCaseClassifier(nn.Module):
    """Shared segmentation encoder + raw/ROI gated fusion + slice attention MIL."""

    def __init__(
        self,
        segmentation_project: str | Path,
        segmentation_checkpoint: str | Path,
        num_segmentation_classes: int = 13,
        hidden_dim: int = 192,
        dropout: float = 0.3,
        roi_dilation: int = 9,
    ):
        super().__init__()
        self.encoder, encoder_channels, self.segmentation_config = load_segmentation_encoder(
            segmentation_project, segmentation_checkpoint
        )
        self.num_segmentation_classes = int(num_segmentation_classes)
        self.roi_dilation = int(roi_dilation)
        pooled_dim = encoder_channels * 2
        stats_dim = self.num_segmentation_classes

        self.raw_projection = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.roi_projection = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.stats_projection = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + stats_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.slice_norm = nn.LayerNorm(hidden_dim)
        self.slice_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    @staticmethod
    def _pool(features: torch.Tensor) -> torch.Tensor:
        average = F.adaptive_avg_pool2d(features, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(features, 1).flatten(1)
        return torch.cat([average, maximum], dim=1)

    def _mask_stats(self, masks: torch.Tensor) -> torch.Tensor:
        fractions = [
            masks.eq(label).float().mean(dim=(-2, -1))
            for label in range(1, self.num_segmentation_classes)
        ]
        foreground = masks.gt(0).float().mean(dim=(-2, -1))
        return torch.stack([foreground, *fractions], dim=-1)

    def set_encoder_mode(self, mode: str) -> None:
        if mode not in {"frozen", "last", "all"}:
            raise ValueError(f"Unknown encoder mode: {mode}")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = mode == "all"
        if mode == "last":
            for parameter in self.encoder.enc4.parameters():
                parameter.requires_grad = True

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
        # images: [B, S, 1, H, W], masks: [B, S, H, W]
        batch, slices, _, height, width = images.shape
        raw = images.reshape(batch * slices, 1, height, width)
        binary = masks.gt(0).float().reshape(batch * slices, 1, height, width)
        if self.roi_dilation > 1:
            padding = self.roi_dilation // 2
            binary = F.max_pool2d(binary, self.roi_dilation, stride=1, padding=padding)
        roi = raw * binary

        # The same pretrained encoder is shared by both branches. This keeps the ROI branch
        # auxiliary instead of doubling the number of trainable backbone parameters.
        raw_vector = self.raw_projection(self._pool(self.encoder(raw)))
        roi_vector = self.roi_projection(self._pool(self.encoder(roi)))
        stats = self._mask_stats(masks).reshape(batch * slices, -1)
        gate = self.gate(torch.cat([raw_vector, roi_vector, stats], dim=1))
        fused = gate * raw_vector + (1.0 - gate) * roi_vector + self.stats_projection(stats)
        fused = self.slice_norm(fused).reshape(batch, slices, -1)

        attention = torch.softmax(self.slice_attention(fused).squeeze(-1), dim=1)
        case_vector = torch.sum(fused * attention.unsqueeze(-1), dim=1)
        logits = self.classifier(case_vector)
        return {
            "logits": logits,
            "attention": attention,
            "mean_gate": gate.reshape(batch, slices, -1).mean(dim=-1),
        }

    def model_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "shared_swinumamba_encoder_raw_roi_gate_slice_attention",
            "segmentation_config": self.segmentation_config,
            "num_segmentation_classes": self.num_segmentation_classes,
            "roi_dilation": self.roi_dilation,
        }
