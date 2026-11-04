from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VSSLiteBlock(nn.Module):
    """CPU-friendly VSS-style block.

    The official Swin-UMamba block relies on selective scan kernels. For a 2GB CPU
    target, this block keeps the residual/gated/local-mixing idea but uses only
    depthwise separable convolutions that export cleanly to TorchScript.
    """

    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.in_proj = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.dw_conv = nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, groups=hidden)
        self.out_proj = nn.Conv2d(hidden, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.ones(1) * 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=1)
        x = self.dw_conv(F.silu(x))
        x = x * torch.sigmoid(gate)
        x = self.out_proj(x)
        return residual + self.scale * x


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int, downsample: bool):
        super().__init__()
        stride = 2 if downsample else 1
        layers: list[nn.Module] = [ConvNormAct(in_channels, out_channels, stride=stride)]
        layers.extend(VSSLiteBlock(out_channels) for _ in range(depth))
        self.stage = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, depth: int):
        super().__init__()
        self.fuse = ConvNormAct(in_channels + skip_channels, out_channels)
        self.blocks = nn.Sequential(*(VSSLiteBlock(out_channels) for _ in range(depth)))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.blocks(x)


class SwinUMambaLite(nn.Module):
    """Small 2D segmentation model for low-memory CPU deployment."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 16,
        depth: int = 1,
    ):
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 6

        self.enc1 = EncoderStage(in_channels, c1, depth=depth, downsample=False)
        self.enc2 = EncoderStage(c1, c2, depth=depth, downsample=True)
        self.enc3 = EncoderStage(c2, c3, depth=depth, downsample=True)
        self.enc4 = EncoderStage(c3, c4, depth=depth, downsample=True)

        self.dec3 = DecoderStage(c4, c3, c3, depth=depth)
        self.dec2 = DecoderStage(c3, c2, c2, depth=depth)
        self.dec1 = DecoderStage(c2, c1, c1, depth=depth)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.enc4(s3)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
