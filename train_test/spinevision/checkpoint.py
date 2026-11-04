from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models import build_model


def save_checkpoint(path: str | Path, model: torch.nn.Module, config: dict[str, Any], **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": config,
        **extra,
    }
    torch.save(payload, path)


def load_model_from_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = build_model(
        model_name=config["model"],
        num_classes=config["num_classes"],
        in_channels=config.get("in_channels", 1),
        base_channels=config.get("base_channels", 16),
        depth=config.get("depth", 1),
        official_base_dim=config.get("official_base_dim", 24),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, config, checkpoint
