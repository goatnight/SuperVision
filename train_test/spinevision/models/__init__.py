from __future__ import annotations

from .swin_umamba_lite import SwinUMambaLite, count_parameters


def build_model(
    model_name: str,
    num_classes: int,
    in_channels: int = 1,
    base_channels: int = 16,
    depth: int = 1,
    official_base_dim: int = 24,
):
    model_name = model_name.lower()
    if model_name in {"lite", "swinumamba_lite", "swin_umamba_lite"}:
        return SwinUMambaLite(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=depth,
        )

    if model_name in {"official", "swinumambad", "swin_umamba_d"}:
        from .official import build_official_swin_umamba

        return build_official_swin_umamba(
            num_classes=num_classes,
            in_channels=in_channels,
            base_dim=official_base_dim,
            depths=(1, 1, 1, 1),
            d_state=8,
            deep_supervision=False,
        )

    raise ValueError(f"Unknown model: {model_name}")
