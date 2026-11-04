from __future__ import annotations

from .SwinUMambaD import SwinUMambaD


def build_official_swin_umamba(
    num_classes: int,
    in_channels: int = 1,
    base_dim: int = 24,
    depths: tuple[int, int, int, int] = (1, 1, 1, 1),
    d_state: int = 8,
    deep_supervision: bool = False,
):
    dims = [base_dim * (2**i) for i in range(4)]
    vss_args = dict(
        in_chans=in_channels,
        patch_size=4,
        depths=list(depths),
        dims=dims,
        d_state=d_state,
        drop_path_rate=0.0,
    )
    decoder_args = dict(
        num_classes=num_classes,
        deep_supervision=deep_supervision,
        features_per_stage=dims,
        drop_path_rate=0.0,
        d_state=d_state,
    )
    return SwinUMambaD(vss_args, decoder_args)
