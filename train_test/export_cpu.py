from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spinevision.checkpoint import load_model_from_checkpoint


class LogitsOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out


def parse_args():
    parser = argparse.ArgumentParser(description="Export a checkpoint for low-memory CPU inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="runs/deploy/model_cpu.ts")
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to checkpoint image_size.")
    parser.add_argument("--num-threads", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    device = torch.device("cpu")
    model, config, _ = load_model_from_checkpoint(args.checkpoint, device)
    model = LogitsOnly(model).eval()
    image_size = args.image_size or int(config["image_size"])
    dummy = torch.zeros(1, 1, image_size, image_size, dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=False)
        traced = torch.jit.optimize_for_inference(traced)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))

    meta = {
        "checkpoint": str(args.checkpoint),
        "image_size": image_size,
        "num_classes": int(config["num_classes"]),
        "model": config["model"],
        "base_channels": config.get("base_channels"),
        "depth": config.get("depth"),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Exported] {output}", flush=True)
    print(f"[Meta] {output.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
