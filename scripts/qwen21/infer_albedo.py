#!/usr/bin/env python3
"""Run a Qwen-Image 2.1 albedo checkpoint on a folder of pictures.

Writes ``<name>.png`` per picture into the output folder: the albedo in sRGB
(Marigold V2's gamma 2.2). A picture with an alpha channel is composited over
black and its alpha is copied to the output. Each picture runs resized so its
long side is ``--long_side`` (sides rounded to multiples of 32), then back.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from marigoldv2.core.builder import build_transforms, register_experiment_modules
from marigoldv2.core.registry import REGISTRY
from marigoldv2.util.config_resolvers import recursive_load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO_ROOT
    / "marigoldv2"
    / "experiments"
    / "20261105_qwen21_albedo"
    / "training_albedo_qwen21.yaml"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _components():
    """The built network components (the registry also holds their loader classes)."""
    return [
        (name, module)
        for name, module in REGISTRY["network_components"].items()
        if isinstance(module, torch.nn.Module)
    ]


def _load_trainables(checkpoint: Path) -> None:
    state = load_file(str(checkpoint / "trainables.safetensors"), device="cpu")
    for name, module in _components():
        prefix = f"{name}."
        part = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if not part:
            continue
        result = module.load_state_dict(part, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"{name}: unexpected keys {result.unexpected_keys[:5]}")
        print(f"[load] {name}: {len(part)} tensors from {checkpoint}")


def _size(width: int, height: int, long_side: int) -> tuple[int, int]:
    scale = long_side / max(width, height)
    return (
        max(32, round(width * scale / 32) * 32),
        max(32, round(height * scale / 32) * 32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--checkpoint",
        help="Folder with trainables.safetensors (none: the base model).",
    )
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument(
        "--model_dir",
        help="Qwen-Image 2.1 folder to run on (default: the config's, Turbo).",
    )
    parser.add_argument("--long_side", type=int, default=1024)
    args = parser.parse_args()

    cfg = recursive_load_config(args.config)
    # The config keeps its paths in paths.<use_paths_from>: an override of a top-level
    # paths key is dropped here, so the model folder is printed and settable directly.
    if "use_paths_from" in cfg.paths:
        cfg.paths = cfg.paths[cfg.paths.use_paths_from]
    if args.model_dir:
        cfg.paths.ckpt_qwen_image21 = args.model_dir
    print(f"[model] {cfg.paths.ckpt_qwen_image21}")
    REGISTRY["cfg"] = cfg
    register_experiment_modules(cfg)
    build_transforms(cfg.network_components, registry_category="network_components")({})
    if args.checkpoint:
        _load_trainables(Path(args.checkpoint))
    for _, module in _components():
        module.eval().requires_grad_(False)
    graph = build_transforms(cfg.network_graph, registry_category="network_graph")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        p
        for p in Path(args.image_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    for path in paths:
        image = Image.open(path)
        alpha = image.getchannel("A") if image.mode in ("RGBA", "LA") else None
        rgb = Image.new("RGB", image.size)
        rgb.paste(image.convert("RGB"), mask=alpha)
        size = _size(*image.size, args.long_side)
        x = np.asarray(rgb.resize(size, Image.LANCZOS), dtype=np.float32) / 127.5 - 1
        batch = {"rgb_norm": torch.from_numpy(x).permute(2, 0, 1)[None], "out": {}}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            graph(batch)
        albedo = batch["out"]["albedo_pred"][0].float().clamp(0, 1).pow(1 / 2.2)
        pixels = (albedo.permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        result = Image.fromarray(pixels).resize(image.size, Image.LANCZOS)
        if alpha is not None:
            result.putalpha(alpha)
        result.save(out_dir / f"{path.stem}.png")
        print(f"{path.name} -> {out_dir / path.stem}.png ({size[0]}x{size[1]})")


if __name__ == "__main__":
    main()
