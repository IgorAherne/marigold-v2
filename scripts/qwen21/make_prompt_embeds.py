#!/usr/bin/env python3
"""Precompute the text-only prompt embedding the Qwen-Image 2.1 albedo model reads.

Writes ``<prefix>_prompt_embeds.pt`` ([1, L, 4096], bf16) and ``<prefix>_prompt_mask.pt``
([1, L]) the way the Qwen-Image 2.1 pipeline encodes a prompt, so the training never
loads the 8B text encoder.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path(os.environ.get("DEPTH_ASSETS_DIR", REPO_ROOT / "assets")).expanduser()
PROMPT = (
    "Show the albedo of this picture: the flat base colour of every surface, "
    "with all lighting removed, no shading, shadows, highlights or reflections."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(ASSETS_DIR / "checkpoints" / "Qwen-Image-2.1-Turbo"),
        help="Qwen-Image 2.1 diffusers folder (text_encoder/ and processor/ are read).",
    )
    parser.add_argument(
        "--output_dir",
        default=str(ASSETS_DIR / "checkpoints" / "qwen21_text_embeddings"),
    )
    parser.add_argument("--prefix", default="qwen21_albedo")
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = QwenImage21Pipeline.from_pretrained(
        args.checkpoint,
        transformer=None,
        vae=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.text_encoder.to(device)
    with torch.no_grad():
        embeds, mask, image_pad_mask = pipe.encode_prompt(
            prompt=args.prompt, device=device
        )
    if image_pad_mask.any():
        raise RuntimeError("A text-only prompt must hold no image slots")
    if mask is None:
        mask = torch.ones(embeds.shape[:2], dtype=torch.long)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(embeds.to(torch.bfloat16).cpu(), out / f"{args.prefix}_prompt_embeds.pt")
    torch.save(mask.to(torch.long).cpu(), out / f"{args.prefix}_prompt_mask.pt")
    print(f"{tuple(embeds.shape)} prompt embedding -> {out}/{args.prefix}_prompt_*.pt")


if __name__ == "__main__":
    main()
