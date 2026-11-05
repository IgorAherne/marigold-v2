"""Arb-Objaverse (IDArb, ICLR 2025): single objects under seven lightings each, with
their albedo, as RGB/albedo pairs composited over black like Spellcast3D's views.

Both PNGs went through Blender 3.3's Filmic view (white albedo is code 206), which
``blender33_filmic_inverse.txt`` undoes. The albedo becomes linear in [0, 1] like
Hypersim's; the lit picture is shown either as rendered (Filmic) or re-encoded as
plain sRGB with clipped highlights, so the model sees both tone curves.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from marigoldv2.core.registry import register

_FILMIC_TO_LINEAR = np.loadtxt(
    Path(__file__).with_name("blender33_filmic_inverse.txt"), dtype=np.float64
)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


_FILMIC_TO_SRGB = np.round(
    _linear_to_srgb(np.clip(_FILMIC_TO_LINEAR, 0.0, 1.0)) * 255.0
).astype(np.uint8)


@register("manifest_transform")
class LoadArbObjaverseManifest:
    """Read ``<manifest>`` lines ``<color png> <albedo png>`` relative to ``base_dir``."""

    def __init__(self, base_dir, manifest, verify_files=False):
        self.base_dir = Path(base_dir)
        self.manifest = self.base_dir / manifest
        self.verify_files = bool(verify_files)

    def __call__(self, manifest):
        if not self.manifest.is_file():
            raise FileNotFoundError(f"Arb-Objaverse manifest missing: {self.manifest}")
        lines = self.manifest.read_text(encoding="utf-8").split("\n")
        for line in filter(None, (s.strip() for s in lines)):
            color, albedo = line.split()
            rgb_path, albedo_path = self.base_dir / color, self.base_dir / albedo
            if self.verify_files and not (rgb_path.is_file() and albedo_path.is_file()):
                raise FileNotFoundError(f"{self.manifest}: missing files for {line}")
            key = f"arb_{color.replace('/', '_')}"
            if key in manifest:
                raise ValueError(f"Duplicate manifest key: {key}")
            manifest[key] = {"rgb_path": str(rgb_path), "albedo_path": str(albedo_path)}
        if not manifest:
            raise ValueError(f"No samples in {self.manifest}")
        return manifest


@register("dataset_transform")
class ReadArbObjaversePair:
    """Lit RGBA picture over black -> ``rgb_norm``/``rgb_int``; Filmic albedo PNG ->
    linear ``albedo_gt``; fully covered pixels the light reached -> ``valid_mask``.

    ``srgb_p`` is the chance the picture is re-encoded as plain sRGB (0 keeps it as
    rendered, for validation). Pixels whose brightest channel is under ``min_code``
    (as rendered) say nothing about their colour, so the loss skips them: the
    point-light renders leave much of an object in the dark.
    """

    def __init__(
        self, srgb_p=0.5, min_code=8, output_key="albedo_gt", mask_key="valid_mask"
    ):
        self.srgb_p = float(srgb_p)
        self.min_code = int(min_code)
        self.output_key = str(output_key)
        self.mask_key = str(mask_key)

    def __call__(self, sample):
        annotation = sample["annotation"]
        rgba = np.asarray(Image.open(annotation["rgb_path"]).convert("RGBA"))
        codes, alpha = rgba[..., :3], rgba[..., 3:].astype(np.float32) / 255.0
        lit = codes.max(axis=-1) >= self.min_code
        if self.srgb_p > 0 and random.random() < self.srgb_p:
            codes = _FILMIC_TO_SRGB[codes]
        rgb = np.round(codes.astype(np.float32) * alpha).astype(np.uint8)
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        sample["rgb_int"] = rgb.int()
        sample["rgb_norm"] = rgb.float() / 255.0 * 2.0 - 1.0
        sample["orig_rgb"] = sample["rgb_int"]
        sample.setdefault("meta", {})["orig_res"] = list(rgb.shape[-2:])

        albedo = np.asarray(Image.open(annotation["albedo_path"]).convert("RGB"))
        linear = np.clip(_FILMIC_TO_LINEAR[albedo], 0.0, 1.0).astype(np.float32)
        valid = torch.from_numpy((rgba[..., 3] >= 250) & lit)[None]
        albedo = torch.from_numpy(linear.transpose(2, 0, 1).copy())
        sample[self.output_key] = torch.where(valid, albedo, torch.zeros_like(albedo))
        sample[self.mask_key] = valid
        return sample
