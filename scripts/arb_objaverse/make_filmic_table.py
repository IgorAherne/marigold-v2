#!/usr/bin/env python3
"""Write the 8-bit inverse of Blender 3.3's Filmic sRGB view, one channel at a time.

Arb-Objaverse rendered its colour and albedo PNGs through BlenderProc 2.5 (Blender
3.3) with the default Filmic view, so white albedo is stored as code 206, not 255.
For values up to scene-linear 1 the view is separable per channel (the desaturation
cube only acts above that), so a 256-entry table undoes it exactly. Needs
``pip install opencolorio`` and Blender 3.3's ``datafiles/colormanagement`` folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import PyOpenColorIO as ocio

OUT = (
    Path(__file__).resolve().parents[2]
    / "marigoldv2"
    / "experiments"
    / "20261105_qwen21_albedo"
    / "blender33_filmic_inverse.txt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("colormanagement", help="Blender 3.3 colormanagement folder")
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()

    config = ocio.Config.CreateFromFile(str(Path(args.colormanagement) / "config.ocio"))
    cpu = config.getProcessor("Linear", "Filmic sRGB").getDefaultCPUProcessor()
    linear = np.concatenate([[0.0], np.geomspace(1e-7, 64.0, 200001)]).astype(
        np.float32
    )
    rgb = np.ascontiguousarray(np.repeat(linear[:, None], 3, axis=1))
    cpu.applyRGB(rgb)
    display = rgb[:, 0].astype(np.float64)
    if np.any(np.diff(display) < 0):
        raise RuntimeError("Filmic curve is not monotonic on the grid")
    codes = np.arange(256) / 255.0
    table = np.interp(codes, display, linear.astype(np.float64))
    lines = [f"{v:.8g}" for v in table]
    Path(args.output).write_text(
        "# Blender 3.3 Filmic sRGB view, 8-bit code -> scene-linear (one per line)\n"
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"code 206 -> {table[206]:.4f}, 255 -> {table[255]:.3f}; -> {args.output}")


if __name__ == "__main__":
    main()
