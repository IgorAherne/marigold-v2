#!/usr/bin/env python3
"""Download a slice of Arb-Objaverse and build the albedo training set from it.

Per group (``data/000-XXX.zip`` of ``lizb6626/Arb-Objaverse``, 9-12 GB, 390-540
objects): keep only the objects whose Objaverse licence is CC-BY or CC0 (from
``allenai/objaverse`` metadata), extract their lit pictures and albedo (not the
normals or materials; about 65 % of the zip) and delete the zip. ``credits.csv``
lists every kept object with its author and page, which CC-BY asks to credit. Then
split by object into train/val/vis manifests of ``<color png> <albedo png>`` lines.
Finished groups are skipped on rerun.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import time
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path(os.environ.get("DEPTH_ASSETS_DIR", REPO_ROOT / "assets")).expanduser()
LICENCES = {"by", "cc0"}
DOWNLOAD_ATTEMPTS = 3
COLOR = re.compile(r"^color_(\d+)_(\d+)\.png$")
ALBEDO = re.compile(r"^albedo_(\d+)\.png$")


def _metadata(group: str, work: Path) -> dict:
    path = hf_hub_download(
        "allenai/objaverse",
        f"metadata/{group}.json.gz",
        repo_type="dataset",
        local_dir=str(work),
    )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _download_zip(group: str, work: Path) -> Path:
    """One group's zip into ``work``. huggingface_hub 2.x never resumes a partial
    file, so a killed attempt's leftover is deleted and the next one starts over."""
    partials = work / ".cache" / "huggingface" / "download"
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        if partials.is_dir():
            for stale in partials.rglob("*.incomplete"):
                stale.unlink()
        try:
            return Path(
                hf_hub_download(
                    "lizb6626/Arb-Objaverse",
                    f"data/{group}.zip",
                    repo_type="dataset",
                    local_dir=str(work),
                )
            )
        except Exception as exc:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            print(f"[retry] {group}.zip after: {exc}", flush=True)
            time.sleep(2**attempt)


def extract_group(zip_path: Path, meta: dict, out_dir: Path) -> list[dict]:
    """Extract the allowed objects' colour/albedo PNGs; return their credit rows."""
    kept = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            parts = info.filename.split("/")
            if len(parts) != 3 or not parts[2]:
                continue
            group, uid, name = parts
            m = meta.get(uid)
            if (
                m is None
                or m.get("license") not in LICENCES
                or m.get("isAgeRestricted")
            ):
                continue
            if not (COLOR.match(name) or ALBEDO.match(name)):
                continue
            target = out_dir / group / uid / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            if uid not in kept:
                user = m.get("user") or {}
                kept[uid] = {
                    "group": group,
                    "uid": uid,
                    "licence": m.get("license"),
                    "name": m.get("name", ""),
                    "author": user.get("username", ""),
                    "url": m.get("viewerUrl", ""),
                }
    return list(kept.values())


def _pairs(object_dir: Path, base: Path, dropped: list[str]) -> list[str]:
    names = {p.name for p in object_dir.iterdir()}
    out = []
    for name in sorted(names):
        m = COLOR.match(name)
        albedo = f"albedo_{m.group(2)}.png" if m else None
        if albedo not in names:
            continue
        # The model takes sides in multiples of 32 and the pair must line up; an
        # odd render would stop the training hours in, so it stays out.
        sizes = {Image.open(object_dir / f).size for f in (name, albedo)}
        if len(sizes) != 1 or any(side % 32 for side in sizes.pop()):
            dropped.append((object_dir / name).relative_to(base).as_posix())
            continue
        color = (object_dir / name).relative_to(base).as_posix()
        out.append(f"{color} {(object_dir / albedo).relative_to(base).as_posix()}")
    return out


CREDITS = ["group", "uid", "licence", "name", "author", "url"]


def write_manifests(out_dir: Path, val_objects: int, vis_objects: int) -> None:
    rows = []
    for part in sorted((out_dir / ".work").glob("credits-*.csv")):
        with part.open(newline="", encoding="utf-8") as handle:
            rows += list(csv.DictReader(handle))
    with (out_dir / "credits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, CREDITS)
        writer.writeheader()
        writer.writerows(rows)
    objects = sorted(p for p in out_dir.glob("000-*/*") if p.is_dir())
    rank = sorted(objects, key=lambda p: hashlib.sha1(p.name.encode()).hexdigest())
    vis, val = (
        set(rank[:vis_objects]),
        set(rank[vis_objects : vis_objects + val_objects]),
    )
    split = {"train": [], "val": [], "vis": []}
    dropped = []
    for obj in objects:
        pairs = _pairs(obj, out_dir, dropped)
        if obj in vis:
            split["vis"] += pairs[:1]
        elif obj in val:
            # The first and last pairs: one under the first HDRI, one under the
            # point lights, which leave the most of an object in the dark.
            split["val"] += pairs[:1] + pairs[-1:]
        else:
            split["train"] += pairs
    for name, lines in split.items():
        (out_dir / f"arb_objaverse_{name}.txt").write_text("\n".join(lines) + "\n")
        print(f"[manifest] {name}: {len(lines)} pairs")
    if dropped:
        (out_dir / "dropped_pairs.txt").write_text("\n".join(dropped) + "\n")
        print(
            f"[manifest] {len(dropped)} pairs of odd size left out: dropped_pairs.txt"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=15, help="Zips 000-000 onward.")
    parser.add_argument(
        "--output_dir", default=str(ASSETS_DIR / "datasets" / "arb_objaverse")
    )
    parser.add_argument(
        "--zip_dir", help="Use zips already here instead of downloading them."
    )
    parser.add_argument("--val_objects", type=int, default=100)
    parser.add_argument("--vis_objects", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    work = out_dir / ".work"
    work.mkdir(parents=True, exist_ok=True)
    for index in range(args.groups):
        group = f"000-{index:03d}"
        done = work / f"{group}.done"
        if done.exists():
            continue
        meta = _metadata(group, work)
        if args.zip_dir:
            zip_path = Path(args.zip_dir) / f"{group}.zip"
        else:
            print(f"[download] {group}.zip", flush=True)
            zip_path = _download_zip(group, work)
        rows = extract_group(zip_path, meta, out_dir)
        with (work / f"credits-{group}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, CREDITS)
            writer.writeheader()
            writer.writerows(rows)
        # The marker goes first: a crash between the two must not cost a re-download.
        done.touch()
        if not args.zip_dir:
            zip_path.unlink()
        print(f"[group] {group}: {len(rows)} objects kept", flush=True)
    write_manifests(out_dir, args.val_objects, args.vis_objects)


if __name__ == "__main__":
    main()
