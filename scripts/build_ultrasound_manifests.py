#!/usr/bin/env python3
"""Create leakage-aware manifests for local rat images and external pretraining.

Use the external Byra manifest for pretraining or baseline benchmarking.
Use the rat manifest for final validation. Do not mix human and rat images in
one random image-level split when making paper claims.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def infer_rat_label(path: Path) -> str | None:
    stem = path.stem.upper()
    if "CHECKPOINT" in stem or ".IPYNB_CHECKPOINTS" in {p.upper() for p in path.parts}:
        return None
    if stem.startswith("LOW_"):
        return "LOW"
    if stem.startswith("MILD_"):
        return "MILD"
    if stem.startswith("SEVERE_"):
        return "SEVERE"
    if stem.startswith("US_") and "CONTROL_3_21D" in stem:
        return "SEVERE"
    return None


def infer_rat_subject(path: Path) -> str:
    stem = path.stem.upper()
    match = re.match(r"(LOW|MILD|SEVERE)_(\d+)", stem)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    if stem.startswith("US_"):
        return stem
    return stem


def build_rat_manifest(data_dir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = infer_rat_label(path)
        if label is None:
            continue
        rows.append(
            {
                "path": str(path),
                "dataset": "local_rat_ccl4",
                "source_domain": "rat_ultrasound",
                "subject_id": infer_rat_subject(path),
                "label": label,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rat-data", default="data")
    parser.add_argument("--out-dir", default="manifests")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rat_rows = build_rat_manifest(Path(args.rat_data))
    if not rat_rows:
        raise SystemExit("No local rat ultrasound images found.")

    write_csv(out_dir / "local_rat_ultrasound_manifest.csv", rat_rows)
    counts = {}
    for row in rat_rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"Wrote {len(rat_rows)} rat images to {out_dir / 'local_rat_ultrasound_manifest.csv'}")
    print(counts)


if __name__ == "__main__":
    main()
