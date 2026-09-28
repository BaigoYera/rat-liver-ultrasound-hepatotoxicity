#!/usr/bin/env python3
"""Dry-run the multimodal data join and late-fusion forward pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    DEFAULT_BIOCHEM_FEATURES,
    LABELS,
    MultimodalRatDataset,
    fit_feature_stats,
    join_ultrasound_biochemistry,
    load_biochemistry_table,
    load_ultrasound_manifest,
)
from model import build_late_fusion_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/local_rat_ultrasound_manifest.csv")
    parser.add_argument("--biochemistry", default="multimodal_baseline/data/rat_biochemistry_from_docx.csv")
    parser.add_argument("--output-dir", default="multimodal_baseline/outputs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    root = Path.cwd()
    ultrasound = load_ultrasound_manifest(args.manifest, project_root=root)
    biochemistry = load_biochemistry_table(args.biochemistry)
    joined = join_ultrasound_biochemistry(ultrasound, biochemistry, DEFAULT_BIOCHEM_FEATURES)
    feature_stats = fit_feature_stats(joined, DEFAULT_BIOCHEM_FEATURES)
    dataset = MultimodalRatDataset(joined, DEFAULT_BIOCHEM_FEATURES, feature_stats=feature_stats)
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=False, num_workers=0)

    batch = next(iter(loader))
    model = build_late_fusion_model(
        biochem_input_dim=len(DEFAULT_BIOCHEM_FEATURES),
        num_classes=len(LABELS),
        imagenet=False,
    )
    model.train()
    logits = model(batch["image"], batch["biochemistry"])
    loss = nn.CrossEntropyLoss()(logits, batch["target"])
    loss.backward()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "status": "ok",
        "validation_type": "smoke_test_only_not_model_validation",
        "ultrasound_manifest": args.manifest,
        "biochemistry_table": args.biochemistry,
        "n_ultrasound_rows": int(len(ultrasound)),
        "n_biochemistry_rows": int(len(biochemistry)),
        "n_joined_rows": int(len(joined)),
        "labels": LABELS,
        "biochemistry_features": DEFAULT_BIOCHEM_FEATURES,
        "batch_size": int(batch["image"].shape[0]),
        "image_tensor_shape": list(batch["image"].shape),
        "biochemistry_tensor_shape": list(batch["biochemistry"].shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "subject_ids_in_first_batch": list(batch["subject_id"]),
        "feature_means": feature_stats.means,
        "feature_stds": feature_stats.stds,
        "scientific_note": "Smoke test only. If using rat_biochemistry_from_docx.csv, rows are extracted from the added rat-level DOCX and still require mapping review before strong scientific claims.",
    }
    (output_dir / "smoke_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joined.to_csv(output_dir / "joined_smoke_manifest.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
