#!/usr/bin/env python3
"""Train/evaluate the rat-level late-fusion multimodal baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import (
    DEFAULT_BIOCHEM_FEATURES,
    LABELS,
    LABEL_TO_IDX,
    MultimodalRatDataset,
    fit_feature_stats,
    join_ultrasound_biochemistry,
    load_biochemistry_table,
    load_ultrasound_manifest,
)
from model import LateFusionInjuryClassifier


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.10, contrast=0.12),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def load_image_checkpoint(model: LateFusionInjuryClassifier, checkpoint_path: str | None) -> int:
    if not checkpoint_path:
        return 0
    path = Path(checkpoint_path)
    if not path.exists():
        return 0
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    backbone_state = model.image_encoder.backbone.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in backbone_state and backbone_state[key].shape == value.shape
    }
    model.image_encoder.backbone.load_state_dict(compatible, strict=False)
    return len(compatible)


def load_biochemistry_checkpoint(model: LateFusionInjuryClassifier, checkpoint_path: str | None) -> int:
    if not checkpoint_path:
        return 0
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Biochemistry checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("biochemistry_encoder_state_dict", checkpoint)
    target_state = model.biochemistry_encoder.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    model.biochemistry_encoder.load_state_dict(compatible, strict=False)
    return len(compatible)


def make_model(args: argparse.Namespace, n_features: int) -> LateFusionInjuryClassifier:
    model = LateFusionInjuryClassifier(
        biochem_input_dim=n_features,
        num_classes=len(LABELS),
        imagenet=False,
        freeze_image_backbone=not args.fine_tune_image_backbone,
    )
    loaded = load_image_checkpoint(model, args.init_checkpoint)
    if loaded:
        print(f"Loaded {loaded} compatible image-backbone tensors from {args.init_checkpoint}")
    elif not args.fine_tune_image_backbone:
        print("Warning: image backbone is frozen but no compatible checkpoint was loaded.")
    biochem_loaded = load_biochemistry_checkpoint(model, args.biochem_checkpoint)
    if biochem_loaded:
        print(f"Loaded {biochem_loaded} biochemical MLP tensors from {args.biochem_checkpoint}")
    return model


def run_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        biochemistry = batch["biochemistry"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(images, biochemistry), targets)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * images.shape[0]
        n += int(images.shape[0])
    return total / max(1, n)


@torch.no_grad()
def predict_rows(model, loader, device) -> list[dict]:
    model.eval()
    rows = []
    for batch in loader:
        logits = model(batch["image"].to(device), batch["biochemistry"].to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        targets = batch["target"].cpu().numpy()
        for idx, subject_id in enumerate(batch["subject_id"]):
            pred_idx = int(np.argmax(probs[idx]))
            true_idx = int(targets[idx])
            rows.append(
                {
                    "subject_id": subject_id,
                    "true_label": LABELS[true_idx],
                    "pred_label": LABELS[pred_idx],
                    "true_idx": true_idx,
                    "pred_idx": pred_idx,
                    "confidence": float(probs[idx, pred_idx]),
                    **{f"prob_{label}": float(probs[idx, label_idx]) for label_idx, label in enumerate(LABELS)},
                }
            )
    return rows


def joined_frame(args: argparse.Namespace) -> pd.DataFrame:
    ultrasound = load_ultrasound_manifest(args.manifest, project_root=Path.cwd())
    biochemistry = load_biochemistry_table(args.biochemistry)
    return join_ultrasound_biochemistry(ultrasound, biochemistry, args.features)


def train_final_model(args: argparse.Namespace, frame: pd.DataFrame, device: torch.device) -> LateFusionInjuryClassifier:
    stats = fit_feature_stats(frame, args.features)
    dataset = MultimodalRatDataset(frame, args.features, feature_stats=stats, image_transform=train_transform())
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True, num_workers=0)
    model = make_model(args, len(args.features)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, loader, criterion, optimizer, device)
        history.append({"epoch": epoch, "train_loss": loss})
        print(f"Final model epoch {epoch:02d}/{args.epochs} train_loss={loss:.4f}")
    model.training_history = history
    model.feature_stats = stats
    return model


def run_loocv(args: argparse.Namespace, frame: pd.DataFrame, output_dir: Path, device: torch.device) -> dict:
    predictions = []
    for fold, test_idx in enumerate(range(len(frame)), start=1):
        train_df = frame.drop(index=test_idx).reset_index(drop=True)
        test_df = frame.iloc[[test_idx]].reset_index(drop=True)
        stats = fit_feature_stats(train_df, args.features)
        train_ds = MultimodalRatDataset(train_df, args.features, feature_stats=stats, image_transform=train_transform())
        test_ds = MultimodalRatDataset(test_df, args.features, feature_stats=stats, image_transform=eval_transform())
        train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(train_ds)), shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

        model = make_model(args, len(args.features)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        for _ in range(args.epochs):
            run_epoch(model, train_loader, criterion, optimizer, device)
        pred = predict_rows(model, test_loader, device)[0]
        pred["fold"] = fold
        pred["correct"] = int(pred["true_label"] == pred["pred_label"])
        predictions.append(pred)
        print(f"Fold {fold:02d}/{len(frame)} {pred['subject_id']}: {pred['true_label']} -> {pred['pred_label']}")

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_dir / "loocv_predictions.csv", index=False)
    y_true = pred_df["true_idx"].astype(int).tolist()
    y_pred = pred_df["pred_idx"].astype(int).tolist()
    metrics = {
        "validation": "leave-one-rat-out cross-validation on joined rat-level multimodal rows",
        "scientific_note": "Exploratory baseline on a very small mapped dataset; requires independent validation before biological or clinical claims.",
        "n_joined_rows": int(len(frame)),
        "labels": LABELS,
        "features": args.features,
        "accuracy": float((pred_df["correct"] == 1).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/local_rat_ultrasound_manifest.csv")
    parser.add_argument("--biochemistry", default="multimodal_baseline/data/rat_biochemistry_from_docx.csv")
    parser.add_argument("--output-dir", default="multimodal_baseline/outputs/real_multimodal")
    parser.add_argument("--init-checkpoint", default="outputs/byra_pretrain/resnet18_manifest.pt")
    parser.add_argument("--biochem-checkpoint", default=None)
    parser.add_argument("--features", nargs="+", default=DEFAULT_BIOCHEM_FEATURES)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-loocv", action="store_true")
    parser.add_argument("--fine-tune-image-backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = joined_frame(args)
    frame.to_csv(output_dir / "joined_multimodal_manifest.csv", index=False)
    print(f"Joined multimodal rows: {len(frame)}")
    print("Label counts:", frame["label"].value_counts().to_dict())

    metrics = None if args.skip_loocv else run_loocv(args, frame, output_dir, device)
    final_model = train_final_model(args, frame, device)
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "labels": LABELS,
            "features": args.features,
            "feature_stats": {
                "means": final_model.feature_stats.means,
                "stds": final_model.feature_stats.stds,
            },
            "history": final_model.training_history,
            "biochemistry": args.biochemistry,
            "manifest": args.manifest,
        },
        output_dir / "late_fusion_multimodal.pt",
    )
    summary = {
        "status": "ok",
        "output_dir": str(output_dir),
        "n_joined_rows": int(len(frame)),
        "features": args.features,
        "loocv_accuracy": None if metrics is None else metrics["accuracy"],
        "loocv_macro_f1": None if metrics is None else metrics["macro_f1"],
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
