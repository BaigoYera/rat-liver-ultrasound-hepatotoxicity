#!/usr/bin/env python3
"""Run leave-one-out cross-validation for local rat injury severity images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


LABELS = ["LOW", "MILD", "SEVERE"]


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, label_to_idx: dict[str, int], transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        target = torch.tensor(self.label_to_idx[row["label"]], dtype=torch.long)
        return image if self.transform is None else self.transform(image), target


def build_train_transform() -> transforms.Compose:
    steps = [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ]
    if hasattr(transforms, "ElasticTransform"):
        steps.append(transforms.ElasticTransform(alpha=30.0, sigma=5.0))
    steps.extend(
        [
            transforms.ColorJitter(brightness=0.12, contrast=0.18),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + 0.015 * torch.randn_like(x)),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(steps)


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def build_model(num_classes: int, init_checkpoint: str | None, imagenet: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if imagenet else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        compatible = {
            key: value
            for key, value in state.items()
            if key in model.state_dict() and model.state_dict()[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
    return model


def train_model(
    train_df: pd.DataFrame,
    label_to_idx: dict[str, int],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    init_checkpoint: str | None,
    imagenet: bool,
) -> nn.Module:
    model = build_model(len(label_to_idx), init_checkpoint, imagenet).to(device)
    loader = DataLoader(
        ManifestDataset(train_df, label_to_idx, build_train_transform()),
        batch_size=min(batch_size, max(1, len(train_df))),
        shuffle=True,
        num_workers=0,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_one(model: nn.Module, row: pd.Series, label_to_idx: dict[str, int], device: torch.device) -> dict:
    dataset = ManifestDataset(pd.DataFrame([row]), label_to_idx, build_eval_transform())
    image, target = dataset[0]
    logits = model.eval()(image.unsqueeze(0).to(device))
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    labels = [label for label, _ in sorted(label_to_idx.items(), key=lambda item: item[1])]
    return {
        "path": row["path"],
        "subject_id": row["subject_id"],
        "true_label": row["label"],
        "pred_label": labels[pred_idx],
        "true_idx": int(target.item()),
        "pred_idx": pred_idx,
        "confidence": float(probs[pred_idx]),
        "severity_score": float(np.dot(probs, np.arange(len(labels)))),
        **{f"prob_{label}": float(probs[idx]) for idx, label in enumerate(labels)},
    }


def specificity_by_class(y_true: list[int], y_pred: list[int], labels: list[str]) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    values = {}
    total = cm.sum()
    for idx, label in enumerate(labels):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        values[label] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return values


def bootstrap_ci(values: np.ndarray, reducer, seed: int, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return 0.0, 0.0
    boots = []
    for _ in range(n_boot):
        sample = values[rng.integers(0, len(values), len(values))]
        boots.append(reducer(sample))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/local_rat_ultrasound_manifest.csv")
    parser.add_argument("--init-checkpoint", default="outputs/byra_pretrain/resnet18_manifest.pt")
    parser.add_argument("--output-dir", default="outputs/rat_loocv")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imagenet", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.manifest)
    frame = frame[frame["label"].isin(LABELS)].reset_index(drop=True)
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = []
    fold_rows = []
    for fold_idx, test_idx in enumerate(range(len(frame)), start=1):
        train_df = frame.drop(index=test_idx).reset_index(drop=True)
        test_row = frame.iloc[test_idx]
        model = train_model(
            train_df,
            label_to_idx,
            args.epochs,
            args.batch_size,
            args.lr,
            device,
            args.init_checkpoint if Path(args.init_checkpoint).exists() else None,
            args.imagenet,
        )
        pred = predict_one(model, test_row, label_to_idx, device)
        pred["fold"] = fold_idx
        pred["correct"] = int(pred["true_label"] == pred["pred_label"])
        predictions.append(pred)
        fold_rows.append({"fold": fold_idx, "held_out": test_row["path"], "correct": pred["correct"]})
        print(f"Fold {fold_idx:02d}/{len(frame)} | {pred['true_label']} -> {pred['pred_label']} | conf={pred['confidence']:.3f}")

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_dir / "loocv_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)

    y_true = pred_df["true_idx"].astype(int).tolist()
    y_pred = pred_df["pred_idx"].astype(int).tolist()
    correct = pred_df["correct"].to_numpy(dtype=float)
    accuracy = float(correct.mean())
    acc_low, acc_high = bootstrap_ci(correct, lambda arr: float(arr.mean()), args.seed)
    macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="weighted", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0))
    specificity = specificity_by_class(y_true, y_pred, LABELS)
    macro_specificity = float(np.mean(list(specificity.values())))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "validation": "leave-one-out cross-validation",
        "n_images": int(len(frame)),
        "labels": LABELS,
        "test_acc": accuracy,
        "accuracy_95ci": [acc_low, acc_high],
        "macro_precision": macro_precision,
        "macro_recall_sensitivity": macro_recall,
        "macro_specificity": macro_specificity,
        "specificity_by_class": specificity,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    final_model = train_model(
        frame,
        label_to_idx,
        args.epochs,
        args.batch_size,
        args.lr,
        device,
        args.init_checkpoint if Path(args.init_checkpoint).exists() else None,
        args.imagenet,
    )
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "labels": LABELS,
            "manifest": args.manifest,
            "label_to_idx": label_to_idx,
        },
        output_dir / "resnet18_all_rat_images.pt",
    )
    print(f"LOOCV accuracy: {accuracy:.3f} (95% CI {acc_low:.3f}-{acc_high:.3f})")
    print(f"Wrote LOOCV outputs to {output_dir}")


if __name__ == "__main__":
    main()
