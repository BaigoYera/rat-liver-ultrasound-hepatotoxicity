#!/usr/bin/env python3
"""Generate enhanced tables and infographics for the MUA liver injury paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models, transforms

try:
    from scipy import stats
except Exception:
    stats = None


LABELS = ["LOW", "MILD", "SEVERE"]
LABEL_ORDER = {"LOW": 0, "MILD": 1, "SEVERE": 2}
BIOMARKER_GROUPS = pd.DataFrame(
    [
        {"Group": "Intact", "Injury_order": -1, "ALT_U_L": 187.87, "AST_U_L": 42.63, "MDA_umol_L": 1.83, "De_Ritis": 0.23, "Liver_index_pct": 3.00, "n": 3},
        {"Group": "LOW", "Injury_order": 0, "ALT_U_L": 109.67, "AST_U_L": 49.76, "MDA_umol_L": 6.97, "De_Ritis": 0.45, "Liver_index_pct": 5.42, "n": 3},
        {"Group": "MILD", "Injury_order": 1, "ALT_U_L": 96.92, "AST_U_L": 60.65, "MDA_umol_L": 7.20, "De_Ritis": 0.63, "Liver_index_pct": 3.97, "n": 4},
        {"Group": "SEVERE", "Injury_order": 2, "ALT_U_L": 229.37, "AST_U_L": 53.81, "MDA_umol_L": 3.40, "De_Ritis": 0.23, "Liver_index_pct": np.nan, "n": 4},
    ]
)


def markdown_table(df: pd.DataFrame) -> str:
    clean = df.copy()
    clean.columns = [str(col) for col in clean.columns]
    rows = [clean.columns.tolist()] + clean.fillna("").astype(str).values.tolist()
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"

    return "\n".join([fmt(rows[0]), "| " + " | ".join("-" * width for width in widths) + " |", *[fmt(row) for row in rows[1:]]])


def save_table(df: pd.DataFrame, tables_dir: Path, name: str) -> None:
    df.to_csv(tables_dir / f"{name}.csv", index=False)
    (tables_dir / f"{name}.md").write_text(markdown_table(df) + "\n", encoding="utf-8")


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.0f}%", ha="center", va="center", color="white" if norm[i, j] > 0.55 else "black")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized proportion")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_per_class_metrics(report: dict, labels: list[str], out_path: Path) -> None:
    rows = []
    for label in labels:
        rows.append({"Class": label, "Precision": report[label]["precision"], "Recall": report[label]["recall"], "F1-score": report[label]["f1-score"]})
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for idx, metric in enumerate(["Precision", "Recall", "F1-score"]):
        ax.bar(x + (idx - 1) * width, df[metric], width=width, label=metric)
    ax.set_xticks(x, df["Class"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Rat LOOCV Per-Class Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_workflow(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    boxes = [
        ("CCl4 rat model", 0.04),
        ("Ultrasound\nBiochemistry\nHistology", 0.23),
        ("Byra external\npretraining", 0.45),
        ("Rat LOOCV\nseverity model", 0.65),
        ("Grad-CAM +\ngroup correlations", 0.84),
    ]
    for text, x in boxes:
        rect = plt.Rectangle((x, 0.35), 0.14, 0.32, fill=True, facecolor="#E8F1F8", edgecolor="#1F4E79", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.07, 0.51, text, ha="center", va="center", fontsize=10)
    for _, x in boxes[:-1]:
        ax.annotate("", xy=(x + 0.18, 0.51), xytext=(x + 0.14, 0.51), arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    ax.set_title("Pilot Multimodal Workflow for CCl4-Induced Liver Injury Severity", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def make_collage(manifest: pd.DataFrame, out_path: Path) -> None:
    selections = []
    for label in LABELS:
        row = manifest[manifest["label"] == label].iloc[0]
        selections.append((label, Path(row["path"])))
    byra_path = Path("data_external/byra_steatosis/processed/manifest.csv")
    if byra_path.exists():
        byra = pd.read_csv(byra_path)
        ext = byra[byra["binary_label"] == "STEATOSIS"].iloc[0]
        selections.append(("External steatosis", Path(ext["path"])))

    tile_w, tile_h = 300, 245
    canvas = Image.new("RGB", (tile_w * len(selections), tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, path) in enumerate(selections):
        img = Image.open(path).convert("RGB")
        img.thumbnail((tile_w, tile_h - 38))
        x = idx * tile_w + (tile_w - img.width) // 2
        y = 28 + (tile_h - 38 - img.height) // 2
        canvas.paste(img, (x, y))
        draw.text((idx * tile_w + 12, 8), f"{chr(65 + idx)}. {label}", fill="black")
    canvas.save(out_path.with_suffix(".png"))


def plot_biomarker_profile(tables_dir: Path, figures_dir: Path) -> None:
    summary = BIOMARKER_GROUPS.copy()
    summary["p_value"] = "NA: animal-level raw values unavailable"
    save_table(summary.round(3), tables_dir, "table_biochemical_markers_group_summary")

    plot_df = summary[summary["Group"].isin(LABELS)].copy()
    metrics = ["ALT_U_L", "AST_U_L", "MDA_umol_L", "De_Ritis"]
    scaled = plot_df[["Group", *metrics]].copy()
    for metric in metrics:
        max_value = scaled[metric].max()
        scaled[metric] = scaled[metric] / max_value if max_value else scaled[metric]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(plot_df))
    for metric in metrics:
        ax.plot(x, scaled[metric], marker="o", label=metric)
    ax.set_xticks(x, plot_df["Group"])
    ax.set_ylabel("Scaled value within metric")
    ax.set_title("Group-Level Biochemical Injury Profile")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig((figures_dir / "figure_biomarker_injury_profile").with_suffix(".png"), dpi=300)
    fig.savefig((figures_dir / "figure_biomarker_injury_profile").with_suffix(".svg"))
    plt.close(fig)


def correlation_outputs(predictions: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    severity = predictions.groupby("true_label", as_index=False)["severity_score"].mean().rename(columns={"true_label": "Group", "severity_score": "Model_severity_score"})
    merged = BIOMARKER_GROUPS.merge(severity, on="Group", how="inner")
    variables = ["Injury_order", "ALT_U_L", "AST_U_L", "MDA_umol_L", "De_Ritis", "Liver_index_pct", "Model_severity_score"]
    corr = merged[variables].corr(method="spearman", min_periods=2)
    corr.to_csv(tables_dir / "table_group_level_spearman_correlation.csv")
    (tables_dir / "table_group_level_spearman_correlation.md").write_text(markdown_table(corr.reset_index().rename(columns={"index": "Variable"}).round(3)) + "\n", encoding="utf-8")

    rows = []
    for metric in ["ALT_U_L", "AST_U_L", "MDA_umol_L", "De_Ritis", "Liver_index_pct"]:
        sub = merged[[metric, "Model_severity_score"]].dropna()
        if stats is not None and len(sub) >= 3:
            rho, p_value = stats.spearmanr(sub[metric], sub["Model_severity_score"])
        else:
            rho, p_value = np.nan, np.nan
        rows.append({"Comparison": f"Model severity score vs {metric}", "Spearman_rho": rho, "p_value": p_value, "n_groups": len(sub)})
    save_table(pd.DataFrame(rows).round(3), tables_dir, "table_model_biomarker_group_correlations")

    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.imshow(corr.fillna(0).values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iloc[i, j]
            ax.text(j, i, "" if pd.isna(val) else f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Exploratory Group-Level Spearman Correlation")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig((figures_dir / "figure_group_level_correlation_heatmap").with_suffix(".png"), dpi=300)
    fig.savefig((figures_dir / "figure_group_level_correlation_heatmap").with_suffix(".svg"))
    plt.close(fig)


def build_model_from_checkpoint(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    labels = checkpoint["labels"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, labels


def gradcam_for_image(model, labels: list[str], image_path: Path, out_path: Path) -> None:
    target_layer = model.layer4[-1]
    activations = None
    gradients = None

    def forward_hook(_, __, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(_, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    original = Image.open(image_path).convert("RGB").resize((224, 224))
    x = transform(original).unsqueeze(0)
    logits = model(x)
    pred_idx = int(torch.argmax(logits, dim=1).item())
    model.zero_grad()
    logits[0, pred_idx].backward()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1)).squeeze().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(original.size)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(original)
    ax.imshow(cam_img, cmap="jet", alpha=0.42)
    ax.set_title(f"Grad-CAM: {labels[pred_idx]}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    h1.remove()
    h2.remove()


def make_gradcam_outputs(manifest: pd.DataFrame, figures_dir: Path) -> None:
    checkpoint_path = Path("outputs/rat_loocv/resnet18_all_rat_images.pt")
    if not checkpoint_path.exists():
        return
    model, labels = build_model_from_checkpoint(checkpoint_path)
    selected = []
    for label in LABELS:
        selected.append(manifest[manifest["label"] == label].iloc[0])
    for row in selected:
        name = f"figure_gradcam_{row['label'].lower()}_{Path(row['path']).stem}"
        gradcam_for_image(model, labels, Path(row["path"]), figures_dir / name)


def model_tables(tables_dir: Path) -> dict:
    rows = []
    class_rows = []
    runs = {
        "Byra binary pretraining": Path("outputs/byra_pretrain/metrics.json"),
        "Byra severity pretraining": Path("outputs/byra_pretrain_severity/metrics.json"),
        "Rat LOOCV injury severity": Path("outputs/rat_loocv/metrics.json"),
    }
    loaded = {}
    for name, path in runs.items():
        if not path.exists():
            continue
        metrics = load_metrics(path)
        loaded[name] = metrics
        report = metrics["classification_report"]
        rows.append(
            {
                "Run": name,
                "Validation": metrics.get("validation", "held-out split"),
                "Accuracy": round(metrics.get("test_acc", 0), 3),
                "Accuracy_95CI": "" if "accuracy_95ci" not in metrics else f"{metrics['accuracy_95ci'][0]:.3f}-{metrics['accuracy_95ci'][1]:.3f}",
                "Macro_precision": round(report["macro avg"]["precision"], 3),
                "Macro_recall": round(report["macro avg"]["recall"], 3),
                "Macro_F1": round(report["macro avg"]["f1-score"], 3),
                "Weighted_F1": round(report["weighted avg"]["f1-score"], 3),
                "Support": int(report["weighted avg"]["support"]),
            }
        )
        for label in metrics["labels"]:
            if label not in report:
                continue
            row = report[label]
            class_rows.append(
                {
                    "Run": name,
                    "Class": label,
                    "Precision": round(row["precision"], 3),
                    "Recall/Sensitivity": round(row["recall"], 3),
                    "F1-score": round(row["f1-score"], 3),
                    "Support": int(row["support"]),
                    "Specificity": round(metrics.get("specificity_by_class", {}).get(label, np.nan), 3) if "specificity_by_class" in metrics else "",
                }
            )
    save_table(pd.DataFrame(rows), tables_dir, "table_model_performance_summary_enhanced")
    save_table(pd.DataFrame(class_rows), tables_dir, "table_rat_loocv_and_pretraining_class_metrics")
    return loaded


def error_analysis(predictions: pd.DataFrame, tables_dir: Path) -> None:
    rows = []
    for _, row in predictions.iterrows():
        if row["true_label"] == row["pred_label"]:
            continue
        distance = abs(LABEL_ORDER[row["true_label"]] - LABEL_ORDER[row["pred_label"]])
        rows.append(
            {
                "Image": row["path"],
                "True_class": row["true_label"],
                "Predicted_class": row["pred_label"],
                "Confidence": round(row["confidence"], 3),
                "Severity_distance": int(distance),
                "Likely_reason": "Adjacent severity overlap / small local training set" if distance == 1 else "Non-adjacent class confusion; inspect image quality and label",
            }
        )
    save_table(pd.DataFrame(rows), tables_dir, "table_rat_loocv_error_analysis")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="paper_outputs_enhanced")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv("manifests/local_rat_ultrasound_manifest.csv")
    byra = pd.read_csv("data_external/byra_steatosis/processed/manifest.csv")
    predictions_path = Path("outputs/rat_loocv/loocv_predictions.csv")
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()

    save_table(manifest.groupby("label").agg(Subjects=("subject_id", "nunique"), Images=("path", "count")).reset_index().rename(columns={"label": "Class"}), tables_dir, "table_local_rat_ultrasound_distribution")
    save_table(byra.groupby("binary_label").agg(Subjects=("patient_id", "nunique"), Images=("path", "count"), Fat_min_pct=("fat_percent", "min"), Fat_max_pct=("fat_percent", "max")).reset_index().rename(columns={"binary_label": "Class"}), tables_dir, "table_byra_binary_distribution")
    save_table(byra.groupby("severity_label").agg(Subjects=("patient_id", "nunique"), Images=("path", "count"), Fat_min_pct=("fat_percent", "min"), Fat_max_pct=("fat_percent", "max")).reset_index().rename(columns={"severity_label": "Class"}), tables_dir, "table_byra_severity_distribution")

    loaded = model_tables(tables_dir)
    if "Rat LOOCV injury severity" in loaded:
        rat_metrics = loaded["Rat LOOCV injury severity"]
        plot_confusion(np.array(rat_metrics["confusion_matrix"]), rat_metrics["labels"], "Rat Injury Severity LOOCV Confusion Matrix", figures_dir / "figure_rat_loocv_confusion_matrix")
        plot_per_class_metrics(rat_metrics["classification_report"], rat_metrics["labels"], figures_dir / "figure_rat_loocv_per_class_metrics")
        pd.DataFrame(rat_metrics["confusion_matrix"], index=rat_metrics["labels"], columns=rat_metrics["labels"]).to_csv(tables_dir / "table_rat_loocv_confusion_matrix.csv")

    plot_workflow(figures_dir / "figure_multimodal_workflow")
    make_collage(manifest, figures_dir / "figure_ultrasound_collage")
    plot_biomarker_profile(tables_dir, figures_dir)
    if not predictions.empty:
        save_table(predictions.round(4), tables_dir, "table_rat_loocv_per_image_predictions")
        error_analysis(predictions, tables_dir)
        correlation_outputs(predictions, tables_dir, figures_dir)
        make_gradcam_outputs(manifest, figures_dir)

    readme = """# Enhanced Paper Outputs

Primary figures:
- `figure_ultrasound_collage.png`
- `figure_multimodal_workflow.png`
- `figure_rat_loocv_confusion_matrix.png`
- `figure_rat_loocv_per_class_metrics.png`
- `figure_biomarker_injury_profile.png`
- `figure_group_level_correlation_heatmap.png`
- `figure_gradcam_*.png`

Primary tables:
- `table_biochemical_markers_group_summary.md`
- `table_model_performance_summary_enhanced.md`
- `table_rat_loocv_and_pretraining_class_metrics.md`
- `table_rat_loocv_per_image_predictions.md`
- `table_rat_loocv_error_analysis.md`
- `table_model_biomarker_group_correlations.md`

Notes:
- Correlations are exploratory group-level analyses because animal-level ultrasound-to-biomarker mapping is unavailable.
- Biochemical p-values are marked unavailable where raw animal-level values are not present.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote enhanced paper outputs to {out_dir}")


if __name__ == "__main__":
    main()
