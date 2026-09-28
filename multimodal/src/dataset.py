"""Dataset utilities for rat ultrasound plus rat-level biochemical features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABELS = ["LOW", "MILD", "SEVERE"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}

REQUIRED_BIOCHEM_COLUMNS = [
    "subject_id",
    "group",
    "injury_label",
    "day",
    "alt_u_l",
    "ast_u_l",
    "mda_umol_l",
]

DEFAULT_BIOCHEM_FEATURES = [
    "alt_u_l",
    "ast_u_l",
    "mda_umol_l",
    "de_ritis",
]

OPTIONAL_BIOCHEM_FEATURES = [
    "liver_index_pct",
    "histology_score",
    "steatosis_score",
    "necrosis_score",
    "inflammation_score",
]


@dataclass(frozen=True)
class FeatureStats:
    means: dict[str, float]
    stds: dict[str, float]


def default_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def _resolve_paths(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    out = frame.copy()
    out["path"] = out["path"].map(lambda value: str((root / value).resolve()) if not Path(str(value)).is_absolute() else str(value))
    return out


def load_ultrasound_manifest(manifest_path: str | Path, project_root: str | Path = ".") -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    root = Path(project_root)
    frame = pd.read_csv(manifest_path)
    missing = {"path", "subject_id", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Ultrasound manifest is missing columns: {sorted(missing)}")
    frame = _resolve_paths(frame, root)
    frame = frame[frame["label"].isin(LABELS)].copy()
    frame = frame[frame["path"].map(lambda value: Path(value).exists())].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No usable ultrasound rows found after label/path filtering.")
    return frame


def load_biochemistry_table(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(REQUIRED_BIOCHEM_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Biochemistry table is missing columns: {sorted(missing)}")
    if "de_ritis" not in frame.columns:
        alt = pd.to_numeric(frame["alt_u_l"], errors="coerce")
        ast = pd.to_numeric(frame["ast_u_l"], errors="coerce")
        frame["de_ritis"] = ast / alt
    return frame


def join_ultrasound_biochemistry(
    ultrasound: pd.DataFrame,
    biochemistry: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    features = features or DEFAULT_BIOCHEM_FEATURES
    missing_features = set(features) - set(biochemistry.columns)
    if missing_features:
        raise ValueError(f"Biochemistry table is missing feature columns: {sorted(missing_features)}")

    bio_cols = ["subject_id", "group", "injury_label", *features]
    optional_cols = [
        "day",
        *OPTIONAL_BIOCHEM_FEATURES,
        "notes",
        "source_file",
        "source_sheet",
        "source_row",
    ]
    bio_cols.extend([col for col in optional_cols if col in biochemistry.columns])

    joined = ultrasound.merge(
        biochemistry[bio_cols],
        on="subject_id",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_biochem"),
    )
    if joined.empty:
        raise ValueError("Join produced zero rows. Check subject_id values in both inputs.")

    label_mismatch = joined["label"].astype(str) != joined["injury_label"].astype(str)
    if label_mismatch.any():
        mismatches = joined.loc[label_mismatch, ["subject_id", "label", "injury_label"]].to_dict("records")
        raise ValueError(f"Ultrasound and biochemistry labels disagree: {mismatches}")

    for feature in features:
        joined[feature] = pd.to_numeric(joined[feature], errors="coerce")
    joined = joined.dropna(subset=features).reset_index(drop=True)
    if joined.empty:
        raise ValueError("Joined rows exist, but all rows have missing biochemical feature values.")
    return joined


def fit_feature_stats(frame: pd.DataFrame, features: list[str]) -> FeatureStats:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for feature in features:
        series = pd.to_numeric(frame[feature], errors="coerce")
        means[feature] = float(series.mean())
        std = float(series.std(ddof=0))
        stds[feature] = std if std > 1e-8 else 1.0
    return FeatureStats(means=means, stds=stds)


class MultimodalRatDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        features: list[str] | None = None,
        feature_stats: FeatureStats | None = None,
        image_transform=None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.features = features or DEFAULT_BIOCHEM_FEATURES
        self.feature_stats = feature_stats or fit_feature_stats(self.frame, self.features)
        self.image_transform = image_transform or default_image_transform()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        image_tensor = self.image_transform(image)
        values = [
            (float(row[feature]) - self.feature_stats.means[feature]) / self.feature_stats.stds[feature]
            for feature in self.features
        ]
        biochem_tensor = torch.tensor(values, dtype=torch.float32)
        target = torch.tensor(LABEL_TO_IDX[str(row["label"])], dtype=torch.long)
        return {
            "image": image_tensor,
            "biochemistry": biochem_tensor,
            "target": target,
            "subject_id": row["subject_id"],
        }
