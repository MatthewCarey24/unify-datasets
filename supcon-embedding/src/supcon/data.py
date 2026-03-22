"""Data loader for SupCon experiment using comprehensiveSft CSV features.

Data sources:
  DFP: *_comprehensiveSft.csv  -- treatment = "drug1" column
  CNS: *_comprehensiveSft.csv  -- treatment from sourceMetadata join
       *_sourceMetadata.csv    -- has "treatment" + join keys

Features: shared numeric columns between DFP & CNS CSVs.
NaN filled with 0, then z-score normalised.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# ── Metadata columns to exclude from features ──────────────────────────
METADATA_COLS = {
    "AnalysisID", "CellId", "CellType", "DishId", "ExperimentID", "FOVNumber",
    "Genotype", "ImagingStart", "Operator", "PlateId", "PlateNumber", "PlateType",
    "Project", "Scope", "SourceID", "SourceNumber", "WellDescription",
    "cellPlateBarcode", "column", "compoundPlateBarcode", "compressionMethod",
    "compressionStatus", "datetime", "div", "expDescription", "experimentFailed",
    "fov", "numericGroupId", "outOfFocus", "repeatNumber", "row", "schmutz",
    "autofluorescence", "cellNumber", "BaselineFluorescence", "protocolIdx",
    "focusGLVA", "focusGLVN", "focusHELM", "focusLLSP", "focusSATU",
    "focusTENG", "focusTENV", "experimentId", "wellId",
    "fovId", "fovPosition", "fullProtocolName", "gliaDensity", "gliaLot",
    "imagingBuffer", "imagingBufferLot", "operatorName", "plateMapName", "round",
    "sampleTime_units", "scopeName", "sourceName", "synapticBlockers",
    "virus1", "virus1Volume", "virus2", "virus2Volume", "sampleTime", "sourceNumber",
    # DFP-only metadata
    "drug1", "drug1Concentration", "drug1ConcentrationUnits", "drug1Concentration_Units",
    "platingDensity",
    # CNS-only metadata
    "cas9Treatment", "cas9Volume", "cellDensity", "libraryPlate",
    "screeningWellContents", "sgRNA", "sgRNAVolume",
    # Injected treatment column
    "treatment",
}


def _find_csv(dataset_dir: str, pattern: str) -> str:
    """Find a CSV matching a glob pattern."""
    p = Path(dataset_dir)
    matches = list(p.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} in {dataset_dir}")
    return str(matches[0])


def discover_feature_columns(dfp_csv: str, cns_csv: str) -> list[str]:
    """Return sorted list of numeric feature columns shared by both CSVs."""
    dfp_head = pd.read_csv(dfp_csv, nrows=50)
    cns_head = pd.read_csv(cns_csv, nrows=50)
    shared = set(dfp_head.columns) & set(cns_head.columns)
    feature_cols = sorted([
        c for c in shared
        if c not in METADATA_COLS
        and dfp_head[c].dtype in ("float64", "float32", "int64", "int32")
    ])
    return feature_cols


def load_sft_dataset(
    csv_path: str,
    perturbation_col: str,
    feature_columns: list[str],
    metadata_csv: str | None = None,
    metadata_join_keys: list[str] | None = None,
    metadata_rename: dict[str, str] | None = None,
    min_samples_per_class: int = 2,
    use_lda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str], LinearDiscriminantAnalysis | None]:
    """Load a comprehensiveSft CSV into (features, labels, label_names, lda).

    Args:
        csv_path: Path to comprehensiveSft CSV.
        perturbation_col: Column name for treatment label.
        feature_columns: Which columns to use as features.
        metadata_csv: Optional sourceMetadata CSV for treatment join.
        metadata_join_keys: Columns to join on.
        metadata_rename: Rename dict applied to metadata before join.
        min_samples_per_class: Drop classes with fewer samples.
        use_lda: If True, apply SVD-LDA after z-score.

    Returns:
        features, labels, label_names, lda_model
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # ── Join with sourceMetadata if treatment column is missing ──
    if perturbation_col not in df.columns and metadata_csv is not None:
        meta = pd.read_csv(metadata_csv)
        if metadata_rename:
            meta = meta.rename(columns=metadata_rename)
        join_keys = metadata_join_keys or ["SourceNumber", "WellDescription"]
        meta_subset = meta[join_keys + [perturbation_col]].drop_duplicates(subset=join_keys)
        df = df.merge(meta_subset, on=join_keys, how="left")
        n_missing = df[perturbation_col].isna().sum()
        if n_missing > 0:
            print(f"  WARNING: {n_missing} rows missing treatment after join")

    # ── Drop rows without treatment ──
    df = df.dropna(subset=[perturbation_col])
    treatments = df[perturbation_col].astype(str)

    # ── Filter rare classes ──
    counts = Counter(treatments)
    valid = {t for t, c in counts.items() if c >= min_samples_per_class}
    mask = treatments.isin(valid)
    df = df[mask].reset_index(drop=True)
    treatments = treatments[mask].reset_index(drop=True)

    # ── Extract features ──
    present_cols = [c for c in feature_columns if c in df.columns]
    features_np = df[present_cols].astype(np.float32).fillna(0.0).values

    # ── Z-score ──
    mean = features_np.mean(axis=0)
    std = features_np.std(axis=0)
    std[std == 0] = 1.0
    features_np = (features_np - mean) / std

    # ── Encode labels ──
    label_names = sorted(treatments.unique())
    label_map = {name: idx for idx, name in enumerate(label_names)}
    labels = treatments.map(label_map).values.astype(np.int64)

    # ── Optional LDA ──
    fitted_lda = None
    if use_lda:
        lda = LinearDiscriminantAnalysis(solver="svd")
        features_np = lda.fit_transform(features_np, labels).astype(np.float32)
        fitted_lda = lda
        lda_mean = features_np.mean(axis=0)
        lda_std = features_np.std(axis=0)
        lda_std[lda_std == 0] = 1.0
        features_np = (features_np - lda_mean) / lda_std
        print(f"  LDA -> {features_np.shape[1]} dims "
              f"(variance ratio sum: {lda.explained_variance_ratio_.sum():.3f})")

    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    print(f"  {len(features_tensor)} samples, {len(label_names)} classes, "
          f"{features_tensor.shape[1]} features")
    return features_tensor, labels_tensor, label_names, fitted_lda


def load_dfp(
    dataset_dir: str,
    feature_columns: list[str],
    min_samples_per_class: int = 2,
    use_lda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str], LinearDiscriminantAnalysis | None]:
    """Load DFP dataset. Treatment = drug1 column."""
    csv_path = _find_csv(dataset_dir, "*_comprehensiveSft.csv")
    return load_sft_dataset(
        csv_path=csv_path,
        perturbation_col="drug1",
        feature_columns=feature_columns,
        min_samples_per_class=min_samples_per_class,
        use_lda=use_lda,
    )


def load_cns(
    dataset_dir: str,
    feature_columns: list[str],
    min_samples_per_class: int = 2,
    use_lda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str], LinearDiscriminantAnalysis | None]:
    """Load CNS dataset. Treatment from sourceMetadata join."""
    csv_path = _find_csv(dataset_dir, "*_comprehensiveSft.csv")
    meta_path = _find_csv(dataset_dir, "*_sourceMetadata.csv")
    return load_sft_dataset(
        csv_path=csv_path,
        perturbation_col="treatment",
        feature_columns=feature_columns,
        metadata_csv=meta_path,
        metadata_join_keys=["SourceNumber", "WellDescription"],
        metadata_rename={"WellDescription_x": "WellDescription"},
        min_samples_per_class=min_samples_per_class,
        use_lda=use_lda,
    )


def load_joint(
    dfp_dir: str,
    cns_dir: str,
    feature_columns: list[str],
    min_samples_per_class: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load both DFP + CNS into a single dataset with shared label space."""
    feat_dfp, lab_dfp, names_dfp, _ = load_dfp(
        dfp_dir, feature_columns, min_samples_per_class,
    )
    feat_cns, lab_cns, names_cns, _ = load_cns(
        cns_dir, feature_columns, min_samples_per_class,
    )

    offset = len(names_dfp)
    lab_cns_offset = lab_cns + offset
    all_names = names_dfp + names_cns

    features = torch.cat([feat_dfp, feat_cns], dim=0)
    labels = torch.cat([lab_dfp, lab_cns_offset], dim=0)

    print(f"Joint: {len(features)} samples, {len(all_names)} classes "
          f"({len(names_dfp)} DFP + {len(names_cns)} CNS)")
    return features, labels, all_names


# ── DataLoader helpers ──────────────────────────────────────────────────

class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def make_dataloader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 1024,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    ds = InMemoryDataset(features, labels)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=shuffle,
        drop_last=True,
    )
