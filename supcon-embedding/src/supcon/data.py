"""Data loader for SupCon experiment using comprehensiveSft CSV files.

Data sources:
  DFP: DFP0395_comprehensiveSft.csv — treatment column = "drug1"
  CNS: CNS0091_comprehensiveSft.csv — treatment from sourceMetadata join

Feature columns: all shared numeric columns between DFP & CNS CSVs,
excluding known metadata columns. NaN filled with 0.

Optional LDA preprocessing (Paul's pipeline for DFP):
  966 features -> z-score -> SVD-LDA -> n_classes-1 dims
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


def discover_feature_columns(
    dfp_csv: str,
    cns_csv: str,
) -> list[str]:
    """Return sorted list of numeric feature columns shared by both CSVs."""
    dfp_head = pd.read_csv(dfp_csv, nrows=0)
    cns_head = pd.read_csv(cns_csv, nrows=0)
    shared = sorted(set(dfp_head.columns) & set(cns_head.columns))

    # Read a small sample to check dtypes
    dfp_sample = pd.read_csv(dfp_csv, nrows=50)
    feature_cols = [
        c for c in shared
        if c not in METADATA_COLS
        and dfp_sample[c].dtype in ("float64", "float32", "int64", "int32")
    ]
    return sorted(feature_cols)


def load_csv_dataset(
    csv_path: str,
    perturbation_col: str,
    feature_columns: list[str],
    metadata_csv: str | None = None,
    metadata_join_keys: list[str] | None = None,
    metadata_rename: dict[str, str] | None = None,
    min_samples_per_class: int = 2,
    use_lda: bool = False,
    lda_model: LinearDiscriminantAnalysis | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str], LinearDiscriminantAnalysis | None]:
    """Load a comprehensiveSft CSV into (features, labels, label_names).

    Args:
        csv_path: Path to comprehensiveSft CSV.
        perturbation_col: Column name for treatment label.
        feature_columns: Which columns to use as features.
        metadata_csv: Optional path to sourceMetadata CSV for treatment join.
        metadata_join_keys: Columns to join on.
        metadata_rename: Rename dict applied to metadata before join.
        min_samples_per_class: Drop classes with fewer samples.
        use_lda: If True, apply SVD-LDA dimensionality reduction.
        lda_model: Pre-fitted LDA model to transform with (skip fitting).
            If None and use_lda=True, fits a new LDA on this data.

    Returns:
        features: [N, F] float32 tensor (F = n_classes-1 if LDA)
        labels:   [N] int64 tensor
        label_names: sorted list mapping int -> treatment string
        lda_model: fitted LDA (or None if use_lda=False)
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # ── Join with sourceMetadata if treatment column is missing ──
    if perturbation_col not in df.columns and metadata_csv is not None:
        meta = pd.read_csv(metadata_csv)
        if metadata_rename:
            meta = meta.rename(columns=metadata_rename)
        join_keys = metadata_join_keys or ["SourceNumber", "WellDescription"]
        meta_cols = join_keys + [perturbation_col]
        meta_subset = meta[meta_cols].drop_duplicates(subset=join_keys)
        df = df.merge(meta_subset, on=join_keys, how="left")
        n_missing = df[perturbation_col].isna().sum()
        if n_missing > 0:
            print(f"  WARNING: {n_missing} rows have no treatment after join")

    # ── Drop rows without treatment label ──
    df = df.dropna(subset=[perturbation_col])
    treatments = df[perturbation_col].astype(str)

    # ── Filter rare classes ──
    counts = Counter(treatments)
    valid = {t for t, c in counts.items() if c >= min_samples_per_class}
    mask = treatments.isin(valid)
    df = df[mask]
    treatments = treatments[mask]

    # ── Extract features ──
    present_cols = [c for c in feature_columns if c in df.columns]
    features_df = df[present_cols].astype(np.float32)
    features_df = features_df.fillna(0.0)

    # ── Per-feature z-score normalisation ──
    mean = features_df.mean()
    std = features_df.std().replace(0.0, 1.0)
    features_df = (features_df - mean) / std

    # ── Encode labels ──
    label_names = sorted(treatments.unique())
    label_map = {name: idx for idx, name in enumerate(label_names)}
    labels = treatments.map(label_map).values

    features_np = features_df.values.astype(np.float32)

    # ── Optional SVD-LDA ──
    fitted_lda = None
    if use_lda:
        if lda_model is not None:
            # Transform with pre-fitted model
            features_np = lda_model.transform(features_np).astype(np.float32)
            print(f"  LDA transform -> {features_np.shape[1]} dims (pre-fitted)")
        else:
            # Fit new LDA on this data
            lda = LinearDiscriminantAnalysis(solver="svd")
            features_np = lda.fit_transform(features_np, labels).astype(np.float32)
            fitted_lda = lda
            print(f"  LDA fit+transform -> {features_np.shape[1]} dims "
                  f"(explained variance ratio sum: "
                  f"{lda.explained_variance_ratio_.sum():.3f})")

    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    print(f"  {len(features_tensor)} samples, {len(label_names)} classes, "
          f"{features_tensor.shape[1]} features")
    return features_tensor, labels_tensor, label_names, fitted_lda


# ── Convenience wrappers per dataset ────────────────────────────────────

def load_dfp(
    csv_path: str = "C:/data/dataset/DFP/DFP0395_comprehensiveSft.csv",
    feature_columns: list[str] | None = None,
    min_samples_per_class: int = 2,
    use_lda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str], LinearDiscriminantAnalysis | None]:
    """Load DFP dataset. Treatment = drug1 column."""
    if feature_columns is None:
        feature_columns = discover_feature_columns(
            csv_path,
            "C:/data/dataset/CNS/CNS0091_comprehensiveSft.csv",
        )
    return load_csv_dataset(
        csv_path=csv_path,
        perturbation_col="drug1",
        feature_columns=feature_columns,
        min_samples_per_class=min_samples_per_class,
        use_lda=use_lda,
    )


def load_dfp_with_dose(
    csv_path: str = "C:/data/dataset/DFP/DFP0395_comprehensiveSft.csv",
    feature_columns: list[str] | None = None,
    min_samples_per_class: int = 2,
    use_lda: bool = False,
) -> dict:
    """Load DFP with compound + dose labels for ordinal loss.

    Returns dict with keys:
        features, labels, label_names, lda_model,
        compound_labels (int), compound_names (list),
        dose_labels (int rank 0..N), dose_values (float conc)
    """
    if feature_columns is None:
        feature_columns = discover_feature_columns(
            csv_path,
            "C:/data/dataset/CNS/CNS0091_comprehensiveSft.csv",
        )

    # Load raw CSV to get compound + dose columns alongside features
    print(f"Loading {csv_path} (with dose info) ...")
    df = pd.read_csv(csv_path)

    # drug1 column has format "COMPOUND:BATCH"
    df["_compound"] = df["drug1"].str.split(":").str[0]

    # Filter + extract features via load_csv_dataset
    features, labels, label_names, lda_model = load_csv_dataset(
        csv_path=csv_path,
        perturbation_col="drug1",
        feature_columns=feature_columns,
        min_samples_per_class=min_samples_per_class,
        use_lda=use_lda,
    )

    # Reload to get the filtered rows aligned with features
    # (load_csv_dataset drops NaN + rare classes, so re-filter df the same way)
    treatments = df["drug1"].astype(str)
    counts = Counter(treatments)
    valid = {t for t, c in counts.items() if c >= min_samples_per_class}
    df_filtered = df[treatments.isin(valid) & treatments.notna()].reset_index(drop=True)

    # Compound labels
    compounds = df_filtered["_compound"]
    compound_names = sorted(compounds.unique())
    compound_map = {name: idx for idx, name in enumerate(compound_names)}
    compound_labels = torch.tensor(
        compounds.map(compound_map).values, dtype=torch.int64
    )

    # Dose rank (per compound, rank concentrations 0..N)
    dose_values = df_filtered["drug1Concentration"].values.astype(np.float32)
    dose_ranks = np.zeros(len(df_filtered), dtype=np.int64)
    for comp in compound_names:
        comp_mask = compounds.values == comp
        concs = np.sort(np.unique(dose_values[comp_mask]))
        conc_to_rank = {c: i for i, c in enumerate(concs)}
        for i in np.where(comp_mask)[0]:
            dose_ranks[i] = conc_to_rank[dose_values[i]]

    print(f"  {len(compound_names)} compounds, dose ranks 0..{dose_ranks.max()}")

    return {
        "features": features,
        "labels": labels,
        "label_names": label_names,
        "lda_model": lda_model,
        "compound_labels": compound_labels,
        "compound_names": compound_names,
        "dose_labels": torch.tensor(dose_ranks, dtype=torch.int64),
        "dose_values": torch.tensor(dose_values, dtype=torch.float32),
    }


def load_cns(
    csv_path: str = "C:/data/dataset/CNS/CNS0091_comprehensiveSft.csv",
    metadata_csv: str = "C:/data/dataset/CNS/CNS0091_sourceMetadata.csv",
    feature_columns: list[str] | None = None,
    min_samples_per_class: int = 2,
    use_lda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str], LinearDiscriminantAnalysis | None]:
    """Load CNS dataset. Treatment from sourceMetadata join."""
    if feature_columns is None:
        feature_columns = discover_feature_columns(
            "C:/data/dataset/DFP/DFP0395_comprehensiveSft.csv",
            csv_path,
        )
    return load_csv_dataset(
        csv_path=csv_path,
        perturbation_col="treatment",
        feature_columns=feature_columns,
        metadata_csv=metadata_csv,
        metadata_join_keys=["SourceNumber", "WellDescription"],
        metadata_rename={"WellDescription_x": "WellDescription"},
        min_samples_per_class=min_samples_per_class,
        use_lda=use_lda,
    )


def load_joint(
    dfp_csv: str = "C:/data/dataset/DFP/DFP0395_comprehensiveSft.csv",
    cns_csv: str = "C:/data/dataset/CNS/CNS0091_comprehensiveSft.csv",
    cns_metadata_csv: str = "C:/data/dataset/CNS/CNS0091_sourceMetadata.csv",
    feature_columns: list[str] | None = None,
    min_samples_per_class: int = 2,
    use_lda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load both DFP + CNS into a single dataset with shared label space."""
    if feature_columns is None:
        feature_columns = discover_feature_columns(dfp_csv, cns_csv)

    feat_dfp, lab_dfp, names_dfp, _ = load_csv_dataset(
        csv_path=dfp_csv,
        perturbation_col="drug1",
        feature_columns=feature_columns,
        min_samples_per_class=min_samples_per_class,
        use_lda=use_lda,
    )
    feat_cns, lab_cns, names_cns, _ = load_csv_dataset(
        csv_path=cns_csv,
        perturbation_col="treatment",
        feature_columns=feature_columns,
        metadata_csv=cns_metadata_csv,
        metadata_join_keys=["SourceNumber", "WellDescription"],
        metadata_rename={"WellDescription_x": "WellDescription"},
        min_samples_per_class=min_samples_per_class,
        use_lda=use_lda,
    )

    # For joint, both must have same feature dim
    if feat_dfp.shape[1] != feat_cns.shape[1]:
        raise ValueError(
            f"Feature dim mismatch: DFP={feat_dfp.shape[1]}, CNS={feat_cns.shape[1]}. "
            f"With LDA each dataset gets n_classes-1 dims. "
            f"Use use_lda=False for joint, or pre-align dimensions."
        )

    # Merge label spaces — offset CNS labels
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
    """In-memory dataset supporting optional extra label tensors."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor,
                 extra: dict[str, torch.Tensor] | None = None):
        self.features = features
        self.labels = labels
        self.extra = extra or {}

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        item = {"features": self.features[idx], "labels": self.labels[idx]}
        for k, v in self.extra.items():
            item[k] = v[idx]
        return item


def make_dataloader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 256,
    shuffle: bool = True,
    extra: dict[str, torch.Tensor] | None = None,
) -> torch.utils.data.DataLoader:
    """Wrap tensors in a DataLoader."""
    ds = InMemoryDataset(features, labels, extra)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=shuffle,
        drop_last=False,
    )
