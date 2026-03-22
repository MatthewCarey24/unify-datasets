"""Data loading for DFP0395 drug embedding.

Loads comprehensiveSft CSV, extracts top-825 numeric features by NaN rate,
z-scores, encodes compound labels (drug1 before `:` separator).
Optional LDA dimensionality reduction.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Metadata columns to exclude from features
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
    "drug1", "drug1Concentration", "drug1ConcentrationUnits", "drug1Concentration_Units",
    "platingDensity", "treatment",
    "cas9Treatment", "cas9Volume", "cellDensity", "libraryPlate",
    "screeningWellContents", "sgRNA", "sgRNAVolume",
    "CellId_sourceFeatureTable", "CellPlateBarcode", "CompoundPlateBarcode",
    "Fov", "Datetime", "Snr", "DFOverF", "Skew", "Area", "SomaRadius",
    "SomaArea", "Brightness", "Snowiness", "Score",
    "SomaMajorAxisLength", "SomaMinorAxisLength", "SomaOrientation",
    "SomaEccentricity", "SomaExtent", "NSourceMaskRegions",
    "TotalNumberOfSpikes", "somaCenterX", "somaCenterY",
}


def _extract_compound(drug1_val: str) -> str:
    """Extract compound ID from drug1 column. Strips batch after ':'."""
    s = str(drug1_val).strip()
    if ":" in s:
        return s.split(":")[0]
    return s


def load_dfp0395(
    csv_path: str,
    n_features: int = 825,
    min_samples_per_class: int = 2,
    use_lda: bool = False,
    exclude_dmso: bool = False,
) -> dict:
    """Load DFP0395 comprehensiveSft CSV.

    Returns dict with keys:
        features: Tensor [N, D]
        labels: Tensor [N] (compound integer labels)
        label_names: list[str] (compound names, sorted)
        doses: Tensor [N] (raw concentration floats)
        dose_ranks: Tensor [N] (integer dose rank per compound, 0=lowest)
        lda_model: fitted LDA or None
        feature_columns: list[str] of selected feature column names
        mean, std: numpy arrays for z-score (for inference)
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Raw: {len(df)} rows, {len(df.columns)} columns")

    # Extract compound labels
    df["compound"] = df["drug1"].apply(_extract_compound)
    doses = df["drug1Concentration"].astype(float).values

    # Optionally exclude DMSO
    if exclude_dmso:
        mask = df["compound"] != "DMSO"
        df = df[mask].reset_index(drop=True)
        doses = doses[mask.values]

    # Identify numeric feature columns (exclude metadata)
    numeric_cols = [
        c for c in df.columns
        if c not in METADATA_COLS
        and c != "compound"
        and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]
    print(f"  {len(numeric_cols)} numeric feature columns found")

    # Select top n_features by lowest NaN rate
    nan_rates = df[numeric_cols].isna().mean()
    sorted_cols = nan_rates.sort_values().index.tolist()
    feature_cols = sorted_cols[:n_features]
    print(f"  Selected top {len(feature_cols)} features by NaN rate "
          f"(worst NaN rate: {nan_rates[feature_cols[-1]]:.3f})")

    # Extract features, fill NaN with 0
    features_np = df[feature_cols].astype(np.float32).fillna(0.0).values

    # Z-score normalize
    mean = features_np.mean(axis=0)
    std = features_np.std(axis=0)
    std[std == 0] = 1.0
    features_np = (features_np - mean) / std

    # Encode compound labels
    compounds = df["compound"].values
    counts = Counter(compounds)
    valid = {c for c, n in counts.items() if n >= min_samples_per_class}
    mask = np.array([c in valid for c in compounds])
    features_np = features_np[mask]
    compounds = compounds[mask]
    doses = doses[mask]

    label_names = sorted(set(compounds))
    label_map = {name: idx for idx, name in enumerate(label_names)}
    labels = np.array([label_map[c] for c in compounds], dtype=np.int64)

    # Compute dose ranks per compound (0 = lowest dose)
    dose_ranks = np.zeros(len(labels), dtype=np.int64)
    for comp_name in label_names:
        comp_mask = compounds == comp_name
        unique_doses = sorted(set(doses[comp_mask]))
        dose_map = {d: r for r, d in enumerate(unique_doses)}
        for i in np.where(comp_mask)[0]:
            dose_ranks[i] = dose_map[doses[i]]

    print(f"  {len(features_np)} samples, {len(label_names)} compounds")
    for name in label_names:
        n = (compounds == name).sum()
        print(f"    {name}: {n} samples")

    # Optional LDA
    lda_model = None
    if use_lda:
        n_classes = len(label_names)
        lda = LinearDiscriminantAnalysis(solver="svd")
        features_np = lda.fit_transform(features_np, labels).astype(np.float32)
        lda_model = lda
        # Re-normalize LDA output
        lda_mean = features_np.mean(axis=0)
        lda_std = features_np.std(axis=0)
        lda_std[lda_std == 0] = 1.0
        features_np = (features_np - lda_mean) / lda_std
        print(f"  LDA -> {features_np.shape[1]} dims "
              f"(explained variance ratio sum: {lda.explained_variance_ratio_.sum():.3f})")

    return {
        "features": torch.tensor(features_np, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "label_names": label_names,
        "doses": torch.tensor(doses, dtype=torch.float32),
        "dose_ranks": torch.tensor(dose_ranks, dtype=torch.int64),
        "lda_model": lda_model,
        "feature_columns": feature_cols,
        "mean": mean,
        "std": std,
    }
