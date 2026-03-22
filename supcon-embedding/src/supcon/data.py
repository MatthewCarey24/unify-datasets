"""Data loader for SupCon experiment using traceMatrix .mat files.

Data sources:
  Each dataset dir contains:
    - *_traceMatrix.mat  (HDF5 v7.3)
        normTraceMatrix: [11459, N] float32 traces
        numericGroupIds: [1, N]     treatment group IDs
    - *_sourceMetadata.csv          treatment name mapping
    - dataset_info.json             epoch boundaries

Features: adaptive-average-pool raw trace from 11459 -> n_bins (default 825).
No NaN, clean signal, identical format for both datasets.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import h5py
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def _find_mat(dataset_dir: str) -> str:
    """Find the *_traceMatrix.mat file in a dataset directory."""
    p = Path(dataset_dir)
    mats = list(p.glob("*_traceMatrix.mat"))
    if not mats:
        raise FileNotFoundError(f"No *_traceMatrix.mat in {dataset_dir}")
    return str(mats[0])


def _find_metadata(dataset_dir: str) -> str:
    """Find the *_sourceMetadata.csv file in a dataset directory."""
    p = Path(dataset_dir)
    csvs = list(p.glob("*_sourceMetadata.csv"))
    if not csvs:
        raise FileNotFoundError(f"No *_sourceMetadata.csv in {dataset_dir}")
    return str(csvs[0])


def load_mat_dataset(
    dataset_dir: str,
    n_bins: int = 825,
    min_samples_per_class: int = 2,
    use_lda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str], LinearDiscriminantAnalysis | None]:
    """Load traces from .mat, pool to n_bins features, return (features, labels, names, lda).

    Args:
        dataset_dir: Directory containing *_traceMatrix.mat and *_sourceMetadata.csv
        n_bins: Number of bins to adaptive-average-pool each trace into
        min_samples_per_class: Drop classes with fewer samples
        use_lda: If True, apply SVD-LDA after pooling + z-score

    Returns:
        features: [N, n_bins] float32 tensor (or [N, n_classes-1] if LDA)
        labels:   [N] int64 tensor
        label_names: sorted list mapping int -> treatment string
        lda_model: fitted LDA or None
    """
    mat_path = _find_mat(dataset_dir)
    meta_path = _find_metadata(dataset_dir)

    print(f"Loading {mat_path} ...")
    with h5py.File(mat_path, "r") as f:
        # normTraceMatrix is [11459, N] — columns are samples
        traces = f["normTraceMatrix"][:].T  # -> [N, 11459]
        group_ids = f["numericGroupIds"][0].astype(int)  # [N]

    print(f"  Raw: {traces.shape[0]} traces x {traces.shape[1]} time points")

    # ── Map numericGroupIds to treatment names via sourceMetadata ──
    meta = pd.read_csv(meta_path)
    # sourceMetadata has NumericGroupID and treatment columns
    id_to_name = {}
    if "NumericGroupID" in meta.columns and "treatment" in meta.columns:
        mapping = meta[["NumericGroupID", "treatment"]].drop_duplicates()
        for _, row in mapping.iterrows():
            gid = int(row["NumericGroupID"])
            name = str(row["treatment"])
            id_to_name[gid] = name
    elif "numericGroupId" in meta.columns and "drug1" in meta.columns:
        mapping = meta[["numericGroupId", "drug1"]].drop_duplicates()
        for _, row in mapping.iterrows():
            gid = int(row["numericGroupId"])
            name = str(row["drug1"])
            id_to_name[gid] = name

    # If no mapping found, try the comprehensiveSft as fallback
    if not id_to_name:
        sft_files = list(Path(dataset_dir).glob("*_comprehensiveSft.csv"))
        if sft_files:
            sft = pd.read_csv(str(sft_files[0]))
            for col in ["drug1", "treatment"]:
                if col in sft.columns and "numericGroupId" in sft.columns:
                    mapping = sft[["numericGroupId", col]].drop_duplicates()
                    for _, row in mapping.iterrows():
                        gid = int(row["numericGroupId"])
                        id_to_name[gid] = str(row[col])
                    break

    if not id_to_name:
        # Last resort: just use numeric IDs as names
        for gid in np.unique(group_ids):
            id_to_name[gid] = f"group_{gid}"

    # ── Map group IDs to treatment strings ──
    treatments = np.array([id_to_name.get(gid, f"group_{gid}") for gid in group_ids])

    # ── Filter rare classes ──
    counts = Counter(treatments)
    valid = {t for t, c in counts.items() if c >= min_samples_per_class}
    mask = np.array([t in valid for t in treatments])
    traces = traces[mask]
    treatments = treatments[mask]

    # ── Adaptive average pool: 11459 -> n_bins ──
    traces_t = torch.tensor(traces, dtype=torch.float32)
    # F.adaptive_avg_pool1d expects [N, C, L] — use C=1
    pooled = F.adaptive_avg_pool1d(traces_t.unsqueeze(1), n_bins).squeeze(1)
    features_np = pooled.numpy()

    # ── Per-feature z-score ──
    mean = features_np.mean(axis=0)
    std = features_np.std(axis=0)
    std[std == 0] = 1.0
    features_np = (features_np - mean) / std

    # ── Encode labels ──
    label_names = sorted(set(treatments))
    label_map = {name: idx for idx, name in enumerate(label_names)}
    labels = np.array([label_map[t] for t in treatments])

    # ── Optional LDA ──
    fitted_lda = None
    if use_lda:
        lda = LinearDiscriminantAnalysis(solver="svd")
        features_np = lda.fit_transform(features_np, labels).astype(np.float32)
        fitted_lda = lda
        print(f"  LDA -> {features_np.shape[1]} dims "
              f"(variance ratio sum: {lda.explained_variance_ratio_.sum():.3f})")

    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    print(f"  {len(features_tensor)} samples, {len(label_names)} classes, "
          f"{features_tensor.shape[1]} features")
    return features_tensor, labels_tensor, label_names, fitted_lda


def load_joint(
    dfp_dir: str = "/data/dataset/DFP",
    cns_dir: str = "/data/dataset/CNS",
    n_bins: int = 825,
    min_samples_per_class: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load both DFP + CNS into a single dataset with shared label space."""
    feat_dfp, lab_dfp, names_dfp, _ = load_mat_dataset(
        dfp_dir, n_bins=n_bins, min_samples_per_class=min_samples_per_class,
    )
    feat_cns, lab_cns, names_cns, _ = load_mat_dataset(
        cns_dir, n_bins=n_bins, min_samples_per_class=min_samples_per_class,
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
