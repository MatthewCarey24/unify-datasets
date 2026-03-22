"""Evaluation metrics for drug embedding quality.

- Per-drug silhouette score (DrSep) — comparable to Paul's DrSep = 0.937
- kNN accuracy (k=10) — fraction of neighbors sharing drug label
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples


def drug_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Compute overall and per-drug silhouette scores.

    Args:
        embeddings: [N, D] L2-normalized embeddings
        labels: [N] integer compound labels

    Returns:
        dict with 'overall', 'per_drug' (dict of label -> score)
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return {"overall": 0.0, "per_drug": {}}

    overall = float(silhouette_score(embeddings, labels))
    sample_scores = silhouette_samples(embeddings, labels)

    per_drug = {}
    for lab in unique:
        mask = labels == lab
        per_drug[int(lab)] = float(sample_scores[mask].mean())

    return {"overall": overall, "per_drug": per_drug}


def knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    max_samples: int = 10000,
) -> float:
    """Fraction of k nearest neighbors sharing the query's drug label.

    Uses cosine similarity (dot product on L2-normed vectors).
    Subsamples if dataset is large for speed.
    """
    n = len(embeddings)
    if n <= 1:
        return 0.0

    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        n = max_samples

    k = min(k, n - 1)
    # Cosine similarity matrix (already L2-normed)
    sim = embeddings @ embeddings.T  # [N, N]
    np.fill_diagonal(sim, -1.0)  # exclude self

    # Top-k neighbors
    topk_idx = np.argpartition(sim, -k, axis=1)[:, -k:]

    correct = 0
    total = 0
    for i in range(n):
        neighbors = topk_idx[i]
        correct += (labels[neighbors] == labels[i]).sum()
        total += k

    return float(correct / total)


def evaluate(embeddings: np.ndarray, labels: np.ndarray, label_names: list[str]) -> dict:
    """Run all eval metrics and print results.

    Returns dict with silhouette_overall, knn_accuracy, per_drug_silhouette.
    """
    sil = drug_silhouette(embeddings, labels)
    knn_acc = knn_accuracy(embeddings, labels, k=10)

    print(f"\n{'='*60}")
    print(f"  Drug Silhouette (DrSep): {sil['overall']:.4f}")
    print(f"  kNN Accuracy (k=10):     {knn_acc:.4f}")
    print(f"{'='*60}")
    print(f"  Per-drug silhouette:")
    for lab_idx, score in sorted(sil["per_drug"].items()):
        name = label_names[lab_idx] if lab_idx < len(label_names) else f"class_{lab_idx}"
        print(f"    {name:>20s}: {score:.4f}")
    print()

    return {
        "silhouette_overall": sil["overall"],
        "knn_accuracy": knn_acc,
        "per_drug_silhouette": {
            label_names[k] if k < len(label_names) else f"class_{k}": v
            for k, v in sil["per_drug"].items()
        },
    }
