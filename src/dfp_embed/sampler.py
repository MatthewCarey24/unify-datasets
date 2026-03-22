"""Balanced batch sampler for SupCon training.

Guarantees P classes per batch with K samples each, so every sample
has at least K-1 positive pairs. This is critical for SupCon to work.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler[list[int]]):
    """Sample P classes, K samples per class -> batch of P*K.

    If a class has fewer than K samples, samples with replacement.
    Yields index lists of length P*K.
    """

    def __init__(
        self,
        labels: torch.Tensor | np.ndarray,
        p_classes: int = 22,
        k_samples: int = 48,
        n_batches: int = 20000,
    ):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.labels = labels
        self.p_classes = p_classes
        self.k_samples = k_samples
        self.n_batches = n_batches

        # Build class -> indices mapping
        self.class_indices: dict[int, np.ndarray] = {}
        for cls in np.unique(labels):
            self.class_indices[cls] = np.where(labels == cls)[0]

        self.all_classes = np.array(sorted(self.class_indices.keys()))
        # Clamp P to available classes
        self.p_classes = min(self.p_classes, len(self.all_classes))

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self.n_batches):
            selected_classes = rng.choice(
                self.all_classes, size=self.p_classes, replace=False
            )
            batch_indices = []
            for cls in selected_classes:
                cls_idx = self.class_indices[cls]
                if len(cls_idx) >= self.k_samples:
                    chosen = rng.choice(cls_idx, size=self.k_samples, replace=False)
                else:
                    chosen = rng.choice(cls_idx, size=self.k_samples, replace=True)
                batch_indices.extend(chosen.tolist())
            yield batch_indices

    def __len__(self) -> int:
        return self.n_batches
