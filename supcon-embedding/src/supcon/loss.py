"""Loss functions for SupCon embedding experiment.

Paul's DFP loss = SupCon + alignment + centroid_repulsion + ordinal_dose
Paul's CNS loss = SupCon + alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Standard SupCon (Khosla et al. 2020).
    Expects L2-normalised embeddings and integer labels."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        B = embeddings.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        sim = embeddings @ embeddings.T / self.temperature

        labels_col = labels.unsqueeze(0)
        mask_pos = (labels_col == labels_col.T).float()
        mask_pos.fill_diagonal_(0.0)
        num_pos = mask_pos.sum(dim=1)

        logits_max = sim.max(dim=1, keepdim=True).values.detach()
        logits = sim - logits_max

        self_mask = torch.eye(B, device=device)
        logits = logits - self_mask * 1e9

        exp_logits = torch.exp(logits) * (1.0 - self_mask)
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        log_prob = logits - log_sum_exp
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (num_pos + 1e-12)

        valid = num_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return -mean_log_prob[valid].mean()


class AlignmentLoss(nn.Module):
    """Pull each embedding toward its class centroid (computed per batch)."""

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        unique_labels = labels.unique()
        if len(unique_labels) < 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = torch.tensor(0.0, device=device)
        count = 0
        for lab in unique_labels:
            mask = labels == lab
            if mask.sum() < 2:
                continue
            members = embeddings[mask]
            centroid = F.normalize(members.mean(dim=0, keepdim=True), dim=-1)
            # 1 - cosine similarity = alignment loss
            cos_sim = (members * centroid).sum(dim=-1)
            loss = loss + (1.0 - cos_sim).mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss / count


class CentroidRepulsionLoss(nn.Module):
    """Push class centroids apart from each other.
    Uses negative cosine similarity between all centroid pairs."""

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        unique_labels = labels.unique()
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute centroids
        centroids = []
        for lab in unique_labels:
            mask = labels == lab
            members = embeddings[mask]
            centroids.append(F.normalize(members.mean(dim=0, keepdim=True), dim=-1))
        centroids = torch.cat(centroids, dim=0)  # [K, D]

        # Cosine similarity between all pairs
        sim = centroids @ centroids.T  # [K, K]
        K = len(unique_labels)
        mask = 1.0 - torch.eye(K, device=device)
        # We want centroids to be far apart: maximize negative similarity
        # Loss = mean of cosine similarities between different centroids
        # (lower = more separated)
        return (sim * mask).sum() / (mask.sum() + 1e-12)


class OrdinalDoseLoss(nn.Module):
    """Within each compound, enforce dose ordering in embedding space.
    Higher dose should be farther from DMSO centroid (the origin).

    dose_labels: integer dose ranks (0=lowest, N=highest) per sample.
    compound_labels: integer compound id per sample.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        compound_labels: torch.Tensor,
        dose_labels: torch.Tensor,
        origin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = embeddings.device
        if origin is None:
            # Use batch mean as proxy for origin
            origin = F.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1)

        # Distance from origin for each sample
        dist = 1.0 - (embeddings * origin).sum(dim=-1)  # [B]

        loss = torch.tensor(0.0, device=device)
        count = 0

        for comp in compound_labels.unique():
            comp_mask = compound_labels == comp
            comp_doses = dose_labels[comp_mask]
            comp_dists = dist[comp_mask]

            unique_doses = comp_doses.unique().sort().values
            if len(unique_doses) < 2:
                continue

            # For consecutive dose pairs, higher dose should have greater distance
            for i in range(len(unique_doses) - 1):
                low_dose = unique_doses[i]
                high_dose = unique_doses[i + 1]
                d_low = comp_dists[comp_doses == low_dose].mean()
                d_high = comp_dists[comp_doses == high_dose].mean()
                # margin ranking: d_high should be > d_low + margin
                violation = self.margin - (d_high - d_low)
                loss = loss + F.relu(violation)
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss / count


class PaulDFPLoss(nn.Module):
    """Combined loss matching Paul's DFP training.

    L = w_supcon * SupCon + w_align * Alignment
        + w_repulse * CentroidRepulsion + w_ordinal * OrdinalDose
    """

    def __init__(
        self,
        temperature: float = 0.40,
        w_supcon: float = 1.0,
        w_align: float = 0.5,
        w_repulse: float = 0.5,
        w_ordinal: float = 0.3,
    ):
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature)
        self.align = AlignmentLoss()
        self.repulse = CentroidRepulsionLoss()
        self.ordinal = OrdinalDoseLoss()
        self.w_supcon = w_supcon
        self.w_align = w_align
        self.w_repulse = w_repulse
        self.w_ordinal = w_ordinal

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        compound_labels: torch.Tensor | None = None,
        dose_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.w_supcon * self.supcon(embeddings, labels)
        loss = loss + self.w_align * self.align(embeddings, labels)
        loss = loss + self.w_repulse * self.repulse(embeddings, labels)

        if compound_labels is not None and dose_labels is not None and self.w_ordinal > 0:
            loss = loss + self.w_ordinal * self.ordinal(
                embeddings, compound_labels, dose_labels
            )
        return loss


class PaulCNSLoss(nn.Module):
    """Combined loss matching Paul's CNS training.

    L = w_supcon * SupCon + w_align * Alignment
    """

    def __init__(
        self,
        temperature: float = 0.30,
        w_supcon: float = 1.0,
        w_align: float = 0.5,
    ):
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature)
        self.align = AlignmentLoss()
        self.w_supcon = w_supcon
        self.w_align = w_align

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.w_supcon * self.supcon(embeddings, labels)
        loss = loss + self.w_align * self.align(embeddings, labels)
        return loss
