"""SupCon loss and auxiliary losses for drug embedding.

Phase 1: SupCon only (compound labels, temp=0.4)
Phase 2+: Add alignment, centroid repulsion, ordinal dose
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al. 2020).

    Expects L2-normalised embeddings and integer labels.
    Temperature = 0.4 is Paul's optimum for drug separation.
    """

    def __init__(self, temperature: float = 0.4):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        B = embeddings.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Cosine similarity scaled by temperature
        sim = embeddings @ embeddings.T / self.temperature  # [B, B]

        # Positive mask: same label, excluding self
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_pos.fill_diagonal_(0.0)
        num_pos = mask_pos.sum(dim=1)

        # Numerical stability: subtract max
        logits_max = sim.max(dim=1, keepdim=True).values.detach()
        logits = sim - logits_max

        # Exclude self from denominator
        self_mask = torch.eye(B, device=device)
        exp_logits = torch.exp(logits) * (1.0 - self_mask)
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-prob over positives
        log_prob = logits - log_sum_exp
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (num_pos + 1e-12)

        valid = num_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return -mean_log_prob[valid].mean()


class AlignmentLoss(nn.Module):
    """Pull each embedding toward its class centroid."""

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
            cos_sim = (members * centroid).sum(dim=-1)
            loss = loss + (1.0 - cos_sim).mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss / count


class CentroidRepulsionLoss(nn.Module):
    """Push class centroids apart (mean cosine sim between centroids)."""

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        unique_labels = labels.unique()
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        centroids = []
        for lab in unique_labels:
            members = embeddings[labels == lab]
            centroids.append(F.normalize(members.mean(dim=0, keepdim=True), dim=-1))
        centroids = torch.cat(centroids, dim=0)

        sim = centroids @ centroids.T
        K = len(unique_labels)
        off_diag = 1.0 - torch.eye(K, device=device)
        return (sim * off_diag).sum() / (off_diag.sum() + 1e-12)


class OrdinalDoseLoss(nn.Module):
    """Enforce dose ordering: higher dose -> farther from origin."""

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        compound_labels: torch.Tensor,
        dose_ranks: torch.Tensor,
        origin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = embeddings.device
        if origin is None:
            origin = F.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1)

        dist = 1.0 - (embeddings * origin).sum(dim=-1)

        loss = torch.tensor(0.0, device=device)
        count = 0
        for comp in compound_labels.unique():
            comp_mask = compound_labels == comp
            comp_doses = dose_ranks[comp_mask]
            comp_dists = dist[comp_mask]

            unique_doses = comp_doses.unique().sort().values
            if len(unique_doses) < 2:
                continue

            for i in range(len(unique_doses) - 1):
                d_low = comp_dists[comp_doses == unique_doses[i]].mean()
                d_high = comp_dists[comp_doses == unique_doses[i + 1]].mean()
                violation = self.margin - (d_high - d_low)
                loss = loss + F.relu(violation)
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss / count


class DFPLoss(nn.Module):
    """Combined loss: SupCon + alignment + centroid repulsion + ordinal dose.

    Phase 1: use_auxiliary=False -> SupCon only
    Phase 2+: use_auxiliary=True -> full Paul-style loss
    """

    def __init__(
        self,
        temperature: float = 0.40,
        w_supcon: float = 1.0,
        w_align: float = 0.5,
        w_repulse: float = 0.5,
        w_ordinal: float = 0.3,
        use_auxiliary: bool = False,
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
        self.use_auxiliary = use_auxiliary

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        compound_labels: torch.Tensor | None = None,
        dose_ranks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.w_supcon * self.supcon(embeddings, labels)

        if self.use_auxiliary:
            loss = loss + self.w_align * self.align(embeddings, labels)
            loss = loss + self.w_repulse * self.repulse(embeddings, labels)
            if compound_labels is not None and dose_ranks is not None:
                loss = loss + self.w_ordinal * self.ordinal(
                    embeddings, compound_labels, dose_ranks
                )

        return loss
