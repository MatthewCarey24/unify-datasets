"""ResidualMLP encoder for SupCon embedding experiment."""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """LayerNorm → Linear → GELU → Linear + skip connection."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        return h + self.skip(x)


class ResidualMLP(nn.Module):
    """
    Input → Linear(512) → ResBlock(512) → ResBlock(256) → ResBlock(128) → 128-dim embedding.
    Embeddings are L2-normalised to the unit sphere.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, embed_dim: int = 128):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.stem = nn.Linear(input_dim, hidden_dims[0])
        blocks = []
        prev = hidden_dims[0]
        for dim in hidden_dims:
            blocks.append(ResBlock(prev, dim))
            prev = dim
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(prev, embed_dim) if prev != embed_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised embeddings [B, embed_dim]."""
        h = self.stem(x)
        h = self.blocks(h)
        h = self.head(h)
        return nn.functional.normalize(h, dim=-1)
