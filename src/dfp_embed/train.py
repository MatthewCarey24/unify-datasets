"""Training loop for DFP0395 drug phenotypic embedding.

Phase 1: SupCon with raw 825 features, compound-only labels, temp=0.4
Phase 2: Add LDA preprocessing (825 -> 21 dims)
Phase 3: Add auxiliary losses (alignment + centroid repulsion + ordinal dose)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from dfp_embed.data import load_dfp0395
from dfp_embed.eval import evaluate
from dfp_embed.loss import DFPLoss
from dfp_embed.model import ResidualMLP
from dfp_embed.sampler import BalancedBatchSampler


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


@torch.no_grad()
def embed_all(
    model: ResidualMLP, features: torch.Tensor, device: torch.device, batch_size: int = 2048
) -> np.ndarray:
    """Embed entire dataset in batches."""
    model.eval()
    parts = []
    for i in range(0, len(features), batch_size):
        batch = features[i : i + batch_size].to(device)
        parts.append(model(batch).cpu().numpy())
    model.train()
    return np.concatenate(parts, axis=0)


class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, dose_ranks):
        self.features = features
        self.labels = labels
        self.dose_ranks = dose_ranks

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.dose_ranks[idx]


def train(cfg: dict) -> Path:
    seed = cfg.get("run", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Data ──
    data_cfg = cfg.get("data", {})
    csv_path = data_cfg.get("csv_path", "data/DFP0395_comprehensiveSft.csv")
    n_features = data_cfg.get("n_features", 825)
    use_lda = data_cfg.get("use_lda", False)
    exclude_dmso = data_cfg.get("exclude_dmso", True)

    data = load_dfp0395(
        csv_path=csv_path,
        n_features=n_features,
        use_lda=use_lda,
        exclude_dmso=exclude_dmso,
    )

    features = data["features"]
    labels = data["labels"]
    label_names = data["label_names"]
    dose_ranks = data["dose_ranks"]
    input_dim = features.shape[1]

    # ── Balanced batch sampler ──
    sampler_cfg = cfg.get("sampler", {})
    p_classes = sampler_cfg.get("p_classes", len(label_names))  # all classes
    k_samples = sampler_cfg.get("k_samples", 48)
    max_steps = cfg.get("optimizer", {}).get("max_steps", 20000)

    dataset = InMemoryDataset(features, labels, dose_ranks)
    sampler = BalancedBatchSampler(
        labels=labels,
        p_classes=p_classes,
        k_samples=k_samples,
        n_batches=max_steps,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    batch_size = p_classes * k_samples
    print(f"  Batch: {p_classes} classes x {k_samples} samples = {batch_size}")

    # ── Model ──
    model_cfg = cfg.get("model", {})
    hidden_dims = model_cfg.get("hidden_dims", [512, 256, 128])
    embed_dim = model_cfg.get("embed_dim", 128)

    model = ResidualMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        embed_dim=embed_dim,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Model: ResidualMLP {hidden_dims} -> {embed_dim}-dim")
    print(f"  Device: {device}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Loss ──
    loss_cfg = cfg.get("loss", {})
    temperature = loss_cfg.get("temperature", 0.4)
    use_auxiliary = loss_cfg.get("use_auxiliary", False)

    criterion = DFPLoss(
        temperature=temperature,
        w_supcon=loss_cfg.get("w_supcon", 1.0),
        w_align=loss_cfg.get("w_align", 0.5),
        w_repulse=loss_cfg.get("w_repulse", 0.5),
        w_ordinal=loss_cfg.get("w_ordinal", 0.3),
        use_auxiliary=use_auxiliary,
    )
    print(f"  Loss: SupCon temp={temperature}"
          f"{' + aux (align+repulse+ordinal)' if use_auxiliary else ''}")

    # ── Optimizer ──
    opt_cfg = cfg.get("optimizer", {})
    lr = opt_cfg.get("lr", 5e-3)
    weight_decay = opt_cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=1e-5
    )
    print(f"  Optimizer: Adam lr={lr}, wd={weight_decay}, cosine -> {max_steps} steps")

    # ── Output ──
    output_dir = Path(cfg.get("run", {}).get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ──
    eval_every = cfg.get("run", {}).get("eval_every", 5000)
    model.train()

    print(f"\n{'='*60}")
    print(f"  Starting training: {max_steps} steps")
    print(f"{'='*60}\n")

    for step, (feat_batch, lab_batch, dose_batch) in enumerate(loader, 1):
        feat_batch = feat_batch.to(device)
        lab_batch = lab_batch.to(device)
        dose_batch = dose_batch.to(device)

        embeddings = model(feat_batch)
        loss = criterion(
            embeddings, lab_batch,
            compound_labels=lab_batch if use_auxiliary else None,
            dose_ranks=dose_batch if use_auxiliary else None,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 500 == 0 or step <= 5 or step == max_steps:
            cur_lr = scheduler.get_last_lr()[0]
            print(f"  step {step:>5d}/{max_steps}  loss={loss.item():.4f}  lr={cur_lr:.2e}")

        if step % eval_every == 0 or step == max_steps:
            embs = embed_all(model, features, device)
            evaluate(embs, labels.numpy(), label_names)
            model.train()

    # ── Save ──
    ckpt_path = output_dir / "checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "label_names": label_names,
        "input_dim": input_dim,
        "step": max_steps,
        "feature_columns": data["feature_columns"],
    }, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")

    meta = {
        "label_names": label_names,
        "input_dim": input_dim,
        "n_samples": len(features),
        "n_classes": len(label_names),
        "final_step": max_steps,
        "use_lda": use_lda,
        "temperature": temperature,
    }
    meta_path = output_dir / "train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train DFP0395 drug embedding")
    parser.add_argument("config", help="Path to TOML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    print(f"Config: {args.config}")
    print(f"  csv: {data_cfg.get('csv_path', 'data/DFP0395_comprehensiveSft.csv')}")
    print(f"  use_lda: {data_cfg.get('use_lda', False)}")
    print(f"  output: {cfg.get('run', {}).get('output_dir', 'outputs/default')}")

    ckpt = train(cfg)
    print(f"\nDone. Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
