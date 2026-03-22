"""Training loop for SupCon experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from supcon.data import discover_feature_columns, load_cns, load_dfp, load_joint, make_dataloader
from supcon.loss import SupConLoss
from supcon.model import ResidualMLP


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def train(cfg: dict) -> Path:
    seed = cfg.get("run", {}).get("seed", 42)
    torch.manual_seed(seed)

    data_cfg = cfg.get("data", {})
    mode = data_cfg.get("mode", "dfp")
    use_lda = data_cfg.get("use_lda", False)
    batch_size = data_cfg.get("batch_size", 1024)
    min_samples = data_cfg.get("min_samples_per_class", 2)
    dfp_dir = data_cfg.get("dfp_dir", "/data/dataset/DFP")
    cns_dir = data_cfg.get("cns_dir", "/data/dataset/CNS")

    # Discover shared feature columns
    dfp_csv = str(Path(dfp_dir) / next(Path(dfp_dir).glob("*_comprehensiveSft.csv")).name)
    cns_csv = str(Path(cns_dir) / next(Path(cns_dir).glob("*_comprehensiveSft.csv")).name)
    feature_columns = discover_feature_columns(dfp_csv, cns_csv)
    print(f"  {len(feature_columns)} shared feature columns")

    # Load data
    if mode == "joint":
        features, labels, label_names = load_joint(
            dfp_dir, cns_dir, feature_columns, min_samples,
        )
    elif mode == "dfp":
        features, labels, label_names, _ = load_dfp(
            dfp_dir, feature_columns, min_samples, use_lda=use_lda,
        )
    elif mode == "cns":
        features, labels, label_names, _ = load_cns(
            cns_dir, feature_columns, min_samples, use_lda=use_lda,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    input_dim = features.shape[1]
    loader = make_dataloader(features, labels, batch_size=batch_size)

    # Model
    model_cfg = cfg.get("model", {})
    model = ResidualMLP(
        input_dim=input_dim,
        hidden_dims=model_cfg.get("hidden_dims", [128, 128]),
        embed_dim=model_cfg.get("embed_dim", 128),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  device: {device}")

    # Loss
    loss_cfg = cfg.get("loss", {})
    criterion = SupConLoss(temperature=loss_cfg.get("temperature", 0.07))

    # Optimizer
    opt_cfg = cfg.get("optimizer", {})
    lr = opt_cfg.get("lr", 5e-3)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
    )
    max_steps = opt_cfg.get("max_steps", 20000)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=1e-5,
    )

    # Output
    output_dir = Path(cfg.get("run", {}).get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    model.train()
    step = 0
    epoch = 0
    while step < max_steps:
        epoch += 1
        for feat_batch, lab_batch in loader:
            if step >= max_steps:
                break
            feat_batch = feat_batch.to(device)
            lab_batch = lab_batch.to(device)

            embeddings = model(feat_batch)
            loss = criterion(embeddings, lab_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 500 == 0 or step <= 5:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  step {step:>5d}/{max_steps}  loss={loss.item():.4f}  lr={cur_lr:.2e}")

    # Save
    ckpt_path = output_dir / "checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "label_names": label_names,
        "input_dim": input_dim,
        "step": step,
    }, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")

    meta_path = output_dir / "train_meta.json"
    meta_path.write_text(json.dumps({
        "label_names": label_names,
        "input_dim": input_dim,
        "n_samples": len(features),
        "n_classes": len(label_names),
        "final_step": step,
    }, indent=2))

    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train SupCon encoder")
    parser.add_argument("config", help="Path to TOML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    print(f"Training with config: {args.config}")
    print(f"  mode: {data_cfg.get('mode', 'dfp')}")
    print(f"  use_lda: {data_cfg.get('use_lda', False)}")
    print(f"  output_dir: {cfg.get('run', {}).get('output_dir', 'outputs/default')}")

    ckpt = train(cfg)
    print(f"Done. Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
