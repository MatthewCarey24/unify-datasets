"""Training loop and CLI entrypoint for SupCon experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from supcon.data import (
    discover_feature_columns,
    load_cns,
    load_dfp,
    load_dfp_with_dose,
    load_joint,
    make_dataloader,
)
from supcon.loss import PaulCNSLoss, PaulDFPLoss, SupConLoss
from supcon.model import ResidualMLP


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def train(cfg: dict) -> Path:
    """Run training and return path to saved checkpoint."""
    seed = cfg.get("run", {}).get("seed", 42)
    torch.manual_seed(seed)

    data_cfg = cfg.get("data", {})
    mode = data_cfg.get("mode", "dfp")
    min_samples = data_cfg.get("min_samples_per_class", 2)
    use_lda = data_cfg.get("use_lda", False)
    batch_size = data_cfg.get("batch_size", 256)

    dfp_csv = data_cfg.get("dfp_csv", "C:/data/dataset/DFP/DFP0395_comprehensiveSft.csv")
    cns_csv = data_cfg.get("cns_csv", "C:/data/dataset/CNS/CNS0091_comprehensiveSft.csv")
    cns_meta = data_cfg.get("cns_metadata_csv", "C:/data/dataset/CNS/CNS0091_sourceMetadata.csv")
    feature_columns = discover_feature_columns(dfp_csv, cns_csv)

    # ── Load data ──
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "supcon")  # supcon | paul_dfp | paul_cns
    extra_tensors = {}

    if mode == "dfp" and loss_type == "paul_dfp":
        # Need compound + dose labels
        data = load_dfp_with_dose(
            csv_path=dfp_csv,
            feature_columns=feature_columns,
            min_samples_per_class=min_samples,
            use_lda=use_lda,
        )
        features = data["features"]
        labels = data["labels"]
        label_names = data["label_names"]
        extra_tensors["compound_labels"] = data["compound_labels"]
        extra_tensors["dose_labels"] = data["dose_labels"]
    elif mode == "dfp":
        features, labels, label_names, _ = load_dfp(
            csv_path=dfp_csv,
            feature_columns=feature_columns,
            min_samples_per_class=min_samples,
            use_lda=use_lda,
        )
    elif mode == "cns":
        features, labels, label_names, _ = load_cns(
            csv_path=cns_csv,
            metadata_csv=cns_meta,
            feature_columns=feature_columns,
            min_samples_per_class=min_samples,
            use_lda=use_lda,
        )
    elif mode == "joint":
        features, labels, label_names = load_joint(
            dfp_csv=dfp_csv,
            cns_csv=cns_csv,
            cns_metadata_csv=cns_meta,
            feature_columns=feature_columns,
            min_samples_per_class=min_samples,
            use_lda=use_lda,
        )
    else:
        raise ValueError(f"Unknown data mode: {mode}")

    input_dim = features.shape[1]
    loader = make_dataloader(features, labels, batch_size=batch_size,
                             extra=extra_tensors if extra_tensors else None)

    # ── Model ──
    model_cfg = cfg.get("model", {})
    model = ResidualMLP(
        input_dim=input_dim,
        hidden_dims=model_cfg.get("hidden_dims", [512, 256, 128]),
        embed_dim=model_cfg.get("embed_dim", 128),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ── Loss ──
    if loss_type == "paul_dfp":
        criterion = PaulDFPLoss(
            temperature=loss_cfg.get("temperature", 0.40),
            w_supcon=loss_cfg.get("w_supcon", 1.0),
            w_align=loss_cfg.get("w_align", 0.5),
            w_repulse=loss_cfg.get("w_repulse", 0.5),
            w_ordinal=loss_cfg.get("w_ordinal", 0.3),
        )
    elif loss_type == "paul_cns":
        criterion = PaulCNSLoss(
            temperature=loss_cfg.get("temperature", 0.30),
            w_supcon=loss_cfg.get("w_supcon", 1.0),
            w_align=loss_cfg.get("w_align", 0.5),
        )
    else:
        criterion = SupConLoss(temperature=loss_cfg.get("temperature", 0.07))

    # ── Optimizer ──
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg.get("lr", 1e-3),
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
    )
    max_steps = opt_cfg.get("max_steps", 5000)

    # ── Output ──
    run_cfg = cfg.get("run", {})
    output_dir = Path(run_cfg.get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    model.train()
    step = 0
    epoch = 0
    while step < max_steps:
        epoch += 1
        for batch in loader:
            if step >= max_steps:
                break

            feat_batch = batch["features"].to(device)
            lab_batch = batch["labels"].to(device)
            embeddings = model(feat_batch)

            if loss_type == "paul_dfp" and "compound_labels" in batch:
                loss = criterion(
                    embeddings, lab_batch,
                    compound_labels=batch["compound_labels"].to(device),
                    dose_labels=batch["dose_labels"].to(device),
                )
            elif loss_type == "paul_cns":
                loss = criterion(embeddings, lab_batch)
            else:
                loss = criterion(embeddings, lab_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % 100 == 0 or step == 1:
                print(f"  step {step:>5d}/{max_steps}  loss={loss.item():.4f}")

    # ── Save checkpoint ──
    ckpt_path = output_dir / "checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "label_names": label_names,
        "feature_columns": feature_columns,
        "input_dim": input_dim,
        "step": step,
    }, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")

    meta_path = output_dir / "train_meta.json"
    meta_path.write_text(json.dumps({
        "label_names": label_names,
        "input_dim": input_dim,
        "n_features": len(feature_columns),
        "n_samples": len(features),
        "n_classes": len(label_names),
        "loss_type": loss_type,
        "final_step": step,
    }, indent=2))

    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train SupCon encoder")
    parser.add_argument("config", help="Path to TOML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})
    print(f"Training with config: {args.config}")
    print(f"  mode: {data_cfg.get('mode', 'dfp')}")
    print(f"  loss: {loss_cfg.get('type', 'supcon')}")
    print(f"  output_dir: {cfg.get('run', {}).get('output_dir', 'outputs/default')}")

    ckpt = train(cfg)
    print(f"Done. Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
