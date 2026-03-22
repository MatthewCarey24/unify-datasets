"""Evaluation: leakage AUROC + silhouette score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score

from supcon.data import discover_feature_columns, load_cns, load_dfp, load_joint
from supcon.model import ResidualMLP


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> ResidualMLP:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_cfg = cfg.get("model", {})
    model = ResidualMLP(
        input_dim=ckpt["input_dim"],
        hidden_dims=model_cfg.get("hidden_dims", [128, 128]),
        embed_dim=model_cfg.get("embed_dim", 128),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def embed(model: ResidualMLP, features: torch.Tensor, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    all_embs = []
    for i in range(0, len(features), batch_size):
        batch = features[i : i + batch_size].to(device)
        emb = model(batch)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def compute_leakage_auroc(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cross-validated AUROC for classifying dataset A vs B in embedding space."""
    from sklearn.model_selection import cross_val_predict
    X = np.concatenate([emb_a, emb_b], axis=0)
    y = np.array([0] * len(emb_a) + [1] * len(emb_b))
    # Subsample if too large (CV on 145K is slow)
    if len(X) > 20000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=20000, replace=False)
        X, y = X[idx], y[idx]
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def compute_silhouette(embeddings: np.ndarray, labels: list[str]) -> float:
    unique = sorted(set(labels))
    if len(unique) < 2:
        return 0.0
    label_map = {name: i for i, name in enumerate(unique)}
    int_labels = np.array([label_map[t] for t in labels])
    n = len(embeddings)
    if n > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=10000, replace=False)
        embeddings = embeddings[idx]
        int_labels = int_labels[idx]
    return float(silhouette_score(embeddings, int_labels))


def evaluate_separate(
    ckpt_dfp: str, ckpt_cns: str,
    dfp_dir: str, cns_dir: str,
    feature_columns: list[str],
    dfp_lda: bool = False, cns_lda: bool = False,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dfp = load_model_from_checkpoint(ckpt_dfp, device)
    model_cns = load_model_from_checkpoint(ckpt_cns, device)

    print("Loading DFP...")
    feat_dfp, lab_dfp, names_dfp, _ = load_dfp(dfp_dir, feature_columns, use_lda=dfp_lda)
    print("Loading CNS...")
    feat_cns, lab_cns, names_cns, _ = load_cns(cns_dir, feature_columns, use_lda=cns_lda)

    print("Embedding DFP...")
    emb_dfp = embed(model_dfp, feat_dfp, device)
    print("Embedding CNS...")
    emb_cns = embed(model_cns, feat_cns, device)

    leakage = compute_leakage_auroc(emb_dfp, emb_cns)
    all_emb = np.concatenate([emb_dfp, emb_cns], axis=0)
    all_labels = [names_dfp[l] for l in lab_dfp.tolist()] + [names_cns[l] for l in lab_cns.tolist()]
    sil = compute_silhouette(all_emb, all_labels)

    return {
        "condition": "separate",
        "leakage_auroc": round(leakage, 4),
        "separation_silhouette": round(sil, 4),
        "n_dfp": len(emb_dfp),
        "n_cns": len(emb_cns),
    }


def evaluate_joint(
    ckpt_joint: str,
    dfp_dir: str, cns_dir: str,
    feature_columns: list[str],
) -> dict:
    """Evaluate joint model using shared z-score (same as training)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(ckpt_joint, device)

    # Load with shared z-score — same as training
    print("Loading joint data (shared z-score)...")
    features, labels, label_names = load_joint(dfp_dir, cns_dir, feature_columns)

    # Split back into DFP/CNS by label prefix
    is_dfp = torch.tensor([label_names[l].startswith("DFP:") for l in labels.tolist()])
    is_cns = ~is_dfp

    print("Embedding all...")
    all_emb = embed(model, features, device)
    emb_dfp = all_emb[is_dfp.numpy()]
    emb_cns = all_emb[is_cns.numpy()]

    leakage = compute_leakage_auroc(emb_dfp, emb_cns)
    all_labels = [label_names[l] for l in labels.tolist()]
    sil = compute_silhouette(all_emb, all_labels)

    return {
        "condition": "joint",
        "leakage_auroc": round(leakage, 4),
        "separation_silhouette": round(sil, 4),
        "n_dfp": int(is_dfp.sum()),
        "n_cns": int(is_cns.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SupCon embeddings")
    parser.add_argument("--dfp-dir", default="/data/dataset/DFP")
    parser.add_argument("--cns-dir", default="/data/dataset/CNS")

    sub = parser.add_subparsers(dest="mode", required=True)

    sep = sub.add_parser("separate")
    sep.add_argument("--ckpt-dfp", required=True)
    sep.add_argument("--ckpt-cns", required=True)
    sep.add_argument("--dfp-lda", action="store_true")
    sep.add_argument("--cns-lda", action="store_true")
    sep.add_argument("-o", "--output", default=None)

    jnt = sub.add_parser("joint")
    jnt.add_argument("--ckpt", required=True)
    jnt.add_argument("-o", "--output", default=None)

    args = parser.parse_args()

    # Discover feature columns
    dfp_csv = str(next(Path(args.dfp_dir).glob("*_comprehensiveSft.csv")))
    cns_csv = str(next(Path(args.cns_dir).glob("*_comprehensiveSft.csv")))
    feature_columns = discover_feature_columns(dfp_csv, cns_csv)

    if args.mode == "separate":
        result = evaluate_separate(
            args.ckpt_dfp, args.ckpt_cns,
            args.dfp_dir, args.cns_dir, feature_columns,
            dfp_lda=args.dfp_lda, cns_lda=args.cns_lda,
        )
    else:
        result = evaluate_joint(
            args.ckpt, args.dfp_dir, args.cns_dir, feature_columns,
        )

    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
