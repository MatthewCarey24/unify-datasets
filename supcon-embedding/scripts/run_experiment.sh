#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== SupCon Embedding Experiment ==="
echo ""

# --- Train DFP (separate) ---
echo ">>> Training DFP (separate condition)..."
python -m supcon.train configs/dfp_separate.toml
echo ""

# --- Train CNS (separate) ---
echo ">>> Training CNS (separate condition)..."
python -m supcon.train configs/cns_separate.toml
echo ""

# --- Train Joint ---
echo ">>> Training Joint condition..."
python -m supcon.train configs/joint.toml
echo ""

# --- Evaluate Separate ---
echo ">>> Evaluating SEPARATE condition..."
python -m supcon.eval separate \
    --ckpt-dfp outputs/dfp_separate/checkpoint.pt \
    --ckpt-cns outputs/cns_separate/checkpoint.pt \
    --dfp-dir C:/data/dataset/DFP \
    --cns-dir C:/data/dataset/CNS \
    --dfp-col drug1 \
    --cns-col treatment \
    -o outputs/eval_separate.json
echo ""

# --- Evaluate Joint ---
echo ">>> Evaluating JOINT condition..."
python -m supcon.eval joint \
    --ckpt outputs/joint/checkpoint.pt \
    --dfp-dir C:/data/dataset/DFP \
    --cns-dir C:/data/dataset/CNS \
    --dfp-col drug1 \
    --cns-col treatment \
    -o outputs/eval_joint.json
echo ""

echo "=== Done ==="
echo "Results:"
echo "  Separate: outputs/eval_separate.json"
echo "  Joint:    outputs/eval_joint.json"
