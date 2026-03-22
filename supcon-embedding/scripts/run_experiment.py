"""Run experiment: Paul's separate setup vs joint training."""

import subprocess
import sys

PYTHON = sys.executable

steps = [
    # Paul's setup
    ("A: DFP with LDA (Paul's)", [PYTHON, "-m", "supcon.train", "configs/A_paul_dfp_lda.toml"]),
    ("A: CNS no LDA (Paul's)",   [PYTHON, "-m", "supcon.train", "configs/A_paul_cns_nolda.toml"]),
    # Joint
    ("D: Joint",                  [PYTHON, "-m", "supcon.train", "configs/D_joint.toml"]),
    # Eval
    ("Eval A: Paul's separate", [
        PYTHON, "-m", "supcon.eval", "separate",
        "--ckpt-dfp", "outputs/A_dfp_lda/checkpoint.pt",
        "--ckpt-cns", "outputs/A_cns_nolda/checkpoint.pt",
        "--dfp-lda",
        "-o", "outputs/eval_A.json",
    ]),
    ("Eval D: Joint", [
        PYTHON, "-m", "supcon.eval", "joint",
        "--ckpt", "outputs/D_joint/checkpoint.pt",
        "-o", "outputs/eval_D.json",
    ]),
]

if __name__ == "__main__":
    for label, cmd in steps:
        print(f"\n{'='*60}")
        print(f">>> {label}")
        print(f"{'='*60}\n")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {label}")
            sys.exit(result.returncode)

    print("\n" + "="*60)
    print("DONE. Results:")
    print("  A (Paul's separate):  outputs/eval_A.json")
    print("  D (joint training):   outputs/eval_D.json")
    print("="*60)
