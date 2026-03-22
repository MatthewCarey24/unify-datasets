"""Quick diagnostic: is the model learning at all?"""
import torch
import torch.nn as nn
from supcon.data import load_mat_dataset, make_dataloader
from supcon.model import ResidualMLP
from supcon.loss import SupConLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DFP with LDA
feat, lab, names, _ = load_mat_dataset("/data/dataset/DFP", n_bins=825, use_lda=True)
print(f"\nInput stats: mean={feat.mean():.4f}, std={feat.std():.4f}, "
      f"min={feat.min():.4f}, max={feat.max():.4f}")
print(f"Shape: {feat.shape}, Classes: {len(names)}")

loader = make_dataloader(feat, lab, batch_size=1024)

# Test 1: Does a SIMPLE model learn? (no ResBlocks, just linear)
print("\n=== Test 1: Simple 2-layer MLP ===")
simple = nn.Sequential(
    nn.Linear(feat.shape[1], 256),
    nn.GELU(),
    nn.Linear(256, 128),
).to(device)
criterion = SupConLoss(temperature=0.07)
opt = torch.optim.Adam(simple.parameters(), lr=5e-3)

for step, batch in enumerate(loader):
    x = batch["features"].to(device)
    y = batch["labels"].to(device)
    emb = nn.functional.normalize(simple(x), dim=-1)
    loss = criterion(emb, y)
    opt.zero_grad()
    loss.backward()

    grad_norm = sum(p.grad.norm().item()**2 for p in simple.parameters() if p.grad is not None)**0.5
    pre_norm = simple(x).norm(dim=-1).mean().item()

    opt.step()
    if step <= 20 or step % 50 == 0:
        print(f"  step {step:3d}  loss={loss.item():.4f}  grad_norm={grad_norm:.4f}  pre_norm_mean={pre_norm:.2f}")
    if step >= 300:
        break

# Test 2: Does the ResidualMLP learn?
print("\n=== Test 2: ResidualMLP ===")
model = ResidualMLP(input_dim=feat.shape[1], hidden_dims=[512, 256, 128], embed_dim=128).to(device)
opt2 = torch.optim.Adam(model.parameters(), lr=5e-3)

for step, batch in enumerate(loader):
    x = batch["features"].to(device)
    y = batch["labels"].to(device)
    emb = model(x)
    loss = criterion(emb, y)
    opt2.zero_grad()
    loss.backward()

    grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5

    opt2.step()
    if step <= 20 or step % 50 == 0:
        print(f"  step {step:3d}  loss={loss.item():.4f}  grad_norm={grad_norm:.4f}")
    if step >= 300:
        break

# Test 3: Same but with raw 825 features (no LDA)
print("\n=== Test 3: ResidualMLP + raw features (no LDA) ===")
feat_raw, lab_raw, names_raw, _ = load_mat_dataset("/data/dataset/DFP", n_bins=825, use_lda=False)
print(f"Raw input stats: mean={feat_raw.mean():.4f}, std={feat_raw.std():.4f}")
loader_raw = make_dataloader(feat_raw, lab_raw, batch_size=1024)

model3 = ResidualMLP(input_dim=825, hidden_dims=[512, 256, 128], embed_dim=128).to(device)
opt3 = torch.optim.Adam(model3.parameters(), lr=5e-3)

for step, batch in enumerate(loader_raw):
    x = batch["features"].to(device)
    y = batch["labels"].to(device)
    emb = model3(x)
    loss = criterion(emb, y)
    opt3.zero_grad()
    loss.backward()

    grad_norm = sum(p.grad.norm().item()**2 for p in model3.parameters() if p.grad is not None)**0.5

    opt3.step()
    if step <= 20 or step % 50 == 0:
        print(f"  step {step:3d}  loss={loss.item():.4f}  grad_norm={grad_norm:.4f}")
    if step >= 300:
        break
