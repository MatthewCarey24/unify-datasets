"""Quick diagnostic: test different optimizers."""
import torch
import torch.nn as nn
from supcon.data import load_mat_dataset, make_dataloader
from supcon.model import ResidualMLP
from supcon.loss import SupConLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feat, lab, names, _ = load_mat_dataset("/data/dataset/DFP", n_bins=825, use_lda=True)
print(f"\nInput: {feat.shape}, {len(names)} classes")
loader = make_dataloader(feat, lab, batch_size=1024)
criterion = SupConLoss(temperature=0.07)


def run_test(name, model, optimizer, steps=300):
    print(f"\n=== {name} ===")
    model.to(device).train()
    for step, batch in enumerate(loader):
        x = batch["features"].to(device)
        y = batch["labels"].to(device)
        emb = model(x)
        loss = criterion(emb, y)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
        optimizer.step()
        if step <= 10 or step % 50 == 0:
            print(f"  step {step:3d}  loss={loss.item():.4f}  grad={grad_norm:.4f}")
        if step >= steps:
            break
    print(f"  FINAL loss={loss.item():.4f}")


# Test 1: Adam lr=5e-3 (current)
m1 = ResidualMLP(input_dim=33, hidden_dims=[128, 128], embed_dim=128)
run_test("Adam lr=5e-3, small model", m1, torch.optim.Adam(m1.parameters(), lr=5e-3))

# Test 2: SGD lr=0.5, momentum=0.9 (SupCon paper)
m2 = ResidualMLP(input_dim=33, hidden_dims=[128, 128], embed_dim=128)
run_test("SGD lr=0.5 mom=0.9, small model", m2,
         torch.optim.SGD(m2.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4))

# Test 3: Adam lr=0.1 (aggressive)
m3 = ResidualMLP(input_dim=33, hidden_dims=[128, 128], embed_dim=128)
run_test("Adam lr=0.1, small model", m3, torch.optim.Adam(m3.parameters(), lr=0.1))

# Test 4: SGD on raw 825 features
feat_raw, lab_raw, _, _ = load_mat_dataset("/data/dataset/DFP", n_bins=825, use_lda=False)
loader_raw = make_dataloader(feat_raw, lab_raw, batch_size=1024)

m4 = ResidualMLP(input_dim=825, hidden_dims=[256, 128], embed_dim=128)
m4.to(device).train()
opt4 = torch.optim.SGD(m4.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4)
print(f"\n=== SGD lr=0.5 mom=0.9, raw 825 features ===")
for step, batch in enumerate(loader_raw):
    x = batch["features"].to(device)
    y = batch["labels"].to(device)
    emb = m4(x)
    loss = criterion(emb, y)
    opt4.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.norm().item()**2 for p in m4.parameters() if p.grad is not None)**0.5
    opt4.step()
    if step <= 10 or step % 50 == 0:
        print(f"  step {step:3d}  loss={loss.item():.4f}  grad={grad_norm:.4f}")
    if step >= 300:
        break
print(f"  FINAL loss={loss.item():.4f}")
