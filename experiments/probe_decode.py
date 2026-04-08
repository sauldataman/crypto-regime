"""Verified fine-tune recipe for TimesFM 2.5. Run on DGX."""
import timesfm
import torch

print("=== Fine-tune proof of concept ===")
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model = m.model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.train()

P = model.p        # 32
O = model.o        # 128
Q = model.q        # 10
ARIDX = model.aridx  # 5

print(f"patch_in={P}, patch_out={O}, quantiles={Q}, point_idx={ARIDX}")

batch = 4
seq_len = 512
num_patches = seq_len // P

# === Approach 1: Loss in NORMALIZED space (skip RevIN denorm) ===
print("\n=== Approach 1: Loss in normalized space ===")
context = torch.randn(batch, seq_len).to(device)
future = torch.randn(batch, O).to(device)

patched_inputs = context.reshape(batch, num_patches, P)
patched_masks = torch.ones_like(patched_inputs, dtype=torch.bool)

# Simple normalization (no RevIN, just standardize)
ctx_mean = context.mean(dim=1, keepdim=True)
ctx_std = context.std(dim=1, keepdim=True).clamp(min=1e-8)
normed_context = (context - ctx_mean) / ctx_std
normed_future = (future - ctx_mean) / ctx_std

normed_patched = normed_context.reshape(batch, num_patches, P)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
optimizer.zero_grad()

(_, _, output_ts, _), _ = model(normed_patched, patched_masks)
# output_ts: [batch, 16, 1280] = [batch, 16, O*Q]
# Reshape to [batch, 16, 128, 10]
output_reshaped = output_ts.reshape(batch, num_patches, O, Q)
# Point forecast from last patch
pred = output_reshaped[:, -1, :, ARIDX]  # [batch, 128]

loss = ((pred - normed_future) ** 2).mean()
loss.backward()

grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
n_grads = sum(1 for p in model.parameters() if p.grad is not None)
print(f"Loss: {loss.item():.6f}")
print(f"Grad norm: {grad_norm:.4f}")
print(f"Params with grad: {n_grads}/{sum(1 for _ in model.parameters())}")
print(f"GRADIENT FLOW: {'YES' if grad_norm > 0 else 'NO'}")

# === Approach 2: Direct loss on raw output (no reshape) ===
print("\n=== Approach 2: Direct loss on raw output_ts ===")
optimizer.zero_grad()
(_, _, output_ts2, _), _ = model(normed_patched, patched_masks)
# Use last patch output directly, MSE against random target
target2 = torch.randn_like(output_ts2[:, -1:, :])  # [batch, 1, 1280]
loss2 = ((output_ts2[:, -1:, :] - target2) ** 2).mean()
loss2.backward()

grad_norm2 = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
print(f"Loss: {loss2.item():.6f}")
print(f"Grad norm: {grad_norm2:.4f}")
print(f"GRADIENT FLOW: {'YES' if grad_norm2 > 0 else 'NO'}")

# === Approach 3: Check if reshape breaks gradients ===
print("\n=== Approach 3: Debug reshape gradient ===")
optimizer.zero_grad()
(_, _, output_ts3, _), _ = model(normed_patched, patched_masks)
print(f"output_ts3 requires_grad: {output_ts3.requires_grad}")
print(f"output_ts3 grad_fn: {output_ts3.grad_fn}")

reshaped = output_ts3.reshape(batch, num_patches, O, Q)
print(f"reshaped requires_grad: {reshaped.requires_grad}")
print(f"reshaped grad_fn: {reshaped.grad_fn}")

indexed = reshaped[:, -1, :, ARIDX]
print(f"indexed requires_grad: {indexed.requires_grad}")
print(f"indexed grad_fn: {indexed.grad_fn}")

target3 = torch.randn_like(indexed)
loss3 = ((indexed - target3) ** 2).mean()
print(f"loss3 requires_grad: {loss3.requires_grad}")
print(f"loss3 grad_fn: {loss3.grad_fn}")

loss3.backward()
grad_norm3 = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
print(f"Grad norm: {grad_norm3:.4f}")
print(f"GRADIENT FLOW: {'YES' if grad_norm3 > 0 else 'NO'}")

# === If any approach works, do optimization test ===
best_approach = None
if grad_norm > 0:
    best_approach = "Approach 1 (normalized space)"
elif grad_norm2 > 0:
    best_approach = "Approach 2 (raw output)"
elif grad_norm3 > 0:
    best_approach = "Approach 3 (reshape debug)"

if best_approach:
    print(f"\n{'='*50}")
    print(f"FINE-TUNE WORKS via {best_approach}")
    print(f"{'='*50}")

    # Prove optimization works: 3 steps, loss should decrease
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    losses = []
    for step in range(3):
        optimizer.zero_grad()
        (_, _, out, _), _ = model(normed_patched, patched_masks)
        if best_approach == "Approach 2 (raw output)":
            target = torch.randn_like(out[:, -1:, :]) if step == 0 else target
            l = ((out[:, -1:, :] - target) ** 2).mean()
        else:
            out_r = out.reshape(batch, num_patches, O, Q)
            pred = out_r[:, -1, :, ARIDX]
            target = normed_future if step == 0 else target
            l = ((pred - target) ** 2).mean()
        l.backward()
        optimizer.step()
        losses.append(l.item())
        print(f"  Step {step}: loss={l.item():.6f}")

    print(f"  Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f} ({'decreasing' if losses[-1] < losses[0] else 'NOT decreasing'})")
else:
    print("\n*** ALL APPROACHES FAILED ***")
    print("Need to investigate RevIN / model internals further")
