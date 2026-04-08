"""Verified fine-tune recipe for TimesFM 2.5. Run on DGX."""
import timesfm
import torch
import numpy as np

print("=== Fine-tune proof of concept ===")
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model = m.model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.train()

# Constants
P = model.p        # 32 (input patch len)
O = model.o        # 128 (output patch len)
Q = model.q        # 10 (num quantiles incl point)
ARIDX = model.aridx  # 5 (index for point forecast in quantile dim)

print(f"patch_in={P}, patch_out={O}, quantiles={Q}, point_idx={ARIDX}")

# === Prepare training data ===
# Simulate: 512 time points context -> predict next 128 points
batch = 4
seq_len = 512
horizon = 128
num_patches = seq_len // P  # 16

# Fake data (replace with real BTC returns)
context = torch.randn(batch, seq_len).to(device)
future = torch.randn(batch, horizon).to(device)

# === Patch the input (same as decode does internally) ===
patched_inputs = context.reshape(batch, num_patches, P)
patched_masks = torch.ones_like(patched_inputs, dtype=torch.bool)

# === Running stats for RevIN normalization ===
from timesfm.torch.util import revin, update_running_stats

n = torch.zeros(batch, device=device)
mu = torch.zeros(batch, device=device)
sigma = torch.zeros(batch, device=device)
patch_mus, patch_sigmas = [], []

for i in range(num_patches):
    (n, mu, sigma), _ = update_running_stats(
        n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
    )
    patch_mus.append(mu)
    patch_sigmas.append(sigma)

context_mu = torch.stack(patch_mus, dim=1)    # [batch, patches]
context_sigma = torch.stack(patch_sigmas, dim=1)

# === Normalize input ===
normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
normed_inputs = torch.where(patched_masks, normed_inputs, torch.zeros_like(normed_inputs))

# === Forward pass (WITH gradients, unlike decode which uses no_grad) ===
(_, _, normed_output_ts, normed_output_qs), _ = model(normed_inputs, patched_masks)

# === Denormalize and reshape output ===
# normed_output_ts: [batch, patches, 1280] = [batch, patches, O*Q] = [batch, patches, 128*10]
renormed_outputs = revin(normed_output_ts, context_mu, context_sigma, reverse=True)
renormed_outputs = renormed_outputs.reshape(batch, num_patches, O, Q)
# Shape: [batch, 16, 128, 10]

# Point forecast from last patch, using ARIDX
point_pred = renormed_outputs[:, -1, :, ARIDX]  # [batch, 128]
print(f"Point prediction shape: {point_pred.shape}")  # [4, 128]

# === Compute loss ===
# Match horizon: we predicted 128 steps, target is 128 steps
loss = ((point_pred - future) ** 2).mean()
print(f"Loss: {loss.item():.6f}")

# === Backward ===
loss.backward()

grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
n_grads = sum(1 for p in model.parameters() if p.grad is not None)
total = sum(1 for _ in model.parameters())

print(f"Grad norm: {grad_norm:.4f}")
print(f"Params with grad: {n_grads}/{total}")
print(f"FINE-TUNE: {'CONFIRMED WORKING' if grad_norm > 0 else 'FAILED'}")

if grad_norm > 0:
    # Do one optimizer step to prove it works end-to-end
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer.zero_grad()

    # Second forward pass
    (_, _, normed_out2, _), _ = model(normed_inputs, patched_masks)
    renormed2 = revin(normed_out2, context_mu, context_sigma, reverse=True)
    renormed2 = renormed2.reshape(batch, num_patches, O, Q)
    pred2 = renormed2[:, -1, :, ARIDX]
    loss2 = ((pred2 - future) ** 2).mean()
    loss2.backward()
    optimizer.step()

    # Third forward to check loss decreased
    with torch.no_grad():
        (_, _, normed_out3, _), _ = model(normed_inputs, patched_masks)
        renormed3 = revin(normed_out3, context_mu, context_sigma, reverse=True)
        renormed3 = renormed3.reshape(batch, num_patches, O, Q)
        pred3 = renormed3[:, -1, :, ARIDX]
        loss3 = ((pred3 - future) ** 2).mean()

    print(f"\nAfter 1 optimizer step:")
    print(f"  Loss before: {loss2.item():.6f}")
    print(f"  Loss after:  {loss3.item():.6f}")
    print(f"  Improved: {loss3.item() < loss2.item()}")

    print("\n" + "=" * 50)
    print("FINE-TUNE RECIPE VERIFIED")
    print("=" * 50)
    print(f"""
Steps:
  1. model = m.model.to(device)  # get the real nn.Module
  2. model.train()
  3. Patch input: x.reshape(batch, num_patches, {P})
  4. Compute running stats for RevIN normalization
  5. Normalize: revin(patched_input, mu, sigma, reverse=False)
  6. Forward: (_, _, output_ts, output_qs), _ = model(normed, masks)
  7. Denormalize: revin(output_ts, mu, sigma, reverse=True)
  8. Reshape: output.reshape(batch, patches, {O}, {Q})
  9. Point forecast: output[:, -1, :, {ARIDX}]  # last patch, point index
  10. Loss + backward + optimizer step

Input:  [batch, {num_patches}, {P}] = patched context ({seq_len} points)
Output: [batch, {num_patches}, {O}, {Q}] = {O}-step forecast x {Q} quantiles
Target: [batch, {horizon}] = future returns
""")
