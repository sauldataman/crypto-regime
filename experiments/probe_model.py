"""Probe TimesFM model internals for fine-tuning. Run on DGX in Docker."""
import timesfm
import torch

print("=== Load model ===")
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model = m.model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.train()

# Model constants from source code:
# input_patch_len = 32, output_patch_len = 128
# tokenizer input = cat([inputs, masks], dim=-1) -> dim = 32+32 = 64
# So inputs shape = [batch, num_patches, 32]
# masks shape = [batch, num_patches, 32]

patch_len = model.p  # 32
output_patch_len = model.o  # 128
num_quantiles = model.q  # 10

print(f"patch_len={patch_len}, output_patch_len={output_patch_len}, num_quantiles={num_quantiles}")
print(f"model_dim={model.md}, num_layers={model.x}, num_heads={model.h}")

# 512 time points = 16 patches of 32
seq_len = 512
num_patches = seq_len // patch_len  # 16
batch = 2

print(f"\n=== Forward pass: batch={batch}, patches={num_patches}, patch_len={patch_len} ===")
x = torch.randn(batch, num_patches, patch_len).to(device)
masks = torch.ones(batch, num_patches, patch_len).to(device)

try:
    out, caches = model(x, masks)
    # out = (input_embeddings, output_embeddings, output_ts, output_quantile_spread)
    input_emb, output_emb, output_ts, output_qs = out
    print(f"input_embeddings:  {input_emb.shape}")
    print(f"output_embeddings: {output_emb.shape}")
    print(f"output_ts (point): {output_ts.shape}")
    print(f"output_qs (quant): {output_qs.shape}")
    print(f"grad_fn on output_ts: {output_ts.grad_fn is not None}")
    print("FORWARD: SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print(f"\n=== Gradient flow test ===")
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

x_train = torch.randn(batch, num_patches, patch_len).to(device)
masks_train = torch.ones(batch, num_patches, patch_len).to(device)
# Target: predict the next output_patch_len values
# output_ts shape should be [batch, num_patches, output_patch_len]
# We'll use MSE on point forecast

optimizer.zero_grad()
try:
    (_, _, output_ts, output_qs), _ = model(x_train, masks_train)
    print(f"output_ts shape: {output_ts.shape}")

    # Use last patch's point forecast as prediction target
    target = torch.randn_like(output_ts)
    loss = ((output_ts - target) ** 2).mean()
    loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())

    print(f"Loss: {loss.item():.6f}")
    print(f"Grad norm: {grad_norm:.6f}")
    print(f"Params with grad: {n_grads}/{total_params}")

    if grad_norm > 0:
        print("*** GRADIENT FLOW: YES ***")
        print("\nGradients by component:")
        components = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                comp = name.split(".")[0]
                if comp not in components:
                    components[comp] = 0
                components[comp] += param.grad.norm().item()
        for comp, gn in sorted(components.items(), key=lambda x: -x[1]):
            print(f"  {comp}: {gn:.6f}")

        print("\n*** TimesFM CAN BE FINE-TUNED via m.model ***")
        print("Training loop recipe:")
        print("  model = m.model.to(device)")
        print("  model.train()")
        print("  x = [batch, num_patches, 32]  # patched input")
        print("  masks = ones_like(x)  # all valid")
        print("  (_, _, point_pred, quantile_pred), _ = model(x, masks)")
        print("  loss = your_loss(point_pred, target)")
        print("  loss.backward()")
    else:
        print("*** GRADIENT FLOW: NO ***")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
