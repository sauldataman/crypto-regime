"""Probe TimesFM model internals for fine-tuning. Run on DGX in Docker."""
import timesfm
import torch
import inspect

print("=== Step 1: Load model ===")
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model = m.model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.train()
print(f"Device: {device}, Params: {sum(p.numel() for p in model.parameters()):,}")

print("\n=== Step 2: Forward with masks ===")
# forward(inputs: Tensor, masks: Tensor, decode_caches=None)
# inputs likely shape: [batch, seq_len] or [batch, seq_len, dim]
# masks likely shape: [batch, seq_len] (1=valid, 0=padding)

batch, seq_len = 2, 512
x = torch.randn(batch, seq_len).to(device)
masks = torch.ones(batch, seq_len).to(device)

try:
    out = model(x, masks)
    if isinstance(out, tuple):
        print(f"model(x, masks) -> tuple len={len(out)}")
        for i, o in enumerate(out):
            if isinstance(o, torch.Tensor):
                print(f"  [{i}] shape={o.shape}, dtype={o.dtype}, grad_fn={o.grad_fn is not None}")
            elif o is None:
                print(f"  [{i}] None")
            else:
                print(f"  [{i}] {type(o).__name__}")
    elif isinstance(out, torch.Tensor):
        print(f"model(x, masks) -> shape={out.shape}, grad_fn={out.grad_fn is not None}")
    else:
        print(f"model(x, masks) -> {type(out)}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Step 3: Check decode method ===")
decode_sig = inspect.signature(model.decode)
print(f"decode signature: {decode_sig}")

print("\n=== Step 4: Check forecast_naive ===")
fn_sig = inspect.signature(model.forecast_naive)
print(f"forecast_naive signature: {fn_sig}")

print("\n=== Step 5: Gradient flow test ===")
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
x_train = torch.randn(batch, seq_len).to(device)
masks_train = torch.ones(batch, seq_len).to(device)

optimizer.zero_grad()
try:
    out = model(x_train, masks_train)
    if isinstance(out, tuple):
        pred = out[0]
    else:
        pred = out

    # Create target matching output shape
    target = torch.randn_like(pred)
    loss = ((pred - target) ** 2).mean()
    loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())

    print(f"Output shape: {pred.shape}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Grad norm: {grad_norm:.6f}")
    print(f"Params with grad: {n_grads}/{total_params}")
    print(f"GRADIENT FLOW: {'YES !!!' if grad_norm > 0 else 'NO'}")

    if grad_norm > 0:
        # Show which layers have gradients
        print("\nGradient by layer:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                gn = param.grad.norm().item()
                if gn > 0:
                    print(f"  {name}: grad_norm={gn:.6f}")
                    break  # just show first few
        print("  ... (truncated)")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
