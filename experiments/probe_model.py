"""Probe TimesFM model internals for fine-tuning. Run on DGX in Docker."""
import timesfm
import torch
import inspect

print("=== Step 1: Find the real nn.Module ===")
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model = m.model  # the real nn.Module
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.train()

print(f"Device: {device}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Children: {[(n, type(c).__name__) for n,c in model.named_children()]}")

print("\n=== Step 2: Forward signature ===")
sig = inspect.signature(model.forward)
print(f"forward signature: {sig}")

methods = [x for x in dir(model) if not x.startswith("_") and callable(getattr(model, x))]
print(f"Methods: {methods[:30]}")

print("\n=== Step 3: Try forward pass ===")
x = torch.randn(2, 512).to(device)

patterns = [
    ("model(x)", lambda: model(x)),
    ("model(x, horizon=30)", lambda: model(x, horizon=30)),
    ("model.forward(x)", lambda: model.forward(x)),
]

for name, fn in patterns:
    try:
        out = fn()
        if isinstance(out, torch.Tensor):
            print(f"{name} -> Tensor shape={out.shape}, grad_fn={out.grad_fn is not None}")
        elif isinstance(out, tuple):
            print(f"{name} -> tuple len={len(out)}")
            for i, o in enumerate(out):
                if isinstance(o, torch.Tensor):
                    print(f"  [{i}] Tensor shape={o.shape}, grad_fn={o.grad_fn is not None}")
                else:
                    print(f"  [{i}] {type(o).__name__}")
        else:
            print(f"{name} -> {type(out)}")
    except Exception as e:
        print(f"{name} FAILED: {e}")

print("\n=== Step 4: Gradient flow test ===")
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
x_train = torch.randn(2, 512).to(device)
target = torch.randn(2, 30).to(device)

optimizer.zero_grad()
try:
    out = model(x_train)
    if isinstance(out, tuple):
        pred = out[0]
    else:
        pred = out
    if pred.shape[-1] > 30:
        pred = pred[:, :30]
    elif pred.shape[-1] < 30:
        target = target[:, :pred.shape[-1]]
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"Loss: {loss.item():.6f}")
    print(f"Grad norm: {grad_norm:.6f}")
    print(f"Params with grad: {n_grads}/{total_params}")
    print(f"GRADIENT FLOW: {'YES' if grad_norm > 0 else 'NO'}")
except Exception as e:
    print(f"Gradient test FAILED: {e}")
    import traceback
    traceback.print_exc()
