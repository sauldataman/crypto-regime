"""Probe TimesFM decode path to understand output mapping. Run on DGX."""
import timesfm
import torch
import inspect

m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model = m.model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).train()

print("=== Output projection configs ===")
cfg_p = model.config.output_projection_point
print(f"Point: input={cfg_p.input_dims}, hidden={cfg_p.hidden_dims}, output={cfg_p.output_dims}")
cfg_q = model.config.output_projection_quantiles
print(f"Quant: input={cfg_q.input_dims}, hidden={cfg_q.hidden_dims}, output={cfg_q.output_dims}")

print(f"\nmodel_dim={model.md}, output_patch_len={model.o}, output_quantile_len={model.os}")
print(f"num_quantiles={model.q}, decode_index={model.aridx}")

print("\n=== decode signature ===")
print(inspect.signature(model.decode))

# Read decode source
print("\n=== decode source ===")
print(inspect.getsource(model.decode))

print("\n=== Forward test ===")
batch, num_patches, patch_len = 2, 16, 32
x = torch.randn(batch, num_patches, patch_len).to(device)
masks = torch.ones(batch, num_patches, patch_len).to(device)

(inp_emb, out_emb, out_ts, out_qs), caches = model(x, masks)
print(f"output_ts: {out_ts.shape}")  # [2, 16, 1280]
print(f"output_qs: {out_qs.shape}")  # [2, 16, 10240]

# The decode method likely:
# 1. Runs forward to get embeddings
# 2. Takes the last patch's output
# 3. Reshapes 1280 -> [output_patch_len] for point forecast
# 4. Auto-regressively generates horizon steps

# Try decode with gradient
print("\n=== decode with gradient ===")
try:
    result = model.decode(horizon=30, inputs=x, masks=masks)
    if isinstance(result, tuple):
        for i, r in enumerate(result):
            if isinstance(r, torch.Tensor):
                print(f"  [{i}] shape={r.shape}, grad_fn={r.grad_fn is not None}")
            elif isinstance(r, list):
                print(f"  [{i}] list len={len(r)}")
            else:
                print(f"  [{i}] {type(r).__name__}")
    print("DECODE: SUCCESS")
except Exception as e:
    print(f"DECODE FAILED: {e}")
    import traceback
    traceback.print_exc()
