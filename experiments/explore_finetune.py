"""
Phase II-E: Fine-tune Exploration (runs on DGX)

Probes TimesFM internals to determine the best fine-tuning approach.
Run this ONCE on DGX to get a report, then decide which path to take.

Three paths explored:
  Path C: Try installing TimesFMFinetuner from PR #223
  Path B: Probe model internals (nn.Module structure, forward pass)
  Path A: Check JAX availability and fine-tune API

Usage:
  python experiments/explore_finetune.py

Output:
  results/finetune_exploration.json
  Console report with recommended path
"""
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

report = {
    "path_c_finetuner_api": {},
    "path_b_model_internals": {},
    "path_a_jax": {},
    "recommendation": None,
}


def explore_path_c():
    """Path C: Check if TimesFMFinetuner exists (current install or PR #223)."""
    logger.info("=" * 60)
    logger.info("PATH C: TimesFMFinetuner API")
    logger.info("=" * 60)

    result = {"available": False, "finetuner_names": [], "source": None}

    # Check current installation
    try:
        import timesfm
        attrs = dir(timesfm)
        finetuner_names = [a for a in attrs if "finetun" in a.lower()]

        if finetuner_names:
            result["available"] = True
            result["finetuner_names"] = finetuner_names
            result["source"] = "current_install"
            logger.info("  FOUND in current install: %s", finetuner_names)

            # Try to inspect the Finetuner
            for name in finetuner_names:
                cls = getattr(timesfm, name)
                logger.info("  %s: %s", name, type(cls))
                if callable(cls):
                    import inspect
                    try:
                        sig = inspect.signature(cls)
                        logger.info("    Signature: %s", sig)
                        result["signature"] = str(sig)
                    except (ValueError, TypeError):
                        logger.info("    Could not inspect signature")
        else:
            logger.info("  NOT found in current install")
            logger.info("  Available attrs: %s", [a for a in attrs if not a.startswith("_")])

    except ImportError:
        logger.error("  timesfm not installed")
        result["error"] = "timesfm not installed"

    # Check submodules
    try:
        import timesfm
        submodules = []
        for attr_name in dir(timesfm):
            attr = getattr(timesfm, attr_name, None)
            if hasattr(attr, "__module__") and "timesfm" in str(getattr(attr, "__module__", "")):
                submodules.append(attr_name)

        # Try importing finetuning module directly
        finetuning_modules = []
        for mod_name in ["timesfm.finetuning_torch", "timesfm.finetuning", "timesfm.finetuner",
                         "timesfm.torch.finetuning", "timesfm.pytorch.finetuning"]:
            try:
                mod = __import__(mod_name, fromlist=[""])
                finetuning_modules.append(mod_name)
                logger.info("  Found module: %s -> %s", mod_name, dir(mod))
                result["finetuning_module"] = mod_name
                result["finetuning_module_attrs"] = [a for a in dir(mod) if not a.startswith("_")]
            except ImportError:
                pass

        if finetuning_modules:
            result["available"] = True
            result["source"] = "submodule"
            logger.info("  Finetuning module found via submodule import")

        result["submodules"] = submodules

    except Exception as e:
        logger.warning("  Submodule check failed: %s", e)

    report["path_c_finetuner_api"] = result
    return result


def explore_path_b():
    """Path B: Probe model internals for manual forward-pass fine-tuning."""
    logger.info("=" * 60)
    logger.info("PATH B: Model Internals")
    logger.info("=" * 60)

    result = {
        "is_nn_module": False,
        "children": [],
        "forward_works": False,
        "forward_error": None,
        "parameters_count": 0,
        "trainable_params": 0,
    }

    try:
        import timesfm
        import torch

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )

        # Basic type info
        result["model_type"] = str(type(model))
        result["is_nn_module"] = isinstance(model, torch.nn.Module)
        logger.info("  Model type: %s", type(model))
        logger.info("  Is nn.Module: %s", result["is_nn_module"])

        # Children (submodules)
        if isinstance(model, torch.nn.Module):
            children = [(name, type(child).__name__) for name, child in model.named_children()]
            result["children"] = children
            logger.info("  Children:")
            for name, ctype in children:
                logger.info("    %s: %s", name, ctype)

            # Parameter count
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            result["parameters_count"] = total
            result["trainable_params"] = trainable
            logger.info("  Parameters: %d total, %d trainable", total, trainable)

            # Named parameters (first few)
            param_names = [n for n, _ in model.named_parameters()]
            result["param_names_sample"] = param_names[:20]
            logger.info("  First 20 param names: %s", param_names[:20])

        # Try different forward pass patterns
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.train()

        test_input = torch.randn(2, 512).to(device)  # batch=2, context=512

        forward_patterns = [
            ("model(x)", lambda: model(test_input)),
            ("model(x, horizon=1)", lambda: model(test_input, horizon=1)),
            ("model.forward(x)", lambda: model.forward(test_input)),
        ]

        # Check for backbone/head pattern
        if hasattr(model, "backbone"):
            forward_patterns.append(
                ("model.backbone(x)", lambda: model.backbone(test_input))
            )
        if hasattr(model, "encoder"):
            forward_patterns.append(
                ("model.encoder(x)", lambda: model.encoder(test_input))
            )

        # Check for _forward_impl or similar
        for attr in dir(model):
            if "forward" in attr.lower() and attr != "forward" and not attr.startswith("_"):
                forward_patterns.append(
                    (f"model.{attr}(x)", lambda a=attr: getattr(model, a)(test_input))
                )

        result["forward_attempts"] = []
        for pattern_name, fn in forward_patterns:
            try:
                with torch.no_grad():
                    out = fn()
                out_info = {
                    "pattern": pattern_name,
                    "success": True,
                    "output_type": str(type(out)),
                }
                if isinstance(out, torch.Tensor):
                    out_info["shape"] = list(out.shape)
                    out_info["requires_grad"] = out.requires_grad
                elif isinstance(out, tuple):
                    out_info["tuple_len"] = len(out)
                    out_info["element_types"] = [str(type(o)) for o in out]
                    if isinstance(out[0], torch.Tensor):
                        out_info["first_shape"] = list(out[0].shape)

                result["forward_attempts"].append(out_info)
                logger.info("  %s -> SUCCESS: %s", pattern_name, out_info)

                if not result["forward_works"]:
                    result["forward_works"] = True
                    result["working_pattern"] = pattern_name

            except Exception as e:
                result["forward_attempts"].append({
                    "pattern": pattern_name,
                    "success": False,
                    "error": str(e)[:200],
                })
                logger.info("  %s -> FAILED: %s", pattern_name, str(e)[:100])

        # Test gradient flow on working pattern
        if result["forward_works"]:
            logger.info("\n  Testing gradient flow...")
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
            test_input_grad = torch.randn(2, 512, requires_grad=False).to(device)
            target = torch.randn(2, 1).to(device)

            optimizer.zero_grad()
            try:
                # Use the working pattern
                if result["working_pattern"] == "model(x)":
                    out = model(test_input_grad)
                elif result["working_pattern"] == "model(x, horizon=1)":
                    out = model(test_input_grad, horizon=1)
                elif result["working_pattern"] == "model.forward(x)":
                    out = model.forward(test_input_grad)
                else:
                    out = model(test_input_grad)

                if isinstance(out, tuple):
                    out = out[0]
                if isinstance(out, torch.Tensor) and out.dim() > 1:
                    pred = out[:, :1]  # take first output step
                else:
                    pred = out

                loss = ((pred - target) ** 2).mean()
                loss.backward()

                grad_norm = sum(
                    p.grad.norm().item() for p in model.parameters()
                    if p.grad is not None
                )
                n_grads = sum(1 for p in model.parameters() if p.grad is not None)

                result["gradient_flow"] = {
                    "loss": float(loss.item()),
                    "grad_norm": float(grad_norm),
                    "n_params_with_grad": n_grads,
                    "total_params": len(list(model.parameters())),
                    "works": grad_norm > 0,
                }
                logger.info("  Gradient flow: loss=%.6f, grad_norm=%.6f, %d/%d params have grad",
                            loss.item(), grad_norm, n_grads, len(list(model.parameters())))

            except Exception as e:
                result["gradient_flow"] = {"works": False, "error": str(e)[:200]}
                logger.info("  Gradient flow test FAILED: %s", str(e)[:100])

    except ImportError as e:
        logger.error("  Cannot probe: %s", e)
        result["error"] = str(e)
    except Exception as e:
        logger.error("  Unexpected error: %s", e)
        result["error"] = str(e)

    report["path_b_model_internals"] = result
    return result


def explore_path_a():
    """Path A: Check JAX availability and fine-tune support."""
    logger.info("=" * 60)
    logger.info("PATH A: JAX Fine-tune")
    logger.info("=" * 60)

    result = {"jax_available": False, "jax_gpu": False, "timesfm_jax": False}

    # Check JAX
    try:
        import jax
        result["jax_available"] = True
        result["jax_version"] = jax.__version__
        devices = jax.devices()
        result["jax_devices"] = [str(d) for d in devices]
        result["jax_gpu"] = any("gpu" in str(d).lower() for d in devices)
        logger.info("  JAX %s, devices: %s", jax.__version__, devices)
    except ImportError:
        logger.info("  JAX not installed")
        result["jax_install_cmd"] = 'pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'

    # Check TimesFM JAX
    try:
        from timesfm import timesfm_2p5
        result["timesfm_jax"] = True
        jax_attrs = dir(timesfm_2p5)
        result["jax_model_attrs"] = [a for a in jax_attrs if not a.startswith("_")]
        logger.info("  TimesFM JAX module found: %s", result["jax_model_attrs"][:10])

        # Check for covariates support
        covariate_attrs = [a for a in jax_attrs if "covariate" in a.lower() or "xreg" in a.lower()]
        result["covariate_support"] = covariate_attrs
        logger.info("  Covariate-related: %s", covariate_attrs)

    except ImportError as e:
        logger.info("  TimesFM JAX module not available: %s", e)
    except Exception as e:
        logger.info("  TimesFM JAX check error: %s", e)

    report["path_a_jax"] = result
    return result


def recommend():
    """Generate recommendation based on exploration results."""
    logger.info("=" * 60)
    logger.info("RECOMMENDATION")
    logger.info("=" * 60)

    pc = report["path_c_finetuner_api"]
    pb = report["path_b_model_internals"]
    pa = report["path_a_jax"]

    # Priority: C > B > A
    if pc.get("available"):
        rec = "PATH_C"
        reason = f"TimesFMFinetuner found: {pc.get('finetuner_names') or pc.get('finetuning_module')}"
        action = "Use the native Finetuner API. See PR #223 for usage examples."
    elif pb.get("forward_works") and pb.get("gradient_flow", {}).get("works"):
        rec = "PATH_B"
        pattern = pb.get("working_pattern", "unknown")
        grad_norm = pb.get("gradient_flow", {}).get("grad_norm", 0)
        reason = f"Manual forward pass works via '{pattern}', gradient norm = {grad_norm:.4f}"
        action = (f"Use '{pattern}' in training loop. Gradient flows correctly. "
                  "Rewrite phase2_finetune.py to use this pattern.")
    elif pb.get("forward_works"):
        rec = "PATH_B_PARTIAL"
        reason = f"Forward pass works via '{pb.get('working_pattern')}' but gradient flow untested/failed"
        action = "Forward pass works but gradients need investigation. Check if model was loaded in eval mode."
    elif pa.get("jax_available") and pa.get("jax_gpu"):
        rec = "PATH_A"
        reason = "JAX with GPU available. PyTorch paths failed."
        action = "Switch to JAX for fine-tuning. Supports covariates natively."
    elif pa.get("jax_available"):
        rec = "PATH_A_CPU"
        reason = "JAX available but no GPU. Fine-tuning will be slow."
        action = "Install JAX CUDA: " + pa.get("jax_install_cmd", "pip install jax[cuda12]")
    else:
        rec = "BLOCKED"
        reason = "No viable fine-tune path found. Neither Finetuner API, manual forward, nor JAX available."
        action = ("Options: (1) Install from PR #223 branch, "
                  "(2) Install JAX with CUDA, "
                  "(3) Wait for TimesFMFinetuner to be released on PyPI")

    report["recommendation"] = {
        "path": rec,
        "reason": reason,
        "action": action,
    }

    logger.info("  Path: %s", rec)
    logger.info("  Reason: %s", reason)
    logger.info("  Action: %s", action)

    # Summary table
    logger.info("\n  ╔═══════════════════════════════════════════╗")
    logger.info("  ║  FINE-TUNE EXPLORATION SUMMARY             ║")
    logger.info("  ╠═══════════════════════════════════════════╣")
    logger.info("  ║  Path C (Finetuner API):  %-16s  ║", "YES" if pc.get("available") else "NO")
    logger.info("  ║  Path B (Manual forward): %-16s  ║",
                "YES" if pb.get("forward_works") else "NO")
    logger.info("  ║  Path B (Gradient flow):  %-16s  ║",
                "YES" if pb.get("gradient_flow", {}).get("works") else "NO")
    logger.info("  ║  Path A (JAX + GPU):      %-16s  ║",
                "YES" if pa.get("jax_gpu") else "NO")
    logger.info("  ╠═══════════════════════════════════════════╣")
    logger.info("  ║  RECOMMENDED: %-28s  ║", rec)
    logger.info("  ╚═══════════════════════════════════════════╝")


def main():
    logger.info("=" * 60)
    logger.info("Phase II-E: Fine-tune Exploration")
    logger.info("=" * 60)

    explore_path_c()
    explore_path_b()
    explore_path_a()
    recommend()

    # Save report
    out_path = RESULTS / "finetune_exploration.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("\nFull report: %s", out_path)


if __name__ == "__main__":
    main()
