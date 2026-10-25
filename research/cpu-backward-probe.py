#!/usr/bin/env python3
"""One bounded FP32 backward through the original 6M QAT model.

Uses the repository's PyTorch KDA/conv/norm references and SDPA, with the
unchanged cooldown quantizers. Input files are a prepared-stream prefix:
token IDs, matching next-token priors, and original int32 article boundaries.
This measures a complete small article; it performs no optimizer update.
"""

import argparse
import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-tokens", type=int, default=1025)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--cpu-seconds", type=int, default=300)
    args = parser.parse_args()
    assert 1 <= args.min_tokens <= args.max_tokens <= 4096
    assert 10 <= args.cpu_seconds <= 300
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        os.environ.setdefault(key, str(args.output.parent / key.lower()))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds - 2, args.cpu_seconds))
    # Conservative address-space ceiling also bounds this process's RSS.
    resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
    started_wall, started_cpu = time.monotonic(), time.process_time()
    result = {"status": "starting", "optimizer_updates": 0,
              "cpu_limit_seconds": args.cpu_seconds, "address_space_limit_gib": 6,
              "threads": 1, "python": sys.version}

    def save(stage, **values):
        result.update(stage=stage, **values)
        result.update(elapsed_wall_seconds=time.monotonic() - started_wall,
                      elapsed_cpu_seconds=time.process_time() - started_cpu,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(args.output)
        print(json.dumps({"stage": stage, **values}), flush=True)

    def cpu_limit(signum, frame):
        save("cpu_limit", status="stopped")
        raise SystemExit(124)

    signal.signal(signal.SIGXCPU, cpu_limit)
    save("importing")
    try:
        import numpy as np
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo))
        from pysrc import kda_reference as ref

        counts = {"kda": 0, "convolution": 0, "gated_norm": 0}

        def chunk_kda(**kw):
            assert kw["use_qk_l2norm_in_kernel"] and kw["use_gate_in_kernel"]
            counts["kda"] += 1
            return ref.kda_recurrent_reference(
                kw["q"], kw["k"], kw["v"], kw["g"], kw["beta"],
                kw["A_log"], kw["dt_bias"], kw["cu_seqlens"],
                dtype=torch.float32), None

        def causal_conv1d(*, x, weight, activation, cu_seqlens):
            assert activation == "silu"
            counts["convolution"] += 1
            return ref.causal_conv1d_silu_reference(x, weight, cu_seqlens), None

        class FusedRMSNormGated(torch.nn.Module):
            def __init__(self, hidden_size, activation):
                super().__init__()
                assert activation == "sigmoid"
                self.weight = torch.nn.Parameter(torch.ones(hidden_size))

            def forward(self, x, gate):
                counts["gated_norm"] += 1
                return ref.fused_rmsnorm_gated_reference(x, gate, self.weight)

        # Local import adapters preserve all production source and checkpoint keys.
        for name in ("fla", "fla.ops", "fla.ops.kda", "fla.modules",
                     "fla.modules.conv", "fla.modules.conv.causal_conv1d"):
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        sys.modules["fla.ops.kda"].chunk_kda = chunk_kda
        sys.modules["fla.modules"].FusedRMSNormGated = FusedRMSNormGated
        sys.modules["fla.modules.conv.causal_conv1d"].causal_conv1d = causal_conv1d

        from pysrc.model import Transformer, TransformerConfig, MatmulQuantization
        from pysrc.quantization import Quantization

        # Reuse cooldown's exact config without its CUDA/DDP/logger imports.
        recipe = repo / "training_recipes/cooldown.py"
        tree = ast.parse(recipe.read_text())
        main_ast = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "main")
        assignments = {}
        for node in main_ast.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                assignments.setdefault(node.target.id, node.value)
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                # Keep the constructor before the later raw_model.to(device).
                assignments.setdefault(node.targets[0].id, node.value)
        ns = dict(torch=torch, TransformerConfig=TransformerConfig,
                  MatmulQuantization=MatmulQuantization, Quantization=Quantization)
        for name in ("quantization_block_size", "weight_quantization",
                     "activation_quantization", "matmul_quantization"):
            ns[name] = eval(compile(ast.Expression(assignments[name]), str(recipe), "eval"), ns)
        original_cfg = eval(compile(ast.Expression(assignments["raw_model"].args[0]),
                                    str(recipe), "eval"), ns)
        cfg = replace(original_cfg, attention_implementation="sdpa")
        assert cfg.window_size == 1024 and cfg.n_layers == 12
        assert sum(cfg.kimi_linear) == 9 and cfg.gradient_checkpointing
        model = Transformer(cfg)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        model.train()
        save("model_loaded", torch_version=torch.__version__,
             checkpoint_tensors=len(model.state_dict()),
             trainable_tensors=sum(p.requires_grad for p in model.parameters()),
             parameters=sum(p.numel() for p in model.parameters()),
             config_source="training_recipes/cooldown.py", query_key_norm_gain=cfg.query_key_norm_gain,
             window=cfg.window_size, activation_dtype=str(cfg.activation_dtype),
             gradient_checkpointing=cfg.gradient_checkpointing, attention="sdpa")

        tokens = np.fromfile(args.tokens, dtype=np.uint8)
        bounds = np.fromfile(args.boundaries, dtype="<i4")
        assert bounds[0] == 0 and (np.diff(bounds) > 0).all()
        assert args.priors.stat().st_size == tokens.size * 205 * 2
        candidates = [(i, int(s), int(e)) for i, (s, e) in enumerate(zip(bounds, bounds[1:]))
                      if e <= len(tokens) and args.min_tokens <= e - s - 1 <= args.max_tokens]
        assert candidates, "no complete article in the requested token range"
        article, start, end = candidates[0]
        n = end - start - 1
        priors = np.memmap(args.priors, dtype="<f2", mode="r", shape=(len(tokens), 205))
        x = torch.from_numpy(tokens[start:end - 1].astype(np.int64)).unsqueeze(0)
        y = torch.from_numpy(tokens[start + 1:end].astype(np.int64)).unsqueeze(0)
        p = torch.from_numpy(np.array(priors[start:end - 1])).unsqueeze(0)
        assert x.max() < 205 and y.max() < 205 and torch.isfinite(p).all()
        del priors
        save("forward_start", input_kind="real_complete_prepared_prefix_article",
             article_index=article, article_start=start, article_end=end,
             input_tokens=n, valid_targets=n, rope_offset=0,
             full_1024_window_exercised=n > 1024,
             original_article_boundaries=True, prior_rows="[start:end-1] predicts [start+1:end]")
        w, c = time.monotonic(), time.process_time()
        loss = model(input_tokens=x, output_tokens=y, priors=p,
                     loss_mask=torch.ones_like(x, dtype=torch.bool),
                     cumulative_sequence_lengths=torch.tensor([[0, n]], dtype=torch.int32))
        assert torch.isfinite(loss).item()
        save("backward_start", loss_sum=loss.item(), mean_nll=loss.item() / n,
             forward_wall_seconds=time.monotonic() - w,
             forward_cpu_seconds=time.process_time() - c)
        w, c = time.monotonic(), time.process_time()
        loss.backward()
        backward_wall, backward_cpu = time.monotonic() - w, time.process_time() - c
        used, missing, nonfinite, nonzero = [], [], [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                missing.append(name)
                continue
            used.append(name)
            if not torch.isfinite(parameter.grad).all().item():
                nonfinite.append(name)
            if torch.count_nonzero(parameter.grad).item():
                nonzero.append(name)
        save("gradients_checked", backward_wall_seconds=backward_wall,
             backward_cpu_seconds=backward_cpu, used_gradient_tensors=len(used),
             nonzero_gradient_tensors=len(nonzero), missing_gradient_names=missing,
             nonfinite_gradient_names=nonfinite, operator_calls=counts)
        assert used and not missing and not nonfinite
        save("complete", status="passed")
    except Exception as exc:
        save("failed", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
