"""CPU loader shared by the bounded backward probe and fixed QAT trajectory."""
import ast
from dataclasses import replace
from pathlib import Path
import sys
import types

import torch


def load_model(checkpoint, kernel=None):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from pysrc import kda_reference as ref
    if kernel:
        import cpu_ops
        cpu_ops.load_kernel(kernel)
        cpu_ops.install_quantizer()
        import pysrc.attention_function as attention_module
        attention_module.sdpa_attention_function = cpu_ops.window_attention

    counts = {"kda": 0, "convolution": 0, "gated_norm": 0}

    def chunk_kda(**kw):
        assert kw["use_qk_l2norm_in_kernel"] and kw["use_gate_in_kernel"]
        counts["kda"] += 1
        fn = cpu_ops.kda if kernel else ref.kda_recurrent_reference
        extra = {} if kernel else {"dtype": torch.float32}
        return fn(
            kw["q"], kw["k"], kw["v"], kw["g"], kw["beta"],
            kw["A_log"], kw["dt_bias"], kw["cu_seqlens"],
            **extra), None

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
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.train()
    return model, counts
