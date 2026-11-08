#!/usr/bin/env python3
"""Small stdlib q7/2KL/cache-miss/source check; optional PyTorch toy VJP.

No complete model, native executable, real cache or optimizer is used here.
The complete cached-piece trainer preflight supplies the deployment check.
"""
import argparse
import ast
from collections import Counter
from fractions import Fraction
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

from q7_teacher_only import (check_teacher_cache_identity, data_objective,
                             q7_recipe, quant7_scale_bits, quant7_value,
                             readonly_teacher_logits, teacher_target_identity)


def power2(exponent):
    return Fraction(2 ** exponent) if exponent >= 0 else Fraction(1, 2 ** -exponent)


def bf16_fraction(bits):
    exponent = (bits >> 7) & 255
    assert 0 < exponent < 255
    value = (128 + (bits & 127)) * power2(exponent - 134)
    return -value if bits & 0x8000 else value


def rational_bf16_rne(value):
    """Independent rational-value BF16 oracle for the fixed normal fixtures."""
    sign = 0x8000 if value < 0 else 0
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < power2(exponent):
        exponent -= 1
    scaled = value / power2(exponent) * 128
    q, remainder = divmod(scaled.numerator, scaled.denominator)
    q += int(2 * remainder > scaled.denominator or
             (2 * remainder == scaled.denominator and q & 1))
    if q == 256:
        exponent += 1
        q = 128
    return sign | ((exponent + 127) << 7) | (q - 128)


def softmax(values):
    maximum = max(values)
    exp = [math.exp(v - maximum) for v in values]
    total = math.fsum(exp)
    return [v / total for v in exp]


def kl_value(logits, teacher):
    student = softmax(logits)
    return math.fsum(q * (math.log(q) - math.log(p)) for p, q in zip(student, teacher))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trainer", type=Path, default=Path(__file__).with_name("integer-coordinate-qat.py"))
    p.add_argument("--torch", action="store_true", help="also run a 205-way toy autograd VJP when PyTorch is available")
    a = p.parse_args()
    started = time.process_time()
    source = a.trainer.read_text()
    tree = ast.parse(source, filename=str(a.trainer))
    spec = importlib.util.spec_from_file_location("q7_checked_integer_trainer", a.trainer)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    legacy = dict(method="fixed-scale-integer-coordinate-v1", epochs=64, lr=.01,
                  betas=[.9, .99], eps=1e-8, weight_decay=0, tau=.5,
                  teacher_kl_coefficient=1, full_targets=586459020, seed=245)
    assert trainer.RECIPE == legacy and trainer.SOFT_LEVELS is None
    recipe = q7_recipe(legacy)
    assert trainer.RECIPE == legacy and recipe["epochs"] == 32 and recipe["seed"] == 414
    assert recipe["soft_symbols"] == list(range(-3, 4)) and recipe["hard_integer_range"] == [-3, 3]
    assert recipe["raw_trainable_tensors"] == recipe["label_ce"] == recipe["final_hidden_mse"] == 0
    assert recipe["teacher_kl_coefficient"] == 2 and recipe["evaluation_every"] == 4
    for key in ("lr", "betas", "eps", "weight_decay", "tau", "full_targets"):
        assert recipe[key] == legacy[key]
    mapped = [quant7_value(q) for q in range(-7, 8)]
    assert mapped == [-3, -3, -2, -2, -1, -1, 0, 0, 0, 1, 1, 2, 2, 3, 3]
    fixtures = {0x3f80: 0x4015, 0x3f81: 0x4016, 0x3f87: 0x401e,
                0x3fdb: 0x4080, 0xbf81: 0xc016}
    for bits in (0x27d9, 0xa7d9, 0x2780, 0xa780):
        fixtures[bits] = rational_bf16_rne(bf16_fraction(bits) * Fraction(7, 3))
    for bits, expected in fixtures.items():
        assert quant7_scale_bits(bits) == expected
        assert bf16_fraction(expected) * bf16_fraction(bits) > 0
        assert expected == rational_bf16_rne(bf16_fraction(bits) * Fraction(7, 3))
    projected = [min(3., max(-3., z)) for z in (-3.21, 3.1, -2.5, 2.5, -.51, .51, 0.)]
    assert all(-3 <= z <= 3 for z in projected)
    assert [min(7, max(-7, round(z))) for z in projected] == [round(z) for z in projected]
    counts = Counter(mapped)
    h7 = -math.fsum(counts[s] * math.log(counts[s] / len(mapped)) for s in range(-3, 4) if counts[s])
    h15 = -math.fsum(counts[s] * math.log(counts[s] / len(mapped)) for s in range(-7, 8) if counts[s])
    assert h7 == h15
    # Full 205-way output objective, with CE deliberately changed between calls.
    logits = [(i - 102) / 64 for i in range(205)]
    reference = softmax([-v / 2 for v in logits])
    probability = softmax(logits)
    objective = lambda values: data_objective(123., kl_value(values, reference), recipe)
    assert data_objective(1., 3., recipe) == data_objective(999., 3., recipe) == 6.
    assert data_objective(1., 3., legacy) == 4.
    epsilon = 1e-5
    gradient_errors = []
    for i in range(205):
        plus, minus = logits.copy(), logits.copy()
        plus[i] += epsilon
        minus[i] -= epsilon
        observed = (objective(plus) - objective(minus)) / (2 * epsilon)
        gradient_errors.append(abs(observed - 2 * (probability[i] - reference[i])))
    assert max(gradient_errors) < 1e-8
    # Cache metadata projects teacher identity while preserving student provenance.
    target = dict(checkpoint="/retained/original.tch", data="/retained/causal-manifest.json")
    old_identity = dict(target, recipe=dict(method="eight-block-causal-distillation-v1",
        teacher_kl=1., cpu_gemm=recipe["cpu_gemm"], kept_layers=[0, 1, 2, 3, 8, 9, 10, 11]))
    new_identity = dict(target, recipe=recipe, teacher_cache_mode="read-only")
    assert old_identity["recipe"]["teacher_kl"] == 1
    assert teacher_target_identity(old_identity) == teacher_target_identity(new_identity)
    absent = Path.cwd() / "temp" / f"q7-checker-absent-cache-{os.getpid()}"
    assert not absent.exists()
    class NoNumpyRead:
        @staticmethod
        def load(*args, **kwargs):
            raise AssertionError("missing cache reached np.load")
    for call in (lambda: readonly_teacher_logits(absent, (0, 100, 1300), NoNumpyRead),
                 lambda: check_teacher_cache_identity(absent, new_identity)):
        try:
            call()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing read-only cache accepted")
    assert not absent.exists()
    # Review actual mode branches and the retained inner hard/export route.
    required = ["q7-teacher-only-v1", "native_scaled_gemm=q7_teacher",
        "SOFT_LEVELS = tuple(RECIPE[\"soft_symbols\"])",
        "scales[key].copy_(s7)", 'model.get_parameter(key[:-7] + ".quantize_weight.scale").copy_(s7)',
        "readonly_teacher_logits(cache, piece, np)", "logits.detach() if q7_teacher else logits",
        "cpu_base = (args.prior_cpu_seconds if q7_teacher",
        "data_objective(ce, kl, RECIPE).backward()", "exact_teacher=not q7_teacher",
        "value.clamp_(-3 if q7_teacher else -7, 3 if q7_teacher else 7)",
        'if not q7_teacher or (epoch + 1) % RECIPE["evaluation_every"] == 0:',
        '"--quant7"', '"q7 zero-step differs from round241 map"',
        '"integer-coordinate zero-step static15 differs"',
        'result["teacher_kl_mean"] == 0', 'result["teacher_kl_mean"] > 0']
    assert all(fragment in source for fragment in required)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("integer_weight", "export_state", "install_coordinates"):
        assert "clamp(-7, 7)" in ast.get_source_segment(source, functions[name])
    # Export and geometry share max_scale_delta. Mapping updates keep the
    # measured geometry value in zero, epoch and preflight metadata.
    metadata = dict(max_scale_delta=1., tensor_count=434)
    geometry = dict(max_scale_delta=0., hard_integer_range=[-3, 3])
    for merged in (dict(metadata), dict(epoch=4), dict(teacher_kl_mean=.1)):
        merged.update(metadata)
        merged.update(geometry)
        assert merged["max_scale_delta"] == 0. and merged["tensor_count"] == 434
    for fragment in ('state["zero_step"] = dict(info)',
                     'state["zero_step"].update(geometry())',
                     'row.update(info)', 'row.update(geometry())',
                     'result.update(check_info)', 'result.update(after)'):
        assert fragment in source
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        assert sum(keyword.arg is None for keyword in call.keywords) <= 1
    helper_source = Path(__file__).with_name("q7_teacher_only.py").read_text()
    helper_tree = ast.parse(helper_source)
    helpers = {node.name: node for node in helper_tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("readonly_teacher_logits", "check_teacher_cache_identity"):
        attrs = {node.attr for node in ast.walk(helpers[name]) if isinstance(node, ast.Attribute)}
        assert not attrs & {"mkdir", "write_text", "write_bytes", "save", "savez", "replace"}
    result = dict(schema="hutter.q7-teacher-only.local-check.v1", status="passed", recipe=recipe,
        quant7_values=mapped, signed_bf16_fixtures=len(fixtures), projection_symbols=7,
        hard_entropy_15_equals_7=True, ce_report_only=True, output_gradient_formula="2*(p-q)",
        finite_difference_gradient_max_abs_error=max(gradient_errors),
        legacy_default_recipe_preserved=True, target_identity_projection_equal=True,
        missing_cache_stops_without_write=True, AST_source_branches_passed=True,
        overlapping_metadata_merges_passed=True,
        local_check_scope="standard-library map/objective/source and absent-cache checks",
        model_forwards=0, native_calls=0, optimizer_updates=0, teacher_cache_files=0,
        teacher_cache_writes=0)
    if a.torch:
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1)
        z = torch.tensor(logits, dtype=torch.float64).requires_grad_()
        t = torch.tensor(reference, dtype=torch.float64)
        ce = F.cross_entropy(z.detach().unsqueeze(0), torch.tensor([3]), reduction="sum")
        kl = F.kl_div(z.log_softmax(-1), t.log(), reduction="sum", log_target=True)
        grad, = torch.autograd.grad(data_objective(ce, kl, recipe), z)
        expected = 2 * (z.detach().softmax(-1) - t)
        torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-12)
        from q7_teacher_only import quant7_scales, quant7_tensor
        assert quant7_tensor(torch.arange(-7, 8, dtype=torch.int8)).tolist() == mapped
        fixture_tensor = torch.tensor(list(fixtures), dtype=torch.uint16).view(torch.bfloat16).float()
        assert quant7_scales(fixture_tensor).to(torch.bfloat16).view(torch.uint16).tolist() == list(fixtures.values())
        result.update(torch_toy_VJP_passed=True, torch_tensor_map_passed=True,
                      torch_gradient_max_abs_error=float((grad - expected).abs().max()))
    result["cpu_seconds"] = time.process_time() - started
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
