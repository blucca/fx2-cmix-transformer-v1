#!/usr/bin/env python3
"""Toy VJP and optional read-only cache identity; imported production helpers."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--trainer", type=Path, default=Path(__file__).with_name("depth-distillation.py"))
for name in ("cache", "checkpoint", "data"):
    p.add_argument("--" + name, type=Path)
a = p.parse_args()
if any(v is not None for v in (a.cache, a.checkpoint, a.data)) and not all(v is not None for v in (a.cache, a.checkpoint, a.data)):
    p.error("--cache, --checkpoint and --data travel together")
spec = importlib.util.spec_from_file_location("production_depth_distillation", a.trainer)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
recipe = dict(m.RECIPE, method="eight-block-teacher-only-distillation-v1", label_ce=0., teacher_kl=2.)
identity = None
if a.cache is not None:
    identity = dict(checkpoint=str(a.checkpoint.resolve()), data=str(a.data.resolve()), recipe=recipe)
    cached = json.loads((a.cache / "identity.json").read_text())
    assert m.teacher_target_identity(cached) == m.teacher_target_identity(identity)
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
z = torch.linspace(-2., 2., 410, dtype=torch.float64).reshape(2, 205).requires_grad_()
t = torch.linspace(1., -1., 410, dtype=torch.float64).reshape(2, 205)
h = torch.tensor([[.2, -.3], [.1, .4]], dtype=torch.float64, requires_grad=True)
ce = F.cross_entropy(z.detach(), torch.tensor([3, 100]), reduction="sum")
kl = F.kl_div(z.log_softmax(-1), t.log_softmax(-1), reduction="sum", log_target=True)
mse = h.square().mean(-1).sum()
objective = m.data_objective(ce, kl, mse, recipe)
assert not ce.requires_grad
torch.testing.assert_close(objective, 2 * kl + .1 * mse)
gz, gh = torch.autograd.grad(objective, (z, h))
expected_z, expected_h = 2 * (z.detach().softmax(-1) - t.softmax(-1)), .2 * h.detach() / h.shape[-1]
torch.testing.assert_close(gz, expected_z, rtol=1e-10, atol=1e-12)
torch.testing.assert_close(gh, expected_h, rtol=1e-10, atol=1e-12)
print(json.dumps(dict(status="passed", recipe=recipe, target_identity=m.teacher_target_identity(identity) if identity is not None else None,
    teacher_cache_mode="read-only" if identity is not None else "unused",
    ce_report_only=True, output_gradient_formula="2*(student_probability-teacher_probability)",
    output_gradient_max_abs_error=float((gz-expected_z).abs().max()),
    hidden_gradient_max_abs_error=float((gh-expected_h).abs().max()),
    toy_objective=float(objective.detach()), cpu_seconds=time.process_time(),
    model_forwards=0, native_calls=0, optimizer_updates=0, teacher_cache_writes=0)))
