"""One offline CUDA smoke for the retained q7 integer-unit linear contract.

This covers existing full12 matrix geometries with generated hard operands.
The full1199 article gate is a separate second stage using original assets.
No model training, installs, networking or persistent machine process.
"""
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
OUT = Path(os.environ.get('RESULT_PATH', 'temp/q7-cuda-linear-resource.json'))
started = time.monotonic()
result = {'status': 'starting', 'scope': 'q7-integer-linear-resource-smoke',
          'full_model': False, 'optimizer_updates': 0, 'wall_limit_seconds': 120}

def save(**data):
    result.update(data)
    result['wall_seconds'] = time.monotonic() - started
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

def expired(signum, frame):
    save(status='stopped', error='resource-smoke-time-limit')
    raise SystemExit(124)

signal.signal(signal.SIGALRM, expired)
signal.alarm(120)
save()
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.is_available(), 'CUDA resource gate failed'
    save(python=platform.python_version(), torch=torch.__version__,
         cuda_build=torch.version.cuda,
         devices=[{'name': torch.cuda.get_device_name(i),
                   'capability': list(torch.cuda.get_device_capability(i)),
                   'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                  for i in range(torch.cuda.device_count())],
         torch_bf16_supported_including_emulation=torch.cuda.is_bf16_supported(),
         hardware_bf16_supported=torch.cuda.is_bf16_supported(including_emulation=False))
    try:
        result['nvidia_smi'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    # Injected at preparation from the existing full12 census: 111 matrices.
    shapes = [(64, 192, 18), (192, 64, 18), (192, 192, 48), (192, 205, 1), (192, 768, 12), (205, 192, 2), (768, 192, 12)]
    assert sum(n for out, kin, n in shapes) == 111
    assert sum(out * kin * n for out, kin, n in shapes) == 5868864
    gen = torch.Generator(device='cpu').manual_seed(508)
    rows = []
    for out, kin, count in shapes:
        assert kin <= 768 and kin * 128 * 3 < 2**24
        a = torch.randint(-128, 128, (1199, kin), generator=gen).float()
        q = torch.randint(-3, 4, (out, kin), generator=gen).float()
        # Existing signed/tiny BF16 scale cases; FP32 is the compute dtype.
        sw = torch.tensor([6e-15, -8e-15, .00390625, -.0078125]).repeat(
            (out + 3) // 4)[:out].to(torch.bfloat16).float()
        sa = torch.tensor(.03125).to(torch.bfloat16).float()
        fold = sa * sw
        ac, zc, fc = a.cuda(), q.cuda().requires_grad_(), fold.cuda()
        hard = zc + (zc.round().clamp(-3, 3) - zc).detach()
        dots = ac @ hard.T
        expected_ints = a[:8].long() @ q.T.long()
        assert torch.equal(dots[:8].detach().cpu(), expected_ints.float())
        y = dots * fc
        grad_out = torch.randint(-2, 3, y.shape, generator=gen).float()
        (y * grad_out.cuda()).sum().backward()
        expected_grad = (grad_out * fold).T @ a
        actual_grad = zc.grad.detach().cpu()
        error = (actual_grad.double() - expected_grad.double()).square().sum()
        norm = expected_grad.double().square().sum()
        relative_l2 = float(torch.sqrt(error / norm.clamp_min(1e-300)))
        assert torch.isfinite(y).all() and torch.isfinite(zc.grad).all()
        assert relative_l2 < 1e-5
        # A few actual batch1199 GEMMs give useful local timing, no model claim.
        with torch.no_grad():
            event1, event2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            event1.record()
            for _ in range(8):
                ac @ zc.T
            event2.record()
            torch.cuda.synchronize()
            gpu_ms = event1.elapsed_time(event2) / 8
        rows.append({'out': out, 'in': kin, 'matrix_count': count,
                     'batch_targets': 1199, 'integer_rows_checked': 8,
                     'integer_dots_exact': True, 'gradient_relative_l2': relative_l2,
                     'fp32_gemm_ms': gpu_ms})
        del ac, zc, fc, hard, dots, y
    save(status='passed', rows=rows, geometry_matrices=111,
         coordinate_elements=5868864,
         peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
except Exception as exc:
    save(status='failed', error=f'{type(exc).__name__}: {exc}')
    raise
finally:
    signal.alarm(0)
print(json.dumps(result, indent=2))
