"""Distinguish explicit cast coverage from a BF16 pointwise precision barrier."""
import json
import os
from pathlib import Path
import resource
import time
import torch
from torch._inductor import config
from torch.fx.experimental.proxy_tensor import make_fx

start = time.monotonic()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
config.compile_threads = config.cpp.threads = 1
out = Path(os.environ['CAST_OUTPUT'])
out.mkdir(parents=True, exist_ok=True)


def cast_only(x):
    return x.to(torch.bfloat16).to(torch.float8_e4m3fn).float()


def pointwise(x):
    return (x.to(torch.bfloat16) * 1.5).to(torch.float8_e4m3fn).float()


x = 1 + torch.arange(65536, dtype=torch.float32) / 65536
reference = pointwise(x)
direct = (x * 1.5).to(torch.float8_e4m3fn).float()
assert torch.count_nonzero(reference != direct).item() == 3071
r = {'scope': 'CPU uniform FP32 [1,2); exact factor1.5 BF16 pointwise control',
     'torch_version': torch.__version__, 'eager_vs_direct_changed': 3071,
     'to_copy_tags': [str(x) for x in torch.ops.aten._to_copy.default.tags],
     'mul_tags': [str(x) for x in torch.ops.aten.mul.Tensor.tags],
     'graphs': {}, 'compiled': []}
with config.patch(emulate_precision_casts=True):
    for fn in (cast_only, pointwise):
        g = make_fx(fn, tracing_mode='fake')(x)
        r['graphs'][fn.__name__] = [
            {'target': str(n.target), 'dtype': str(getattr(n.meta.get('val'), 'dtype', None)),
             'low_precision_pointwise_barrier': n.meta.get('low_precision_pointwise_barrier', False)}
            for n in g.graph.nodes if n.op == 'call_function']
cache = Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])
for emulate in (False, True):
    torch._dynamo.reset()
    prior = set(cache.rglob('*.cpp')) if cache.exists() else set()
    row = {'emulate_precision_casts': emulate}
    try:
        with config.patch(emulate_precision_casts=emulate):
            y = torch.compile(pointwise, backend='inductor', fullgraph=True)(x)
        assert y.dtype == torch.float32 and y.shape == reference.shape
        row.update(status='measured', mismatches_vs_eager=int(torch.count_nonzero(y != reference)),
                   mismatches_vs_direct=int(torch.count_nonzero(y != direct)),
                   max_abs_vs_eager=(y-reference).abs().max().item())
    except Exception as e:
        row.update(status='failed', error_type=type(e).__name__, error=str(e)[:5000])
    row['generated_cpp'] = []
    for i, p in enumerate(sorted(set(cache.rglob('*.cpp')) - prior)):
        name = f'pointwise-{str(emulate).lower()}-{i}.cpp'
        (out/name).write_bytes(p.read_bytes())
        row['generated_cpp'].append(name)
    r['compiled'].append(row)
children = resource.getrusage(resource.RUSAGE_CHILDREN)
r['resources'] = {'cpu_seconds': time.process_time()+children.ru_utime+children.ru_stime,
                  'wall_seconds': time.monotonic()-start,
                  'python_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'child_peak_rss_kib': children.ru_maxrss}
(out/'result.json').write_text(json.dumps(r, indent=2)+'\n')
print(json.dumps(r), flush=True)
