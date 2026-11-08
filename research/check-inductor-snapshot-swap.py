"""A real CPU snapshot-swap fault: returned values and original storage are checked.

Original reproducer: pytorch/pytorch#203806. The upfront explicit-copy and
standalone stack variants are local controls, not upstream patch validation.
All runs use the installed 2.15 CPU package and one thread.
"""
import json
import os
from pathlib import Path
import resource
import statistics
import time

START = time.monotonic()
OUT = Path(os.environ['SNAPSHOT_OUTPUT'])
OUT.mkdir(parents=True, exist_ok=True)
RESULT = {'status': 'started', 'source_issue': 'https://github.com/pytorch/pytorch/issues/203806',
          'contract': 'two float32 CPU inputs; capture both old values before any write; swap values; preserve original storage',
          'sizes': [3, 1048576], 'cases': []}


def save():
    RESULT['resources'] = {'cpu_seconds': time.process_time(), 'wall_seconds': time.monotonic()-START,
                           'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (OUT/'result.json').write_text(json.dumps(RESULT, indent=2)+'\n')


def foreach_snapshot(a, b):
    torch._foreach_copy_([a, b], [b.clone(), a.clone()])
    return a, b


def explicit_snapshot(a, b):
    sources = (b.clone(), a.clone())  # Capture BOTH before either copy_.
    for dest, source in zip((a, b), sources):
        dest.copy_(source)
    return a, b


def stack_snapshot(a, b):
    sources = torch.stack((b, a))
    torch._foreach_copy_([a, b], list(sources.unbind()))
    return a, b


def inputs(n):
    a = torch.arange(n, dtype=torch.float32)
    b = a + 2097152
    return a, b


def check(fn, n):
    a, b = inputs(n)
    gold_a, gold_b = b.clone(), a.clone()
    # External aliases witness the pre-call storage, independently of returns.
    alias_a, alias_b = a.view(-1), b.view(-1)
    pointers = (a.data_ptr(), b.data_ptr())
    returned = fn(a, b)
    errors = {name: int(torch.count_nonzero(got != gold)) for name, got, gold in (
        ('return_a', returned[0], gold_a), ('return_b', returned[1], gold_b),
        ('input_a', a, gold_a), ('input_b', b, gold_b),
        ('alias_a', alias_a, gold_a), ('alias_b', alias_b, gold_b))}
    preserved = pointers == (a.data_ptr(), b.data_ptr())
    return {'mismatches': errors, 'storage_preserved': preserved,
            'returned_input_storage': [returned[0].data_ptr()==a.data_ptr(), returned[1].data_ptr()==b.data_ptr()],
            'passed': not any(errors.values()) and preserved}


try:
    import torch
    from torch._inductor import config
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config.compile_threads = config.cpp.threads = 1
    RESULT['version'] = {'torch': torch.__version__, 'git': torch.version.git_version,
                         'cuda': torch.version.cuda, 'cpu_capability': torch.backends.cpu.get_cpu_capability(),
                         'python': os.sys.version, 'emulate_precision_casts': config.emulate_precision_casts}
    assert str(torch.__version__) == '2.15.0+cpu', RESULT['version']
    methods = (foreach_snapshot, explicit_snapshot, stack_snapshot)
    # Minimal executable check of the shared snapshot/mutation contract.
    assert all(check(fn, 3)['passed'] for fn in methods)
    for fn in methods:
        for n in (3, 1048576):
            if n > 3 and not RESULT['cases'][-1]['compiled']['passed']:
                break  # A failed workaround gets no large timing/compilation budget.
            torch._dynamo.reset()
            before = set(Path(os.environ['TORCHINDUCTOR_CACHE_DIR']).rglob('*'))
            row = {'method': fn.__name__, 'n': n, 'status': 'started'}
            RESULT['cases'].append(row); save()
            begin = time.monotonic()
            compiled = torch.compile(fn, backend='inductor', fullgraph=True, dynamic=False)
            row['compiled'] = check(compiled, n)
            row['compile_and_first_call_wall_seconds'] = time.monotonic()-begin
            # Compare timings of CORRECT implementations only. Reset is outside timing.
            if n > 3 and row['compiled']['passed']:
                a, b = inputs(n)
                a0, b0 = a.clone(), b.clone()
                samples = []
                for trial in range(22):
                    a.copy_(a0); b.copy_(b0)
                    tick = time.perf_counter_ns(); compiled(a, b); elapsed = time.perf_counter_ns()-tick
                    if trial >= 2:
                        samples.append(elapsed/1e6)
                row['timing_ms'] = {'median': statistics.median(samples), 'min': min(samples),
                                    'max': max(samples), 'samples': samples,
                                    'definition': 'warm CPU call only; two warmups; reset excluded; one thread'}
                assert not torch.count_nonzero(a != b0) and not torch.count_nonzero(b != a0)
            created = set(Path(os.environ['TORCHINDUCTOR_CACHE_DIR']).rglob('*')) - before
            saved = []
            dest = OUT/'kernels'/f'{fn.__name__}-{n}'; dest.mkdir(parents=True, exist_ok=True)
            for i, path in enumerate(sorted(p for p in created if p.is_file() and p.suffix in ('.cpp', '.py'))):
                name = f'{i}{path.suffix}'
                (dest/name).write_bytes(path.read_bytes()); saved.append(str(dest.relative_to(OUT)/name))
            row['generated_files'] = saved
            row['status'] = 'measured'; save()
    RESULT['status'] = 'measured'
except Exception as error:
    RESULT.update(status='failed', error_type=type(error).__name__, error=str(error)[:3000])
finally:
    save(); print(json.dumps(RESULT), flush=True)
