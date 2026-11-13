#!/usr/bin/env python3
"""Record one stock runtime and its installed RowwiseMoments AVX2 headers."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SHAPES = (1024, 2048, 4096, 175616)


def runtime(out):
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    torch.manual_seed(0)
    rows = []
    for n in SHAPES:
        x = torch.randn(4, n)
        xd = x.double()
        mean = xd.mean(-1, keepdim=True)
        var = ((xd - mean) ** 2).mean(-1, keepdim=True)
        reference = (xd - mean) / torch.sqrt(var + 1e-5)
        # Independent NumPy reference checks the reference, not another layer_norm.
        a = xd.numpy()
        independent = (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-5)
        reference_delta = float(np.max(np.abs(reference.numpy() - independent)))
        assert reference_delta < 1e-12
        for name, value in [('float32', x), ('float64', xd)]:
            actual = F.layer_norm(value, (n,), eps=1e-5).double()
            error = float((actual - reference).abs().max())
            rows.append({'n': n, 'dtype': name, 'max_abs_error': error,
                         'reference_delta': reference_delta,
                         'above_reported_test_tolerance': error > 2e-4})
    result = {'torch': torch.__version__, 'git_version': torch.version.git_version,
              'platform': platform.platform(), 'python': platform.python_version(),
              'requested_cpu_capability': os.environ.get('ATEN_CPU_CAPABILITY'),
              'actual_cpu_capability': torch.backends.cpu.get_cpu_capability(),
              'build_config': torch.__config__.show(), 'rows': rows}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('temp/layernorm-moments'))
    parser.add_argument('--runtime-only', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.runtime_only:
        out.parent.mkdir(parents=True, exist_ok=True)
        runtime(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for capability in ('avx2', 'default'):
        target = out / f'runtime-{capability}.json'
        env = dict(os.environ, ATEN_CPU_CAPABILITY=capability, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [sys.executable, str(Path(__file__).resolve()), '--runtime-only', '--out', str(target)]
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
        (out / f'runtime-{capability}.log').write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f'{capability} measurement failed; read its saved log')
        results.append(json.loads(target.read_text()))
    import torch
    include = Path(torch.__file__).parent / 'include'
    libraries = Path(torch.__file__).parent / 'lib'
    source = Path(__file__).resolve().with_name('moments.cpp')
    executable = out / ('moments.exe' if os.name == 'nt' else 'moments')
    header = include / 'ATen/native/cpu/moments_utils.h'
    shutil.copyfile(header, out / 'installed-moments_utils.h')
    if os.name == 'nt':
        command = ['cl', '/nologo', '/O2', '/std:c++20', '/arch:AVX2', '/EHsc', '/DNDEBUG',
                   '/DCPU_CAPABILITY=AVX2', '/DCPU_CAPABILITY_AVX2', '/I' + str(include),
                   str(source), '/Fo' + str(out / 'moments.obj'), '/Fe' + str(executable),
                   '/link', '/LIBPATH:' + str(libraries), 'c10.lib', 'torch_cpu.lib']
    else:
        command = [os.environ.get('CXX', 'c++'), '-O3', '-std=c++20', '-mavx2', '-mfma', '-DNDEBUG',
                   '-DCPU_CAPABILITY=AVX2', '-DCPU_CAPABILITY_AVX2', '-I', str(include),
                   str(source), '-L', str(libraries), '-Wl,-rpath,' + str(libraries),
                   '-lc10', '-ltorch_cpu', '-o', str(executable)]
    (out / 'build.json').write_text(json.dumps({'command': command, 'torch': torch.__version__,
                                              'platform': platform.platform()}, indent=2) + '\n')
    completed = subprocess.run(command, cwd=out, capture_output=True, text=True, timeout=180)
    (out / 'build.log').write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError('Native build failed; read build.log')
    env = dict(os.environ, PATH=str(libraries) + os.pathsep + os.environ.get('PATH', ''))
    completed = subprocess.run([str(executable)], env=env, capture_output=True, text=True, timeout=60)
    (out / 'native.log').write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError('Native probe failed; read native.log')
    native = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith('{')]
    assert len(native) == 8 and {(r['dtype'], r['n']) for r in native} == {
        (dtype, n) for dtype in ('float32', 'float64') for n in SHAPES}
    result = {'runtime': results, 'native': native}
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'complete', 'torch': torch.__version__, 'native_rows': len(native),
                      'runtime_rows': sum(len(r['rows']) for r in results)}))


if __name__ == '__main__':
    main()
