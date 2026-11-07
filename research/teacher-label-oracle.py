#!/usr/bin/env python3
"""Output-only CE+KL optimum on the original frozen 205-column teacher cache."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'cache', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    sys.dont_write_bytecode = True
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    cores = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {1 if 1 in cores else cores[-1]})
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (178, 180))
    start = time.monotonic()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='hutter.frozen-teacher-output-oracle.v1', status='running',
                  scope='independent 205 output logits for every original training context',
                  formula='r=(q+one_hot(y))/2; CE(r)=ln2-ln(1+qy); KL(q||r)=ln2+qy*ln(qy)-qy*ln(1+qy)',
                  cpu_limit_seconds=180, wall_limit_seconds=240,
                  address_space_limit_bytes=1 << 30, teacher_forwards=0,
                  native_calls=0, optimizer_updates=0, teacher_cache_writes=0,
                  scored_pieces=0, scored_targets=0, per_piece=[])
    sums = {k: [] for k in ('teacher_ce', 'oracle_ce', 'oracle_kl', 'qy')}

    def save():
        result.update(cpu_seconds=time.process_time(), wall_seconds=time.monotonic()-start,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      cpu_affinity=sorted(os.sched_getaffinity(0)))
        tmp = out / 'result.tmp'
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        tmp.replace(out / 'result.json')

    def limit(signum, frame):
        raise TimeoutError('cpu_limit' if signum == signal.SIGXCPU else 'wall_limit')

    signal.signal(signal.SIGXCPU, limit)
    signal.signal(signal.SIGALRM, limit)
    signal.alarm(240)
    save()
    try:
        import numpy as np
        manifest = json.loads(a.manifest.read_text())
        identity = json.loads((a.cache/'identity.json').read_text())
        assert identity['data'] == str(a.manifest.resolve())
        assert identity['recipe']['cpu_gemm'] == 'integer-units-before-native-BF16-fold'
        assert manifest['method'] == 'causal-whole-piece-v1'
        samples = manifest['samples']
        pieces = [(i, lo, hi) for i, s in enumerate(samples) for lo, hi in s['train_pieces']]
        assert len(pieces) == manifest['training_pieces'] == 1313
        assert sum(hi-lo-1 for _, lo, hi in pieces) == manifest['training_targets'] == 2941307
        result.update(identity=identity, expected_pieces=1313, expected_targets=2941307,
                      columns=205, cache=str(a.cache), manifest=str(a.manifest))
        for index, (i, lo, hi) in enumerate(pieces):
            n = hi-lo-1
            y = np.fromfile(samples[i]['tokens'], dtype=np.uint8, count=n, offset=lo+1)
            assert y.shape == (n,) and int(y.max()) < 205
            path = a.cache/f'sample-{i:04d}-{lo}-{hi}.npz'
            with np.load(path, allow_pickle=False) as data:
                logits = data['logits']
            assert logits.dtype == np.float32 and logits.shape == (1, n, 205)
            assert bool(np.isfinite(logits).all())
            local = {k: [] for k in sums}
            # A 2048-row tile bounds FP64 temporaries; original caches stay read-only.
            for begin in range(0, n, 2048):
                z = logits[0, begin:begin+2048].astype(np.float64)
                maxz = z.max(axis=1)
                logz = maxz + np.log(np.exp(z-maxz[:, None]).sum(axis=1))
                logqy = z[np.arange(len(z)), y[begin:begin+len(z)]]-logz
                qy = np.exp(logqy)
                log1p = np.log1p(qy)
                ce = math.log(2)-log1p
                kl = math.log(2)+qy*logqy-qy*log1p
                assert bool(((qy >= 0) & (qy <= 1+1e-12)).all())
                assert bool((ce+kl <= -logqy+1e-12).all())
                for key, value in zip(local, (-logqy, ce, kl, qy)):
                    local[key].append(float(value.sum(dtype=np.float64)))
            row = dict(piece=index, sample=i, local_token_range=[lo, hi], targets=n,
                       **{k+'_mean': math.fsum(v)/n for k, v in local.items()})
            result['per_piece'].append(row)
            result['scored_pieces'] += 1
            result['scored_targets'] += n
            for key in sums:
                sums[key].append(math.fsum(local[key]))
            del logits, y, z, maxz, logz, logqy, qy, log1p, ce, kl
            if index % 32 == 31:
                save()
        assert result['scored_targets'] == result['expected_targets']
        result['means'] = {k: math.fsum(v)/result['scored_targets'] for k, v in sums.items()}
        m = result['means']
        m['oracle_old_output_objective'] = m['oracle_ce']+m['oracle_kl']
        m['teacher_old_output_objective'] = m['teacher_ce']
        m['old_objective_descent_from_teacher'] = m['teacher_ce']-m['oracle_old_output_objective']
        result['status'] = 'passed'
        save()
        print(json.dumps(result['means']), flush=True)
    except Exception as exc:
        result.update(status='bounded_stop' if isinstance(exc, TimeoutError) else 'failed',
                      error=f'{type(exc).__name__}: {exc}')
        save()
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
