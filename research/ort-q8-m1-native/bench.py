#!/usr/bin/env python3
"""One complete M1 comparison, native positive/negative controls, ORT 1.31.1."""
import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

# In a source checkout, the existing stock-Q8/Two-Q4 reference lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ort_split8 as baseline
from q8_m1 import native_m1


def native_session(model, library):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.register_custom_ops_library(str(library.resolve()))
    return ort.InferenceSession(model, options, providers=['CPUExecutionProvider'])


def measure(sess, a):
    for _ in range(30):
        y = sess.run(None, {'A': a})[0]
    times = []
    for _ in range(200):
        start = time.perf_counter()
        sess.run(None, {'A': a})
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return float(np.median(times)), float(times[189]), y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cpuinfo = Path('/proc/cpuinfo')
    raw = cpuinfo.read_text() if cpuinfo.exists() else ''
    flags = next((line.split(':', 1)[1].split() for line in raw.splitlines() if line.startswith('flags')), [])
    cpu = next((line.split(':', 1)[1].strip() for line in raw.splitlines() if line.startswith('model name')), platform.processor())
    result = {'status': 'running', 'startedAt': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'cpu': cpu,
              'features': {name: name in flags if flags else None for name in ['avx2','fma','avx512f','avx512_vnni','avx_vnni']},
              'python': platform.python_version(), 'ort': ort.__version__, 'onnx': onnx.__version__,
              'numpy': np.__version__, 'threads': 1, 'warmup': 30, 'iterations': 200,
              'controls': {}, 'M': 1, 'K': 768, 'N': 3072, 'modes': {}}
    def save():
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    save()
    try:
        # Duplicate the 256-code table to N16, the native tile's explicit contract.
        b = np.tile(np.arange(256, dtype=np.uint8).reshape(8,1,32), (2,1,1))
        s = np.full(16, .125, dtype=np.float32)
        sess = native_session(native_m1(b,s)[0], args.library)
        actual = np.concatenate([sess.run(None, {'A': row.reshape(1,32)})[0]
                                 for row in np.eye(32,dtype=np.float32)])
        ref = baseline.dequant(b,s)
        result['controls']['all256_identity_equal'] = bool(np.array_equal(actual,ref))
        assert result['controls']['all256_identity_equal']
        # A deliberately wrong zero encoding must be detected by the identity control.
        wrong = native_session(native_m1(b ^ np.uint8(128),s)[0], args.library)
        wrong_y = np.concatenate([wrong.run(None, {'A': row.reshape(1,32)})[0]
                                  for row in np.eye(32,dtype=np.float32)])
        result['controls']['wrong_zero_encoding_max_abs'] = baseline.error(wrong_y,ref)[0]
        assert result['controls']['wrong_zero_encoding_max_abs'] == 16
        rng = np.random.default_rng(562)
        b = rng.integers(0,256,size=(16,2,32),dtype=np.uint8)
        s = np.linspace(.00013,.00471,32,dtype=np.float32)
        a = rng.standard_normal((1,64)).astype(np.float32)
        ref = a @ baseline.dequant(b,s)
        y = native_session(native_m1(b,s)[0],args.library).run(None,{'A':a})[0]
        result['controls']['arbitrary_scales_norm_max_error'] = baseline.error(y,ref)[1]
        assert result['controls']['arbitrary_scales_norm_max_error'] <= 2e-6
        wrong_s = s.reshape(16,2)[:,::-1].copy().reshape(-1)
        wrong_y = native_session(native_m1(b,wrong_s)[0],args.library).run(None,{'A':a})[0]
        result['controls']['wrong_block_scales_norm_max_error'] = baseline.error(wrong_y,ref)[1]
        assert result['controls']['wrong_block_scales_norm_max_error'] > 1e-3
        save()
        # Preserve the original reporter's full seed-0 generation order, timing M1 only.
        rng = np.random.default_rng(0)
        for m,k,n in [(197,768,3072),(197,3072,768),(1,768,3072)]:
            w = rng.standard_normal((k,n)).astype(np.float32) * np.float32(.02)
            a = rng.standard_normal((m,k)).astype(np.float32)
        b,s,deq = baseline.quant8(w)
        ref = a @ deq
        models = [('fp32_same_weights',baseline.dense(1,k,n,deq)),
                  ('split_two_q4',baseline.split8(1,k,n,b,s)),
                  ('native_q8_m1',native_m1(b,s))]
        for name,(model,payload) in models:
            sess = native_session(model,args.library) if name=='native_q8_m1' else baseline.session(model)
            p50,p95,y = measure(sess,a)
            absolute,relative = baseline.error(y,ref)
            result['modes'][name] = {'p50_ms':p50,'p95_ms':p95,'max_abs_error':absolute,
                                    'norm_max_error':relative,'initializer_bytes':payload}
            print(name,json.dumps(result['modes'][name]),flush=True)
            save()
        native = result['modes']['native_q8_m1']
        result['speedup_vs_split8'] = result['modes']['split_two_q4']['p50_ms'] / native['p50_ms']
        result['speedup_vs_fp32'] = result['modes']['fp32_same_weights']['p50_ms'] / native['p50_ms']
        result['prepack_cache_tensor_bytes'] = b.nbytes + s.nbytes
        result['decision'] = {'m1_precision_pass':native['norm_max_error']<=2e-6,
                              'speedup_vs_split8_pass':result['speedup_vs_split8']>=2,
                              'initializer_payload_pass':native['initializer_bytes']==b.nbytes+s.nbytes}
        result['decision']['passed'] = all(result['decision'].values())
        assert result['decision']['m1_precision_pass']
        result['status'] = 'completed'
    except Exception as exc:
        result['status']='failed'
        result['error']=repr(exc)
        raise
    finally:
        result['completedAt']=datetime.now(timezone.utc).isoformat()
        save()


if __name__=='__main__':
    main()
