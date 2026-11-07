#!/usr/bin/env python3
"""Extract the two existing causal pieces and original MLP histograms; NumPy only."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz', type=Path,
                        default=HERE/'results/depth-middle-bridge.npz')
    parser.add_argument('--census', type=Path,
                        default=HERE/'results/weights-census.json')
    parser.add_argument('--out', type=Path,
                        default=Path('temp/native-mlp-rank64/inputs'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='hutter.native-mlp-inputs.round455.v1',
                  npz=str(args.npz),
                  census=str(args.census),
                  scope='existing full causal F16 priors; NN reset at each original complete piece',
                  new_ppmd_capture_calls=0, model_instances=0, pieces=[])
    with np.load(args.npz, allow_pickle=False) as arrays:
        for label, rows in [('train', 1199), ('held', 1606)]:
            x = arrays[label + '_input_tokens']
            y = arrays[label + '_target_tokens']
            p = arrays[label + '_priors_f16']
            b = arrays[label + '_bounds']
            assert x.shape == y.shape == (1, rows)
            assert x.dtype.kind in 'iu' and y.dtype.kind in 'iu'
            assert p.shape == (1, rows, 205) and p.dtype.kind == 'f' and p.dtype.itemsize == 2
            assert b.shape == (1, 2) and np.array_equal(b, [[0, rows]])
            assert np.all((x >= 0) & (x < 205)) and np.all((y >= 0) & (y < 205))
            assert np.array_equal(x[0, 1:], y[0, :-1])
            assert np.isfinite(p).all()
            values = [
                (label + '-input.u8', np.ascontiguousarray(x.reshape(rows), dtype=np.uint8)),
                (label + '-target.u8', np.ascontiguousarray(y.reshape(rows), dtype=np.uint8)),
                (label + '-priors.f16', np.ascontiguousarray(p.reshape(rows, 205), dtype='<f2')),
            ]
            files = {}
            for filename, data in values:
                with (args.out / filename).open('xb') as f:
                    f.write(data.tobytes(order='C'))
                assert (args.out / filename).read_bytes() == data.tobytes(order='C')
                files[filename] = dict(bytes=data.nbytes)
            result['pieces'].append(dict(name=label, rows=rows, prior_columns=205,
                                         rope_position_offset=0, bounds=b.tolist(), files=files))
    census = json.loads(args.census.read_text())
    by_name = {t['name']: t for t in census['per_tensor']}
    histograms = []
    names = []
    for layer in range(12):
        for kind, shape in [('up', [768, 192]), ('down', [192, 768])]:
            name = f'blocks.{layer}.mlp.{kind}.weight.q'
            t = by_name[name]
            assert t['dtype'] == 0 and t['shape'] == shape and t['elements'] == 147456
            assert len(t['histogram']) == 15 and sum(t['histogram']) == t['elements']
            names.append(name)
            histograms.append(t['histogram'])
    with (args.out / 'mlp-qhist.u32').open('xb') as f:
        f.write(np.asarray(histograms, dtype='<u4').tobytes(order='C'))
    result.update(mlp_histogram_order=names, mlp_histogram_shape=[24, 15],
                  mlp_histogram_dtype='uint32le')
    with (args.out / 'input-manifest.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': 'prepared', 'output': str(args.out),
                      'piece_rows': [1199, 1606], 'native_runs': 0}))


if __name__ == '__main__':
    main()
