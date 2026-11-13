#!/usr/bin/env python3
"""Check the published fixed-input baseline and the single no-inline control."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
linux = json.loads((D / 'baseline-linux.json').read_text())
windows = json.loads((D / 'noinline-windows.json').read_text())


def accurate(row):
    tolerance = 2e-6 if row['dtype'] == 'float32' else 2e-12
    return all(value['mean_absdiff'] <= tolerance and value['variance_absdiff'] <= tolerance
               for value in row['variants'])


shape = {(dtype, n) for dtype in ('float32', 'float64') for n in (1024, 2048, 4096, 175616)}
for rows in (linux['native'], windows['native'], windows['native_noinline']):
    assert len(rows) == 8 and {(r['dtype'], r['n']) for r in rows} == shape
assert all(accurate(r) for r in linux['native'] + windows['native_noinline'])
assert {(r['dtype'], r['n']) for r in windows['native'] if not accurate(r)} == {
    ('float64', 2048), ('float32', 4096), ('float64', 4096),
    ('float32', 175616), ('float64', 175616)}
for result in linux['runtime'] + windows['runtime']:
    assert len(result['rows']) == 8 and all(r['max_abs_error'] < 2e-4 for r in result['rows'])
print('PASS: Linux 8 / Windows no-inline 8; Windows baseline 5 inaccurate rows; stock runtime 32 accurate rows')
