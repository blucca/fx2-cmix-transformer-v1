#!/usr/bin/env python3
"""Replay the completed original-input identity and inverse resource contract."""
import json
from pathlib import Path
s=json.loads((Path(__file__).resolve().parent/'results/prepared-original-provenance.json').read_text())
f=s['completed_inverse_inventory']['file_sizes']
assert s['status']=='passed' and s['restored']=={'bytes':1000000000,'md5':'e206c3450ac99950df65bf70ef61a12d'}
assert s['prepared']['bytes']==586459020 and s['range_neural_calls']==0
assert f['.intro_decomp']+f['.main_decomp_restored']+f['.coda_decomp']==1000000000
assert f['.input_decomp']==f['.main_decomp']+f['.intro_decomp']+f['.coda_decomp']
assert sum(f.values())==s['completed_inverse_inventory']['logical_bytes']
assert 0<s['cpu']['total_completed_seconds']<=s['budget']['cpu_limit_seconds']==1800
assert s['retained_files']==['enwik9_uncompressed'] and s['removed_duplicate_intermediates']==len(f)-1
print('Original 1 GB provenance, inverse inventory and resource contract passed.')
