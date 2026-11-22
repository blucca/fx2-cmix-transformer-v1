from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import tied_runner

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
common=SimpleNamespace(PAST=(16,128,512),dump=dump)
features=dict(avx2=True,fma=True,avx512f=False,avx512_vnni=False,avx_vnni=False)
with TemporaryDirectory(dir=Path('temp').resolve(),prefix='tied-comparison-') as directory:
    root=Path(directory)
    for name in ('old-before','dense-before','native','q8_acc0','dense-after','old-after'):
        fast=name=='native'
        dump(root/'runs'/'test'/name/'result.json',dict(status='completed',host={'features':features},
             rows=[{'p50_ms':.5 if fast else 1.} for _ in common.PAST],peak_rss_kib=500 if fast else 1000,
             all_outputs_passed=True))
    x=tied_runner.compare(common,root,'test');assert x['target_compute_gate_passed']
    path=root/'runs'/'test'/'q8_acc0'/'result.json'
    x=json.loads(path.read_text());x['all_outputs_passed']=False;dump(path,x)
    x=tied_runner.compare(common,root,'test');assert not x['numerical_pass'] and not x['target_compute_gate_passed']
print('TIED_COMPARISON_POSITIVE_NEGATIVE_PASS')
