"""Reconstruct the new dense vocabulary from the canonical small Q8 transfer.

All unchanged bins come from the existing local complete-model artifacts.
ONNX 1.23 external files need single-link ordinary copies, shared by graphs.
"""
import argparse
import filecmp
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--base-root',type=Path,required=True)
a=p.parse_args();root=a.root.resolve();base=a.base_root.resolve()
assert sys.byteorder=='little' and root!=base
m=json.loads((root/'manifest.json').read_text());models=root/'models'
assert m['vocab']==50257 and m['padded_vocab']==50272 and m['K']==768 and m['block_size']==32
for item in m['copied_unchanged_files']:
    name=item['file'];assert Path(name).name==name
    source=base/'models'/name;target=models/name
    assert source.stat().st_size==item['bytes']
    if not target.exists():shutil.copyfile(source,target)
    assert target.stat().st_nlink==1 and filecmp.cmp(source,target,shallow=False)
new={x['file']:x for x in m['new_weight_files']}
bname='__blucca_tied_vocab_q8_B.bin';sname='__blucca_tied_vocab_q8_S.bin';ename='__blucca_tied_vocab_q8_dense_E.bin'
for name in (bname,sname):assert (models/name).stat().st_size==new[name]['bytes']
b=np.memmap(models/bname,mode='r',dtype=np.uint8,shape=(50272,24,32))
s=np.memmap(models/sname,mode='r',dtype='<f4',shape=(50272,24,1))
assert np.all(b[50257:]==128) and np.isfinite(s).all() and np.all(s>0)
e=((b.astype(np.float32)-np.float32(128))*s).reshape(50272,768)[:50257]
assert e.flags.c_contiguous and e.dtype==np.float32
with (models/ename).open('wb') as f:
    view=memoryview(e).cast('B');assert f.write(view)==new[ename]['bytes']
lib=root/'libq8_m1.so'
if not lib.exists():shutil.copyfile(base/'libq8_m1.so',lib)
assert filecmp.cmp(base/'libq8_m1.so',lib,shallow=False)
for mode in ('dense','native','q8_acc0'):
    assert (models/(mode+'.onnx')).stat().st_size==m['models'][mode]['file_bytes']
    for name in m['models'][mode]['external_files']:assert (models/name).is_file()
(root/'target-assembly.json').write_text(json.dumps({'status':'complete','canonical_Q8_vocabulary_reconstructed':True,
    'old_native_library_byte_equal':True,'unchanged_source_files':len(m['copied_unchanged_files']),
    'vocab':50257,'padded_vocab':50272,'dense_E_bytes':len(view)},indent=2))
print('TARGET_ASSEMBLY_PASS',len(view))
