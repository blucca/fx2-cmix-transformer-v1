#!/usr/bin/env python3
"""Replay existing half buffers through original ByteModel and mixer input code."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--compiler',default='clang++-17')
a=p.parse_args()
root=Path(__file__).resolve().parent
out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
flags=json.loads((root/'compiler-flags.json').read_text())
sources=['src/models/byte-model.cpp','src/mixer/sigmoid.cpp',
         'src/mixer/mixer-input.cpp','source-half.cpp','observer.cpp']
objects=[]
for name in sources:
    source=root/name;obj=out/(source.stem+'.o')
    subprocess.run([a.compiler,*flags['compile_flags'],'-I'+str(root/'src'),
                    '-c',str(source),'-o',str(obj)],check=True)
    objects.append(str(obj))
subprocess.run([a.compiler,*flags['link_flags'],*objects,'-o',str(out/'observer')],check=True)
for name in ('teacher.f16','oracle.f16'):
    shutil.copyfile(root.parent/'results'/('byte-path-permutation-'+name),out/name)
shutil.copyfile(root/'ids.txt',out/'ids.txt')
subprocess.run([str(out/'observer'),str(out)],check=True)
print(out/'observations.tsv')
