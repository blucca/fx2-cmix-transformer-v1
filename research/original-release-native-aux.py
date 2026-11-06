#!/usr/bin/env python3
"""One bounded original-release dictionary/order and native section delivery.

NN/range payload and full enwik9 inverse are separate original stages. The
section fixture has an explicit marker payload, and creates actual cmix and
archive9 envelopes with the unchanged native header/extractor.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--job',type=Path,required=True)
a=p.parse_args();R=a.root.resolve();J=a.job.resolve();A=J/'assets'
assert not (J/'state.json').exists(), 'inspect the existing single job'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
 os.environ[key]='1'
start=time.monotonic();child_before=resource.getrusage(resource.RUSAGE_CHILDREN)
row={'schema':'hutter.original-release-native-aux.v1','status':'running',
     'cpu_limit_seconds':1800,'address_space_limit_gib':24,'threads':1,
     'prepared_transformer_calls':0,'prepared_payload_coding_calls':0,'full_original_restoration_calls':0,
     'fixture_payload':'section-extraction marker','commands':[]}

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def cpu():
 u=resource.getrusage(resource.RUSAGE_CHILDREN)
 return time.process_time()+u.ru_utime+u.ru_stime-child_before.ru_utime-child_before.ru_stime

def save(stage,**extra):
 row.update(stage=stage,updated_at_unix=time.time(),cpu_seconds=cpu(),wall_seconds=time.monotonic()-start,**extra)
 tmp=J/'state.tmp';tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(J/'state.json')
 print(json.dumps({'stage':stage,'status':row['status'],'cpu_seconds':row['cpu_seconds'],**extra}),flush=True)

def run(stage,command,cwd=J,descendants=1):
 # Native extractors spawn at most two generic decoder commands. Allocate
 # their CPU ceiling from the same remaining original 0.5 CPU-hour ledger.
 remaining=int(1800-cpu())-10
 ceiling=remaining//descendants
 assert ceiling>=10,'original shared CPU budget exhausted'
 def limits():
  resource.setrlimit(resource.RLIMIT_CPU,(ceiling,ceiling))
  resource.setrlimit(resource.RLIMIT_AS,(24<<30,24<<30))
 save(stage,child_cpu_limit_seconds=ceiling)
 before=resource.getrusage(resource.RUSAGE_CHILDREN)
 with (J/(stage+'.log')).open('w') as log:
  out=subprocess.run(list(map(str,command)),cwd=cwd,stdout=log,stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,preexec_fn=limits,timeout=1800)
 after=resource.getrusage(resource.RUSAGE_CHILDREN)
 used=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime)
 row['commands'].append({'stage':stage,'command':list(map(str,command)),
  'exit_code':out.returncode,'cpu_seconds':used,'peak_rss_kib':after.ru_maxrss})
 (J/(stage+'.exit')).write_text(str(out.returncode)+'\n')
 save(stage+'_terminal',last_exit_code=out.returncode)
 assert out.returncode==0,stage+' failed; inspect its log'
 assert cpu()<1800,'shared CPU budget gate'

save('input_gate')
try:
 assets=json.loads((A/'manifest.json').read_text())
 for name,identity in assets['files'].items():
  path=A/name
  assert path.stat().st_size==identity['bytes'] and digest(path)==identity['sha256'],name
 B=A/'cmix.upx';W=A/'original-static15.tfwc3'
 assert B.stat().st_size==181956 and B.read_bytes()[:4]==b'\x7fELF'
 assert W.stat().st_size==2904325
 original_dict=R/'upstream/dictionary/english.dic'
 original_order=R/'upstream/src/readalike_prepr/data/new_article_order'
 for name,path in [('dictionary',original_dict),('order',original_order)]:
  expected=assets['original_input_identities'][name]
  assert path.stat().st_size==expected['bytes'] and digest(path)==expected['sha256']
 assert original_dict.stat().st_size==411996 and original_order.stat().st_size==1094862
 available=os.statvfs('/data');free=available.f_bavail*available.f_frsize
 assert free>=2<<30,'need 2 GiB available for this short original auxiliary stage'
 row['asset_provenance']=assets
 aux=J/'aux';aux.mkdir()
 D=aux/'comp_dict';O=aux/'comp_order'
 for label,source,out in [('dictionary',original_dict,D),('order',original_order,O)]:
  existing=A/out.name
  if existing.exists():out.write_bytes(existing.read_bytes())
  else:run('encode_'+label,[B,'-c',source,out])
 source=J/'native-selfextract-check.cpp'
 source.write_text('#include <cstring>\n#include "'+str(R/'upstream/src/readalike_prepr/self_extract.h')+'"\n'
  'static_assert(sizeof(HeaderInfo)==16);\n'
  'int main(int argc,char**argv){if(argc!=2)return 2;'
  'if(!strcmp(argv[1],"compressor"))return selfextract_comp();'
  'if(!strcmp(argv[1],"archive"))return selfextract_decomp();return 2;}\n')
 checker=J/'native-selfextract-check'
 run('compile_native_extract',['c++','-O0','-std=c++17',source,'-o',checker])
 marker=J/'section-marker.bin';marker.write_bytes(b'SECTION EXTRACTION ONLY\n')
 package=J/'package'
 run('assemble_section_fixture',['python3','-B',J/'archive-package.py',
  '--binary',B,'--compressed-dictionary',D,'--compressed-order',O,'--weights',W,
  '--raw-payload',marker,'--vocab',R/'prepared/vocab.bin','--prepared-bytes','10000','--out',package])
 run('native_compressor_extract',[checker,'compressor'],cwd=package,descendants=3)
 assert (package/'.dict').read_bytes()==original_dict.read_bytes()
 assert (package/'.new_article_order').read_bytes()==original_order.read_bytes()
 assert (package/'.tfweights').read_bytes()==W.read_bytes()
 assert (package/'.decomp_bin').read_bytes()==B.read_bytes()
 save('native_compressor_passed')
 run('native_archive_extract',[checker,'archive'],cwd=package,descendants=2)
 assert (package/'.dict').read_bytes()==original_dict.read_bytes()
 assert (package/'.tfweights').read_bytes()==W.read_bytes()
 package_result=json.loads((package/'package.json').read_text())
 payload=marker.read_bytes()
 header=(10000|(1<<39)).to_bytes(5,'big')+(R/'prepared/vocab.bin').read_bytes()
 assert (package/'.ready4cmix_decomp').read_bytes()==header+payload
 components={name:{'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)}
  for name,path in [('binary',B),('compressed_dictionary',D),('compressed_order',O),('weights',W)]}
 common_charge=2*B.stat().st_size+2*D.stat().st_size+O.stat().st_size+2*W.stat().st_size+69
 assert package_result['full_s']==common_charge+len(payload)
 save('complete',status='passed',components=components,common_two_copy_charge_bytes=common_charge,
  actual_dictionary_roundtrip=True,actual_order_roundtrip=True,
  actual_packed_native_compressor_sections=True,actual_packed_native_archive_sections=True,
  fixture_package=package_result,formal_fullstream_s=None)
 (J/'job.exit').write_text('0\n')
except Exception as exc:
 save('failed',status='failed',failure_stage=row.get('stage'),error=type(exc).__name__+': '+str(exc))
 (J/'job.exit').write_text('1\n')
 raise
