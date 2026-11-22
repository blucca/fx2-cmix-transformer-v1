"""Read one pinned public ONNX source and its exact graph contracts."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import urllib.request

ROOT=Path(__file__).resolve().parent
URL='https://huggingface.co/Xenova/distilgpt2/resolve/a41c10485c18a64b6606729b6a082330cbd8f49e/onnx/decoder_with_past_model.onnx'
BYTES=328939971
OID='2c9b7dd7f11facae49943e28ef08ba91d37850232962b5e945bb8bc6ede5495b'
MODEL=ROOT/'source.onnx'

def save(stage,**extra):
    (ROOT/'prepare-state.json').write_text(json.dumps({'stage':stage,'at':datetime.now(timezone.utc).isoformat(),**extra},indent=2))

save('download')
if not MODEL.exists():
    part=MODEL.with_suffix('.onnx.part')
    with urllib.request.urlopen(URL,timeout=120) as response,part.open('wb') as out:
        size=0
        while True:
            block=response.read(1024*1024)
            if not block:break
            out.write(block);size+=len(block)
            assert size<=BYTES,'Pinned model byte budget exceeded'
            if size%(32*1024*1024)==0:
                save('download',bytes=size);print('DOWNLOAD',size,flush=True)
    assert size==BYTES,(size,BYTES)
    part.rename(MODEL)
digest=hashlib.file_digest(MODEL.open('rb'),'sha256').hexdigest()
assert MODEL.stat().st_size==BYTES and digest==OID,'HF LFS source identity mismatch'
save('inspect',bytes=BYTES)
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
m=onnx.load(MODEL)
inits={x.name:x for x in m.graph.initializer}
selected=[]
for i,n in enumerate(m.graph.node):
    if n.op_type in ('Gemm','MatMul'):
        attrs={a.name:onnx.helper.get_attribute_value(a) for a in n.attribute}
        w=inits.get(n.input[1])
        selected.append({'index':i,'op':n.op_type,'name':n.name,'inputs':list(n.input),'outputs':list(n.output),'attrs':attrs,
                         'B_constant':w is not None,'B_shape':list(w.dims) if w else None,'B_dtype':w.data_type if w else None})
def info(x):
    t=x.type.tensor_type
    return {'name':x.name,'dtype':t.elem_type,'dims':[d.dim_value if d.HasField('dim_value') else d.dim_param for d in t.shape.dim]}
raw=Path('/proc/cpuinfo').read_text()
flags=next(l.split(':',1)[1].split() for l in raw.splitlines() if l.startswith('flags'))
summary={'source_url':URL,'source_oid':OID,'source_bytes':BYTES,'python':sys.version,'numpy':np.__version__,'onnx':onnx.__version__,'ort':ort.__version__,
         'cpu':next(l.split(':',1)[1].strip() for l in raw.splitlines() if l.startswith('model name')),
         'features':{f:f in flags for f in ('avx2','fma','avx512f','avx512_vnni','avx_vnni')},
         'nodes':len(m.graph.node),'ops':dict(Counter(n.op_type for n in m.graph.node)),
         'opsets':[(x.domain,x.version) for x in m.opset_import],
         'inputs':[info(x) for x in m.graph.input],'outputs':[info(x) for x in m.graph.output],
         'Gemm_MatMul':selected,'initializers':len(inits)}
assert ort.__version__=='1.31.1'
(ROOT/'graph-contract.json').write_text(json.dumps(summary,indent=2))
print('PREPARED',summary['cpu'],summary['features'],'nodes',summary['nodes'],'Gemm/MatMul',len(selected),flush=True)
save('prepared',graph_contract='graph-contract.json')
(ROOT/'PREPARED').write_text('Source identity and whole graph inventory complete\n')
