#!/usr/bin/env python3
"""Original ONNXRuntime#32918 fixture: six pass controls + pre-Q boundaries.

Run: python3 research/ort-qdq-fusion-control.py minimal_model.onnx temp/ort32918
Dependencies: numpy==2.5.3 onnxruntime==1.30.0. Original attachment stays upstream.
"""
from pathlib import Path
import argparse
import json
import platform
import struct

import numpy as np
import onnxruntime as ort

def varint(b,p):
 v=s=0
 while True:
  c=b[p]; p+=1; v|=(c&127)<<s
  if c<128:return v,p
  s+=7

def fields(b):
 p=0; out=[]
 while p<len(b):
  k,p=varint(b,p); no,w=k>>3,k&7
  if w==0:v,p=varint(b,p)
  elif w==2:n,p=varint(b,p); v=b[p:p+n]; p+=n
  elif w in (1,5):n=8 if w==1 else 4; v=b[p:p+n]; p+=n
  else:raise ValueError((no,w,p))
  out.append((no,w,v))
 return out

def vals(b,n):return [v for k,w,v in fields(b) if k==n]
def one(b,n,default=None):return next(iter(vals(b,n)),default)
def s(v):return v.decode() if isinstance(v,bytes) else v


def encode_varint(n):
 out=bytearray()
 while n>127:
  out.append((n&127)|128);n>>=7
 out.append(n)
 return bytes(out)


def pack(no,w,value):
 return encode_varint((no<<3)|w)+(encode_varint(value) if w==0
     else encode_varint(len(value))+value if w==2 else value)


ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('model',type=Path)
ap.add_argument('out',type=Path)
args=ap.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
model=args.model.read_bytes();graph=one(model,7)
constants={s(one(t,8)):t for t in vals(graph,5)}
scale=struct.unpack('<f',one(constants['ncf_qdq_scale_241607'],9))[0]
zero=one(constants['ncf_qdq_zero_468157'],9)[0]
assert scale==.03125 and zero==128
rng=np.random.default_rng(42)
feed={'x':rng.standard_normal((1,3,64,64)).astype(np.float32)}
rows={};outputs={}
MODES=[('reference','ORT_DISABLE_ALL',set()),
       ('all','ORT_ENABLE_ALL',set()),
       ('no_matmul_add','ORT_ENABLE_ALL',{'MatMulAddFusion'}),
       ('no_conv_bn','ORT_ENABLE_ALL',{'ConvBNFusion'}),
       ('no_nchwc','ORT_ENABLE_ALL',{'NchwcTransformer'}),
       ('no_conv_bn_nchwc','ORT_ENABLE_ALL',{'ConvBNFusion','NchwcTransformer'})]


def run(path,name,level,disabled):
 options=ort.SessionOptions()
 options.graph_optimization_level=getattr(ort.GraphOptimizationLevel,level)
 options.intra_op_num_threads=1;options.inter_op_num_threads=1
 options.optimized_model_filepath=str(args.out/(name+'.onnx'))
 session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'],
                             disabled_optimizers=disabled)
 out=session.run(None,feed)
 assert out and all(np.isfinite(x).all() for x in out)
 return out


for name,level,disabled in MODES:
 out=run(args.model,name,level,disabled)[0]
 assert list(out.shape)==[1,48,64,64]
 outputs[name]=out
 optimized=one((args.out/(name+'.onnx')).read_bytes(),7)
 nodes=[{'name':s(one(n,3,b'')),'op':s(one(n,4,b'')),
         'domain':s(one(n,7,b''))} for n in vals(optimized,1)]
 rows[name]={'level':level,'disabled':sorted(disabled),'graph_nodes':nodes}
reference=outputs['reference']
for name,row in rows.items():
 difference=outputs[name].astype(np.float64)-reference.astype(np.float64)
 row['max_abs_vs_reference']=float(np.max(np.abs(difference)))
 row['changed_elements_vs_reference']=int(np.count_nonzero(difference))
 row['equal_to_all']=bool(np.array_equal(outputs[name],outputs['all']))
 row['graph_op_counts']={op:sum(n['op']==op for n in row['graph_nodes'])
                        for op in sorted({n['op'] for n in row['graph_nodes']})}

# This fixture's t_12 has the same FLOAT32 shape as its existing final output.
# For other tensor types/models, construct ValueInfo with the ONNX library.
original_output=vals(graph,12)[0]
extra=b''.join(pack(no,w,b't_12' if no==1 else v)
               for no,w,v in fields(original_output))
probe_graph=graph+pack(12,2,extra)
probe_model=b''.join(pack(no,w,probe_graph if no==7 else v)
                     for no,w,v in fields(model))
assert [s(one(v,1)) for v in vals(one(probe_model,7),12)]==[
    s(one(original_output,1)),'t_12']
probe_path=args.out/'pre-q-probe.onnx'
probe_path.write_bytes(probe_model)
probes={};probe_rows={}
for name,level,disabled in (MODES[0],MODES[1],MODES[-1]):
 out=run(probe_path,'probe_'+name,level,disabled)
 assert len(out)==2 and np.array_equal(out[0],outputs[name]),name
 probes[name]=out[1]
 probe_rows[name]={'original_endpoint_equal':True}

boundaries=[]
for p in np.argwhere(outputs['all']!=reference):
 idx=tuple(p);a=probes['reference'][idx];b=probes['all'][idx]
 def q(x):
  return int(np.clip(np.rint(x/np.float32(scale))+zero,0,255))
 qa,qb=q(a),q(b)
 assert (qa-zero)*scale==float(reference[idx])
 assert (qb-zero)*scale==float(outputs['all'][idx])
 boundaries.append({'index':[int(x) for x in p],
  'pre_q_reference':float(a),'pre_q_all':float(b),
  'reference_fp32_hex':hex(struct.unpack('<I',struct.pack('<f',a))[0]),
  'all_fp32_hex':hex(struct.unpack('<I',struct.pack('<f',b))[0]),
  'threshold':(float(reference[idx])+float(outputs['all'][idx]))/2,
  'reference_uint8':qa,'all_uint8':qb,
  'reference_output':float(reference[idx]),'all_output':float(outputs['all'][idx])})

result={'issue':'https://github.com/microsoft/onnxruntime/issues/32918',
 'runtime_version':ort.__version__,'numpy_version':np.__version__,
 'python':platform.python_version(),'machine':platform.machine(),
 'rng_seed':42,'input_shape':[1,3,64,64],'output_shape':[1,48,64,64],
 'scale':scale,'zero_point':zero,'sessions':rows,'probe_sessions':probe_rows,
 'pre_q_max_abs_diff':float(np.max(np.abs(probes['all'].astype(np.float64)
                                       -probes['reference'].astype(np.float64)))),
 'quantization_boundaries':boundaries}
(args.out/'control-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({name:{k:v for k,v in row.items() if k not in
                       ('graph_nodes','graph_op_counts')} for name,row in rows.items()},indent=2))
print('Readout:',args.out/'control-result.json')
