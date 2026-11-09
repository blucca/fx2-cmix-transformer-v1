"""One original full12 q7 article forward/VJP on a free CUDA resource.

KDA uses the pure-PyTorch recurrent reference in place of the CPU C ABI;
conv and gated norm keep the same reference functions on CUDA. Attention uses
dense math-SDPA. FP32 integer-unit linear arithmetic, the q7 initialization,
fixed scales, article, cached teacher and objective keep their original forms.
Optimizer updates and native evaluations are zero.

The measured v3 compute body is retained. RESULT_PATH, Q7_CUDA_INPUT_DIR and
Q7_CUDA_SOURCE_DIR provide portable paths; prepare the one-assert source copy
as documented in q7-cuda-resource.md. Kaggle's mounted bundle remains supported.
"""
import json,os,sys,time,signal,traceback
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
started=time.monotonic()
OUT=Path(os.environ.get('RESULT_PATH', '/kaggle/working/q7-full1199-resource.json' if Path('/kaggle/working').is_dir() else 'temp/q7-cuda-full1199-resource.json'))
OUT.parent.mkdir(parents=True,exist_ok=True)
r={'status':'starting','scope':'original-q7-full12-1199-resource','targets':1199,'optimizer_updates':0,'native_evaluations':0,'wall_limit_seconds':840}
def save(**d):
 r.update(d);r['wall_seconds']=time.monotonic()-started;OUT.write_text(json.dumps(r,indent=2));print(json.dumps({'stage':r.get('stage'),'status':r['status'],'wall_seconds':r['wall_seconds']}),flush=True)
def expired(*_):
 save(status='stopped',error='resource-gate-time-limit');raise SystemExit(124)
signal.signal(signal.SIGALRM,expired);signal.alarm(840)
try:
 # Kaggle extracts uploaded ZIP datasets before mounting them.
 root=Path(os.environ['Q7_CUDA_INPUT_DIR']) if os.environ.get('Q7_CUDA_INPUT_DIR') else next(Path('/kaggle/input').rglob('sample.npz')).parent
 src=Path(os.environ.get('Q7_CUDA_SOURCE_DIR',str(root/'source')));sys.path[:0]=[str(src/'research'),str(src),str(root/'deps')]
 import torch,numpy as np
 from torch.nn import functional as F
 from torch.nn.attention import sdpa_kernel,SDPBackend
 import cpu_model,importlib.util
 spec=importlib.util.spec_from_file_location('q7_original',src/'research/integer-coordinate-qat.py');qat=importlib.util.module_from_spec(spec);spec.loader.exec_module(qat);qat.SOFT_LEVELS=list(range(-3,4))
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(414)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 assert torch.cuda.is_available()
 save(stage='loading',python=sys.version.split()[0],torch=torch.__version__,cuda_build=torch.version.cuda,devices=[{'name':torch.cuda.get_device_name(i),'capability':list(torch.cuda.get_device_capability(i)),'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())],used_device=0,compute_dtype='float32',native_hardware_bf16=torch.cuda.is_bf16_supported(including_emulation=False),kda_backend='pure-torch-FP32-recurrent-reference',attention_backend='dense-math-SDPA',activation_quantizer='original-untiled-STE',source_cpu_abi_unchanged=True)
 model,counts=cpu_model.load_model(src/'models/6m-q4-fp32.tch',kernel=None,native_scaled_gemm=True)
 z,scales,q,frozen=qat.install_q7_coordinates(model)
 assert len(z)==111 and sum(v.numel() for v in z.values())==5868864 and len(frozen)==429
 assert model.cfg.n_layers==12 and sum(model.cfg.kimi_linear)==9 and model.cfg.d_model==192 and model.cfg.gradient_checkpointing and model.cfg.dropout==0 and model.cfg.window_size==1024
 for v in z.values():v.data=v.data.cuda()
 for s in scales.values():s.data=s.data.cuda()
 model.cuda()
 assert all(torch.equal(model.get_parameter(k[:-7]+'.quantize_weight.scale'),scales[k]) for k in z)
 with np.load(root/'sample.npz',allow_pickle=False) as f:
  tokens=f['tokens'];priors=f['priors'];teacher=f['teacher_logits']
 assert tokens.shape==(1200,) and priors.shape==teacher.shape==(1,1199,205) and priors.dtype==np.float16 and teacher.dtype==np.float32
 x=torch.from_numpy(tokens[:-1].astype(np.int64)).unsqueeze(0).cuda();y=torch.from_numpy(tokens[1:].astype(np.int64)).unsqueeze(0).cuda();p=torch.from_numpy(priors).cuda();reference=torch.from_numpy(teacher).cuda();bounds=torch.tensor([[0,1199]],dtype=torch.int32,device='cuda')
 save(stage='forward',coordinate_tensors=111,coordinate_elements=5868864,frozen_tensors=429,article=[0,5321,6521],data_readback=json.loads((root/'input-readback.json').read_text()))
 torch.cuda.synchronize();t=time.monotonic()
 with sdpa_kernel(backends=[SDPBackend.MATH]):
  logits=model.compute_logits(input_tokens=x,priors=p,cumulative_sequence_lengths=bounds)
  torch.cuda.synchronize();forward_seconds=time.monotonic()-t
  assert logits.shape==(1,1199,205) and torch.isfinite(logits).all()
  ce=F.cross_entropy(logits.detach().view(-1,205),y.flatten(),reduction='none').float().sum()
  kl=F.kl_div(logits.log_softmax(-1),reference.log_softmax(-1),reduction='sum',log_target=True)
  save(stage='backward',forward_seconds=forward_seconds,ce_mean=float(ce)/1199,teacher_kl_mean=float(kl.detach())/1199)
  t=time.monotonic();(2*kl).backward();torch.cuda.synchronize();data_backward_seconds=time.monotonic()-t
 for v in z.values():
  assert v.grad is not None and torch.isfinite(v.grad).all();v.grad.div_(1199)
 t=time.monotonic();soft_nats=0.
 for v in z.values():
  h=qat.entropy(v);assert torch.isfinite(h);soft_nats+=float(h.detach());(2*h/qat.FULL_TARGETS).backward()
 torch.cuda.synchronize();rate_backward_seconds=time.monotonic()-t
 finite=0;nonzero=0;norm2=0.
 for key,v in z.items():
  assert v.grad is not None and torch.isfinite(v.grad).all();finite+=1;nonzero+=int((v.grad!=0).any());norm2+=float(v.grad.double().square().sum())
  assert torch.equal(v.detach().cpu(),q[key].float()),key
 for key,parameter in model.named_parameters():
  assert not parameter.requires_grad and parameter.grad is None and torch.equal(parameter.detach().cpu(),frozen[key]),key
 total=forward_seconds+data_backward_seconds+rate_backward_seconds
 save(status='passed',stage='complete',forward_seconds=forward_seconds,data_backward_seconds=data_backward_seconds,rate_backward_seconds=rate_backward_seconds,complete_forward_backward_seconds=total,targets_per_compute_second=1199/total,soft_entropy_nats=soft_nats,objective=2*float(kl.detach())/1199+2*soft_nats/qat.FULL_TARGETS,finite_gradient_tensors=finite,nonzero_gradient_tensors=nonzero,gradient_l2=norm2**.5,coordinates_and_frozen_state_unchanged=True,adapter_counts=counts,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
except Exception as e:
 save(status='failed',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc());raise
finally:signal.alarm(0)
