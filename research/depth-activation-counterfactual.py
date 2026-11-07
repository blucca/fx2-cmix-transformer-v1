#!/usr/bin/python3
"""One snapshot, one existing full piece, two no-grad activation counterfactuals."""
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
import argparse
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--job', '--original-root', dest='original_root', type=Path, required=True,
    help='original tree with source/research, incoming/data-manifest.json and training/teacher-cache')
p.add_argument('--kernel', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cpu', '--cpu-core', dest='cpu', type=int, default=1)
p.add_argument('--cpu-seconds', type=int, default=180)
p.add_argument('--wall-seconds', type=int, default=220)
p.add_argument('--address-space-gib', type=int, default=3)
p.add_argument('--source', type=Path, help='research source directory, default original-root/source/research')
p.add_argument('--checkpoint', type=Path, help='training hard-coordinate checkpoint, default original-root/training/latest.pt')
p.add_argument('--initialization', type=Path, help='relocated original12-layer initialization; default checkpoint identity value')
p.add_argument('--data', type=Path, help='relocated original data manifest, default original-root/incoming/data-manifest.json')
p.add_argument('--identity-data-path', help='original checkpoint identity data path when --data is relocated')
p.add_argument('--teacher-cache', type=Path, help='existing cache including unchanged identity.json')
p.add_argument('--dependency-root', type=Path, help='optional already-installed dependency directory')
a = p.parse_args()
assert 10 <= a.cpu_seconds <= 180 and 10 <= a.wall_seconds <= 220
assert 1 <= a.address_space_gib <= 3 and a.cpu >= 0
ORIGINAL = a.original_root.resolve()
OUT = a.output.resolve()
KERNEL = a.kernel.resolve()
SOURCE = (a.source or ORIGINAL/'source/research').resolve()
LATEST = (a.checkpoint or ORIGINAL/'training/latest.pt').resolve()
MANIFEST = (a.data or ORIGINAL/'incoming/data-manifest.json').resolve()
TEACHER_CACHE = (a.teacher_cache or ORIGINAL/'training/teacher-cache').resolve()
IDENTITY_DATA = a.identity_data_path or str(MANIFEST)
INITIALIZATION = a.initialization.resolve() if a.initialization else None
OUT.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(SOURCE), str(SOURCE.parent)]
if a.dependency_root:
    sys.path.append(str(a.dependency_root.resolve()))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_seconds-2,a.cpu_seconds))
START = time.monotonic()
result = dict(schema='hutter.depth8.activation-counterfactual.round449.v1',
    status='running', stage='preflight', started_at=time.time(),
    scope='one original current hard-coordinate snapshot and manifest piece 8; A original activation quantization, B floating activations with identical native-scaled FP32 GEMM',
    top_level_student_forwards=0, optimizer_updates=0, optimizer_instances=0,
    gradient_calls=0, new_teacher_calls=0, new_cache_files=0, capture_calls=0,
    native_calls=0, codec_calls=0, model_exports=0, scale_searches=0,
    scale_calibrations=0, source_writes=0, checkpoint_reads=0)


def save(stage, **updates):
    result.update(stage=stage, cpu_seconds=time.process_time(),
        wall_seconds=time.monotonic()-START,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        updated_at=time.time(), **updates)
    tmp = OUT / 'result.tmp'
    tmp.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
    tmp.replace(OUT / 'result.json')


def limit(signum, frame):
    raise TimeoutError('cpu_limit' if signum == signal.SIGXCPU else 'wall_deadline')


def main():
    os.sched_setaffinity(0, {a.cpu})
    resource.setrlimit(resource.RLIMIT_AS, (a.address_space_gib << 30, a.address_space_gib << 30))
    signal.signal(signal.SIGXCPU, limit)
    signal.signal(signal.SIGALRM, limit)
    signal.alarm(a.wall_seconds)
    source = SOURCE
    required = [source/n for n in ('depth-distillation.py', 'depth_student.py',
        'integer-coordinate-qat.py', 'cpu_model.py', 'cpu_ops.py')]
    required += [LATEST,
        TEACHER_CACHE/'identity.json',
        MANIFEST, KERNEL]
    save('preflight', dependencies=[str(p) for p in required],
        cpu_affinity=list(os.sched_getaffinity(0)), address_space_limit_bytes=a.address_space_gib << 30,
        cpu_limit_seconds=list(resource.getrlimit(resource.RLIMIT_CPU)),
        thread_environment={k:os.environ.get(k) for k in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    for p in required:
        assert p.is_file() and os.access(p, os.R_OK), 'missing readable dependency: '+str(p)
    save('importing')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from depth_student import load_student, student_source_key
    from pysrc.quantization import Quantize
    spec = importlib.util.spec_from_file_location('original_depth_distillation', source/'depth-distillation.py')
    distill = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(distill)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest_path = MANIFEST
    manifest = json.loads(manifest_path.read_text())
    samples = manifest['samples']
    pieces = [(i,lo,hi) for i,s in enumerate(samples) for lo,hi in s['train_pieces']]
    assert manifest['method']=='causal-whole-piece-v1' and manifest['source_bytes']==586459020
    assert len(pieces)==manifest['training_pieces']==1313
    assert sum(hi-lo-1 for _,lo,hi in pieces)==manifest['training_targets']==2941307
    piece_index = 7
    i,lo,hi = pieces[piece_index]
    assert (i,lo,hi)==(0,5321,6521)
    n = hi-lo-1
    assert n==1199
    sample = samples[i]
    tokens_path, priors_path = Path(sample['tokens']), Path(sample['priors'])
    assert 0 <= lo < hi <= tokens_path.stat().st_size
    assert priors_path.stat().st_size==tokens_path.stat().st_size*410
    tokens = np.fromfile(tokens_path,dtype=np.uint8,count=hi-lo,offset=lo)
    prior = np.fromfile(priors_path,dtype='<f2',count=n*205,offset=lo*410).reshape(n,205)
    x = torch.from_numpy(tokens[:-1].astype(np.int64)).unsqueeze(0)
    y = torch.from_numpy(tokens[1:].astype(np.int64)).unsqueeze(0)
    pp = torch.from_numpy(prior).unsqueeze(0)
    bounds = torch.tensor([[0,n]],dtype=torch.int32)
    assert tokens.shape==(n+1,) and bool(torch.isfinite(pp).all())
    assert int(x.max())<205 and int(y.max())<205
    cache_path = TEACHER_CACHE/f'sample-{i:04d}-{lo}-{hi}.npz'
    with np.load(cache_path,allow_pickle=False) as data:
        reference,feature = torch.from_numpy(data['logits']),torch.from_numpy(data['hidden'])
    assert reference.dtype==feature.dtype==torch.float32
    assert tuple(reference.shape)==(1,n,205) and tuple(feature.shape)==(1,n,192)
    assert bool(torch.isfinite(reference).all() & torch.isfinite(feature).all())
    save('inputs_ready', piece=dict(manifest_index_zero_based=piece_index,
        manifest_ordinal=8,sample_index=i,sample_name=sample['name'],lo=lo,hi=hi,
        targets=n,bounds=[[0,n]],selection='same manifest piece as round446',
        tokens_path=str(tokens_path),priors_path=str(priors_path),priors_dtype='float16',
        vocabulary_columns=205), manifest=dict(path=str(manifest_path),
        training_pieces=len(pieces),training_targets=manifest['training_targets']),
        teacher_cache=dict(path=str(cache_path),bytes=cache_path.stat().st_size,
        logits_shape=list(reference.shape),hidden_shape=list(feature.shape),dtype='float32',cache_hits=1))
    latest = LATEST
    with latest.open('rb') as stream:
        st = os.fstat(stream.fileno())
        ckpt = torch.load(stream,map_location='cpu',weights_only=True)
    result['checkpoint_reads'] = 1
    identity = ckpt['identity']
    assert identity['recipe']==distill.RECIPE and identity['data']==IDENTITY_DATA
    assert json.loads((TEACHER_CACHE/'identity.json').read_text())==identity
    assert ckpt['epoch']>=8
    save('loading_student', checkpoint=dict(path=str(latest),bytes=st.st_size,
        inode=st.st_ino,mtime_ns=st.st_mtime_ns,
        cursor={k:ckpt[k] for k in ('epoch','step','next_piece','seen_targets','epoch_targets')},
        epoch_semantics='completed full epochs; cursor is in epoch+1',identity=identity))
    model,adapter_counts,initialization = load_student(INITIALIZATION or Path(identity['checkpoint']),KERNEL,native_scaled_gemm=True)
    from pysrc.model import CastedLinear
    native_source = inspect.getsource(CastedLinear.forward)
    folding = 'linear(x / sa, weight.type_as(x) / sw) * (sa * sw.flatten())'
    b_supported = folding in native_source and 'torch.int' not in native_source
    z,scales,initial_q,raw,frozen,mode = distill.install_student_coordinates(model)
    assert len(z)==75 and len(raw)==59
    assert ckpt['z'].keys()==z.keys() and ckpt['raw'].keys()==raw.keys()
    with torch.no_grad():
        for values,saved in ((z,ckpt['z']),(raw,ckpt['raw'])):
            for key,value in values.items():
                assert value.shape==saved[key].shape and saved[key].dtype==torch.float32
                value.copy_(saved[key])
                assert bool(torch.isfinite(value).all())
        for value in z.values():
            assert bool(((value>=-7)&(value<=7)).all())
    torch.set_rng_state(ckpt['rng'])
    del ckpt,initial_q,initialization
    mode['hard'] = True
    model.train()
    assert model.cfg.dropout==0 and model.cfg.gradient_checkpointing
    named = [('z/'+k,p) for k,p in sorted(z.items())]+[('raw/'+k,p) for k,p in sorted(raw.items())]
    tobytes = lambda p:p.detach().contiguous().numpy().tobytes()
    before = {k:tobytes(p) for k,p in named}
    model_before = {k:tobytes(p) for k,p in model.state_dict().items()}
    activation_modules = {k:m for k,m in model.named_modules() if isinstance(m,Quantize) and m.batched}
    weight_modules = {k:m for k,m in model.named_modules() if isinstance(m,Quantize) and not m.batched}
    assert len(activation_modules)==80 and len(weight_modules)==75
    originals = {k:m.forward for k,m in activation_modules.items()}
    original_weights = {k:m.forward for k,m in weight_modules.items()}
    native_modules = [k for k,m in model.named_modules() if isinstance(m,CastedLinear) and m.native_scaled_gemm]
    rows = []
    fields = ('calls','elements','below_qmin','above_qmax','rounded_below_qmin',
        'rounded_above_qmax','round_to_zero','nonzero_round_to_zero','input_zero','output_zero',
        'quantization_error_squared_sum','input_squared_sum','out_of_range_error_squared_sum')
    for name,module in activation_modules.items():
        q = module.quant
        effective = module.scale.detach().to(q.scale_dtype).to(q.arithmetic_dtype).flatten()
        assert q.buckets==256 and bool(torch.isfinite(effective).all() & (effective!=0).all())
        if q.epsilon is not None:
            effective = (2*(effective>=0).to(effective.dtype)-1)*effective.abs().clamp_min(q.epsilon)
        teacher_name = student_source_key(name)
        rows.append(dict(name=name,teacher_name=teacher_name,
            student_layer=int(name.split('.')[1]) if name.startswith('blocks.') else None,
            teacher_layer=int(teacher_name.split('.')[1]) if teacher_name.startswith('blocks.') else None,
            size=list(module.size),n_blocks=module.n_blocks,block_size=q.block_size,
            qmin=q.qmin,qmax=q.qmax,buckets=q.buckets,
            scale=dict(count=effective.numel(),minimum=float(effective.min()),maximum=float(effective.max()),
                positive=int((effective>0).sum()),negative=int((effective<0).sum()),zero=int((effective==0).sum()),
                scale_dtype=str(q.scale_dtype),arithmetic_dtype=str(q.arithmetic_dtype),epsilon=q.epsilon,
                raw_minimum=float(module.scale.detach().min()),raw_maximum=float(module.scale.detach().max())),
            A={k:0 for k in fields},B={k:0 for k in fields}))
    by_name = {row['name']:row for row in rows}
    phase = {'label':'A'}
    def make_hook(name):
        def hook(module,args,kwargs,output):
            input = args[0] if args else kwargs['input']
            steps = kwargs.get('initialize_scales_steps',args[1] if len(args)>1 else None)
            return_ints = kwargs.get('return_ints',args[2] if len(args)>2 else None)
            assert steps is None and return_ints is False
            assert input.dtype==output.dtype==torch.float32 and input.shape==output.shape
            assert bool(torch.isfinite(input).all() & torch.isfinite(output).all())
            if phase['label']=='B':
                assert output is input
            q = module.quant
            blocks = module.reshape_input(input.detach()).to(q.arithmetic_dtype)
            scale = module.scale.detach().view(-1).to(q.scale_dtype).type_as(blocks)[None,:,None]
            if q.epsilon is not None:
                scale = (2*(scale>=0).to(scale.dtype)-1)*scale.abs().clamp_min(q.epsilon)
            scaled = blocks/scale
            rounded = scaled.round()
            low,high = scaled<q.qmin,scaled>q.qmax
            error = (module.reshape_input(output.detach()).double()-blocks.double()).square()
            counts = dict(calls=1,elements=input.numel(),below_qmin=int(low.sum()),above_qmax=int(high.sum()),
                rounded_below_qmin=int((rounded<q.qmin).sum()),rounded_above_qmax=int((rounded>q.qmax).sum()),
                round_to_zero=int((rounded==0).sum()),nonzero_round_to_zero=int(((rounded==0)&(blocks!=0)).sum()),
                input_zero=int((blocks==0).sum()),output_zero=int((output==0).sum()),
                quantization_error_squared_sum=float(error.sum()),input_squared_sum=float(blocks.double().square().sum()),
                out_of_range_error_squared_sum=float(error.masked_select(low|high).sum()))
            target = by_name[name][phase['label']]
            for key,value in counts.items():
                target[key] += value
            target.setdefault('input_shapes',[]).append(list(input.shape))
        return hook
    handles = [m.register_forward_hook(make_hook(k),with_kwargs=True) for k,m in activation_modules.items()]
    save('student_ready', counts=dict(z_tensors=75,raw_tensors=59,mutable_tensors=134,
        z_elements=sum(p.numel() for p in z.values()),raw_elements=sum(p.numel() for p in raw.values()),
        activation_quantizers=80,weight_quantizers=75,model_state_tensors=len(model_before),
        native_scaled_linear_modules=len(native_modules)),
        hard_mode=True,dropout=model.cfg.dropout,model_training=model.training,
        gradient_checkpointing=model.cfg.gradient_checkpointing,torch_version=torch.__version__,
        torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
        native_scaled_gemm=True,native_scaled_gemm_modules=native_modules,
        floating_activation_support=dict(supported=b_supported,source=str(inspect.getsourcefile(CastedLinear)),
        preserved_expression=folding,reason='FP32 GEMM accepts floating x/sa; original scale fold is retained' if b_supported else 'original native branch source differs or requires integer dtype; B stopped'),
        counter_definitions=dict(out_of_range='input/effective_signed_BF16_scale < qmin or > qmax before rounding',
            round_to_zero='round(input/scale)==0 including original input zeros',
            nonzero_round_to_zero='round(input/scale)==0 and input!=0',
            error='actual hook output minus input; B identity gives zero error',
            B_counts='shadow old-scale range/rounding counts on B floating inputs; no quantization is applied',
            calls='actual forward hook calls; no one-call-per-module assumption'),activation_quantizers=rows)
    outputs,losses,reconstructed = {},{},{}
    for label in ('A','B'):
        if label=='B':
            if not b_supported:
                result['B_status']='stopped'
                break
            def identity_activation(input,initialize_scales_steps,return_ints):
                assert initialize_scales_steps is None and return_ints is False
                return input
            for module in activation_modules.values():
                module.forward = identity_activation
        phase['label'] = label
        save('forward_'+label)
        with torch.no_grad():
            hidden = model.compute_last_activation(input_tokens=x,priors=pp,cumulative_sequence_lengths=bounds)
            logits = model.last_activation_to_logits(hidden,priors=pp)
            result['top_level_student_forwards'] += 1
            assert tuple(logits.shape)==(1,n,205) and tuple(hidden.shape)==(1,n,192)
            assert bool(torch.isfinite(logits).all() & torch.isfinite(hidden).all())
            ce = F.cross_entropy(logits.reshape(-1,205),y.flatten(),reduction='sum')/n
            kl = F.kl_div(logits.log_softmax(-1),reference.log_softmax(-1),reduction='sum',log_target=True)/n
            wh = .1*(hidden-feature).square().mean(-1).sum()/n
            values = dict(ce=float(ce),kl=float(kl),hidden_weighted=float(wh),hidden_unweighted=float(wh)/.1)
            values['old_data_objective']=values['ce']+values['kl']+values['hidden_weighted']
            values['proposed_2KL_data_objective']=2*values['kl']+values['hidden_weighted']
            lp,tq = logits.double().log_softmax(-1),reference.double().log_softmax(-1)
            expected = dict(ce=float(-lp.gather(-1,y[...,None]).mean()),
                kl=float((tq.exp()*(tq-lp)).sum()/n),hidden_weighted=float(.1*(hidden.double()-feature.double()).square().mean()))
            for key,value in expected.items():
                assert math.isfinite(values[key]) and math.isclose(values[key],value,rel_tol=2e-5,abs_tol=2e-6), label+' loss reconstruction '+key
            reconstructed[label] = dict(float64_reconstruction=expected,
                absolute_errors={key:abs(values[key]-value) for key,value in expected.items()})
            losses[label] = values
            outputs[label] = (logits,hidden)
        save('losses_'+label,losses=losses,loss_reconstruction=reconstructed,activation_quantizers=rows)
    for name,module in activation_modules.items():
        module.forward = originals[name]
    for handle in handles:
        handle.remove()
    assert all(m.forward is original_weights[k] for k,m in weight_modules.items())
    assert all(before[k]==tobytes(p) for k,p in named)
    assert all(model_before[k]==tobytes(p) for k,p in model.state_dict().items())
    assert all(p.grad is None for _,p in named)
    aggregates = {}
    for label in outputs:
        aggregate = {key:sum(row[label][key] for row in rows) for key in fields}
        aggregate['visited_modules'] = sum(row[label]['calls']>0 for row in rows)
        aggregate['uncalled_modules'] = [row['name'] for row in rows if not row[label]['calls']]
        aggregate['call_multiplicity'] = {str(v):sum(row[label]['calls']==v for row in rows) for v in sorted({row[label]['calls'] for row in rows})}
        for row in rows:
            m = row[label]
            for key in ('below_qmin','above_qmax','rounded_below_qmin','rounded_above_qmax','round_to_zero','nonzero_round_to_zero','input_zero','output_zero'):
                assert 0<=m[key]<=m['elements']
            assert m['below_qmin']+m['above_qmax']<=m['elements']
            assert m['nonzero_round_to_zero']<=m['round_to_zero'] and m['input_zero']<=m['round_to_zero']
            for key in ('quantization_error_squared_sum','input_squared_sum','out_of_range_error_squared_sum'):
                assert math.isfinite(m[key]) and m[key]>=0
            m['out_of_range_fraction'] = (m['below_qmin']+m['above_qmax'])/m['elements'] if m['elements'] else None
            m['nonzero_round_to_zero_fraction'] = m['nonzero_round_to_zero']/m['elements'] if m['elements'] else None
            m['error_to_input_squared_ratio'] = m['quantization_error_squared_sum']/m['input_squared_sum'] if m['input_squared_sum'] else None
        aggregate['out_of_range_fraction'] = (aggregate['below_qmin']+aggregate['above_qmax'])/aggregate['elements']
        aggregate['nonzero_round_to_zero_fraction'] = aggregate['nonzero_round_to_zero']/aggregate['elements']
        aggregate['error_to_input_squared_ratio'] = aggregate['quantization_error_squared_sum']/aggregate['input_squared_sum']
        aggregate['out_of_range_error_fraction'] = aggregate['out_of_range_error_squared_sum']/aggregate['quantization_error_squared_sum'] if aggregate['quantization_error_squared_sum'] else 0.
        aggregates[label] = aggregate
    differences = {}
    if 'B' in outputs:
        assert result['top_level_student_forwards']==2
        assert aggregates['B']['quantization_error_squared_sum']==0
        for index,kind in enumerate(('logits','hidden')):
            a,b = outputs['A'][index].double(),outputs['B'][index].double()
            d = b-a
            differences[kind] = dict(elements=d.numel(),squared_error_sum=float(d.square().sum()),
                mean_squared_difference=float(d.square().mean()),root_mean_squared_difference=float(d.square().mean().sqrt()),
                mean_absolute_difference=float(d.abs().mean()),maximum_absolute_difference=float(d.abs().max()),
                relative_l2_to_A=float(d.norm()/a.norm()))
        result['loss_delta_B_minus_A'] = {k:losses['B'][k]-losses['A'][k] for k in losses['A']}
        result['B_status']='passed'
    arrays = dict(input_tokens=x.numpy(),target_tokens=y.numpy(),priors_f16=pp.numpy(),bounds=bounds.numpy(),
        teacher_logits=reference.numpy(),teacher_hidden=feature.numpy())
    for label,(logits,hidden) in outputs.items():
        arrays[label+'_logits']=logits.numpy()
        arrays[label+'_hidden']=hidden.numpy()
    raw_path = OUT/'raw-readout.npz'
    with raw_path.open('xb') as f:
        np.savez(f,**arrays)
    signal.alarm(0)
    save('complete',status='passed',completed_at=time.time(),losses=losses,loss_reconstruction=reconstructed,
        activation_quantizers=rows,activation_aggregates=aggregates,output_differences=differences,
        adapter_operator_counts=adapter_counts,
        native_calls_scope='new native evaluator or codec executions; original CPU KDA shared library remains in use',
        raw_readout=dict(path=str(raw_path),bytes=raw_path.stat().st_size,format='npz without compression',keys=sorted(arrays),
            contents='scientific inputs/teacher cache arrays/A-B logits and hidden only; model weights absent'),
        checks=dict(losses_finite=True,losses_reconstructed=True,counts_bounded=True,
            all_134_z_raw_byte_identical=True,all_model_state_byte_identical=True,
            all_155_scale_tensors_byte_identical=True,all_75_weight_forward_functions_unchanged=True,
            parameter_grad_slots_empty=True,activation_forward_methods_restored=True,
            no_grad=True,dropout_zero=True,hard_coordinates_fixed=True))
    print('ACTIVATION_SUMMARY '+json.dumps({k:result[k] for k in ('status','checkpoint','losses',
        'activation_aggregates','output_differences','cpu_seconds','wall_seconds','peak_rss_kib')},separators=(',',':')),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        signal.alarm(0)
        save('failed',status='failed',error_kind=type(error).__name__,error=str(error),failed_stage=result['stage'],completed_at=time.time())
        traceback.print_exc()
        print('ACTIVATION_FAILURE '+json.dumps({k:result.get(k) for k in
            ('status','failed_stage','error_kind','error','cpu_seconds','wall_seconds','peak_rss_kib')},separators=(',',':')),flush=True)
        raise SystemExit(124 if isinstance(error,TimeoutError) else 1)
