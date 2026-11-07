#!/usr/bin/python3
"""Single-lambda train-piece residual bridge and three fixed held-piece paths.

The only insertion hook is at student block4, after the production skip pop.
It uses the unmodified block3 output saved by a read-only forward hook.
Teacher W0 and zero8 use original hard weights and fixed signed-BF16 scales.
"""
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
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--job','--original-root',dest='original_root',type=Path,required=True,
    help='original asset tree with incoming/data-manifest.json and its train/validation captures')
p.add_argument('--checkpoint',type=Path,required=True,
    help='original twelve-layer W0 float32 checkpoint; no trained/latest checkpoint is used')
p.add_argument('--kernel',type=Path,required=True,help='already-built original cpu-kda.so')
p.add_argument('--output',type=Path,required=True,help='fresh directory for scientific result/arrays')
p.add_argument('--source',type=Path,help='original research source; default original-root/source/research')
p.add_argument('--data',type=Path,help='original mapped manifest; default original-root/incoming/data-manifest.json')
p.add_argument('--dependency-root',type=Path,help='optional already-installed NumPy/Torch/typing dependency directory')
p.add_argument('--cpu','--cpu-core',dest='cpu',type=int,default=1)
p.add_argument('--cpu-seconds',type=int,default=180)
p.add_argument('--wall-seconds',type=int,default=220)
p.add_argument('--address-space-gib',type=int,default=4)
a=p.parse_args()
assert 10<=a.cpu_seconds<=180 and 10<=a.wall_seconds<=220
assert 1<=a.address_space_gib<=4 and a.cpu>=0
ORIGINAL=a.original_root.resolve()
OUT=a.output.resolve()
SOURCE=(a.source or ORIGINAL/'source/research').resolve()
KERNEL=a.kernel.resolve()
CHECKPOINT=a.checkpoint.resolve()
MANIFEST=(a.data or ORIGINAL/'incoming/data-manifest.json').resolve()
OUT.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(SOURCE),str(SOURCE.parent)]
if a.dependency_root:sys.path.append(str(a.dependency_root.resolve()))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('TMPDIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR'):
    path=OUT/key.lower()
    path.mkdir()
    os.environ[key]=str(path)
resource.setrlimit(resource.RLIMIT_CPU,(a.cpu_seconds-2,a.cpu_seconds))
START = time.monotonic()
result = dict(schema='hutter.depth8.middle-bridge.round450.v1', status='running',
    stage='preflight',started_at=time.time(),
    scope='one fixed train piece, one original held piece; floating RMS-normalized residual ridge diagnostic',
    top_level_teacher_forwards=0,top_level_student_forwards=0,
    optimizer_updates=0,optimizer_instances=0,gradient_calls=0,
    teacher_checkpoint_reads=0,student_initialization_checkpoint_reads=0,
    trained_checkpoint_reads=0,new_cache_files=0,ppmd_capture_calls=0,
    native_evaluator_calls=0,codec_calls=0,model_exports=0,
    scale_searches=0,scale_calibrations=0,ridge_solves=0,
    native_int4_bridge_runs=0,full_training_runs=0,real_coded_stream_runs=0,
    original_source_writes=0,original_checkpoint_writes=0,
    original_cache_writes=0,original_native_writes=0)


def save(stage,**updates):
    result.update(stage=stage,cpu_seconds=time.process_time(),
        wall_seconds=time.monotonic()-START,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        updated_at=time.time(),**updates)
    tmp=OUT/'result.tmp'
    tmp.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
    tmp.replace(OUT/'result.json')


def limit(signum,frame):
    raise TimeoutError('cpu_limit' if signum==signal.SIGXCPU else 'wall_deadline')


def main():
    os.sched_setaffinity(0,{a.cpu})
    resource.setrlimit(resource.RLIMIT_AS,(a.address_space_gib<<30,a.address_space_gib<<30))
    signal.signal(signal.SIGXCPU,limit)
    signal.signal(signal.SIGALRM,limit)
    signal.alarm(a.wall_seconds)
    required=[SOURCE/n for n in ('depth-distillation.py','depth_student.py',
        'integer-coordinate-qat.py','cpu_model.py','cpu_ops.py')]+[CHECKPOINT,MANIFEST,KERNEL]
    save('preflight',dependencies=[str(p) for p in required],
        cpu_affinity=list(os.sched_getaffinity(0)),
        address_space_limit_bytes=a.address_space_gib<<30,cpu_limit_seconds=list(resource.getrlimit(resource.RLIMIT_CPU)),
        thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    for path in required:
        assert path.is_file() and os.access(path,os.R_OK),'missing readable dependency: '+str(path)
    save('importing')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from cpu_model import load_model
    from depth_student import load_student,student_source_key,SKIP_SOURCE_INDICES
    spec=importlib.util.spec_from_file_location('original_depth_distillation',SOURCE/'depth-distillation.py')
    distill=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(distill)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(450)
    assert torch.get_num_threads()==torch.get_num_interop_threads()==1
    manifest=json.loads(MANIFEST.read_text())
    assert manifest['method']=='causal-whole-piece-v1' and manifest['source_bytes']==586459020
    samples=manifest['samples']
    assert [s['name'] for s in samples]==['causal','long-audit']
    train_pieces=[(i,lo,hi) for i,s in enumerate(samples) for lo,hi in s['train_pieces']]
    assert len(train_pieces)==manifest['training_pieces']==1313
    assert sum(hi-lo-1 for _,lo,hi in train_pieces)==manifest['training_targets']==2941307
    assert train_pieces[7]==(0,5321,6521)
    score=samples[0]['score']
    assert score['targets']==1082505
    boundaries_path=Path(score['boundaries'])
    boundaries=np.fromfile(boundaries_path,dtype='<i4')
    assert boundaries.shape==(474,) and boundaries[0]==0
    assert bool(np.all(boundaries[1:]>boundaries[:-1]))
    assert int(boundaries[-1])==Path(score['tokens']).stat().st_size==1082978
    assert sum(int(hi-lo-1) for lo,hi in zip(boundaries[:-1],boundaries[1:]))==1082505
    held_index=next(i for i,(lo,hi) in enumerate(zip(boundaries[:-1],boundaries[1:])) if 1025<=hi-lo-1<=2048)
    held_lo,held_hi=map(int,boundaries[held_index:held_index+2])
    assert (held_index,held_lo,held_hi)==(7,14457,16064)
    roles={
        'train':dict(record=samples[0],lo=5321,hi=6521,plan_piece_index=7,
            role='train',article=1447,source_token_range=[1256446,1257646],
            selection='original manifest ordinal 8; same piece as rounds446 and449'),
        'held':dict(record=score,lo=held_lo,hi=held_hi,plan_piece_index=held_index,
            role='validation',article=3363,source_token_range=[5055675,5057282],
            selection='first original validation complete piece with 1025–2048 targets')}
    data={}
    pieces={}
    for name,info in roles.items():
        rec=info['record'];lo,hi=info['lo'],info['hi'];n=hi-lo-1
        tp,pp=Path(rec['tokens']),Path(rec['priors'])
        assert tp.resolve().is_relative_to((ORIGINAL/'incoming').resolve())
        assert pp.resolve().is_relative_to((ORIGINAL/'incoming').resolve())
        assert 0<=lo<hi<=tp.stat().st_size and pp.stat().st_size==tp.stat().st_size*410
        tok=np.fromfile(tp,dtype=np.uint8,count=hi-lo,offset=lo)
        prior=np.fromfile(pp,dtype='<f2',count=n*205,offset=lo*410).reshape(n,205)
        x=torch.from_numpy(tok[:-1].astype(np.int64))[None]
        y=torch.from_numpy(tok[1:].astype(np.int64))[None]
        priors=torch.from_numpy(prior)[None]
        bounds=torch.tensor([[0,n]],dtype=torch.int32)
        assert tok.shape==(n+1,) and bool(torch.isfinite(priors).all())
        assert int(x.max())<205 and int(y.max())<205
        data[name]=(x,y,priors,bounds)
        pieces[name]={k:v for k,v in info.items() if k!='record'}
        pieces[name].update(targets=n,local_token_range=[lo,hi],
            tokens_path=str(tp),priors_path=str(pp),priors_dtype='float16',
            vocabulary_columns=205,bounds=[[0,n]],
            causal_history='unchanged original full-pass PPMd f16 capture; NN reset at this complete piece')
    assert pieces['train']['tokens_path']!=pieces['held']['tokens_path']
    assert roles['train']['article']!=roles['held']['article']
    save('inputs_ready',pieces=pieces,manifest=dict(path=str(MANIFEST),training_pieces=1313,
        training_targets=2941307,held_pieces=473,held_targets=1082505,boundaries_path=str(boundaries_path)),
        role_contract='train rows enter ridge; validation rows enter metrics/oracle only')
    teacher,teacher_adapter=load_model(CHECKPOINT,KERNEL,native_scaled_gemm=True)
    from pysrc.model import rms_norm,CastedLinear
    result['teacher_checkpoint_reads']+=1
    teacher.requires_grad_(False).eval()
    student,student_adapter,initialization=load_student(CHECKPOINT,KERNEL,native_scaled_gemm=True)
    result['student_initialization_checkpoint_reads']+=1
    z,scales,initial_q,raw,frozen,mode=distill.install_student_coordinates(student)
    assert len(z)==75 and len(raw)==59
    mode['hard']=True
    student.requires_grad_(False).eval()
    for value in z.values():value.requires_grad_(False)
    for model in (teacher,student):
        assert model.cfg.dropout==0 and model.cfg.window_size_multipliers is None
        assert model.cfg.value_embedding_pattern is None
        assert model.cfg.attention_implementation=='sdpa' and model.cfg.gradient_checkpointing
    assert teacher.cfg.n_layers==12 and student.cfg.n_layers==8
    ts=teacher.state_dict();ss=student.state_dict()
    for key,value in ss.items():
        reference=ts[student_source_key(key)]
        if key=='skip_connection_weights.value':reference=reference[list(SKIP_SOURCE_INDICES)]
        assert torch.equal(value.view(torch.int32),reference.view(torch.int32)),key
    byte_data=lambda p:p.detach().contiguous().numpy().tobytes()
    before_t={k:byte_data(p) for k,p in ts.items()}
    before_s={k:byte_data(p) for k,p in ss.items()}
    before_z={k:byte_data(p) for k,p in z.items()}
    del ts,ss,initial_q,frozen,initialization
    save('models_ready',checkpoint=dict(path=str(CHECKPOINT),bytes=CHECKPOINT.stat().st_size,
        teacher='original W0 twelve layers',student='hard zero8 mapped only from original W0'),
        mapping=dict(kept_original_layers=[0,1,2,3,8,9,10,11],skip_source_indices=[2,3,4,5],
            teacher_skip_pairs=[[3,8],[2,9],[1,10],[0,11]],student_skip_pairs=[[3,4],[2,5],[1,6],[0,7]]),
        bridge_contract=dict(main='x3 + RMSnorm(x3) @ A.T + b',
            insertion='student block4 prehook, after production source3 skip is popped',
            pre_block4='main(x3) + original_teacher_skip[2] * unmodified x3',
            oracle='teacher x7 + original_teacher_skip[2] * unmodified x3',
            rms_norm_source=inspect.getsource(rms_norm),bridge_arithmetic='floating FP32',
            ridge_solution='float64; FP32 inference matrix and bias'),
        counts=dict(teacher_state_tensors=len(before_t),student_state_tensors=len(before_s),
            student_hard_coordinates=75,student_raw_tensors=59),
        torch_version=torch.__version__,torch_threads=1,torch_interop_threads=1,
        hard_mode=True,fixed_scales=True,native_scaled_gemm=True)
    captures={};context={}
    def teacher_hook(layer):
        def hook(module,args,output):
            assert layer not in context['capture']
            context['capture'][layer]=output.detach().clone()
        return hook
    teacher_handles=[teacher.blocks[i].register_forward_hook(teacher_hook(i)) for i in (3,7,8,9,10,11)]
    def student_x3(module,args,output):
        assert 'x3' not in context['student']
        context['student']['x3']=output.detach().clone()
        assert torch.equal(output.view(torch.int32),context['teacher'][3].view(torch.int32)),'first four layers mismatch'
    sh=student.blocks[3].register_forward_hook(student_x3)
    def insertion(module,args,kwargs):
        x3=context['student']['x3']
        skip=student.skip_connection_weights()[0]*x3
        expected=x3+skip
        assert torch.equal(args[0].view(torch.int32),expected.view(torch.int32)),'source3 skip input contract'
        phase=context['phase']
        if phase=='zero':return None
        main=context['teacher'][7] if phase=='oracle' else x3+F.linear(rms_norm(x3),A,b)
        assert bool(torch.isfinite(main).all())
        context['student']['bridge_main']=main.detach().clone()
        return (main+skip,)+args[1:],kwargs
    bh=student.blocks[4].register_forward_pre_hook(insertion,with_kwargs=True)
    def oracle_tail(index):
        def hook(module,args,output):
            if context.get('phase')=='oracle':
                assert torch.equal(output.view(torch.int32),context['teacher'][index+4].view(torch.int32)),'oracle tail block mismatch '+str(index)
        return hook
    tail_handles=[student.blocks[i].register_forward_hook(oracle_tail(i)) for i in range(4,8)]
    def forward(model,name):
        x,y,pp,bounds=data[name]
        hidden=model.compute_last_activation(input_tokens=x,priors=pp,cumulative_sequence_lengths=bounds)
        logits=model.last_activation_to_logits(hidden,priors=pp)
        n=y.numel()
        assert tuple(hidden.shape)==(1,n,192) and tuple(logits.shape)==(1,n,205)
        assert bool(torch.isfinite(hidden).all() & torch.isfinite(logits).all())
        return logits,hidden
    loss_checks={}
    def metrics(output,reference,name,label):
        logits,hidden=output;tl,th=reference;y=data[name][1];n=y.numel()
        ce=F.cross_entropy(logits.reshape(-1,205),y.flatten(),reduction='sum')/n
        kl=F.kl_div(logits.log_softmax(-1),tl.log_softmax(-1),reduction='sum',log_target=True)/n
        hmse=(hidden-th).square().mean()
        values=dict(ce=float(ce),kl_teacher_q_to_student_p=float(kl),hidden_mse=float(hmse),hidden_weighted=.1*float(hmse))
        lp,tq=logits.double().log_softmax(-1),tl.double().log_softmax(-1)
        check=dict(ce=float(-lp.gather(-1,y[...,None]).mean()),
            kl_teacher_q_to_student_p=float((tq.exp()*(tq-lp)).sum()/n),
            hidden_mse=float((hidden.double()-th.double()).square().mean()))
        for key,v in check.items():
            assert math.isfinite(values[key]) and math.isclose(values[key],v,rel_tol=2e-5,abs_tol=2e-6),name+' '+label+' loss reconstruction '+key
        loss_checks[name+'_'+label]=dict(float64_reconstruction=check,
            absolute_errors={k:abs(values[k]-v) for k,v in check.items()})
        return values
    arrays={};losses={};references={}
    with torch.no_grad():
        for name in ('train','held'):
            context.clear();context['capture']={}
            save('teacher_'+name)
            references[name]=forward(teacher,name)
            result['top_level_teacher_forwards']+=1
            captures[name]=context['capture']
            for key,value in zip(('input_tokens','target_tokens','priors_f16','bounds'),data[name]):
                arrays[name+'_'+key]=value.numpy()
            for index in (3,7):arrays[name+'_teacher_x'+str(index)]=captures[name][index].numpy()
            for key,value in zip(('logits','hidden'),references[name]):arrays[name+'_teacher_'+key]=value.numpy()
            losses[name]={'teacher':metrics(references[name],references[name],name,'teacher')}
            if name=='train':
                context.update(teacher=captures[name],student={},phase='zero')
                save('student_train_zero')
                train_zero=forward(student,name)
                result['top_level_student_forwards']+=1
                losses[name]['zero']=metrics(train_zero,references[name],name,'zero')
                assert torch.equal(context['student']['x3'],captures[name][3])
                X=rms_norm(captures[name][3]).reshape(-1,192).double()
                D=captures[name][7].reshape(-1,192).double()-captures[name][3].reshape(-1,192).double()
                design=torch.cat((X,torch.ones(X.shape[0],1,dtype=torch.float64)),dim=1)
                xtx=design.T@design;xty=design.T@D
                lam=1e-3*float((X.T@X).trace())/192
                ridge=torch.eye(193,dtype=torch.float64)*lam;ridge[-1,-1]=0
                save('ridge_solve',ridge=dict(lambda_rule='1e-3 * trace(X.T X) / 192',
                    lambda_value=lam,bias_penalized=False,train_rows=int(X.shape[0]),
                    input_dim=192,output_dim=192,lambda_candidates=1,rank_candidates=1,
                    target='teacher x7 - teacher x3',features='production float32 RMSnorm(x3), promoted to float64'))
                theta=torch.linalg.solve(xtx+ridge,xty)
                result['ridge_solves']+=1
                assert bool(torch.isfinite(theta).all())
                A=theta[:192].T.float().contiguous();b=theta[192].float().contiguous()
                equation=(xtx+ridge)@theta-xty
                result['ridge']['normal_equation_relative_l2']=float(equation.norm()/xty.norm())
                for key,value in dict(bridge_A_f32=A,bridge_b_f32=b,ridge_XTX_f64=xtx,
                    ridge_XTY_f64=xty,ridge_theta_f64=theta).items():arrays[key]=value.numpy()
            else:
                for phase in ('zero','fitted','oracle'):
                    context.update(teacher=captures[name],student={},phase=phase)
                    save('student_held_'+phase)
                    output=forward(student,name)
                    result['top_level_student_forwards']+=1
                    losses[name][phase]=metrics(output,references[name],name,phase)
                    for key,value in zip(('logits','hidden'),output):arrays[name+'_'+phase+'_'+key]=value.numpy()
                    if phase=='fitted':arrays['held_fitted_bridge_main']=context['student']['bridge_main'].numpy()
                    if phase=='oracle':
                        for actual,expected in zip(output,references[name]):
                            assert torch.equal(actual.view(torch.int32),expected.view(torch.int32)),'oracle full output mismatch'
        bridge_error={}
        for name in ('train','held'):
            x3,x7=captures[name][3],captures[name][7]
            fit=x3+F.linear(rms_norm(x3),A,b)
            old=(x3.double()-x7.double()).square().mean()
            new=(fit.double()-x7.double()).square().mean()
            assert math.isfinite(float(old)) and math.isfinite(float(new)) and float(old)>0
            bridge_error[name]=dict(unbridged_x7_mse=float(old),fitted_x7_mse=float(new),
                residual_error_ratio=float(new/old),residual_error_reduction_fraction=1-float(new/old))
    for handle in teacher_handles+[sh,bh]+tail_handles:handle.remove()
    assert all(before_t[k]==byte_data(p) for k,p in teacher.state_dict().items())
    assert all(before_s[k]==byte_data(p) for k,p in student.state_dict().items())
    assert all(before_z[k]==byte_data(p) for k,p in z.items())
    assert all(p.grad is None for m in (teacher,student) for p in m.parameters())
    assert all(p.grad is None for p in z.values())
    assert result['top_level_teacher_forwards']==2 and result['top_level_student_forwards']==4
    assert result['ridge_solves']==1 and result['trained_checkpoint_reads']==0
    delta={k:losses['held']['fitted'][k]-losses['held']['zero'][k] for k in losses['held']['zero']}
    gap=losses['held']['zero']['ce']-losses['held']['teacher']['ce']
    recovery=(losses['held']['zero']['ce']-losses['held']['fitted']['ce'])/gap if gap else None
    arrays_path=OUT/'raw-readout.npz'
    with arrays_path.open('xb') as f:np.savez(f,**arrays)
    signal.alarm(0)
    save('complete',status='passed',completed_at=time.time(),losses=losses,
        loss_reconstruction=loss_checks,bridge_midpoint_error=bridge_error,
        held_loss_delta_fitted_minus_zero=delta,held_ce_gap_recovery_fraction=recovery,
        bridge_payload_estimate=dict(matrix_rows=192,matrix_columns=192,int4_matrix_bytes=18432,
            fp32_bias_bytes=768,bf16_row_scale_bytes=384,nominal_payload_bytes=19584,
            float32_diagnostic_matrix_bias_bytes=(192*192+192)*4,
            scope='new-family payload estimate; exporter/codec metadata, compression and coded package unmeasured'),
        raw_readout=dict(path=str(arrays_path),bytes=arrays_path.stat().st_size,
            format='npz without compression',keys=sorted(arrays),
            contents='scientific input/capture/output arrays and learned bridge ridge coefficients; original checkpoint absent'),
        checks=dict(all_losses_finite=True,all_losses_reconstructed=True,
            oracle_logits_bit_exact=True,oracle_hidden_bit_exact=True,oracle_all_four_tail_blocks_bit_exact=True,
            first_four_layers_bit_exact_train_and_all_held_paths=True,
            original_x3_skip_saved_before_bridge=True,production_pre_bridge_skip_input_bit_exact=True,
            train_held_roles_disjoint=True,held_rows_in_ridge=0,
            teacher_student_all_original_state_byte_identical=True,student_coordinates_byte_identical=True,
            all_parameter_grad_slots_empty=True,no_grad=True,all_fixed_scales_unchanged=True,
            ridge_solves=1,lambda_candidates=1,model_forward_hooks_removed=True),
        adapter_operator_counts=dict(teacher=teacher_adapter,student=student_adapter,
            scope='shared cpu_model import-adapter counters; model-depth call counts are explicit above'))
    print('MIDDLE_BRIDGE_SUMMARY '+json.dumps({k:result[k] for k in ('status','pieces','losses',
        'bridge_midpoint_error','held_loss_delta_fitted_minus_zero','held_ce_gap_recovery_fraction',
        'cpu_seconds','wall_seconds','peak_rss_kib')},separators=(',',':')),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        signal.alarm(0)
        save('failed',status='failed',error_kind=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        traceback.print_exc()
        raise SystemExit(1)
