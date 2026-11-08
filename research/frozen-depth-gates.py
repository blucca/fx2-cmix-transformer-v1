#!/usr/bin/env python3
"""Frozen selected-e12 hard-W, four zero-gate VJPs and one fixed +0.001 control."""
import argparse, bisect, hashlib, json, math, os, resource, signal, struct, sys, time, traceback
from dataclasses import replace
from pathlib import Path

sys.dont_write_bytecode = True
EXPECTED_W = '8dd9113b676830598d2dad13999f0a90ce738a84fe5ed482fa749aed9db7ce9e'
BANKS = [0,1,2,3,0,1,2,3,4,5,6,7]
TINY = 1e-3
HIDDEN_ATOL = LOGIT_ATOL = 2e-6
RTOL, NLL_ATOL = 2e-6, 1e-10


def read_c3_material(path, np):
    """Existing v2 models plus C++ static15 case; RoPE tables carry no payload."""
    from pysrc import weights_compress as w
    data = path.read_bytes()
    assert data[:8] == b'FX2TFWC3'
    count, = struct.unpack('<I', data[8:12]); assert count == 294
    dec, models = w.BinDecoder(data[12:]), w._Models()
    def meta():
        b = dec.decode_tree(models.meta_tree(), 8); models.meta_prev = b; return b
    tensors, derived = {}, {}
    for _ in range(count):
        length = meta(); chars = bytearray(); c2 = c1 = 0
        for _ in range(length):
            ch = dec.decode_tree(models.name_tree(c2,c1),8); chars.append(ch); c2,c1=c1,ch
        name = chars.decode(); dtype, ndim = meta(), meta()
        shape = tuple(sum(meta() << (8*k) for k in range(4)) for _ in range(ndim))
        n = math.prod(shape); encoding = meta(); typ = np.dtype(w._DTYPE_TO_NUMPY[dtype])
        if encoding == 6:
            assert dtype == 0
            freq = [meta() | (meta()<<8) for _ in range(14)]; freq.append(4096-sum(freq))
            assert all(f > 0 for f in freq)
            cumulative = [0]
            for f in freq: cumulative.append(cumulative[-1]+f)
            values = np.empty(n,dtype=np.int8)
            for i in range(n):
                unit = dec.range >> 12; value = dec.code // unit
                assert value < 4096
                sym = bisect.bisect_right(cumulative,value)-1
                dec.code -= unit*cumulative[sym]; dec.range = unit*freq[sym]
                while dec.range < 1<<24:
                    dec.code = ((dec.code<<8)|dec._byte()) & 0xffffffff; dec.range <<= 8
                values[i] = sym-7
            array = values.reshape(shape)
        elif encoding == w.ENC_INT4:
            array = np.array([dec.decode_tree(models.int4,4)-7 for _ in range(n)],dtype=np.int8).reshape(shape)
        elif encoding == w.ENC_BF16:
            values = np.empty(n,dtype=np.uint16)
            for i in range(n):
                hi=dec.decode_tree(models.bf16_hi,8); lo=dec.decode_tree(models.bf16_lo_tree(hi),8)
                values[i]=(hi<<8)|lo
            array=values.reshape(shape)
        elif encoding == w.ENC_PLANE4:
            array=np.frombuffer(bytes(dec.decode_tree(models.plane[i&3],8) for i in range(n*4)),dtype=typ).reshape(shape)
        elif encoding in (w.ENC_ROPE_SIN,w.ENC_ROPE_COS):
            assert name in ('rope.sin','rope.cos') and dtype == 2 and shape == (131072,32)
            derived[name]={'shape':list(shape),'encoding':encoding,'source':'CUDA-derived format metadata; CPU uses recipe-generated piece-local RoPE'}
            continue
        else:
            assert encoding == w.ENC_RAW
            array=np.frombuffer(bytes(dec.decode_tree(models.raw,8) for _ in range(n*typ.itemsize)),dtype=typ).reshape(shape)
        assert name not in tensors and bool(np.isfinite(array).all())
        tensors[name]=(dtype,array)
    assert len(tensors)+len(derived)==294
    assert tuple(tensors['config.ints'][1]) == (205,192,8,64,3,768,1024,64,3,4,10000)
    assert tensors['config.kimi'][1].tolist() == [1,1,1,0]*2
    assert tensors['rope.inv_freq'][1].shape == (32,)
    return tensors,derived


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','hard-weights','kernel','data','selection','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); timestamp=time.time()
    r=dict(schema='hutter.frozen-e12.gated-depth-probe.round482.v1',status='running',stage='preflight',
        started_at_epoch=timestamp,banks=BANKS,gates_initial=[0.]*4,
        active_gates=[1.]*8,tiny_control={'gate_vector':[TINY]*4,'selection':'predeclared uniform positive control'},
        tolerance=dict(hidden_atol=HIDDEN_ATOL,logits_atol=LOGIT_ATOL,rtol=RTOL,target_nll_atol=NLL_ATOL),
        teacher_forwards=0,fixed8_forwards=0,gated12_forwards=0,gate_vjps=0,
        optimizer_updates=0,optimizer_instances=0,gate_fits=0,native_evaluator_calls=0,new_native_builds=0,
        teacher_cache_reads=0,teacher_cache_writes=0,model_exports=0,pieces=[],
        gradient_scope='four raw whole-block gates at zero; original hard-activation STE and CPU adjoints; W and BF16 scales frozen',
        finite_change_scope='one all-positive 1e-3 vector; actual hard rounding; report-only held pieces',
        cpu_limit_seconds=580,wall_limit_seconds=750,memory_max_bytes=6<<30,
        paths={k:str(getattr(a,k).resolve()) for k in ('source','checkpoint','hard_weights','kernel','data','selection')})
    def save(stage=None):
        if stage: r['stage']=stage
        r.update(cpu_seconds=time.process_time(),wall_seconds=time.monotonic()-started,
                 peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        f=out/'result.tmp'; f.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n'); f.replace(out/'result.json')
        receipt={k:r[k] for k in ('schema','status','stage','started_at_epoch','cpu_seconds','wall_seconds','peak_rss_kib',
                 'teacher_forwards','fixed8_forwards','gated12_forwards','gate_vjps','optimizer_updates','new_native_builds')}
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def limit(sig,frame): raise TimeoutError('resource_stop_signal_'+str(sig))
    try:
        for env in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[env]='1'
        core=min(os.sched_getaffinity(0)); os.sched_setaffinity(0,{core})
        resource.setrlimit(resource.RLIMIT_AS,(6<<30,6<<30)); resource.setrlimit(resource.RLIMIT_CPU,(578,580))
        for sig in (signal.SIGXCPU,signal.SIGALRM,signal.SIGTERM): signal.signal(sig,limit)
        signal.alarm(750); save()
        for name in ('checkpoint','hard_weights','kernel','data','selection'):
            assert getattr(a,name).is_file(),'missing input: '+str(getattr(a,name))
        sys.path[:0]=[str(a.source/'research'),str(a.source),'/data/workspace/temp/hutter/round414/deps']
        import numpy as np
        import torch
        import torch.nn.functional as F
        from cpu_model import load_model
        from depth_student import load_student,state_counts
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        # build_model installs the existing CPU FLA import adapters first.
        model,_,init=load_student(a.checkpoint,a.kernel,native_scaled_gemm=True)
        from pysrc.export_weights import bf16_bits,quantize_weight_rows
        r['torch_version']=torch.__version__; r['cpu_affinity']=[core]
        r['input_sha256']={name:hashlib.sha256(getattr(a,name).read_bytes()).hexdigest() for name in
            ('checkpoint','kernel','data','selection')}
        r['source_sha256']={name:hashlib.sha256((a.source/name).read_bytes()).hexdigest() for name in
            ('pysrc/model.py','pysrc/quantization.py','research/cpu_model.py','research/cpu_ops.py','research/depth_student.py','training_recipes/cooldown.py')}
        wbytes=a.hard_weights.read_bytes(); wsha=hashlib.sha256(wbytes).hexdigest()
        assert wsha==EXPECTED_W and len(wbytes)==1960982
        material,derived=read_c3_material(a.hard_weights,np)
        state=model.state_dict(); mapped={}; used=set()
        def floating(name):
            dtype,arr=material[name]; used.add(name)
            if dtype==1: arr=(arr.astype(np.uint32)<<16).view(np.float32)
            else: assert dtype==2,name
            return torch.from_numpy(arr.copy())
        for key,target in state.items():
            if key+'.q' in material:
                dtype,q=material[key+'.q']; assert dtype==0; used.add(key+'.q')
                scale=floating(key+'.scale'); value=torch.from_numpy(q.astype(np.float32))*scale[:,None]
            elif key.endswith('.quantize_weight.scale'):
                value=floating(key[:-len('.quantize_weight.scale')]+'.weight.scale')
            else: value=floating(key)
            assert value.dtype==target.dtype==torch.float32 and value.shape==target.shape,key
            mapped[key]=value
        assert used==set(material)-{'config.ints','config.kimi','rope.inv_freq'}
        model.load_state_dict(mapped,strict=True); model.requires_grad_(False).eval()
        for key,value in model.state_dict().items():
            assert torch.equal(value.view(torch.int32),mapped[key].view(torch.int32)),key
            if key+'.q' in material:
                q,s=quantize_weight_rows(value,model.state_dict()[key[:-7]+'.quantize_weight.scale'])
                assert np.array_equal(q.numpy(),material[key+'.q'][1]),key
                assert np.array_equal(bf16_bits(s),material[key+'.scale'][1]),key
        r['actual_w']=dict(sha256=wsha,bytes=len(wbytes),selected_epoch=12,selected_step=949,
            whole_material_tensors=294,model_state_tensors=289,all_payloads_loaded=True,
            all_loaded_f32_bits_equal=True,hard_requantized_int4_and_bf16_scale_equal=True,
            counts=state_counts(model),derived_rope=derived,provenance='original complete CE+KL selected-e12 hard export')
        del mapped,state,material,wbytes,init
        teacher,_=load_model(a.checkpoint,a.kernel,native_scaled_gemm=True)
        teacher.requires_grad_(False).eval()
        assert model.cfg.dropout==teacher.cfg.dropout==0 and model.cfg.window_size==1024
        assert model.cfg.prior_embedding and model.cfg.value_embedding_pattern is None
        r['forward_contract']=dict(prior='original score f16 PPMd 205 columns',rope='piece-local positions start at offset zero',
            causal_bounds='one complete held piece, cumulative bounds [[0,n]]',native_scaled_gemm=True,
            activation_and_weight_scale_dtype='BF16',activation_dtype='FP32',w_updates=0,
            state='each CPU KDA invocation allocates its own recurrent state/conv context; 12 position wrappers')
        chosen=json.loads(a.selection.read_text()); assert [x['piece'] for x in chosen['pieces']]==[0,7,14]
        manifest=json.loads(a.data.read_text()); sample=manifest['samples'][0]; assert sample['name']=='causal'
        score=sample['score']; bounds=np.fromfile(score['boundaries'],dtype='<i4').tolist()
        assert len(bounds)==474 and score['targets']==1082505
        gates=torch.nn.Parameter(torch.zeros(4,dtype=torch.float32))
        banks=model.blocks; cfg=model.cfg; oldskip=model.skip_connection_weights.value
        skips=torch.nn.Parameter(torch.cat((torch.zeros(2),oldskip.detach())),requires_grad=False)
        r['twelve_skip_vector']=skips.tolist()
        class Position(torch.nn.Module):
            def __init__(self,position,bank):
                super().__init__(); self.position=position; self.bank=bank
            def forward(self,x,**kw):
                branch=self.bank(x,**kw)
                if 4<=self.position<8:
                    g=gates[self.position-4]
                    return (1-g)*x+g*branch
                return branch
        slots=torch.nn.ModuleList([Position(i,banks[b]) for i,b in enumerate(BANKS)])
        assert len({id(p) for p in slots})==12 and len({id(p.bank) for p in slots})==8
        assert all(slots[i].bank is slots[i+4].bank for i in range(4))
        def mode(gated):
            model.blocks=slots if gated else banks
            model.cfg=replace(cfg,n_layers=12,attention_weight_sharing_pattern=BANKS,
                              mlp_weight_sharing_pattern=BANKS) if gated else cfg
            model.skip_connection_weights.value=skips if gated else oldskip
        def forward(m,x,pp,nb):
            h=m.compute_last_activation(input_tokens=x,priors=pp,cumulative_sequence_lengths=nb)
            z=m.last_activation_to_logits(h,priors=pp)
            assert bool(torch.isfinite(h).all() & torch.isfinite(z).all())
            return h,z
        def measures(z,ref,y,n):
            lp=z.double().log_softmax(-1); tp=ref.double().log_softmax(-1)
            return dict(ce=float(F.nll_loss(lp.reshape(-1,205),y,reduction='mean')),
                        teacher_kl=float(F.kl_div(lp,tp,reduction='sum',log_target=True)/n))
        save('models_ready')
        for selected in chosen['pieces']:
            piece=selected['piece']; lo,hi=bounds[piece:piece+2]; n=hi-lo-1
            assert [lo,hi]==selected['local_token_range'] and n==selected['targets']
            tok=np.fromfile(score['tokens'],dtype=np.uint8,count=n+1,offset=lo)
            prior=np.fromfile(score['priors'],dtype='<f2',count=n*205,offset=lo*410).reshape(n,205)
            x=torch.from_numpy(tok[:-1].astype(np.int64))[None]; y=torch.from_numpy(tok[1:].astype(np.int64))
            pp=torch.from_numpy(prior)[None]; nb=torch.tensor([[0,n]],dtype=torch.int32)
            row=dict(piece=piece,regime=selected['regime'],targets=n,local_token_range=[lo,hi],
                tokens_sha256=hashlib.sha256(tok.tobytes()).hexdigest(),priors_sha256=hashlib.sha256(prior.tobytes()).hexdigest())
            with torch.no_grad():
                th,ref=forward(teacher,x,pp,nb); r['teacher_forwards']+=1
                row['teacher_ce']=float(F.cross_entropy(ref.reshape(-1,205).double(),y))
                assert abs(row['teacher_ce']-selected['teacher_raw_nll'])<1e-8
                mode(False); h8,z8=forward(model,x,pp,nb); r['fixed8_forwards']+=1
            mode(True)
            h,z=forward(model,x,pp,nb); r['gated12_forwards']+=1
            torch.testing.assert_close(h.detach(),h8,atol=HIDDEN_ATOL,rtol=RTOL)
            torch.testing.assert_close(z.detach(),z8,atol=LOGIT_ATOL,rtol=RTOL)
            base=measures(z.detach(),ref,y,n); same=measures(z8,ref,y,n)
            assert abs(base['ce']-same['ce'])<=NLL_ATOL
            row.update(g0_control=dict(hidden_max_abs=float((h.detach()-h8).abs().max()),
                logits_max_abs=float((z.detach()-z8).abs().max()),target_nll_abs=abs(base['ce']-same['ce']),passed=True),g0=base)
            ce=F.cross_entropy(z.reshape(-1,205),y,reduction='sum')/n
            kl=F.kl_div(z.log_softmax(-1),ref.log_softmax(-1),reduction='sum',log_target=True)/n
            assert ce.requires_grad and kl.requires_grad
            gc=torch.autograd.grad(ce,gates,retain_graph=True,allow_unused=False)[0]
            gk=torch.autograd.grad(kl,gates,allow_unused=False)[0]; r['gate_vjps']+=2
            assert bool(torch.isfinite(gc).all() & torch.isfinite(gk).all())
            row['gate_derivatives']=dict(ce=gc.tolist(),teacher_kl=gk.tolist(),
                jointly_improving_positive_coordinates=[i for i in range(4) if gc[i]<0 and gk[i]<0],
                predeclared_uniform_direction_ce=float(gc.sum()),predeclared_uniform_direction_kl=float(gk.sum()))
            del h,z,ce,kl,th,h8,z8
            with torch.no_grad():
                gates.fill_(TINY); ht,zt=forward(model,x,pp,nb); r['gated12_forwards']+=1
                tiny=measures(zt,ref,y,n); gates.zero_()
            row['tiny_control']=dict(gates=[TINY]*4,means=tiny,
                delta={k:tiny[k]-base[k] for k in base},
                ste_linear_prediction=dict(ce=TINY*float(gc.sum()),teacher_kl=TINY*float(gk.sum())))
            r['pieces'].append(row); r['targets']=sum(x['targets'] for x in r['pieces']); save('piece_'+str(piece)+'_complete')
            del x,y,pp,nb,tok,prior,ref,ht,zt,gc,gk
        assert r['targets']==3934 and r['gate_vjps']==6
        total=r['targets']
        r['target_weighted']={label:{k:math.fsum(p['targets']*p[label][k] for p in r['pieces'])/total for k in ('ce','teacher_kl')} for label in ('g0',)}
        r['target_weighted']['tiny_delta']={k:math.fsum(p['targets']*p['tiny_control']['delta'][k] for p in r['pieces'])/total for k in ('ce','teacher_kl')}
        r['target_weighted']['gate_derivatives']={k:[math.fsum(p['targets']*p['gate_derivatives'][k][i] for p in r['pieces'])/total for i in range(4)] for k in ('ce','teacher_kl')}
        assert bool((gates.detach()==0).all()) and all(not p.requires_grad for b in banks for p in b.parameters())
        r.update(status='passed',held_usage='report-only; gate vector, tiny step, and tolerances fixed before execution')
        save('complete'); print(json.dumps({k:r[k] for k in ('status','targets','cpu_seconds','peak_rss_kib','target_weighted')}))
    except BaseException as exc:
        r.update(status='bounded_stop' if isinstance(exc,TimeoutError) else 'failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        save('stopped'); raise
    finally:
        signal.alarm(0)


if __name__=='__main__': main()
