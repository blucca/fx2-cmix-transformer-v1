#!/usr/bin/env python3
"""One read-only, hard-coordinate, complete-piece gradient decomposition."""
import argparse
import importlib.util
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
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--job', type=Path, required=True,
                    help='Original eight-block job with incoming/, training/ and teacher-cache')
parser.add_argument('--kernel', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True, help='Fresh experiment scratch directory')
parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent)
parser.add_argument('--cpu', type=int, default=1)
args = parser.parse_args()
OUT, ORIGINAL, SOURCE, KERNEL = (p.resolve() for p in
                                (args.output, args.job, args.source, args.kernel))
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT/'result.json').exists(), 'Use a fresh result directory'
START = time.monotonic()
LABELS = ['ce', 'kl', 'hidden_weighted', 'rate', 'old_total', 'proposed_total']
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
DIRECTIONS = [(4, 1), (4, 0), (5, 1), (5, 0)]
result = dict(schema='hutter.depth8.single-piece-gradient.round446.v1',
              status='running', stage='preflight', started_at=time.time(),
              scope='ordinary Euclidean parameter gradients; one original train piece',
              optimizer_updates=0, optimizer_instances=0, new_teacher_calls=0,
              new_cache_files=0, native_calls=0, capture_calls=0, codec_calls=0,
              top_level_student_forwards=0, data_gradient_vjps=0, rate_gradient_vjps=0,
              norm_labels=LABELS, component_pairs=[list(p) for p in PAIRS],
              recipe_to_data_pairs=[list(p) for p in DIRECTIONS])


def save(stage, **updates):
    result.update(stage=stage, cpu_seconds=time.process_time(),
                  wall_seconds=time.monotonic()-START,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  updated_at=time.time(), **updates)
    p = OUT / 'result.json'
    tmp = OUT / 'result.tmp'
    tmp.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
    tmp.replace(p)


def limit(signum, frame):
    raise TimeoutError('cpu_limit' if signum == signal.SIGXCPU else 'wall_deadline')



def main():
    os.sched_setaffinity(0, {args.cpu})
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (178, 180))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1' 
    signal.signal(signal.SIGXCPU, limit)
    signal.signal(signal.SIGALRM, limit)
    signal.alarm(220)
    save('preflight', cpu_affinity=list(os.sched_getaffinity(0)),
         address_space_limit_bytes=4 << 30,
         cpu_limit_seconds=list(resource.getrlimit(resource.RLIMIT_CPU)),
         thread_environment={k:os.environ.get(k) for k in
                             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    source = SOURCE
    sys.path.insert(0, str(source))
    required = [source / n for n in ('depth-distillation.py', 'depth_student.py',
                                    'integer-coordinate-qat.py', 'cpu_model.py', 'cpu_ops.py')]
    required += [ORIGINAL / 'training/latest.pt',
                 ORIGINAL / 'training/teacher-cache/identity.json',
                 ORIGINAL / 'incoming/data-manifest.json',
                 KERNEL]
    for p in required:
        assert p.is_file() and os.access(p, os.R_OK), 'missing readable dependency: '+str(p)
    save('importing')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from depth_student import load_student
    spec = importlib.util.spec_from_file_location('original_depth_distillation', source / 'depth-distillation.py')
    distill = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(distill)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest_path = ORIGINAL / 'incoming/data-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    samples = manifest['samples']
    pieces = [(i, lo, hi) for i,s in enumerate(samples) for lo,hi in s['train_pieces']]
    assert manifest['method'] == 'causal-whole-piece-v1'
    assert manifest['source_bytes'] == 586459020
    assert len(pieces) == manifest['training_pieces'] == 1313
    assert sum(hi-lo-1 for _,lo,hi in pieces) == manifest['training_targets'] == 2941307
    piece_index, piece = next((j,p) for j,p in enumerate(pieces) if 1025 <= p[2]-p[1]-1 <= 2048)
    i,lo,hi = piece
    n = hi-lo-1
    sample = samples[i]
    tokens_path, priors_path = Path(sample['tokens']), Path(sample['priors'])
    assert tokens_path.is_file() and priors_path.is_file()
    assert 0 <= lo < hi <= tokens_path.stat().st_size
    assert priors_path.stat().st_size == tokens_path.stat().st_size*410
    tokens = np.fromfile(tokens_path, dtype=np.uint8, count=hi-lo, offset=lo)
    assert tokens.shape == (n+1,)
    prior = np.fromfile(priors_path, dtype='<f2', count=n*205, offset=lo*410).reshape(n,205)
    x = torch.from_numpy(tokens[:-1].astype(np.int64)).unsqueeze(0)
    y = torch.from_numpy(tokens[1:].astype(np.int64)).unsqueeze(0)
    pp = torch.from_numpy(prior).unsqueeze(0)
    bounds = torch.tensor([[0,n]], dtype=torch.int32)
    assert bool(torch.isfinite(pp).all()) and int(x.max()) < 205 and int(y.max()) < 205
    cache_path = ORIGINAL / 'training/teacher-cache' / f'sample-{i:04d}-{lo}-{hi}.npz'
    assert cache_path.is_file() and os.access(cache_path, os.R_OK), 'existing teacher cache required'
    with np.load(cache_path, allow_pickle=False) as cache:
        reference, feature = torch.from_numpy(cache['logits']), torch.from_numpy(cache['hidden'])
    assert reference.dtype == feature.dtype == torch.float32
    assert tuple(reference.shape) == (1,n,205) and tuple(feature.shape) == (1,n,192)
    assert bool(torch.isfinite(reference).all() & torch.isfinite(feature).all())
    save('inputs_ready', piece=dict(manifest_index=piece_index, sample_index=i,
         sample_name=sample['name'], lo=lo, hi=hi, targets=n,
         selection='first eligible complete piece in manifest order', bounds=[[0,n]],
         priors_dtype=str(pp.dtype), tokens_path=str(tokens_path), priors_path=str(priors_path)),
         manifest=dict(path=str(manifest_path),
                       training_pieces=len(pieces), training_targets=manifest['training_targets']),
         teacher_cache=dict(path=str(cache_path), bytes=cache_path.stat().st_size,
                            logits_shape=list(reference.shape),
                            hidden_shape=list(feature.shape), dtype='float32', cache_hits=1))
    latest = ORIGINAL / 'training/latest.pt'
    with latest.open('rb') as stream:
        st = os.fstat(stream.fileno())
        ckpt = torch.load(stream, map_location='cpu', weights_only=True)
    identity = ckpt['identity']
    assert identity['recipe'] == distill.RECIPE
    assert identity['data'] == str(manifest_path)
    assert json.loads((ORIGINAL / 'training/teacher-cache/identity.json').read_text()) == identity
    assert ckpt['epoch'] >= 8
    save('loading_student', checkpoint=dict(path=str(latest), bytes=st.st_size,
         inode=st.st_ino, mtime_ns=st.st_mtime_ns,
         cursor={k:ckpt[k] for k in ('epoch','step','next_piece','seen_targets','epoch_targets')},
         epoch_semantics='completed full epochs; cursor in epoch+1', identity=identity))
    model, adapter_counts, initialization = load_student(
        Path(identity['checkpoint']), KERNEL,
        native_scaled_gemm=True)
    z,scales,initial_q,raw,frozen,mode = distill.install_student_coordinates(model)
    assert len(z) == 75 and len(raw) == 59
    assert ckpt['z'].keys() == z.keys() and ckpt['raw'].keys() == raw.keys()
    with torch.no_grad():
        for values,saved in ((z,ckpt['z']), (raw,ckpt['raw'])):
            for key,value in values.items():
                assert value.shape == saved[key].shape and saved[key].dtype == torch.float32
                value.copy_(saved[key])
                assert torch.equal(value, saved[key])
                assert bool(torch.isfinite(value).all())
        for value in z.values():
            assert bool(((value >= -7) & (value <= 7)).all())
    torch.set_rng_state(ckpt['rng'])
    del ckpt,initial_q,initialization
    mode['hard'] = True
    model.train()
    assert model.cfg.dropout == 0 and model.cfg.gradient_checkpointing
    named = [('z/'+k,p) for k,p in sorted(z.items())] + [('raw/'+k,p) for k,p in sorted(raw.items())]
    params = [p for _,p in named]
    save('student_ready', counts=dict(z_tensors=len(z), raw_tensors=len(raw),
         trainable_tensors=len(params), trainable_elements=sum(p.numel() for p in params),
         z_elements=sum(p.numel() for p in z.values()), raw_elements=sum(p.numel() for p in raw.values()),
         frozen_tensors=len(frozen)), hard_mode=True, torch_version=torch.__version__,
         torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads())
    hidden = model.compute_last_activation(input_tokens=x, priors=pp,
                                           cumulative_sequence_lengths=bounds)
    logits = model.last_activation_to_logits(hidden, priors=pp)
    result['top_level_student_forwards'] = 1
    ce = F.cross_entropy(logits.reshape(-1,205), y.flatten(), reduction='sum')/n
    kl = F.kl_div(logits.log_softmax(-1), reference.log_softmax(-1),
                  reduction='sum', log_target=True)/n
    weighted_hidden = .1*(hidden-feature).square().mean(-1).sum()/n
    assert all(bool(torch.isfinite(loss)) for loss in (ce,kl,weighted_hidden))
    save('losses_ready', losses=dict(ce=float(ce.detach()), kl=float(kl.detach()),
         hidden_weighted=float(weighted_hidden.detach()), hidden_unweighted=float(weighted_hidden.detach())/.1),
         student_logits_shape=list(logits.shape), student_hidden_shape=list(hidden.shape))
    unused = {}

    def grad(loss, label, retain=True):
        values = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
        result['data_gradient_vjps'] += 1
        unused[label] = [named[j][0] for j,g in enumerate(values) if g is None]
        values = [torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,values)]
        assert all(bool(torch.isfinite(g).all()) for g in values), 'nonfinite '+label+' gradient'
        save('gradient_'+label, unused_parameters=unused,
             component_parameter_coverage={k:dict(used=134-len(v), unused=len(v),
                 finite_after_zero_projection=134) for k,v in unused.items()})
        return values

    gc,gk,gh = grad(ce,'ce'),grad(kl,'kl'),grad(weighted_hidden,'hidden_weighted')
    go_data = grad(ce+kl+weighted_hidden,'old_data_direct')
    gn_data = grad(2*kl+weighted_hidden,'proposed_data_direct',retain=False)
    del logits,hidden,ce,kl,weighted_hidden,reference,feature,x,y,pp
    gr = [torch.zeros_like(p) for p in params]
    rate = soft_nats = 0.
    for j,(name,p) in enumerate(named):
        if name.startswith('z/'):
            entropy = distill._old.entropy(p, distill.RECIPE['entropy_tau'])
            term = 2*entropy/586459020
            g, = torch.autograd.grad(term,p)
            result['rate_gradient_vjps'] += 1
            assert bool(torch.isfinite(entropy)) and bool(torch.isfinite(g).all())
            soft_nats += float(entropy.detach())
            rate += float(term.detach())
            gr[j] = g.detach()
            del entropy,term,g
    result['losses'].update(rate=rate, soft_entropy_nats=soft_nats)
    result['component_parameter_coverage']['rate'] = dict(used=75,unused=59,
        finite_after_zero_projection=134, unused_group='all raw parameters')
    result['losses']['old_total'] = sum(result['losses'][k] for k in LABELS[:4])
    result['losses']['proposed_total'] = 2*result['losses']['kl']+result['losses']['hidden_weighted']+rate
    gradients = [gc,gk,gh,gr]
    groups = {k:dict(squared_norms=[0.]*6, component_pair_dots=[0.]*6,
                    recipe_to_data_dots=[0.]*4) for k in ('all','z','raw')}
    rows = []
    residuals = dict(old_squared_error=0., proposed_squared_error=0.,
                     old_direct_squared_norm=0., proposed_direct_squared_norm=0., max_absolute_error=0.)
    finite = {k:0 for k in LABELS}
    nonzero = {k:0 for k in LABELS}
    for j,(name,p) in enumerate(named):
        a,b,c,d = [g[j].double() for g in gradients]
        old,new = go_data[j].double()+d,gn_data[j].double()+d
        values = [a,b,c,d,old,new]
        expected_old,expected_new = a+b+c+d,2*b+c+d
        errors = [old-expected_old,new-expected_new]
        residuals['old_squared_error'] += float(errors[0].square().sum())
        residuals['proposed_squared_error'] += float(errors[1].square().sum())
        residuals['old_direct_squared_norm'] += float(old.square().sum())
        residuals['proposed_direct_squared_norm'] += float(new.square().sum())
        residuals['max_absolute_error'] = max(residuals['max_absolute_error'],*(float(e.abs().max()) for e in errors))
        squared = [float(v.square().sum()) for v in values]
        pairdots = [float((values[l]*values[r]).sum()) for l,r in PAIRS]
        directiondots = [float((values[l]*values[r]).sum()) for l,r in DIRECTIONS]
        for label,v in zip(LABELS,values):
            assert bool(torch.isfinite(v).all())
            finite[label] += 1
            nonzero[label] += int(bool((v != 0).any()))
        for group in ('all',name.split('/')[0]):
            for field,vals in [('squared_norms',squared),('component_pair_dots',pairdots),('recipe_to_data_dots',directiondots)]:
                groups[group][field] = [u+v for u,v in zip(groups[group][field],vals)]
        norms = [math.sqrt(v) for v in squared]
        cosine = lambda dot,l,r: dot/(norms[l]*norms[r]) if norms[l]*norms[r] > 0 else None
        rows.append(dict(name=name,shape=list(p.shape),elements=p.numel(),norms=norms,
            component_pair_dots=pairdots,component_pair_cosines=[cosine(v,*pair) for v,pair in zip(pairdots,PAIRS)],
            recipe_to_data_dots=directiondots,recipe_to_data_cosines=[cosine(v,*pair) for v,pair in zip(directiondots,DIRECTIONS)]))
    for group in groups.values():
        norms = [math.sqrt(v) for v in group.pop('squared_norms')]
        group['norms'] = norms
        for field,pairs in [('component_pair',PAIRS),('recipe_to_data',DIRECTIONS)]:
            group[field+'_cosines'] = [dot/(norms[l]*norms[r]) if norms[l]*norms[r]>0 else None
                for dot,(l,r) in zip(group[field+'_dots'],pairs)]
    for recipe in ('old','proposed'):
        residuals[recipe+'_relative_l2_error'] = math.sqrt(residuals[recipe+'_squared_error']/
            max(residuals[recipe+'_direct_squared_norm'],1e-300))
        assert residuals[recipe+'_relative_l2_error'] < 1e-4, 'gradient reconstruction failed'
    assert len(rows) == 134 and all(v == 134 for v in finite.values())
    assert all(p.grad is None for p in params)
    assert all(torch.equal(dict(model.named_parameters())[k],v) for k,v in frozen.items())
    assert math.isclose(rate,2*soft_nats/586459020,rel_tol=2e-7,abs_tol=1e-12)
    # Derived directions use the same component gradients, with zero extra VJPs.
    direction_names = ('old_to_kl','old_to_ce','proposed_to_kl','proposed_to_ce')
    lr_dots = {key:.01*groups['z']['recipe_to_data_dots'][j]+
                   .0001*groups['raw']['recipe_to_data_dots'][j]
               for j,key in enumerate(direction_names)}
    scalar = min(rows, key=lambda row:row['recipe_to_data_dots'][0])
    scalar_lr = .01 if scalar['name'].startswith('z/') else .0001
    result['lr_scaled_ordinary_direction'] = dict(
        scope='direction v=-diag(original hard group LR)*ordinary gradient; history-free comparison',
        actual_adam_scope='Original AdamW has historical moments and elementwise preconditioning',
        original_group_learning_rates=dict(z=.01,raw=.0001), gradient_dots=lr_dots,
        directional_derivatives_under_negative_lr_scaled_gradient={k:-v for k,v in lr_dots.items()},
        largest_old_to_kl_negative_parameter=dict(name=scalar['name'],shape=scalar['shape'],
            ordinary_old_to_kl_dot=scalar['recipe_to_data_dots'][0],
            ordinary_proposed_to_kl_dot=scalar['recipe_to_data_dots'][2],
            lr_scaled_old_to_kl_dot=scalar_lr*scalar['recipe_to_data_dots'][0],
            lr_scaled_proposed_to_kl_dot=scalar_lr*scalar['recipe_to_data_dots'][2],
            rest_ordinary_old_to_kl_dot=groups['all']['recipe_to_data_dots'][0]-scalar['recipe_to_data_dots'][0]))
    result['execution'] = dict(child_exit_code=0,cpu_affinity=list(os.sched_getaffinity(0)),
        total_cpu_seconds=time.process_time(),wall_seconds=time.monotonic()-START,
        address_space_limit_bytes=4 << 30,reruns=0,
        resource_scope='Standalone process CPU at result write; RLIMIT_CPU covers exit cleanup')
    signal.alarm(0)
    save('complete', status='passed', groups=groups, parameters=rows,
         reconstruction=residuals, finite_gradient_tensors=finite, nonzero_gradient_tensors=nonzero,
         adapter_operator_counts=adapter_counts,
         checks=dict(losses_finite=True, gradients_finite=True, component_reconstruction=True,
                     counts_and_cache_shapes=True,
                     frozen_values_unchanged=True, parameter_grad_slots_empty=True), completed_at=time.time())
    print('GRADIENT_SUMMARY '+json.dumps({k:result[k] for k in
        ('status','stage','checkpoint','piece','losses','groups','reconstruction','counts',
         'top_level_student_forwards','data_gradient_vjps','rate_gradient_vjps',
         'optimizer_updates','cpu_seconds','wall_seconds','peak_rss_kib')},separators=(',', ':')),flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        signal.alarm(0)
        save('failed', status='failed', error_kind=type(error).__name__,
             error=str(error), failed_stage=result['stage'], completed_at=time.time())
        traceback.print_exc()
        print('GRADIENT_SUMMARY '+json.dumps({k:result.get(k) for k in
              ('status','failed_stage','error_kind','error','cpu_seconds','wall_seconds','peak_rss_kib')},
              separators=(',', ':')),flush=True)
        raise SystemExit(124 if isinstance(error,TimeoutError) else 1)
