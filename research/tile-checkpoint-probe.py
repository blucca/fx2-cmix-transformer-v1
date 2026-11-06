#!/usr/bin/env python3
"""One whole-piece zero-step check of opt-in exact SDPA tile checkpointing.

The original CPU route computes a no-grad teacher before the same model's
111 integer weight coordinates are installed. The student uses exact tiles
with non-reentrant activation checkpointing and complete-piece BPTT. No
optimizer is constructed. Uniform f16 priors select a resource/numerical
check; compression quality uses real causal PPMd priors in a separate job.
"""
import argparse
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

sys.dont_write_bytecode = True


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def process_memory():
    keys = {'VmPeak', 'VmSize', 'VmHWM', 'VmRSS'}
    rows = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in keys:
            rows[key + '_kib'] = int(value.split()[0])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'kernel', 'manifest', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', required=True, help='case key in manifest["cases"]')
    parser.add_argument('--compare-original-gradients', action='store_true',
                        help='short piece: save original-route CE+KL z adjoints, then compare exact tiles')
    parser.add_argument('--cpu-seconds', type=int, default=1500)
    args = parser.parse_args()
    assert 10 <= args.cpu_seconds <= 1800
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), 'use a fresh result path'
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[key] = '1'
    for key in ['TMPDIR', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_EXTENSIONS_DIR']:
        path = args.output.parent / (args.output.stem + '-' + key.lower())
        path.mkdir(exist_ok=True)
        os.environ[key] = str(path)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (7 << 30, 7 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds - 2, args.cpu_seconds))
    started_wall, started_cpu = time.monotonic(), time.process_time()
    result = dict(schema='hutter.exact-tile-full-model-zero.v1', status='running',
                  case=args.case, optimizer_updates=0, new_income=0, threads=1,
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  cpu_limit_seconds=args.cpu_seconds, address_space_limit_gib=7,
                  bptt='whole_piece_all_targets_no_truncation',
                  production_model_source_changes=0)

    def save(stage, **updates):
        result.update(stage=stage, updated_at=time.time(), **updates)
        result.update(elapsed_wall_seconds=time.monotonic()-started_wall,
                      elapsed_cpu_seconds=time.process_time()-started_cpu,
                      process_cpu_seconds=time.process_time(),
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      memory=process_memory())
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2)+'\n')
        tmp.replace(args.output)
        print(json.dumps({'stage': stage, 'case': args.case,
                          'cpu_seconds': result['elapsed_cpu_seconds'],
                          'peak_rss_kib': result['peak_rss_kib'], **updates}), flush=True)

    def cpu_limit(signum, frame):
        save('cpu_limit', status='cpu_budget')
        raise SystemExit(124)

    signal.signal(signal.SIGXCPU, cpu_limit)
    save('importing')
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from cpu_model import load_model

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(416)
        manifest = json.loads(args.manifest.read_text())
        case = manifest['cases'][args.case]
        tokens_path, priors_path = Path(case['tokens']), Path(case['priors'])
        tokens = np.fromfile(tokens_path, dtype=np.uint8)
        n = len(tokens)-1
        assert 0 < n <= 131071 and n == case['targets']
        assert priors_path.stat().st_size == len(tokens)*410
        assert sha256(tokens_path) == case['tokens_sha256']
        assert sha256(priors_path) == case['priors_sha256']
        x = torch.from_numpy(tokens[:-1].astype(np.int64)).unsqueeze(0)
        y = torch.from_numpy(tokens[1:].astype(np.int64)).unsqueeze(0)
        prior = np.fromfile(priors_path, dtype='<f2', count=n*205).reshape(n, 205)
        p = torch.from_numpy(prior).unsqueeze(0)
        bounds = torch.tensor([[0, n]], dtype=torch.int32)
        assert int(x.max()) < 205 and int(y.max()) < 205 and bool(torch.isfinite(p).all())
        save('inputs_loaded', input={key: case[key] for key in
             ['targets', 'source_begin', 'source_end', 'source_article_index',
              'source_article_begin', 'source_article_end', 'native_piece_limit',
              'prior_kind', 'tokens_sha256', 'priors_sha256']},
             source_stream_sha256=manifest['source_stream_sha256'],
             source_vocab_sha256=manifest['source_vocab_sha256'],
             source_bounds_sha256=manifest['source_bounds_sha256'],
             compression_quality_evaluation=case['prior_kind'] == 'real_causal_ppmd_f16',
             prior_rows='[0:n] predicts tokens[1:n+1]', rope_offset=0)

        model, counts = load_model(args.checkpoint, args.kernel)
        model.requires_grad_(False).eval()
        assert model.cfg.dropout == 0 and model.cfg.vocabulary_size == 205
        assert model.cfg.n_layers == 12 and sum(model.cfg.kimi_linear) == 9
        assert model.cfg.window_size == 1024 and model.cfg.gradient_checkpointing
        import cpu_ops
        import pysrc.attention_function as attention_module
        assert attention_module.sdpa_attention_function is cpu_ops.window_attention
        save('teacher_forward_start', torch_version=torch.__version__, python=sys.version,
             checkpoint_sha256=sha256(args.checkpoint), kernel_sha256=sha256(args.kernel),
             cpu_ops_sha256=sha256(cpu_ops.__file__), model_layers=12, kda_layers=9,
             attention_layers=3, window=1024, query_tile=1024,
             block_checkpointing=True, teacher_route='window_attention(checkpoint_tiles=False)/no_grad')
        w, c = time.monotonic(), time.process_time()
        with torch.no_grad():
            reference = model.compute_logits(input_tokens=x, priors=p,
                                             cumulative_sequence_lengths=bounds).detach()
        assert reference.dtype == torch.float32 and bool(torch.isfinite(reference).all())
        save('teacher_complete', teacher_forward_wall_seconds=time.monotonic()-w,
             teacher_forward_cpu_seconds=time.process_time()-c,
             teacher_operator_calls=dict(counts))

        spec = importlib.util.spec_from_file_location(
            'integer_coordinate_probe', Path(__file__).with_name('integer-coordinate-qat.py'))
        integer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(integer)
        z, scales, original_q, original = integer.install_coordinates(model)
        model.train()
        original_gradients = None
        if args.compare_original_gradients:
            assert n <= 2048
            save('original_adjoint_start')
            w, c = time.monotonic(), time.process_time()
            original_logits = model.compute_logits(input_tokens=x, priors=p,
                                                   cumulative_sequence_lengths=bounds)
            assert torch.equal(original_logits.detach(), reference)
            original_ce = F.cross_entropy(original_logits.reshape(-1, 205), y.flatten(), reduction='sum')
            original_kl = F.kl_div(original_logits.log_softmax(-1), reference.log_softmax(-1),
                                   reduction='sum', log_target=True)
            assert float(original_kl.detach()) == 0.0
            (original_ce+original_kl).backward()
            assert all(value.grad is not None and bool(torch.isfinite(value.grad).all()) for value in z.values())
            original_gradients = {key: value.grad.detach().clone() for key, value in z.items()}
            gradient_path = args.output.parent/(args.output.stem+'-original-ce-kl-gradients.pt')
            assert not gradient_path.exists()
            torch.save(original_gradients, gradient_path)
            for value in z.values():
                value.grad = None
            del original_logits, original_ce, original_kl
            save('original_adjoint_complete', original_adjoint_wall_seconds=time.monotonic()-w,
                 original_adjoint_cpu_seconds=time.process_time()-c,
                 original_gradient_file=str(gradient_path),
                 original_gradient_bytes=gradient_path.stat().st_size,
                 original_gradient_sha256=sha256(gradient_path))
        attention_module.sdpa_attention_function = functools.partial(
            cpu_ops.window_attention, checkpoint_tiles=True)
        assert len(z) == 111 and len(original) == 429
        save('student_forward_start', integer_coordinate_tensors=len(z),
             integer_coordinates=sum(value.numel() for value in z.values()),
             frozen_model_tensors=len(original),
             student_route='window_attention(checkpoint_tiles=True)/use_reentrant=False')
        w, c = time.monotonic(), time.process_time()
        logits = model.compute_logits(input_tokens=x, priors=p,
                                      cumulative_sequence_lengths=bounds)
        assert logits.dtype == torch.float32 and bool(torch.isfinite(logits).all())
        exact = torch.equal(logits.detach(), reference)
        save('student_complete', student_forward_wall_seconds=time.monotonic()-w,
             student_forward_cpu_seconds=time.process_time()-c,
             teacher_student_bit_exact=exact,
             logits_max_abs_error=0.0 if exact else float((logits.detach()-reference).abs().max()))
        assert exact, 'zero-step original teacher / exact-tile student logits differ'

        save('objective_start')
        ce = F.cross_entropy(logits.reshape(-1, 205), y.flatten(), reduction='sum')
        kl = F.kl_div(logits.log_softmax(-1), reference.log_softmax(-1),
                      reduction='sum', log_target=True)
        assert bool(torch.isfinite(ce) & torch.isfinite(kl))
        assert float(kl.detach()) == 0.0, 'bit-exact zero-step KL differs from zero'
        save('backward_start', ce_sum=float(ce.detach()), ce_mean=float(ce.detach())/n,
             teacher_kl_sum=float(kl.detach()), teacher_kl_mean=float(kl.detach())/n)
        w, c = time.monotonic(), time.process_time()
        (ce+kl).backward()
        backward_wall, backward_cpu = time.monotonic()-w, time.process_time()-c
        del reference, logits, ce, kl
        missing = [key for key, value in z.items() if value.grad is None]
        nonfinite = [key for key, value in z.items()
                     if value.grad is not None and not bool(torch.isfinite(value.grad).all())]
        assert not missing and not nonfinite, {'missing': missing, 'nonfinite': nonfinite}
        save('bptt_complete', backward_wall_seconds=backward_wall,
             backward_cpu_seconds=backward_cpu, bptt_finite_gradient_tensors=len(z),
             missing_gradient_names=missing, nonfinite_gradient_names=nonfinite)
        if original_gradients is not None:
            exact_tensors, max_abs, error_squared, reference_squared = 0, 0.0, 0.0, 0.0
            for key, value in z.items():
                expected, actual = original_gradients[key], value.grad
                exact_tensors += int(torch.equal(expected, actual))
                delta = actual.double()-expected.double()
                max_abs = max(max_abs, float(delta.abs().max()))
                error_squared += float(delta.square().sum())
                reference_squared += float(expected.double().square().sum())
            save('adjoints_compared', ce_kl_adjoint_bit_exact=exact_tensors == len(z),
                 ce_kl_adjoint_exact_tensors=exact_tensors, ce_kl_adjoint_tensors=len(z),
                 ce_kl_adjoint_max_abs_error=max_abs,
                 ce_kl_adjoint_relative_l2=math.sqrt(error_squared/max(reference_squared, 1e-30)))
            assert exact_tensors == len(z), 'short original-route / tile-route CE+KL z adjoints differ'
            del original_gradients
        for value in z.values():
            value.grad.div_(n)
        save('entropy_backward_start')
        w, c = time.monotonic(), time.process_time()
        soft_nats = 0.0
        for value in z.values():
            h = integer.entropy(value)
            assert bool(torch.isfinite(h))
            soft_nats += float(h.detach())
            (2*h/integer.FULL_TARGETS).backward()
        entropy_wall, entropy_cpu = time.monotonic()-w, time.process_time()-c
        nonfinite = [key for key, value in z.items() if not bool(torch.isfinite(value.grad).all())]
        assert not nonfinite, nonfinite
        nonzero = sum(int(bool((value.grad != 0).any())) for value in z.values())
        norm_squared = sum(float(value.grad.double().square().sum()) for value in z.values())
        save('gradients_checked', finite_gradient_tensors=len(z), nonzero_gradient_tensors=nonzero,
             gradient_norm=math.sqrt(norm_squared), soft_entropy_nats=soft_nats,
             entropy_backward_wall_seconds=entropy_wall, entropy_backward_cpu_seconds=entropy_cpu,
             objective=result['ce_mean']+result['teacher_kl_mean']+2*soft_nats/integer.FULL_TARGETS)

        save('frozen_state_check')
        frozen_preserved, scale_bits_preserved, scale_count = True, True, 0
        for key, parameter in model.named_parameters():
            assert not parameter.requires_grad and parameter.grad is None, key
            frozen_preserved = frozen_preserved and torch.equal(parameter, original[key])
            if key.endswith('.scale'):
                scale_count += 1
                scale_bits_preserved = scale_bits_preserved and torch.equal(
                    parameter.detach().to(torch.bfloat16).view(torch.int16),
                    original[key].to(torch.bfloat16).view(torch.int16))
        changed_coordinates = sum(int((value.detach() != original_q[key].float()).sum())
                                  for key, value in z.items())
        assert frozen_preserved and scale_bits_preserved and changed_coordinates == 0
        save('complete', status='passed', frozen_model_tensors_preserved=frozen_preserved,
             frozen_scale_tensors=scale_count, frozen_bf16_scale_bits_preserved=scale_bits_preserved,
             changed_z_coordinates=changed_coordinates, operator_calls=dict(counts))
    except Exception as exc:
        save('failed', status='failed', failure_stage=result.get('stage'),
             error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
