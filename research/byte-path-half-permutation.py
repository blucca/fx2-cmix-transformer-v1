#!/usr/bin/env python3
"""One domain-preserving half permutation on 12 saved rows; standard library only."""
import argparse
import json
import math
import resource
import struct
import time
from pathlib import Path


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


FLOOR = f32(1e-6)
UNITS = FLOOR.as_integer_ratio()[1]  # 2**43: all guarded values are exact integers.


def half_words(logits):
    exp = [math.exp(z - max(logits)) for z in logits]
    total = math.fsum(exp)
    raw = [f32(v / total) for v in exp]
    return raw, [struct.unpack('<H', struct.pack('<e', v))[0] for v in raw]


def decode(words):
    tail = len(words) // 8 * 8
    out = []
    for i, h in enumerate(words):
        value = struct.unpack('<e', struct.pack('<H', h))[0]
        if i >= tail and h & 0x7c00 == 0 and h & 0x3ff:
            value *= .5  # Source scalar subnormal decoder, exponent 112-e.
        out.append(max(FLOOR, value))
    return out


def groups(vocab, target):
    out = [[] for _ in range(9)]
    for i, byte in enumerate(vocab):
        out[8 - (byte ^ vocab[target]).bit_length()].append(i)
    return [g for g in out if g]


def normalize(p):
    total = math.fsum(p)
    return [v / total for v in p]


def kl(p, q):
    return math.fsum(a * math.log(a / b) for a, b in zip(p, q))


def observe(p, vocab, target):
    """ByteModel source order: right sum, then left sum seeded with right sum."""
    full = [0.] * 256
    for byte, value in zip(vocab, p):
        full[byte] = f32(value)
    q, lookup, override = [], [], []
    for depth in range(8):
        width = 1 << (8 - depth)
        lo = vocab[target] // width * width
        hi, mid = lo + width, lo + width // 2
        num = 0.
        for value in full[mid:hi]:
            num = f32(num + value)
        den = num
        for value in full[lo:mid]:
            den = f32(den + value)
        prob = f32(num / den) if den else .5
        q.append(prob)
        override.append(int(prob) if prob in (0., 1.) else -1)
        clamped = min(max(prob, f32(1e-4)), f32(1 - 1e-4))
        lookup.append(min(100000, max(0, int(f32(clamped * 100001)))))
    return dict(conditionals=q, lookup=lookup, override=override)


def units(p):
    out = []
    for value in p:
        n, d = value.as_integer_ratio()
        assert UNITS % d == 0
        out.append(n * (UNITS // d))
    return out


def run(evidence, control, output):
    started, cpu = time.perf_counter(), time.process_time()
    data = json.loads(evidence.read_text())
    prior = json.loads(control.read_text())
    vocab, examples = data['vocab_bytes'], data['example_rows']
    assert len(vocab) == 205 and len(examples) == 12 and prior['path_bits'] == 96
    keys = lambda r: (r['regime'], r['piece'], r['row'], r['target'])
    old = {keys(r): r for r in prior['cases']}
    teacher_buffer, oracle_buffer, cases = bytearray(), bytearray(), []
    for item in examples:
        tf32, th = half_words(item['teacher_logits'])
        _, sh = half_words(item['student_logits'])
        t, s = decode(th), decode(sh)
        gs, tail = groups(vocab, item['target']), 200
        oh, mapping, cells = th.copy(), list(range(205)), []
        for group in gs:
            for scalar_tail in (False, True):
                cell = [i for i in group if (i >= tail) == scalar_tail]
                if not cell:
                    continue
                # Rearrangement inequality: maximize sum R_i log S_i.
                src = sorted(cell, key=lambda i: (t[i], i))
                dst = sorted(cell, key=lambda i: (s[i], i))
                for source, dest in zip(src, dst):
                    oh[dest], mapping[dest] = th[source], source
                cells.append(cell)
        r = decode(oh)
        tu, ru = units(t), units(r)
        assert oh[item['target']] == th[item['target']]
        assert all((i >= tail) == (mapping[i] >= tail) for i in range(205))
        assert all(sorted(oh[i] for i in c) == sorted(th[i] for i in c) for c in cells)
        assert all(sorted((r[i] for i in sorted(c, key=lambda i: (s[i], i)))) ==
                   [r[i] for i in sorted(c, key=lambda i: (s[i], i))] for c in cells)
        assert [struct.unpack('<H', struct.pack('<e', tf32[i]))[0] for i in mapping] == oh
        group_errors = [abs(sum(tu[i] for i in g) - sum(ru[i] for i in g)) for g in gs]
        domain_errors = [abs(sum(tu[i] for i in c) - sum(ru[i] for i in c)) for c in cells]
        assert max(group_errors + domain_errors) == 0
        nt, ns, nr = map(normalize, (t, s, r))
        original, permuted = observe(t, vocab, item['target']), observe(r, vocab, item['target'])
        previous = old[keys(item)]
        assert original['lookup'] == previous['teacher_codes']
        full, oracle_kl, teacher_oracle_kl = kl(nt, ns), kl(nr, ns), kl(nt, nr)
        student_groups = [math.fsum(ns[i] for i in g) for g in gs]
        path = kl([math.fsum(nt[i] for i in g) for g in gs], student_groups)
        oracle_path = kl([math.fsum(nr[i] for i in g) for g in gs], student_groups)
        assert abs(full - previous['full_kl']) < 2e-12
        assert abs(path - previous['path_kl']) < 2e-12
        assert abs(path - oracle_path) < 2e-12
        assert oracle_kl <= full + 2e-12
        lookup_changes = sum(a != b for a, b in zip(original['lookup'], permuted['lookup']))
        override_changes = sum(a != b for a, b in zip(original['override'], permuted['override']))
        tbytes, obytes = struct.pack('<205H', *th), struct.pack('<205H', *oh)
        teacher_buffer.extend(tbytes)
        oracle_buffer.extend(obytes)
        cases.append(dict(regime=item['regime'], piece=item['piece'], row=item['row'],
            target_index=item['target'], target_byte=vocab[item['target']], groups=len(gs),
            group_decoder_cells=len(cells), full_kl_teacher_student=full,
            full_kl_oracle_student=oracle_kl, full_kl_teacher_oracle=teacher_oracle_kl,
            path_kl_teacher_student=path, path_kl_oracle_student=oracle_path,
            relative_target_distance_reduction=(full - oracle_kl) / full,
            exact_group_max_error_units=max(group_errors),
            exact_decoder_cell_max_error_units=max(domain_errors),
            native_condition_max_error=max(abs(a - b) for a, b in zip(
                original['conditionals'], permuted['conditionals'])),
            lookup_changes=lookup_changes, override_changes=override_changes,
            previous_rehalf_lookup_changes=previous['rehalf_bin_changes'],
            teacher_observation=original, oracle_observation=permuted,
            moved_half_words=sum(a != b for a, b in zip(th, oh)),
            changed_effective_values=sum(a != b for a, b in zip(t, r)),
            teacher_tail_words=th[tail:], oracle_tail_words=oh[tail:],
            source_index_per_destination_u8_hex=bytes(mapping).hex(),
            ))
    mean = lambda k: math.fsum(c[k] for c in cases) / len(cases)
    ts, rs, tr, path = map(mean, ('full_kl_teacher_student', 'full_kl_oracle_student',
                                'full_kl_teacher_oracle', 'path_kl_teacher_student'))
    totals = {k: sum(c[k] for c in cases) for k in ('lookup_changes', 'override_changes',
              'previous_rehalf_lookup_changes', 'moved_half_words', 'changed_effective_values')}
    assert totals['previous_rehalf_lookup_changes'] == 31
    assert totals['lookup_changes'] == totals['override_changes'] == 0
    assert totals['changed_effective_values'] > 0 and rs < ts
    artifacts = {}
    for name, content in (('teacher.f16', teacher_buffer), ('oracle.f16', oracle_buffer)):
        (output / name).write_bytes(content)
        artifacts[name] = dict(bytes=len(content),
                               layout='12 consecutive 205-word little-endian uint16 rows; cases order')
    inputs = {name: dict(path=str(p))
              for name, p in (('saved_logits', evidence), ('prior_half_guard_control', control))}
    result = dict(schema='hutter.saved-logit-half-permutation-oracle.round464.v1', status='passed',
        scope='12 saved real-logit rows / 12 actual targets / 96 path bits; output-buffer certificate',
        mechanism='single sorted permutation per nonempty target sibling/leaf x decoder domain',
        arithmetic='FP64 softmax -> FP32 -> RNE half; source decode/guard; serial FP32 source-order observation',
        decoder_domains=dict(f16c_indices=[0, 199], scalar_tail_indices=[200, 204],
                             scalar_subnormal_scale=.5),
        objective='min KL(R||S) within each fixed teacher-half-word decoder-cell multiset',
        normalization='KL uses independently normalized effective teacher/student/oracle rows; coder uses absolute rows',
        floor=FLOOR, exact_group_error_unit_denominator=UNITS, rows=12, path_bits=96,
        means=dict(full_kl_teacher_student=ts, full_kl_oracle_student=rs,
                   full_kl_teacher_oracle=tr, path_kl_teacher_student=path,
                   path_kl_oracle_student=path, target_distance_reduction=ts-rs,
                   relative_target_distance_reduction=(ts-rs)/ts,
                   relative_within_sibling_kl_reduction=(ts-rs)/(ts-path)),
        totals=totals, exact_group_max_error_units=0, exact_decoder_cell_max_error_units=0,
        maximum_absolute_group_mass_error=0.0,
        maximum_native_condition_error=max(c['native_condition_max_error'] for c in cases),
        teacher_override_counts={str(v): sum(c['teacher_observation']['override'].count(v) for c in cases)
                                 for v in (-1, 0, 1)},
        certificate=dict(target_leaf_half_fixed=True, per_decoder_cell_half_multiset_preserved=True,
                         sorted_rearrangement_minimizer=True, pre_half_fp32_permutation_rounding_matches=True),
        deferred_evaluation='learned architectural realizability and complete native fast/O3 payload after original 2KL terminal decision',
        model_forwards=0, optimizer_updates=0, new_learning_engineering=0, remote_calls=0,
        native_calls=0, codec_runs=0, inputs=inputs, half_buffers=artifacts,
        cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, cases=cases)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('control', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args.evidence, args.control, args.output_dir)
