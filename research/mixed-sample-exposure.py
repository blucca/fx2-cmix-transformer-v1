#!/usr/bin/env python3
"""Replay direct seed414 exposure for the retained mixed-codec samples.

Positions use the prepared stream. A sampled piece [b,e) exposes every input
byte, with next-token training targets [b+1,e). This is the original frozen
sampling contract; upstream W0 pretraining and online coder history are separate.
"""
from pathlib import Path
import json
import sys

CASES = {
    'prefix8': (0, 8388608, None, 0),
    'prefix8_mature': (7344758, 8386448, 1041397, 0),
    'q2': (293232482, 294281058, 1048367, 293232482),
    'q2_legacy_W64_held_window': (293763521, 294280015, 516391, 293232482),
}

def intersect(begin, end, lo, hi):
    return max(0, min(end, hi) - max(begin, lo))


def scope(plan, lo, hi, neural_targets, trace_begin=0):
    roles = {r: {'pieces': 0, 'input_bytes': 0, 'target_positions': 0,
                 'input_ranges': []} for r in ('train', 'validation')}
    for row in plan['intervals']:
        b, e, r = row['begin'], row['end'], row['role']
        overlap = intersect(b, e, lo, hi)
        if overlap:
            s = roles[r]
            s['pieces'] += 1
            s['input_bytes'] += overlap
            s['target_positions'] += intersect(b + 1, e, lo, hi)
            s['input_ranges'].append([max(lo, b), min(hi, e)])
    # The frozen pieces are disjoint; a sorted subtraction suffices.
    cursor, remaining = lo, []
    for b, e in roles['train']['input_ranges']:
        if cursor < b:
            remaining.append([cursor, b])
        cursor = e
    if cursor < hi:
        remaining.append([cursor, hi])
    unseen_bytes = sum(e - b for b, e in remaining)
    assert unseen_bytes + roles['train']['input_bytes'] == hi - lo
    return {'source_begin': lo, 'source_end': hi, 'input_bytes': hi - lo,
            'neural_targets_in_retained_trace_scope': neural_targets,
            'direct_exposure': roles,
            'outside_sampled_training_input': {'bytes': unseen_bytes,
                'source_ranges': remaining,
                'scope_local_ranges': [[b - lo, e - lo] for b, e in remaining],
                'trace_local_ranges': [[b - trace_begin, e - trace_begin] for b, e in remaining]},
            'training_target_share_of_trace_targets':
                roles['train']['target_positions'] / neural_targets if neural_targets else None}


def summarize(plan):
    previous = 0
    for row in plan['intervals']:
        assert previous <= row['begin'] < row['end'] <= plan['source_bytes']
        assert row['targets'] == row['end'] - row['begin'] - 1
        previous = row['end']
    for role, total in plan['roles'].items():
        selected = [r for r in plan['intervals'] if r['role'] == role]
        assert len(selected) == total['pieces']
        assert sum(r['targets'] for r in selected) == total['targets']
    return {name: scope(plan, lo, hi, nt, tb) for name, (lo, hi, nt, tb) in CASES.items()}


if __name__ == '__main__':
    # One clipped-piece check covers first-row masking and subtraction edges.
    fixture = {'intervals': [{'begin': 10, 'end': 15, 'role': 'train'}]}
    a, b = scope(fixture, 12, 17, None), scope(fixture, 10, 17, None)
    assert (a['direct_exposure']['train']['input_bytes'], a['direct_exposure']['train']['target_positions'],
            b['direct_exposure']['train']['input_bytes'], b['direct_exposure']['train']['target_positions']) == (3, 3, 5, 4)
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('results') / 'mixed-sample-exposure.json'
    data = json.loads(path.read_text())
    plan = data.get('original_selected_piece_plan', data)
    actual = summarize(plan)
    if 'scopes' in data:
        assert data['scopes'] == actual
    print(json.dumps({'status': 'passed', 'scopes': actual}, indent=2))
