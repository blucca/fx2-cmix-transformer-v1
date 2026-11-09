#!/usr/bin/env python3
"""Exact seven-level scalar screen from the existing INT4 census; stdlib only.

One shared integer alphabet {0,+/-a,+/-b,+/-c}; one positive scale per tensor.
On every nearest-assignment interval, minimize the exact quadratic in scale.
Selection is count-weighted original-q squared error, then alphabet lexicographic.
Nearest ties go toward zero; scale ties select the smaller positive scale.
Input is the published 111-tensor integer histogram census.
"""
import argparse
import hashlib
import itertools
import json
import math
import resource
import time
from fractions import Fraction as F
from pathlib import Path

Q = tuple(range(-7, 8))
N = 5_868_864
TENSORS = 111
TRAIN_TARGETS = 586_459_020
W0_ACTUAL = 2_904_325
UNIFORM7_ACTUAL = 2_044_814


def nearest_positive(q, scale, levels):
    return min(levels, key=lambda c: (abs(F(q) - scale*c), c))


def remap(hist, levels, scale):
    mapping = [(-1 if q < 0 else 1)*nearest_positive(abs(q), scale, levels) for q in Q]
    new_hist = [0]*15
    sse = F(0)
    for q, count, code in zip(Q, hist, mapping):
        new_hist[code+7] += count
        sse += count*(F(q)-scale*code)**2
    return mapping, new_hist, sse


def fit_scale(hist, positive):
    levels = (0,)+tuple(positive)
    folded = [hist[7-q]+hist[7+q] for q in range(1, 8)]
    if sum(folded) == 0:
        return F(1), F(0)
    # q=r*(c_j+c_(j+1))/2 lists every possible assignment change.
    breaks = sorted({F(2*q, left+right) for q in range(1, 8)
                     for left, right in zip(levels, levels[1:])})
    boundaries = [F(0)]+breaks+[None]
    second = sum(count*q*q for q, count in enumerate(folded, 1))
    best = None
    for low, high in zip(boundaries, boundaries[1:]):
        probe = (low+high)/2 if high is not None else low+1
        codes = [nearest_positive(q, probe, levels) for q in range(1, 8)]
        a = sum(count*code*code for count, code in zip(folded, codes))
        b = sum(count*q*code for q, (count, code) in enumerate(zip(folded, codes), 1))
        scale = F(b, a) if a else (low if low > 0 else F(1))
        scale = max(scale, low)
        if high is not None:
            scale = min(scale, high)
        if scale <= 0:
            continue
        error = F(second)-2*b*scale+a*scale*scale
        key = (error, scale)
        if best is None or key < best:
            best = key
    error, scale = best
    return scale, error


def frequencies(hist):
    n = sum(hist)
    freq = [1+count*4081//n for count in hist]
    remainder = [count*4081 % n for count in hist]
    while sum(freq) < 4096:
        i = max(range(15), key=lambda j: (remainder[j], -j))
        freq[i] += 1
        remainder[i] = 0
    return freq


def code_stats(hist):
    n = sum(hist)
    entropy = -sum(count/n*math.log2(count/n) for count in hist if count)
    freq = frequencies(hist)
    data_bits = sum(count*math.log2(4096/f) for count, f in zip(hist, freq) if count)
    return {'entropy_bits_per_value': entropy, 'entropy_data_bytes': n*entropy/8,
            'c3_frequencies_15': freq, 'c3_expected_data_bits': data_bits,
            'c3_nominal_table_bytes': 28, 'c3_expected_bytes_with_table': data_bits/8+28}


def exact_fraction(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def build_row(tensor, positive, scale, error):
    hist = tensor['histogram']
    mapping, new_hist, actual_error = remap(hist, (0,)+tuple(positive), scale)
    assert actual_error == error and error >= 0
    count = tensor['elements']
    return {'name': tensor['name'], 'elements': count, 'shape': tensor['shape'],
            'scale_multiplier': float(scale), 'scale_multiplier_exact': exact_fraction(scale),
            'effective_positive_levels_in_original_q': [float(scale*c) for c in positive],
            'mapping_original_q_minus7_to7': mapping, 'histogram_15': new_hist,
            'squared_error_sum': float(error), 'mse_original_q': float(error/count),
            'squared_error_sum_exact': exact_fraction(error), **code_stats(new_hist)}


def summarize(rows, error_sum, original_c3_bytes):
    global_hist = [sum(row['histogram_15'][i] for row in rows) for i in range(15)]
    c3_bytes = sum(row['c3_expected_bytes_with_table'] for row in rows)
    entropy_bytes = sum(row['entropy_data_bytes'] for row in rows)
    predicted_w = W0_ACTUAL+(c3_bytes-original_c3_bytes)
    return {'elements': sum(row['elements'] for row in rows),
            'squared_error_sum': float(error_sum), 'squared_error_sum_exact': exact_fraction(error_sum),
            'weighted_mse_original_q': float(error_sum/N),
            'global_histogram_15': global_hist,
            'global_entropy_bits_per_value': code_stats(global_hist)['entropy_bits_per_value'],
            'weighted_tensor_entropy_bits_per_value': entropy_bytes*8/N,
            'oracle_tensor_data_bytes': entropy_bytes,
            'c3_expected_int4_bytes_with_nominal_tables': c3_bytes,
            'c3_expected_int4_bits_per_value_including_tables': c3_bytes*8/N,
            'predicted_complete_W_bytes_anchored_to_W0': predicted_w,
            'predicted_single_W_saved_vs_W0_bytes': W0_ACTUAL-predicted_w,
            'predicted_double_W_saved_vs_W0_bytes': 2*(W0_ACTUAL-predicted_w),
            'actual_new_complete_W_bytes': None,
            'actual_new_range_coded_bytes': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census', type=Path, default=Path('research/results/weights-census.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    raw = args.census.read_bytes()
    data = json.loads(raw)
    tensors = data['per_tensor']
    assert data['status'] == 'complete' and data['int4_elements'] == N
    assert len(tensors) == TENSORS and sum(t['elements'] for t in tensors) == N
    assert len({t['name'] for t in tensors}) == TENSORS
    assert all(len(t['histogram']) == 15 and sum(t['histogram']) == t['elements'] for t in tensors)
    # A nonnegative-loss toy with a known exactly representable optimum proves
    # the scale solver reaches the optimum and preserves odd signed mapping.
    toy = [0]*15
    for q in (-6, -4, -2, 0, 2, 4, 6):
        toy[q+7] = 1
    toy_scale, toy_error = fit_scale(toy, (1, 2, 3))
    assert toy_scale == 2 and toy_error == 0
    toy_map, toy_remapped, toy_sse = remap(toy, (0, 1, 2, 3), toy_scale)
    assert toy_sse == 0 and sum(toy_remapped) == 7
    assert all(toy_map[7-q] == -toy_map[7+q] for q in range(8))

    original_rows = []
    original_c3_bytes = 0.
    for t in tensors:
        stats = code_stats(t['histogram'])
        # Existing census comparison is metadata arithmetic, not another census.
        assert abs(stats['c3_expected_bytes_with_table']*8-t['static4096_with_28B_table_bits']) < .0001
        original_c3_bytes += stats['c3_expected_bytes_with_table']
        original_rows.append({'name': t['name'], 'elements': t['elements'],
                              'histogram_15': t['histogram'], 'mse_original_q': 0., **stats})
    uniform_rows = []
    uniform_error = F(0)
    for t in tensors:
        scale = F(7, 3)
        mapping, hist, error = remap(t['histogram'], (0, 1, 2, 3), scale)
        expected = [(-1 if q < 0 else 1)*((3*abs(q)+3)//7) for q in Q]
        assert mapping == expected
        uniform_error += error
        uniform_rows.append(build_row(t, (1, 2, 3), scale, error))

    candidates = []
    chosen = None
    for positive in itertools.combinations(range(1, 8), 3):
        fitted = [fit_scale(t['histogram'], positive) for t in tensors]
        error = sum((e for _, e in fitted), F(0))
        rows = [build_row(t, positive, scale, e) for t, (scale, e) in zip(tensors, fitted)]
        summary = summarize(rows, error, original_c3_bytes)
        candidates.append({'positive_integer_levels': list(positive), **summary})
        key = (error, positive)
        if chosen is None or key < chosen[0]:
            chosen = (key, positive, rows, summary)
    assert len(candidates) == 35
    _, positive, rows, selected = chosen
    original = summarize(original_rows, F(0), original_c3_bytes)
    uniform = summarize(uniform_rows, uniform_error, original_c3_bytes)
    optimized_uniform = next(c for c in candidates if c['positive_integer_levels'] == [1, 2, 3])
    delta_w_vs_uniform = selected['predicted_complete_W_bytes_anchored_to_W0']-uniform['predicted_complete_W_bytes_anchored_to_W0']
    physical_second = sum(count*q*q for t in tensors for q, count in zip(Q, t['histogram']))
    selected['mse_reduction_vs_uniform_fraction'] = 1-selected['weighted_mse_original_q']/uniform['weighted_mse_original_q']
    selected['spacing_mse_reduction_vs_reoptimized_uniform_fraction'] = 1-selected['weighted_mse_original_q']/optimized_uniform['weighted_mse_original_q']
    selected['predicted_single_W_change_vs_uniform7_bytes'] = delta_w_vs_uniform
    selected['predicted_double_W_change_vs_uniform7_bytes'] = 2*delta_w_vs_uniform
    selected['nll_gain_required_to_pay_extra_double_W_vs_uniform7_nats_per_target'] = 2*delta_w_vs_uniform*math.log(256)/TRAIN_TARGETS
    selected['nll_headroom_vs_W0_from_double_W_saving_nats_per_target'] = selected['predicted_double_W_saved_vs_W0_bytes']*math.log(256)/TRAIN_TARGETS
    selected['distortion_as_fraction_of_original_q_second_moment'] = selected['squared_error_sum']/physical_second
    uniform['distortion_as_fraction_of_original_q_second_moment'] = float(uniform_error/physical_second)
    uniform['known_existing_actual_complete_W_bytes'] = UNIFORM7_ACTUAL
    uniform['predicted_minus_known_actual_complete_W_bytes'] = uniform['predicted_complete_W_bytes_anchored_to_W0']-UNIFORM7_ACTUAL
    for row, u, old in zip(rows, uniform_rows, original_rows):
        row['original_histogram_15'] = old['histogram_15']
        row['original_mse_original_q'] = 0.
        row['original_entropy_bits_per_value'] = old['entropy_bits_per_value']
        row['original_c3_expected_bytes_with_table'] = old['c3_expected_bytes_with_table']
        row['uniform7_mse_original_q'] = u['mse_original_q']
        row['uniform7_histogram_15'] = u['histogram_15']
        row['uniform7_entropy_bits_per_value'] = u['entropy_bits_per_value']
        row['uniform7_c3_expected_bytes_with_table'] = u['c3_expected_bytes_with_table']
        row['mse_reduction_vs_uniform_fraction'] = 1-row['mse_original_q']/u['mse_original_q'] if u['mse_original_q'] else 0.
        row['predicted_double_W_component_change_vs_uniform_bytes'] = 2*(row['c3_expected_bytes_with_table']-u['c3_expected_bytes_with_table'])
    improved = sum(row['mse_original_q'] < row['uniform7_mse_original_q'] for row in rows)
    selected['tensor_mse_improved_count'] = improved
    selected['tensor_mse_worse_count'] = TENSORS-improved
    selected['tensor_mse_reduction_fraction_min'] = min(row['mse_reduction_vs_uniform_fraction'] for row in rows)
    selected['tensor_mse_reduction_fraction_max'] = max(row['mse_reduction_vs_uniform_fraction'] for row in rows)
    selected['distinct_tensor_mappings_count'] = len({tuple(row['mapping_original_q_minus7_to7']) for row in rows})
    support = [-c for c in reversed(positive)]+[0]+list(positive)
    selected['seven_code_masses'] = [{'code': code, 'count': selected['global_histogram_15'][code+7],
                                      'probability': selected['global_histogram_15'][code+7]/N} for code in support]
    scales = [r['scale_multiplier'] for r in rows]
    result = {
        'status':'complete_exact_scalar_fit',
        'source':{'path':str(args.census), 'sha256':hashlib.sha256(raw).hexdigest(),
                  'status':data['status'], 'int4_tensors':TENSORS,'int4_elements':N,
                  'histogram_bin_order':list(Q), 'existing_census_measurement':data.get('measurement')},
        'algorithm':{'alphabet_shared_across_tensors':True,'candidate_count':35,
                     'support_cardinality':7,'positive_integer_levels':list(positive),
                     'signed_integer_alphabet':[-c for c in reversed(positive)]+[0]+list(positive),
                     'scale_per_tensor':True,'scale_multiplier_min':min(scales),'scale_multiplier_max':max(scales),
                     'objective':'sum_tensor sum_original_q count(q)*(q-scale_tensor*nearest_code(q))^2',
                     'scale_solver':'exact Fraction assignment breakpoints; clipped quadratic B/A on each interval',
                     'selection_ties':'alphabet lexicographic; smaller positive scale',
                     'nearest_ties':'smaller absolute integer code (toward zero)',
                     'rate_is_selection_objective':False},
        'original':original, 'uniform7':uniform,'reoptimized_uniform_control':optimized_uniform,'selected':selected,
        'per_tensor':rows,'all_fixed_cardinality_candidates':candidates,
        'checks':{'passed':True,'source_counts_shapes_and_histograms':True,
                  'original_c3_cost_matches_existing_census':True,'uniform_mapping_matches_original_formula':True,
                  'optimal_toy_scale':2,'optimal_toy_sse':0,'signed_toy_mapping':True,
                  'every_selected_tensor_exact_sse_and_count':True},
        'units_and_scope':{
            'mse':'Original q integer-coordinate squared error. Original W0 is the reference, MSE 0.',
            'entropy':'Seven supported code masses, global and count-weighted per-tensor separately.',
            'c3':'Existing static15: 15 positive frequencies total4096, one reserve each, 4081 proportional; expected data bits plus nominal 28 B per tensor.',
            'predicted_W':'Existing actual static15 W0 plus difference of expected INT4+table bytes. Metadata compression, BF16-scale byte entropy and final range rounding remain unmeasured.',
            'actual_new_code_length':None,
            'actual_BF16_original_scale_range':None,
            'actual_BF16_scaled_range_and_rounding':None,
            'actual_NN_or_native_or_training':None,
            'physical_error_formula_before_BF16_rounding':'For row j in tensor t, physical_SSE_tj = original_signed_BF16_scale_tj**2 * SSE_q_tj before new BF16 rounding; tensor/overall physical error requires rowwise errors and original scales, while this census aggregates q counts per tensor.',
            'future_native_scale_contract':'Keep the sign of original BF16 scale and multiply by the fitted positive ratio once; real BF16 RNE and exported native identity belong to a future full-method admission.',
        },
        'known_existing_true_lengths':{'static15_W0':W0_ACTUAL,'uniform7_initial_W':UNIFORM7_ACTUAL,
                                      'source':'research/results/weights-static15.json and research/results/weights-quant7.json'},
    }
    cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    result['computation']={'elapsed_seconds':time.monotonic()-started,
                           'cpu_seconds':cpu_after.ru_utime+cpu_after.ru_stime-cpu_before.ru_utime-cpu_before.ru_stime,
                           'max_rss_kib':cpu_after.ru_maxrss}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'nonuniform-seven-histogram-fit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'alphabet':result['algorithm']['signed_integer_alphabet'],
                      'scale_range':[min(scales),max(scales)],'original':original,
                      'uniform7':uniform,'selected':selected,'computation':result['computation']},indent=2))


if __name__ == '__main__':
    main()
