"""Exact scalar function/gradient check for a proposed shared residual block.

This is algebra on a ReLU-squared positive-input family. It measures zero
Transformer rows, native observers, training steps or compressed bytes.
"""
from fractions import Fraction as Q
import json

def base(x, a):
    return x + a * max(x, Q(0)) ** 2

def shared(x, a, g):
    z = base(x, a)
    return z + g * a * max(z, Q(0)) ** 2

for x in map(Q, (-2, -1, 0, 1, 2, 3)):
    for a in map(Q, (0, 1, 2)):
        assert shared(x, a, Q(0)) == base(x, a)

xs = [Q(1), Q(2)]
a = g = Q(1)
ys = [shared(x, a, g) for x in xs]
assert ys == [Q(6), Q(42)]
# One base coefficient fits the first input exactly and misses the second.
fitted_a = ys[0] - xs[0]
assert base(xs[0], fitted_a) == ys[0]
assert base(xs[1], fitted_a) == Q(22)
assert base(xs[1], fitted_a) != ys[1]

rows = []
for x in xs:
    z = base(x, a)
    # Two occurrence derivatives sum into the ONE shared parameter a.
    first_occurrence = x*x*(1 + 2*g*a*z)
    second_occurrence = g*z*z
    total = x*x + g*(z*z + 2*a*z*x*x)
    assert first_occurrence + second_occurrence == total
    rows.append({'x': int(x), 'output': int(shared(x,a,g)),
                 'first_occurrence_d_a': int(first_occurrence),
                 'second_occurrence_d_a': int(second_occurrence),
                 'one_shared_d_a': int(total),
                 'zero_gate_d_a': int(x*x),
                 'zero_gate_d_g': int(a*z*z)})
assert [r['one_shared_d_a'] for r in rows] == [9, 88]
det = rows[0]['zero_gate_d_a']*rows[1]['zero_gate_d_g'] - rows[1]['zero_gate_d_a']*rows[0]['zero_gate_d_g']
assert det == 20

# A carry-stream scale set to zero removes the identity path.
x=Q(2)
assert 0*x + 0 == 0 and 0*x + 0 != x
def execute(bank_map, gates, skips, x):
    """Mirror the source's reverse skip pairing, using affine block stubs."""
    source = []
    n = len(bank_map)
    for i, (bank, gate) in enumerate(zip(bank_map, gates)):
        if i >= n // 2:
            x += skips[i - n // 2] * source[n - i - 1]
        block = (bank + 2) * x + (1 - bank)
        x = (1 - gate) * x + gate * block
        if i < n // 2:
            source.append(x)
    return x

old_bank = list(range(8))
new_bank = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7]
new_gate = [1] * 4 + [0] * 4 + [1] * 4
old_skip = [Q(1, 8), Q(1, 7), Q(1, 6), Q(1, 5)]
new_skip = [Q(0), Q(0)] + old_skip
for x in map(Q, (-2, 0, 2)):
    assert execute(old_bank, [1]*8, old_skip, x) == execute(new_bank, new_gate, new_skip, x)
wrong_skip = old_skip + old_skip[:2]
assert execute(old_bank, [1]*8, old_skip, Q(2)) != execute(new_bank, new_gate, wrong_skip, Q(2))

# One weight bank serves two state slots. The slots carry different histories.
independent = [Q(0), Q(0)]
aliased = Q(0)
for _ in range(2):
    independent[0] += 1
    independent[1] += 3
    first_aliased_output = aliased + 1
    aliased = first_aliased_output + 3
assert independent == [Q(2), Q(6)] and first_aliased_output == Q(5)

result = {'scope':'exact scalar ReLU-squared family on positive test inputs; architecture/byte performance unmeasured',
          'zero_gate_inclusion_checked': True,
          'strict_expansion_witness': {'inputs':[1,2], 'shared_outputs':[6,42], 'single_base_fit_a':5, 'single_base_outputs':[6,22]},
          'shared_gradient_rows': rows,
          'zero_gate_two_parameter_output_jacobian_determinant': det,
          'stream_carry_zero_is_zero_when_residual_is_zero': True,
          'reverse_skip_identity_cases': 3, 'extra_zero_skips_required': True,
          'two_slot_state_witness': {'independent_slot0_after_two_steps':2, 'aliased_slot0_after_two_steps':5},
          'transformer_targets': 0, 'native_observations':0, 'trained_models':0, 'coded_byte_measurements':0}
print(json.dumps(result,indent=2))
