# Carry a native bit residual through CPU log odds

The forward objective reads the actual native lookup/override coordinates.
Its backward can be expressed directly in **binary log-odds coordinates**:

```text
∂ KL(Bern(t) || Bern(sigmoid(eta))) / ∂ eta = s - t
```

At the current native value s, the first-order gradient carrier is therefore
`(s - t) × eta_shadow`. The residual is detached from the native observation;
`eta_shadow` is the declared differentiable CPU log odds.

## An observed forward with an explicit VJP

```python
# obs_kl is the actual detached U_obs: sum eight bits, mean over NN targets.
# residual is the detached native r_student - r_teacher at each visited bit.
# eta_shadow is finite; set it to zero on deterministic vocabulary branches.
carrier = (residual * eta_shadow).sum(-1).mean()
loss = obs_kl + (carrier - carrier.detach())
```

The identical-carrier subtraction adds an exact zero for finite values. Thus
the forward keeps the observed KL, while its VJP is the native bit residual
times the CPU log-odds VJP. An actual observation match gives zero loss and zero
carrier gradient, even when the CPU/native rows have their own arithmetic
difference. [The probability-coordinate contract](native-observer-training-contract.md)
and this log-odds carrier express two explicit relaxations; the latter supplies
a direct binary residual with magnitude at most one.

## A concrete effective-row shadow

A future bridge can form `ell_hard = log(guarded_half_row_from_CPU)` with the
declared half encoder, scalar-tail decoder and floor. To attach an identity
logit relaxation to the CPU model's z, use:

```python
ell_shadow = ell_hard.detach() + (z - z.detach())
```

For each target's at-most-nine groups, sum `exp(ell_shadow)` into group masses.
The one/zero children of each visited prefix are **positive group sums**.
`eta_shadow = log(one_mass) - log(zero_mass)`; deterministic branches use zero.
A fixed 205-by-205 group-index table and one row-wise scatter-add can serve all
eight nodes. This retains the existing literal raw-byte/vocabulary partition.

This is an **effective-row-logit shadow** with a declared identity relaxation
to z. The actual forward still requires the current hard-exported W's complete
native NN and consumer. It uses the cached teacher's native observations, not
the CPU row's observations, for t and s. The current 2KL trajectory stays fixed;
this document supplies a candidate VJP contract for a complete future comparison.

## Frozen native replay

The 96 existing teacher/oracle pairs give **95 zero residuals**. The one changed
lookup, 52721 to 52722, gives a positive log-odds residual
`1/100001 = 9.999900001e-6`. The clipped scalar control, teacher index20 and
student index10, gives `−10/100001 = −9.999900001e-5`. A finite difference
in the relaxed log-odds coordinate agrees in both cases. Model forwards, new
native captures and learner implementation are zero.

```sh
python3 research/native-log-odds-shadow-check.py
```

[Saved scalar result](results/native-log-odds-shadow-check.json).
Complete learning, native held/long/giant generalization and selected-weight
mixer/u16/range bytes retain their full-method measurement.
