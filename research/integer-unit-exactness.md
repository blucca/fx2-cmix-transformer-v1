# When a quantized dot product is exact in FP32

With hard operands and normal BF16 scales, the fixed CPU path first recovers
integer units, computes a FP32 dot product, then multiplies by the two scales.
Its integer stage has a useful exactness certificate: **768 × 128 × 7 = 688,128**, well inside FP32's exact-integer
range. The seven-symbol method reduces that absolute partial-sum bound to
**294,912**.

## Hard coordinates and scale recovery

The admitted [q7 runner](integer-coordinate-qat.py) projects each coordinate
into `[-3, 3]` before its next forward or export. Its hard STE is

```
q = z + (round(z) - z).detach()
```

For a nonzero nearest integer `q`, `z` and `q` have the same sign and their
magnitudes lie within a factor of two. Sterbenz's lemma makes their FP32
subtraction exact; the following addition returns that integer exactly.
For the zero bin, the subtraction is `-z` and the sum is zero. This covers
the projected forward value; the STE still supplies a continuous derivative.

A normal BF16 scale has eight significant bits. Multiplication by an integer
of magnitude at most 128 needs at most fifteen significant bits. With a
finite product, that product is exactly representable in FP32. Dividing it
by the same scale recovers the integer exactly under round-to-nearest.
The activation quantizer's custom forward returns the rounded-and-clamped
integer times its scale directly; its backward is implemented separately.

## Dot product and the actual fold

Each INT8-by-INT4 product is an exact integer with magnitude at most 896.
With at most 768 terms, **every partial sum**, in any reduction order, has
magnitude at most 688,128. Products, additions and FMA reductions therefore
retain the exact integer dot. This argument also covers seven-symbol weights.

The [CPU source](../pysrc/model.py) computes

```
linear(x / sa, weight / sw) * (sa * sw)
```

In the default signed-INT8 branch (`FX2_POSITIVE_U8=0`), the
[native arena](../cpp_infer/src/opt/arena_build.cpp) stores
`fold[o] = s_act * bf16_to_f32(weight_scale[o])`. Both scales originate from
BF16; **the fold itself is a FP32 multiplication**. There is zero additional
BF16 conversion of their product. The opt-in positive-U8 MLP branch instead
uses an effective activation scale of `s_act / 2` and its corresponding operand
range; it retains its separate arithmetic contract.

The certificate applies to hard bounded coordinates, nonzero normal BF16
scales, finite dequantized products, FP32 round-to-nearest arithmetic and the
stated integer reduction bound. Continuous coordinates have their own
floating-point path. Nonlinear state, attention, complete native quality and
range bytes continue through their existing independent measurements.

## Reproduce without a model

```
python3 -B research/integer-unit-exactness.py
```

The standard-library control checks FP32 neighbours around the hard thresholds,
all 65,024 signed normal BF16 scales with fifteen INT4 symbols and four INT8
endpoints, three reduction orders and the exact source expressions. Overflow
cases are counted outside the finite-product domain. A larger-integer control
shows rounding beyond the bound. [Saved result](results/integer-unit-exactness.json).

This scalar control and algebraic proof describe the linear arithmetic stages;
the [running q7 native curve and same-weight codec](q7-teacher-only.md) supply
the experiment's end-to-end measurements.
