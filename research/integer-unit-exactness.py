"""Scalar binary32 certificate for bounded integer-unit GEMM; stdlib only."""
import json
import struct
from pathlib import Path


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def ste(z):
    # The admitted runner projects z before the next hard forward/export.
    assert -3 <= z <= 3
    return f32(z + f32(round(z) - z))


def check():
    root = Path(__file__).resolve().parents[1]
    model = (root / "pysrc/model.py").read_text()
    native = (root / "cpp_infer/src/opt/arena_build.cpp").read_text()
    qmat = (root / "cpp_infer/src/opt/qmat.h").read_text()
    qat = (root / "research/integer-coordinate-qat.py").read_text()
    assert "assert weight.shape[1] <= 768" in model
    assert "linear(x / sa, weight.type_as(x) / sw) * (sa * sw.flatten())" in model
    assert "fold[o] = s.s_act * bf16_to_f32(ws.bf16_bits()[o]);" in native
    assert "#define FX2_POSITIVE_U8 0" in qmat
    assert "q = z + (z.round().clamp(-7, 7) - z).detach()" in qat

    # Grid plus representable neighbours of every in-domain half-integer.
    points = {f32(k / 4096) for k in range(-12288, 12289)}
    for half in (.5, 1.5, 2.5):
        bits = struct.unpack("<I", struct.pack("<f", half))[0]
        for d in range(-2, 3):
            value = from_bits(bits + d)
            points.update((value, -value))
    points.update((from_bits(1), -from_bits(1), 0.0, -0.0))
    assert all(ste(z) == round(z) for z in points)

    # All signed normal BF16 scales: 254 exponents x 128 mantissas x 2 signs.
    # All INT4 symbols plus the two INT8 magnitude endpoints. Overflow is
    # counted outside the finite-product domain; it is never used as a result.
    symbols = list(range(-7, 8)) + [-128, -127, 127, 128]
    checked = overflow = 0
    for exponent in range(1, 255):
        for mantissa in range(128):
            for sign in (0, 1):
                s = from_bits(((sign << 15) | (exponent << 7) | mantissa) << 16)
                for q in symbols:
                    exact = q * s  # <=15 significant bits, exact in Python binary64.
                    try:
                        dequantized = f32(exact)
                    except OverflowError:
                        overflow += 1
                        continue
                    assert dequantized == exact
                    assert f32(dequantized / s) == q
                    checked += 1

    bounds = {"q15": 768 * 128 * 7, "q7": 768 * 128 * 3}
    assert bounds == {"q15": 688128, "q7": 294912}
    assert all(b < 2**24 for b in bounds.values())
    dot_checks = 0
    for wmax in (3, 7):
        for mode in ("positive", "negative", "cancelling"):
            terms = [128 * wmax * (-1 if mode == "negative" or
                     (mode == "cancelling" and i % 2) else 1) for i in range(768)]
            for ordered in (terms, sorted(terms), list(reversed(terms))):
                acc = 0.0
                for term in ordered:
                    acc = f32(acc + term)
                assert acc == sum(terms)
                dot_checks += 1
    # One larger-integer control explains the role of the partial-sum bound.
    assert f32(f32(2**24 + 1) - 2**24) == 0 and (2**24 + 1) - 2**24 == 1
    return {
        "status": "passed",
        "scope": "hard bounded coordinates; normal BF16 scales; finite products; binary32 RNE integer GEMM",
        "ste_scalar_points": len(points),
        "normal_bf16_scales": 254 * 128 * 2,
        "scale_symbol_cases_passed": checked,
        "overflow_cases_outside_finite_product_domain": overflow,
        "dot_absolute_partial_sum_bounds": bounds,
        "dot_order_controls": dot_checks,
        "fold": "FP32 product of the two BF16-derived scales; no second BF16 cast",
        "operations": {"torch_imports": 0, "model_runs": 0, "native_runs": 0, "network_calls": 0},
    }


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
