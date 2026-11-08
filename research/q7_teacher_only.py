"""Small q7 recipe helpers: exact BF16 mapping, objective and read-only targets.

Importing this module uses the standard library only. The complete trainer owns
the model, its fixed-scale snapshot and the cumulative CPU ledger.
"""
import json
from pathlib import Path


def q7_recipe(original):
    return dict(original, method="q7-teacher-only-fixed-scale-integer-coordinate-v1",
                epochs=32, evaluation_every=4, seed=414,
                final_batch="merge-short-tail", label_ce=0.,
                teacher_kl_coefficient=2., final_hidden_mse=0.,
                raw_trainable_tensors=0, coordinate_tensors=111,
                hard_integer_range=[-3, 3], soft_symbols=list(range(-3, 4)),
                quant7_map="sign(q)*floor((3*abs(q)+3)/7)",
                weight_scale_map="signed-BF16-exact-RNE-times-7/3",
                native_scaled_gemm=True,
                cpu_gemm="integer-units-before-native-BF16-fold")


def quant7_value(q):
    """The actual round241 integer map, including q=+-1 mapping to zero."""
    assert isinstance(q, int) and -7 <= q <= 7
    magnitude = (3 * abs(q) + 3) // 7
    return -magnitude if q < 0 else magnitude


def quant7_scale_bits(bits):
    """Mirror weights-recode.cpp's signed normal-BF16 integer RNE exactly."""
    assert isinstance(bits, int) and 0 <= bits <= 65535
    exponent = (bits >> 7) & 255
    assert 0 < exponent < 255, "q7 source scale requires finite nonzero normal BF16"
    numerator = 7 * (128 + (bits & 127))
    denominator = 6
    exponent += 1
    if numerator >= 256 * denominator:
        denominator = 12
        exponent += 1
    q, remainder = divmod(numerator, denominator)
    q += int(2 * remainder > denominator or
             (2 * remainder == denominator and q & 1))
    if q == 256:
        q = 128
        exponent += 1
    assert exponent < 255, "q7 scale overflow"
    return (bits & 0x8000) | (exponent << 7) | (q - 128)


def quant7_tensor(q):
    """Vectorized integer map; the scalar helper is its stdlib check oracle."""
    import torch
    assert q.dtype == torch.int8 and bool((q >= -7).all() & (q <= 7).all())
    return (q.sign() * torch.div(3 * q.abs() + 3, 7,
                                rounding_mode="floor")).to(torch.int8)


def quant7_scales(scale):
    import torch
    assert scale.dtype == torch.float32 and scale.device.type == "cpu" and scale.ndim == 1
    bits = scale.to(torch.bfloat16).view(torch.uint16).tolist()
    mapped = torch.tensor([quant7_scale_bits(b) for b in bits], dtype=torch.uint16)
    return mapped.view(torch.bfloat16).float()


def data_objective(ce, kl, recipe):
    """Shared scalar/VJP expression; label CE is detached before this call."""
    return recipe.get("label_ce", 1.) * ce + recipe["teacher_kl_coefficient"] * kl


def teacher_target_identity(identity):
    """Same target projection as the existing depth teacher-cache loader."""
    return dict(checkpoint=identity["checkpoint"], data=identity["data"],
                cpu_gemm=identity["recipe"]["cpu_gemm"])


def check_teacher_cache_identity(cache, identity):
    path = Path(cache) / "identity.json"
    if not path.is_file():
        raise FileNotFoundError(f"existing read-only teacher cache identity required: {path}")
    target = teacher_target_identity(identity)
    assert teacher_target_identity(json.loads(path.read_text())) == target, "teacher target identity mismatch"
    return target


def readonly_teacher_logits(cache, piece, np):
    """Read only the complete f32 205-way logits from the original depth npz."""
    i, lo, hi = piece
    path = Path(cache) / f"sample-{i:04d}-{lo}-{hi}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"read-only teacher cache miss: {path}")
    with np.load(path, allow_pickle=False) as data:
        logits = data["logits"]
    assert logits.dtype == np.float32 and logits.shape == (1, hi - lo - 1, 205)
    assert bool(np.isfinite(logits).all())
    return logits
