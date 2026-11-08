#!/usr/bin/env python3
"""Local Adam coordinate control for the fixed signed-BF16 q7 scale map.

This controls pre-round latent displacement with prescribed proportional
gradient histories. The admitted model, optimizer and trajectory stay fixed.
Only standard-library scalar arithmetic and the existing scale mapper are used.
"""
import json
import math
from pathlib import Path
import struct

from q7_teacher_only import quant7_scale_bits


def bf16(bits):
    return struct.unpack(">f", struct.pack(">I", bits << 16))[0]


def displacement_factor(ratio, old_vroot, epsilon):
    """Same proportional history; theta=s*z displacement before projection."""
    assert ratio > 0 and old_vroot >= 0 and epsilon > 0
    return ratio * (old_vroot + epsilon) / (old_vroot + epsilon / ratio)


def adam_history(scale, physical_gradients, lr=.01, beta1=.9, beta2=.99, epsilon=1e-8):
    """Prescribed scalar linear losses; decay zero, no rounding/projection."""
    m = v = 0.
    rows = []
    for step, physical_gradient in enumerate(physical_gradients, 1):
        gradient = scale * physical_gradient
        m = beta1 * m + (1-beta1) * gradient
        v = beta2 * v + (1-beta2) * gradient * gradient
        mhat, vhat = m/(1-beta1**step), v/(1-beta2**step)
        delta_z = -lr * mhat / (math.sqrt(vhat) + epsilon)
        rows.append(dict(step=step, mhat=mhat, vroot=math.sqrt(vhat),
                         delta_z=delta_z, delta_scaled_latent=scale*delta_z))
    return rows


def readout():
    # A successful normal-BF16 map has the same ratio at every exponent.
    # Exponent 127 covers all 128 mantissas without overflow or subnormals.
    ratios = []
    for sign in (0, 0x8000):
        for mantissa in range(128):
            bits = sign | (127 << 7) | mantissa
            mapped = quant7_scale_bits(bits)
            ratio = bf16(mapped) / bf16(bits)
            assert ratio > 0
            for exponent in (1, 80, 200, 251):
                other = sign | (exponent << 7) | mantissa
                assert bf16(quant7_scale_bits(other)) / bf16(other) == ratio
            ratios.append(dict(source_bits=bits, mapped_bits=mapped, ratio=ratio))
    lo, hi = min(ratios, key=lambda x:x["ratio"]), max(ratios, key=lambda x:x["ratio"])
    assert math.isclose(lo["ratio"], 272/117, abs_tol=1e-15)
    assert math.isclose(hi["ratio"], 260/111, abs_tol=1e-15)
    examples = []
    for sign in (0, 0x8000):
        s15 = bf16(sign | 0x3f80)
        s7 = bf16(quant7_scale_bits(sign | 0x3f80))
        ratio = s7/s15
        for magnitude in (1e-4, 1e-8, 1e-12):
            history = [magnitude, -magnitude/2, 2*magnitude]
            old, new = adam_history(s15, history), adam_history(s7, history)
            for a, b in zip(old, new):
                assert a["delta_scaled_latent"] != 0
                observed = b["delta_scaled_latent"]/a["delta_scaled_latent"]
                expected = displacement_factor(ratio, a["vroot"], 1e-8)
                assert math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
                assert ratio <= observed <= ratio*ratio
                assert all(abs(sum(r["delta_z"] for r in old[:i])) < .5 for i in range(1,4))
                assert all(abs(sum(r["delta_z"] for r in new[:i])) < .5 for i in range(1,4))
            if sign == 0:
                examples.append(dict(physical_gradient_magnitude=magnitude,
                    old_scale=s15, new_scale=s7, scale_ratio=ratio,
                    last_old_coordinate_vroot=old[-1]["vroot"],
                    last_scaled_latent_displacement_ratio=observed))
    return dict(schema="hutter.q7.coordinate-step.v1",
        scope="local pre-round latent displacement; prescribed proportional gradient history; before projection",
        mapper="signed normal BF16_RNE(s15 * 7/3)", mantissas=128, signs=2,
        scale_ratio_min=lo, scale_ratio_max=hi,
        gradient_dominant_displacement_ratio_bounds=[lo["ratio"],hi["ratio"]],
        epsilon_dominant_limiting_ratio_bounds=[lo["ratio"]**2,hi["ratio"]**2],
        formula="r * (sqrt(vhat15) + eps) / (sqrt(vhat15) + eps/r)",
        control_parameters=dict(lr=.01,betas=[.9,.99],eps=1e-8,weight_decay=0),
        examples=examples,
        conditions=["fixed signed scales with positive ratio r",
            "same complete gradient history up to factor r, including any covariantly transformed rate term",
            "same Adam parameters and proportional moment states, including fresh zero states",
            "scaled latent coordinate theta=s*z before rounding and projection"],
        actual_method=dict(q7_recipe_unchanged=True,hard_weight_displacement="determined by threshold crossings",
            actual_gradient_history="distinct loss, alphabet/rate centers and CPU forward math in the two runs",
            measured_model_moment_distribution="outside this scalar control"),
        operations=dict(model_runs=0,native_runs=0,model_training_steps=0,network_calls=0,
                        scalar_control_histories=12,scalar_control_steps=36))


if __name__ == "__main__":
    result = readout()
    output = Path(__file__).with_name("results") / "q7-coordinate-step.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status="passed", scale_ratio_bounds=[
        result["scale_ratio_min"]["ratio"],result["scale_ratio_max"]["ratio"]],
        epsilon_dominant_limiting_ratio_bounds=result["epsilon_dominant_limiting_ratio_bounds"],
        examples=result["examples"], heavy_calls=0), indent=2))
