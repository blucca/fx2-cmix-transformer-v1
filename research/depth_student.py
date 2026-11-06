"""The single fixed eight-block CPU student and its complete v2 weight file.

Checkpoint initialization keeps original blocks 0,1,2,3,8,9,10,11 and skip
indices 2,3,4,5. The CPU adapters, cooldown config, quantization arithmetic and
gradient checkpointing come from cpu_model. Heavy imports are local to the
runtime entry points so the mapping contract can be checked with the stdlib.
"""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path


RETAINED_LAYERS = (0, 1, 2, 3, 8, 9, 10, 11)
SKIP_SOURCE_INDICES = (2, 3, 4, 5)
STUDENT_CONFIG_INTS = (205, 192, 8, 64, 3, 768, 1024, 64, 3, 4, 10000)
TEACHER_CONFIG_INTS = (205, 192, 12, 64, 3, 768, 1024, 64, 3, 4, 10000)
STUDENT_KIMI = (True, True, True, False) * 2
SKIP_KEY = "skip_connection_weights.value"


def student_source_key(key):
    """Map a student state/file tensor name to its original teacher name."""
    if key.startswith("blocks."):
        _, layer, suffix = key.split(".", 2)
        index = int(layer)
        assert 0 <= index < len(RETAINED_LAYERS), key
        return f"blocks.{RETAINED_LAYERS[index]}.{suffix}"
    return key


def teacher_shapes(student_shapes):
    """Expand actual student tensor shapes to the original unshared schema.

    Blocks 0..3 supply one complete [KDA,KDA,KDA,attention] group. This works
    for checkpoint state names and for the exported .q/.scale tensor names.
    """
    shapes = {key: tuple(shape) for key, shape in student_shapes.items()
              if not key.startswith("blocks.")}
    shapes[SKIP_KEY] = (6,)
    groups = {layer: {} for layer in range(4)}
    for key, shape in student_shapes.items():
        if key.startswith("blocks."):
            _, layer, suffix = key.split(".", 2)
            layer = int(layer)
            assert 0 <= layer < 8, key
            if layer < 4:
                groups[layer][suffix] = tuple(shape)
    assert all(groups.values())
    for layer in range(12):
        for suffix, shape in groups[layer % 4].items():
            shapes[f"blocks.{layer}.{suffix}"] = shape
    return shapes


def _student_config(cfg):
    assert cfg.n_layers == 12 and cfg.kimi_linear == [True, True, True, False] * 3
    assert cfg.attention_weight_sharing_pattern == list(range(12))
    assert cfg.mlp_weight_sharing_pattern == list(range(12))
    assert cfg.window_size_multipliers is None and cfg.value_embedding_pattern is None
    assert cfg.gradient_checkpointing and cfg.skip_connections
    return replace(cfg, n_layers=8, attention_weight_sharing_pattern=list(range(8)),
                   mlp_weight_sharing_pattern=list(range(8)),
                   kimi_linear=[cfg.kimi_linear[layer] for layer in RETAINED_LAYERS])


def _assert_student(model):
    from pysrc.model import Attention, KimiLinearAttention

    cfg = model.cfg
    ints = (cfg.vocabulary_size, cfg.d_model, cfg.n_layers, cfg.d_head,
            cfg.n_query_heads, cfg.d_mlp, cfg.window_size,
            cfg.kimi_linear_d_head, cfg.kimi_linear_n_heads,
            cfg.kimi_linear_convolution_size, cfg.rope_base)
    assert ints == STUDENT_CONFIG_INTS, ints
    assert tuple(cfg.kimi_linear) == STUDENT_KIMI
    assert cfg.attention_weight_sharing_pattern == list(range(8))
    assert cfg.mlp_weight_sharing_pattern == list(range(8))
    assert cfg.gradient_checkpointing and cfg.attention_implementation == "sdpa"
    assert len(model.blocks) == 8
    kinds = []
    for index, block in enumerate(model.blocks):
        assert block.layer == index, (index, block.layer)
        expected = KimiLinearAttention if STUDENT_KIMI[index] else Attention
        assert isinstance(block.attention, expected), index
        kinds.append("KDA" if STUDENT_KIMI[index] else "attention")
    assert model.skip_connection_weights.value.shape == (4,)
    return kinds


def _assert_shapes(values, shapes):
    assert isinstance(values, Mapping)
    assert set(values) == set(shapes), {
        "missing": sorted(set(shapes) - set(values)),
        "extra": sorted(set(values) - set(shapes)),
    }
    for key, value in values.items():
        assert tuple(value.shape) == tuple(shapes[key]), (key, value.shape, shapes[key])


def state_counts(model, state_dict=None):
    """Read the realized student state and precision groups from the model."""
    from pysrc.export_weights import (
        _ACTIVATION_SCALE_SUFFIXES, _UNQUANTIZED_FP32_SUFFIXES,
    )

    kinds = _assert_student(model)
    actual = model.state_dict()
    state = actual if state_dict is None else state_dict
    _assert_shapes(state, {key: value.shape for key, value in actual.items()})
    quantized = {key for key in state if key.endswith(".weight")
                 and key[:-7] + ".quantize_weight.scale" in state}
    weight_scales = {key for key in state if key.endswith(".quantize_weight.scale")}
    activations = {key for key in state if key.endswith(_ACTIVATION_SCALE_SUFFIXES)}
    raw = set(state) - quantized - weight_scales - activations
    assert weight_scales == {key[:-7] + ".quantize_weight.scale" for key in quantized}
    assert all(key.endswith(_UNQUANTIZED_FP32_SUFFIXES) for key in raw), sorted(raw)
    result = {
        "blocks": len(model.blocks), "kda_blocks": kinds.count("KDA"),
        "attention_blocks": kinds.count("attention"), "block_types": kinds,
        "state_tensors": len(actual),
        "state_elements": sum(value.numel() for value in actual.values()),
        "quantized_weights": len(quantized),
        "int4_values": sum(state[key].numel() for key in quantized),
        "weight_scale_elements": sum(state[key].numel() for key in weight_scales),
        "activation_scales": len(activations),
        "activation_scale_elements": sum(state[key].numel() for key in activations),
        "raw_fp32": len(raw), "raw_fp32_elements": sum(state[key].numel() for key in raw),
    }
    assert (result["quantized_weights"], result["activation_scales"],
            result["raw_fp32"], result["state_tensors"]) == (75, 80, 59, 289), result
    return result


def load_student(checkpoint, kernel=None, native_scaled_gemm=False):
    """Construct the fixed student; strictly map all original checkpoint keys.

    Returns (model, cpu_import_adapter_counts, initialization_info). Structural
    counts in initialization_info come from the actual model state. The second
    result retains cpu_model's existing shared import-adapter diagnostics.
    """
    import torch
    from cpu_model import build_model

    model, counts = build_model(kernel, config_transform=_student_config,
                                native_scaled_gemm=native_scaled_gemm)
    target = model.state_dict()
    original = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _assert_shapes(original, teacher_shapes({key: value.shape for key, value in target.items()}))
    for key, value in original.items():
        assert value.device.type == "cpu" and value.dtype == torch.float32, key
        assert bool(torch.isfinite(value).all()), key
    mapped = {}
    key_map = {}
    for key, value in target.items():
        source = student_source_key(key)
        data = original[source]
        if key == SKIP_KEY:
            data = data[list(SKIP_SOURCE_INDICES)]
        assert data.shape == value.shape and data.dtype == value.dtype, key
        mapped[key] = data
        key_map[key] = source
    model.load_state_dict(mapped, strict=True)
    for key, value in model.state_dict().items():
        assert torch.equal(value.view(torch.int32), mapped[key].view(torch.int32)), key
    info = {
        "method": "fixed-eight-block-outer-pairs-v1",
        "retained_original_layers": list(RETAINED_LAYERS),
        "dropped_original_layers": [4, 5, 6, 7],
        "skip_source_indices": list(SKIP_SOURCE_INDICES),
        "skip_teacher_pairs": [[3, 8], [2, 9], [1, 10], [0, 11]],
        "skip_student_pairs": [[3, 4], [2, 5], [1, 6], [0, 7]],
        "teacher_state_tensors": len(original),
        "teacher_state_elements": sum(value.numel() for value in original.values()),
        "mapped_tensors": len(mapped), "mapped_keys": key_map,
        "all_checkpoint_keys_shapes_checked": True,
        "all_mapped_values_bit_exact": True,
        "gradient_checkpointing": model.cfg.gradient_checkpointing,
        "native_scaled_gemm": native_scaled_gemm,
        "operator_counts_scope": "existing cpu_model shared import-adapter diagnostics",
        "state_counts": state_counts(model),
    }
    del mapped, original, target
    model.train()
    return model, counts, info


def logits_and_hidden(model, input_tokens, priors, cumulative_sequence_lengths,
                      override_window_size=None, initialize_scales=None):
    """Reuse compute_logits' two production calls and return the normalized RMS.

    The hidden is the input to unembedding before its activation quantizer.
    Autograd ownership follows the caller's context, including teacher no-grad
    cache generation and student final-feature MSE. Both model depths work.
    """
    scales = {} if initialize_scales is None else {"initialize_scales": initialize_scales}
    hidden = model.compute_last_activation(
        input_tokens=input_tokens, priors=priors,
        cumulative_sequence_lengths=cumulative_sequence_lengths,
        override_window_size=override_window_size, **scales)
    logits = model.last_activation_to_logits(hidden, priors=priors, **scales)
    return logits, hidden


def export_student(model, template_original_weights, output_v2, output_raw=None,
                   *, state_dict=None):
    """Write a genuine shortened FX2TFWC2 model and optional full RAW file.

    Callers can supply a complete state_dict with z*s weights and fixed scales.
    All 289 trainable payloads are regenerated with the existing CPU row
    quantizer/signed BF16 conversion. Three decoded original RoPE tensors are
    retained byte-for-byte; config.ints/config.kimi are rebuilt for eight blocks.
    The existing recoder's default lossless mode accepts this 294-tensor v2.
    """
    import numpy as np
    import torch
    from pysrc.export_weights import (
        _ACTIVATION_SCALE_SUFFIXES, DTYPE_BF16_BITS, DTYPE_FLOAT32,
        DTYPE_INT8, DTYPE_INT32, ROPE_SEQUENCE_LENGTH,
        bf16_bits, quantize_weight_rows, read_tensor_file, write_tensor_file,
    )
    from pysrc.weights_compress import (
        read_tensor_file_v2, write_tensor_file_v2,
    )

    with torch.inference_mode():
        state = model.state_dict() if state_dict is None else state_dict
        summary = state_counts(model, state)
        quantized = {key for key in state if key.endswith(".weight")
                     and key[:-7] + ".quantize_weight.scale" in state}
        entries = []
        negative_rows = negative_matrices = 0
        for key, value in state.items():
            value = value.detach()
            assert value.device.type == "cpu" and value.dtype == torch.float32, key
            assert bool(torch.isfinite(value).all()), key
            if key in quantized:
                scale = state[key[:-7] + ".quantize_weight.scale"].detach()
                ints, scale_used = quantize_weight_rows(value, scale)
                assert bool(torch.isfinite(scale_used).all() & (scale_used != 0).all()), key
                assert int(ints.abs().max()) <= 7, key
                n_negative = int((scale_used < 0).sum())
                negative_rows += n_negative
                negative_matrices += n_negative > 0
                entries.append((key + ".q", DTYPE_INT8, ints.numpy()))
                entries.append((key + ".scale", DTYPE_BF16_BITS, bf16_bits(scale)))
            elif key.endswith(".quantize_weight.scale"):
                continue
            elif key.endswith(_ACTIVATION_SCALE_SUFFIXES):
                scale_used = value.to(torch.bfloat16).to(torch.float32)
                assert bool(torch.isfinite(scale_used).all() & (scale_used > 0).all()), key
                entries.append((key, DTYPE_BF16_BITS, bf16_bits(value)))
            else:
                entries.append((key, DTYPE_FLOAT32, value.numpy()))
        assert len(entries) == summary["state_tensors"]

        template = read_tensor_file_v2(str(template_original_weights))
        shapes = teacher_shapes({name: data.shape for name, _, data in entries})
        codes = {student_source_key(name): code for name, code, _ in entries}
        for layer in range(12):
            for name, code, _ in entries:
                if name.startswith(f"blocks.{layer % 4}."):
                    suffix = name.split(".", 2)[2]
                    codes[f"blocks.{layer}.{suffix}"] = code
        for name, shape in (("rope.inv_freq", (model.cfg.d_head // 2,)),
                            ("rope.sin", (ROPE_SEQUENCE_LENGTH, model.cfg.d_head // 2)),
                            ("rope.cos", (ROPE_SEQUENCE_LENGTH, model.cfg.d_head // 2))):
            shapes[name] = shape
            codes[name] = DTYPE_FLOAT32
        shapes.update({"config.ints": (11,), "config.kimi": (12,)})
        codes.update({"config.ints": DTYPE_INT32, "config.kimi": DTYPE_INT32})
        assert set(template) == set(shapes) and len(template) == 434
        for name, (code, data) in template.items():
            assert code == codes[name] and data.shape == shapes[name], name
        assert tuple(template["config.ints"][1].tolist()) == TEACHER_CONFIG_INTS
        assert template["config.kimi"][1].tolist() == [1, 1, 1, 0] * 3
        for name in ("rope.inv_freq", "rope.sin", "rope.cos"):
            code, data = template[name]
            assert bool(np.isfinite(data).all()), name
            entries.append((name, code, data))
        entries.extend([
            ("config.ints", DTYPE_INT32, np.array(STUDENT_CONFIG_INTS, dtype=np.int32)),
            ("config.kimi", DTYPE_INT32, np.array(STUDENT_KIMI, dtype=np.int32)),
        ])
        assert len(entries) == 294
        output_v2 = Path(output_v2)
        assert output_v2.resolve() != Path(template_original_weights).resolve()
        if output_raw is not None:
            output_raw = Path(output_raw)
            assert output_raw.resolve() not in {output_v2.resolve(),
                                                Path(template_original_weights).resolve()}
            write_tensor_file(str(output_raw), entries)
            _verify_entries(entries, read_tensor_file(str(output_raw)))
        write_tensor_file_v2(str(output_v2), entries)
        _verify_entries(entries, read_tensor_file_v2(str(output_v2)))
        return dict(summary, tensor_count=len(entries),
                    derived_tensors=5, negative_scale_matrices=negative_matrices,
                    negative_scale_rows=negative_rows,
                    original_template_tensors=len(template),
                    original_template_shapes_checked=True,
                    original_rope_bytes_equal=True, decoded_tensor_bytes_equal=True,
                    v2_bytes=output_v2.stat().st_size,
                    raw_bytes=output_raw.stat().st_size if output_raw else None)


def _verify_entries(entries, decoded):
    assert list(decoded) == [name for name, _, _ in entries]
    for name, code, data in entries:
        decoded_code, decoded_data = decoded[name]
        assert decoded_code == code and decoded_data.shape == data.shape, name
        assert decoded_data.dtype == data.dtype, name
        assert decoded_data.tobytes() == data.tobytes(), name
