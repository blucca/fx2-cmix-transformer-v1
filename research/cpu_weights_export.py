"""Export the fixed CPU QAT model's 429 payloads for weights-recode.

Load the CPU model first (cpu_model.load_model installs its CPU imports), then
call export_payloads(model.state_dict(), path). The recoder retains the five
original RoPE/config tensors and their original compressed encodings.
"""

from pathlib import Path

import numpy as np
import torch


@torch.inference_mode()
def export_payloads(state_dict, output_path):
    from pysrc.export_weights import (
        _ACTIVATION_SCALE_SUFFIXES,
        _UNQUANTIZED_FP32_SUFFIXES,
        DTYPE_BF16_BITS,
        DTYPE_FLOAT32,
        DTYPE_INT8,
        bf16_bits,
        quantize_weight_rows,
        read_tensor_file,
        write_tensor_file,
    )

    assert len(state_dict) == 429, len(state_dict)
    quantized = {
        key for key in state_dict
        if key.endswith(".weight")
        and key[:-len(".weight")] + ".quantize_weight.scale" in state_dict
    }
    assert len(quantized) == 111, len(quantized)
    entries = []
    n_activation = n_raw = 0
    for key, value in state_dict.items():
        value = value.detach()
        assert value.device.type == "cpu" and value.dtype == torch.float32, key
        assert bool(torch.isfinite(value).all()), key
        if key in quantized:
            scale = state_dict[key[:-len(".weight")] + ".quantize_weight.scale"].detach()
            ints, scale_used = quantize_weight_rows(value, scale)
            assert bool(torch.isfinite(scale_used).all() & (scale_used != 0).all()), key
            assert int(ints.abs().max()) <= 7, key
            entries.append((key + ".q", DTYPE_INT8, ints.numpy()))
            entries.append((key + ".scale", DTYPE_BF16_BITS, bf16_bits(scale)))
        elif key.endswith(".quantize_weight.scale"):
            assert key[:-len(".quantize_weight.scale")] + ".weight" in quantized, key
        elif key.endswith(_ACTIVATION_SCALE_SUFFIXES):
            scale_used = value.to(torch.bfloat16).to(torch.float32)
            assert bool(torch.isfinite(scale_used).all() & (scale_used > 0).all()), key
            entries.append((key, DTYPE_BF16_BITS, bf16_bits(value)))
            n_activation += 1
        else:
            assert key.endswith(_UNQUANTIZED_FP32_SUFFIXES), key
            entries.append((key, DTYPE_FLOAT32, value.numpy()))
            n_raw += 1
    assert (n_activation, n_raw, len(entries)) == (119, 88, 429)
    output_path = Path(output_path)
    write_tensor_file(str(output_path), entries)
    # This check also exercises the existing RAW reader used by CPU exports.
    read_back = read_tensor_file(str(output_path))
    assert len(read_back) == len(entries)
    for name, dtype, data in entries:
        read_dtype, read_data = read_back[name]
        assert read_dtype == dtype and read_data.shape == data.shape, name
        assert read_data.tobytes() == np.ascontiguousarray(data).tobytes(), name
    return {"tensor_count": len(entries), "quantized_weights": len(quantized),
            "activation_scales": n_activation, "raw_fp32": n_raw,
            "output_bytes": output_path.stat().st_size}
