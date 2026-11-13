# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Blucca
#!/usr/bin/env python3
"""OpenVINO #39490: stock reproduction and an in-process helper-only fix.

The original M and BF16 bytes follow Somay-kousis's self-contained report.
No model download, runtime package write, C++ build, or benchmark is performed.
Run with the official OpenVINO wheel and a CPU PyTorch wheel installed.
"""
import hashlib
import inspect
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import openvino as ov
from openvino import opset13 as ops
from openvino.tools.ovc.moc_frontend import pytorch_frontend_utils as frontend

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
ROOT = Path(__file__).resolve().parent
report = {
    "issue": "https://github.com/openvinotoolkit/openvino/issues/39490",
    "original_report_author": "Somay-kousis",
    "platform": platform.platform(),
    "versions": {"python": sys.version, "torch": torch.__version__,
                 "torch_cuda": torch.version.cuda, "openvino": ov.__version__,
                 "openvino_build": ov.get_version(), "numpy": np.__version__},
    "threads": {"torch": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
                "openvino_cpu": 2, "openvino_streams": 1},
}
assert torch.version.cuda is None, "This reproduction requires the CPU-only PyTorch wheel"

class M(torch.nn.Module):
    def forward(self, x):
        return x * 2


def describe(tensor):
    return {"dtype": str(tensor.dtype), "values": tensor.tolist(),
            "bits_u16": [f"0x{int(x):04x}" for x in tensor.view(torch.int16).numpy().view(np.uint16)]}


stock_helper = frontend.to_torch_tensor
source = inspect.getsource(stock_helper)
package_path = Path(inspect.getfile(stock_helper))
package_sha_before = hashlib.sha256(package_path.read_bytes()).hexdigest()
(ROOT / "installed-helper-stock.py").write_text(source)
report["installed_helper"] = {"module": stock_helper.__module__, "path": str(package_path),
                              "function_sha256": hashlib.sha256(source.encode()).hexdigest(),
                              "package_file_sha256": package_sha_before}

ref = torch.tensor([1.5, -2.0], dtype=torch.bfloat16)
t = ov.Tensor(ov.Type.bf16, [2])
t.data.view(np.int16)[:] = ref.view(torch.int16).numpy()
stock = stock_helper(t)
stock_model = ov.convert_model(M(), example_input=t)
report["stock"] = {"tensor_data_numpy_dtype": str(t.data.dtype),
                   "helper": describe(stock),
                   "converted_input_type": str(stock_model.input(0).element_type)}
print("STOCK", json.dumps(report["stock"]), flush=True)
assert stock.dtype == torch.float16 and stock.tolist() == [1.9375, -2.0]
assert stock_model.input(0).element_type == ov.Type.f16

# Numeric dtype casting repairs the dtype label after the original misread,
# while retaining the changed numeric payload. Compare this with the byte view.
naive = torch.tensor(t.data).to(torch.bfloat16)
naive_model = ov.convert_model(M(), example_input=naive)
assert naive.dtype == torch.bfloat16 and naive.tolist() == [1.9375, -2.0]
assert naive_model.input(0).element_type == ov.Type.bf16
report["numeric_cast_control"] = {"helper": describe(naive),
                                  "converted_input_type": str(naive_model.input(0).element_type)}
print("NUMERIC_CAST", json.dumps(report["numeric_cast_control"]), flush=True)

# Positive control: passing a native Torch BF16 example has the intended values/type.
direct_model = ov.convert_model(M(), example_input=ref)
report["direct_torch_bf16"] = {"helper": describe(stock_helper(ref)),
                               "converted_input_type": str(direct_model.input(0).element_type)}
assert stock_helper(ref) is ref and direct_model.input(0).element_type == ov.Type.bf16
print("DIRECT", json.dumps(report["direct_torch_bf16"]), flush=True)

fp16 = ov.Tensor(ov.Type.f16, [2])
fp16.data[:] = [1.5, -2.0]
stock_fp16 = stock_helper(fp16)

# Surgical overlay: replace exactly the ov.Tensor branch, keeping its copy behavior.
# Recompiling in the same globals also routes tuple/list/dict recursion and
# prepare_torch_inputs / convert_model through the shared patched helper.
needle = "    if isinstance(tensor, Tensor):\n        return torch.tensor(tensor.data)\n"
replacement = (
    "    if isinstance(tensor, Tensor):\n"
    "        if tensor.element_type == Type.bf16:\n"
    "            return torch.tensor(tensor.data.view(np.int16)).view(torch.bfloat16)\n"
    "        return torch.tensor(tensor.data)\n"
)
assert source.count(needle) == 1, "Installed helper source has a different ov.Tensor branch"
patched_source = source.replace(needle, replacement, 1)
(ROOT / "installed-helper-overlay.py").write_text(patched_source)
frontend.Type = ov.Type
exec(compile(patched_source, "<bf16-helper-overlay>", "exec"), frontend.__dict__)
helper = frontend.to_torch_tensor
fixed = helper(t)
assert fixed.dtype == torch.bfloat16 and torch.equal(fixed.view(torch.int16), ref.view(torch.int16))

# One bit-preservation assertion covers finite values, signed zero, infinities,
# and a NaN payload; equality is checked as bytes, with no NaN numeric comparison.
edge_bits = np.array([0x0000, 0x8000, 0x3FC0, 0xC000, 0x7F80, 0xFF80, 0x7FC1], dtype=np.uint16)
edge = ov.Tensor(ov.Type.bf16, [len(edge_bits)])
edge.data.view(np.uint16)[:] = edge_bits
assert np.array_equal(helper(edge).view(torch.int16).numpy().view(np.uint16), edge_bits)
report["edge_bits_preserved"] = [f"0x{int(x):04x}" for x in edge_bits]

copied = helper(t)
t.data.view(np.int16)[0] = 0
assert torch.equal(copied.view(torch.int16), ref.view(torch.int16))
t.data.view(np.int16)[:] = ref.view(torch.int16).numpy()
report["copy_semantics"] = "PASS: mutating the ov.Tensor storage leaves the Torch copy unchanged"

nested = helper({"tuple": (t,), "list": [t], "dict": {"x": t}})
leaves = [nested["tuple"][0], nested["list"][0], nested["dict"]["x"]]
assert all(x.dtype == torch.bfloat16 and torch.equal(x.view(torch.int16), ref.view(torch.int16)) for x in leaves)
assert isinstance(nested["tuple"], tuple) and isinstance(nested["list"], tuple) and isinstance(nested["dict"], dict)
report["nested"] = {"leaf_dtypes": [str(x.dtype) for x in leaves],
                    "list_result_type": type(nested["list"]).__name__, "tuple_dict": "PASS"}

fixed_fp16 = helper(fp16)
assert fixed_fp16.dtype == torch.float16 and torch.equal(fixed_fp16, stock_fp16)
fp16_model = ov.convert_model(M(), example_input=fp16)
assert fp16_model.input(0).element_type == ov.Type.f16
report["ordinary_fp16_control"] = {"helper": describe(fixed_fp16),
                                   "converted_input_type": str(fp16_model.input(0).element_type)}

fixed_model = ov.convert_model(M(), example_input=t)
assert fixed_model.input(0).element_type == ov.Type.bf16
# Make the output explicitly f32 in the graph, so this input-boundary test
# reads an ordinary NumPy f32 result rather than a BF16 output data view.
f32_model = ov.Model([ops.convert(fixed_model.get_results()[0].input_value(0), ov.Type.f32)], fixed_model.get_parameters())
assert f32_model.output(0).element_type == ov.Type.f32
core = ov.Core()
compiled = core.compile_model(f32_model, "CPU", {"INFERENCE_NUM_THREADS": 2,
                              "NUM_STREAMS": "1", "INFERENCE_PRECISION_HINT": ov.Type.f32})
result = compiled.create_infer_request().infer({0: t})[compiled.output(0)]
assert result.dtype == np.float32
np.testing.assert_array_equal(result, np.array([3.0, -4.0], dtype=np.float32))

# Reuse this same CPU M graph once with the numerically cast BF16 payload.
naive_t = ov.Tensor(ov.Type.bf16, [2])
naive_t.data.view(np.int16)[:] = naive.view(torch.int16).numpy()
naive_result = compiled.create_infer_request().infer({0: naive_t})[compiled.output(0)]
np.testing.assert_array_equal(naive_result, np.array([3.875, -4.0], dtype=np.float32))
report["numeric_cast_control"].update({"same_cpu_graph_input_type": str(compiled.input(0).element_type),
                                       "inference_output_numpy_dtype": str(naive_result.dtype),
                                       "inference_values": naive_result.tolist()})
report["fixed"] = {"helper": describe(fixed), "converted_input_type": str(fixed_model.input(0).element_type),
                   "output_type_in_graph": str(f32_model.output(0).element_type),
                   "cpu_device": core.get_property("CPU", "FULL_DEVICE_NAME"),
                   "compiled_input_type": str(compiled.input(0).element_type),
                   "compiled_output_type": str(compiled.output(0).element_type),
                   "inference_output_numpy_dtype": str(result.dtype), "inference_values": result.tolist()}
print("FIXED", json.dumps(report["fixed"]), flush=True)
print("CONTROLS", json.dumps({k: report[k] for k in ("numeric_cast_control", "ordinary_fp16_control", "nested", "edge_bits_preserved", "copy_semantics")}), flush=True)

assert hashlib.sha256(package_path.read_bytes()).hexdigest() == package_sha_before
frontend.to_torch_tensor = stock_helper
report["package_file_unchanged"] = True
report["status"] = "PASS"
(ROOT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
print("PASS: stock BF16 misread reproduced; typed-byte helper fix passes native CPU conversion/inference", flush=True)
