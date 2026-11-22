#!/usr/bin/env python3
"""Fixed DistilGPT2 tied-vocab Q8 extension of the existing whole-decode graphs.

Input: --base-root/models/{dense,native,q8_acc0}.onnx plus their existing bins.
Output: --out-dir/models; unchanged bins are copied once per output directory, never modified at source.
Reuses converter.quant8/dequant and the existing native API31 without changes.
This script converts/checks graphs; complete runtime gates belong to the runner.
"""
import argparse
from collections import Counter
import gc
import filecmp
import shutil
import importlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import ExternalDataInfo, uses_external_data

N, NPAD, K, BLOCK = 50257, 50272, 768, 32
KB = K // BLOCK
MODES = ("dense", "native", "q8_acc0")
WTE = "transformer.wte.weight"
WTE_TRANSPOSED = "transformer.wte.weight_transposed"
EMBED_NODE = "/transformer/wte/Gather"
EMBED_OUT = "/transformer/wte/Gather_output_0"
HEAD_NODE = "/lm_head/MatMul"
PREFIX = "__blucca_tied_vocab_q8_"
BN, SN = PREFIX + "B", PREFIX + "S"
BFILE, SFILE, EFILE = BN + ".bin", SN + ".bin", PREFIX + "dense_E.bin"
OUTPUTS = ["logits"] + [f"present.{i}.{kind}" for i in range(6) for kind in ("key", "value")]
INPUTS = ["input_ids"] + [f"past_key_values.{i}.{kind}" for i in range(6) for kind in ("key", "value")] + ["attention_mask"]


def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def shape(value):
    return tuple(d.dim_value if d.HasField("dim_value") else d.dim_param
                 for d in value.type.tensor_type.shape.dim)


def consumers(graph, name):
    return [node for node in graph.node if name in node.input]


def unique(items, label):
    items = list(items)
    assert len(items) == 1, (label, len(items))
    return items[0]


def payload(tensor):
    return math.prod(tensor.dims) * np.dtype(helper.tensor_dtype_to_np_dtype(tensor.data_type)).itemsize


def ext_signature(tensor):
    assert uses_external_data(tensor), tensor.name
    info = ExternalDataInfo(tensor)
    location = info.location
    assert location and Path(location).name == location, (tensor.name, location)
    offset = int(info.offset or 0)
    length = int(info.length) if info.length is not None else payload(tensor)
    assert offset == 0 and length == payload(tensor), (tensor.name, offset, length)
    return location, offset, length


def set_external(tensor, location, length):
    # Metadata-only tensors keep new/saved graphs small and source bytes intact.
    for field in ("raw_data", "float_data", "int32_data", "int64_data", "double_data", "uint64_data", "string_data"):
        tensor.ClearField(field)
    del tensor.external_data[:]
    for key, value in (("location", location), ("offset", "0"), ("length", str(length))):
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value
    tensor.data_location = TensorProto.EXTERNAL


def external_tensor(name, dtype, dims, location):
    tensor = TensorProto()
    tensor.name, tensor.data_type = name, dtype
    tensor.dims.extend(dims)
    set_external(tensor, location, payload(tensor))
    return tensor


def array_file_once(path, array):
    assert array.flags.c_contiguous
    view = memoryview(array).cast("B")
    if path.exists():
        assert path.stat().st_size == len(view), (path, "existing size")
        with path.open("rb") as stream:
            for start in range(0, len(view), 4 * 1024 * 1024):
                expected = view[start:start + 4 * 1024 * 1024]
                assert stream.read(len(expected)) == expected, (path, "existing bytes", start)
    else:
        with path.open("xb") as stream:
            for start in range(0, len(view), 4 * 1024 * 1024):
                chunk = view[start:start + 4 * 1024 * 1024]
                assert stream.write(chunk) == len(chunk), path
    return {"file": path.name, "bytes": len(view)}


def inspect_base(model, mode):
    graph = model.graph
    assert not model.functions
    assert all(a.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
               for node in graph.node for a in node.attribute)
    assert [x.name for x in graph.input] == INPUTS
    assert [x.name for x in graph.output] == OUTPUTS
    assert shape(graph.input[0]) == (1, 1)
    assert graph.input[0].type.tensor_type.elem_type == TensorProto.INT64
    assert shape(graph.output[0]) == (1, 1, N)
    assert graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    for value in graph.input[1:-1]:
        assert value.type.tensor_type.elem_type == TensorProto.FLOAT
        assert shape(value)[0:2] == (1, 12) and shape(value)[3] == 64
    assert graph.input[-1].type.tensor_type.elem_type == TensorProto.INT64
    assert len(shape(graph.input[-1])) == 2 and shape(graph.input[-1])[0] == 1
    for value in graph.output[1:]:
        assert value.type.tensor_type.elem_type == TensorProto.FLOAT
        assert shape(value)[0:2] == (1, 12) and shape(value)[3] == 64
    assert [(x.domain, x.version) for x in model.opset_import if x.domain == ""] == [("", 13)]
    names = [t.name for t in graph.initializer]
    assert len(names) == len(set(names))
    assert not any(name.startswith(PREFIX) for node in graph.node for name in (*node.input, *node.output))
    assert not any(name.startswith(PREFIX) for name in names)
    inits = {t.name: t for t in graph.initializer}
    e = inits[WTE]
    assert e.data_type == TensorProto.FLOAT and tuple(e.dims) == (N, K)
    e_sig = ext_signature(e)
    emb = unique((n for n in graph.node if n.name == EMBED_NODE), "wte Gather")
    assert emb.domain == "" and emb.op_type == "Gather"
    assert list(emb.input) == [WTE, "/transformer/Reshape_output_0"]
    assert list(emb.output) == [EMBED_OUT]
    assert set(attrs(emb)) <= {"axis"} and int(attrs(emb).get("axis", 0)) == 0
    transpose = unique((n for n in graph.node if WTE_TRANSPOSED in n.output), "wte Transpose")
    assert transpose.domain == "" and transpose.op_type == "Transpose"
    assert list(transpose.input) == [WTE] and list(transpose.output) == [WTE_TRANSPOSED]
    assert set(attrs(transpose)) <= {"perm"}
    assert list(attrs(transpose).get("perm", [1, 0])) == [1, 0]
    head = unique((n for n in graph.node if n.name == HEAD_NODE), "LM head")
    assert head.domain == "" and head.op_type == "MatMul" and not attrs(head)
    assert list(head.input) == ["/transformer/Reshape_3_output_0", WTE_TRANSPOSED]
    assert list(head.output) == ["logits"]
    assert [n.name for n in consumers(graph, WTE)] == [emb.name, transpose.name]
    assert [n.name for n in consumers(graph, WTE_TRANSPOSED)] == [head.name]
    counts = Counter(n.op_type for n in graph.node)
    expected = {"dense": (24, 0, 0), "native": (0, 24, 0), "q8_acc0": (0, 0, 24)}[mode]
    assert (counts["Gemm"], counts["Q8MatMulM1"], counts["MatMulNBits"]) == expected, (mode, counts)
    assert counts["MatMul"] == 13
    assert len(graph.node) == (1600 if mode == "dense" else 1648)
    return {"embedding": emb, "transpose": transpose, "head": head, "E": e, "E_external": e_sig}


def native_pack_bytes(graph):
    result = 0
    for node in graph.node:
        if node.domain == "org.blucca" and node.op_type == "Q8MatMulM1":
            a = attrs(node)
            k, n = int(a["K"]), int(a["N"])
            assert k % BLOCK == 0 and n % 16 == 0
            result += n * k + n * (k // BLOCK) * 4
    return result


def rewrite(model, mode, contract):
    graph = model.graph
    before_nodes = [n.SerializeToString() for n in graph.node]
    before_inits = {t.name: t.SerializeToString() for t in graph.initializer}
    before_outputs = [v.SerializeToString() for v in graph.output]
    before_inputs = [v.SerializeToString() for v in graph.input]
    before_value_info = {v.name: v.SerializeToString() for v in graph.value_info}
    old_pack = native_pack_bytes(graph)

    if mode == "dense":
        set_external(contract["E"], EFILE, N * K * 4)
        assert [n.SerializeToString() for n in graph.node] == before_nodes
    else:
        emb, transpose, head = contract["embedding"], contract["transpose"], contract["head"]
        removed_names = {emb.name, transpose.name, head.name}
        expected_nodes = [n.SerializeToString() for n in graph.node if n.name not in removed_names]
        nodes, added = [], []

        def init(suffix, array):
            name = PREFIX + suffix
            added.append(numpy_helper.from_array(array, name))
            return name

        def static(name, dtype, dims):
            value = helper.make_tensor_value_info(name, dtype, dims)
            existing = [v for v in graph.value_info if v.name == name]
            if existing:
                assert len(existing) == 1
                existing[0].CopyFrom(value)
            else:
                graph.value_info.append(value)

        scale_shape = init("scale_shape", np.array([NPAD, KB, 1], dtype=np.int64))
        zero_point = init("zero_point", np.array(128, dtype=np.float32))
        embed_shape = init("embed_shape", np.array([1, 1, K], dtype=np.int64))
        a_shape = init("A_shape", np.array([1, K], dtype=np.int64))
        starts = init("slice_starts", np.array([0], dtype=np.int64))
        ends = init("slice_ends", np.array([N], dtype=np.int64))
        axes = init("slice_axes", np.array([1], dtype=np.int64))
        steps = init("slice_steps", np.array([1], dtype=np.int64))
        logits_shape = init("logits_shape", np.array([1, 1, N], dtype=np.int64))
        codes, floats, centered = (PREFIX + x for x in ("embed_codes", "embed_float", "embed_centered"))
        scale_rows, selected_scales, blocks = (PREFIX + x for x in ("scale_rows", "embed_scales", "embed_blocks"))
        a2d, ypad, ysliced = (PREFIX + x for x in ("A2D", "Ypad", "Ysliced"))
        for name, dtype, dims in ((codes, TensorProto.UINT8, [1, 1, KB, BLOCK]),
                                  (floats, TensorProto.FLOAT, [1, 1, KB, BLOCK]),
                                  (centered, TensorProto.FLOAT, [1, 1, KB, BLOCK]),
                                  (scale_rows, TensorProto.FLOAT, [NPAD, KB, 1]),
                                  (selected_scales, TensorProto.FLOAT, [1, 1, KB, 1]),
                                  (blocks, TensorProto.FLOAT, [1, 1, KB, BLOCK]),
                                  (EMBED_OUT, TensorProto.FLOAT, [1, 1, K]),
                                  (a2d, TensorProto.FLOAT, [1, K]),
                                  (ypad, TensorProto.FLOAT, [1, NPAD]),
                                  (ysliced, TensorProto.FLOAT, [1, N])):
            static(name, dtype, dims)
        for node in graph.node:
            if node.name == emb.name:
                ids = emb.input[1]
                nodes.extend([
                    helper.make_node("Gather", [BN, ids], [codes], name=PREFIX + "embed_gather_B", axis=0),
                    helper.make_node("Cast", [codes], [floats], name=PREFIX + "embed_cast", to=TensorProto.FLOAT),
                    helper.make_node("Sub", [floats, zero_point], [centered], name=PREFIX + "embed_center",),
                    helper.make_node("Reshape", [SN, scale_shape], [scale_rows], name=PREFIX + "scale_reshape"),
                    helper.make_node("Gather", [scale_rows, ids], [selected_scales], name=PREFIX + "embed_gather_S", axis=0),
                    helper.make_node("Mul", [centered, selected_scales], [blocks], name=PREFIX + "embed_multiply"),
                    helper.make_node("Reshape", [blocks, embed_shape], [EMBED_OUT], name=PREFIX + "embed_reshape"),
                ])
            elif node.name == transpose.name:
                continue
            elif node.name == head.name:
                domain = "org.blucca" if mode == "native" else "com.microsoft"
                op = "Q8MatMulM1" if mode == "native" else "MatMulNBits"
                options = {"K": K, "N": NPAD}
                if mode == "q8_acc0":
                    options.update(bits=8, block_size=BLOCK, accuracy_level=0)
                assert [(o.domain, o.version) for o in model.opset_import if o.domain == domain] == [(domain, 1)]
                nodes.extend([
                    helper.make_node("Reshape", [head.input[0], a_shape], [a2d], name=PREFIX + "head_reshape_A"),
                    helper.make_node(op, [a2d, BN, SN], [ypad], name=PREFIX + "head_matmul", domain=domain, **options),
                    helper.make_node("Slice", [ypad, starts, ends, axes, steps], [ysliced], name=PREFIX + "head_slice"),
                    helper.make_node("Reshape", [ysliced, logits_shape], ["logits"], name=PREFIX + "head_reshape_logits"),
                ])
            else:
                nodes.append(node)
        del graph.node[:]
        graph.node.extend(nodes)
        keep = [t for t in graph.initializer if t.name != WTE]
        del graph.initializer[:]
        graph.initializer.extend(keep)
        graph.initializer.extend([
            external_tensor(BN, TensorProto.UINT8, [NPAD, KB, BLOCK], BFILE),
            external_tensor(SN, TensorProto.FLOAT, [NPAD * KB], SFILE),
            *added,
        ])
        # Only stale vocabulary weight value_info is removed.
        keep_info = [v for v in graph.value_info if v.name not in {WTE, WTE_TRANSPOSED}]
        del graph.value_info[:]
        graph.value_info.extend(keep_info)
        assert [n.SerializeToString() for n in graph.node if not n.name.startswith(PREFIX)] == expected_nodes
        assert len(graph.node) == 1656
        assert not consumers(graph, WTE) and not consumers(graph, WTE_TRANSPOSED)
        assert not any(WTE_TRANSPOSED in n.output for n in graph.node)
        assert {n.name for n in consumers(graph, BN)} == {PREFIX + "embed_gather_B", PREFIX + "head_matmul"}
        assert {n.name for n in consumers(graph, SN)} == {PREFIX + "scale_reshape", PREFIX + "head_matmul"}
        assert unique((n for n in graph.node if n.name == PREFIX + "embed_gather_S"), "scale Gather").input[0] == scale_rows
        assert native_pack_bytes(graph) == old_pack + (NPAD * K + NPAD * KB * 4 if mode == "native" else 0)

    after_inits = {t.name: t for t in graph.initializer}
    assert len(after_inits) == len(graph.initializer)
    for name, raw in before_inits.items():
        if name != WTE:
            assert after_inits[name].SerializeToString() == raw, (mode, "changed old initializer", name)
    if mode == "dense":
        assert len(after_inits) == len(before_inits)
        assert {n.name for n in consumers(graph, WTE)} == {EMBED_NODE, contract["transpose"].name}
        assert BN not in after_inits and SN not in after_inits
    else:
        assert WTE not in after_inits and len(after_inits) == len(before_inits) + 10
    assert [v.SerializeToString() for v in graph.output] == before_outputs
    assert [v.SerializeToString() for v in graph.input] == before_inputs
    after_info = {v.name: v.SerializeToString() for v in graph.value_info}
    for name, raw in before_value_info.items():
        if mode == "dense" or name not in {WTE, WTE_TRANSPOSED, EMBED_OUT}:
            assert after_info[name] == raw, (mode, "changed unrelated value_info", name)
    counts = Counter(n.op_type for n in graph.node)
    expected = {"dense": (24, 0, 0, 13), "native": (0, 25, 0, 12), "q8_acc0": (0, 0, 25, 12)}[mode]
    assert (counts["Gemm"], counts["Q8MatMulM1"], counts["MatMulNBits"], counts["MatMul"]) == expected
    return {"nodes": len(graph.node), "initializer_count": len(graph.initializer),
            "initializer_payload_bytes": sum(payload(t) for t in graph.initializer),
            "op_counts": dict(counts), "preserved_old_initializer_count": len(before_inits) - 1,
            "non_vocabulary_nodes_preserved_byte_for_byte": True,
            "non_vocabulary_initializers_preserved_byte_for_byte": True,
            "input_output_metadata_preserved_byte_for_byte": True,
            "native_old_prepack_payload_bytes": old_pack,
            "native_new_head_prepack_payload_bytes": NPAD * K + NPAD * KB * 4 if mode == "native" else 0,
            "native_total_prepack_payload_bytes": native_pack_bytes(graph),
            "weight_tie": {"kind": "shared_FP32_dequantized_E" if mode == "dense" else "shared_Q8_B_S",
                           "embedding_and_head_same_quantized_rows": True,
                           "canonical_initializers": [WTE] if mode == "dense" else [BN, SN],
                           "canonical_initializer_consumers": {name: [n.name for n in consumers(graph, name)]
                                                               for name in ([WTE] if mode == "dense" else [BN, SN])}}}


def copy_unchanged(model, source_dir, target_dir, links):
    for tensor in model.graph.initializer:
        if not uses_external_data(tensor):
            continue
        location, offset, length = ext_signature(tensor)
        if location in {BFILE, SFILE, EFILE}:
            assert (target_dir / location).stat().st_size == length
            continue
        source, target = source_dir / location, target_dir / location
        assert source.is_file() and source.stat().st_size == length, source
        # ONNX 1.23 rejects external tensors with multiple hardlinks. One
        # ordinary copy can still be shared by all three graph files here.
        if not target.exists():
            shutil.copyfile(source, target)
        assert target.stat().st_nlink == 1 and target.stat().st_size == length
        assert filecmp.cmp(source, target, shallow=False), (source, target, "copy bytes")
        links[location] = {"file": location, "bytes": length, "source": str(source), "source_bytes_preserved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    base_root, out_dir = args.base_root.resolve(), args.out_dir.resolve()
    source_dir, target_dir = base_root / "models", out_dir / "models"
    assert source_dir != target_dir and sys.byteorder == "little"
    sys.path.insert(0, str(base_root))
    old = importlib.import_module("converter")
    assert old.BLOCK == BLOCK
    assert old.SOURCE_REF == "Xenova/distilgpt2@a41c10485c18a64b6606729b6a082330cbd8f49e"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Inspect metadata only. Existing 24-Gemm weights are never loaded/rewritten.
    signatures = []
    for mode in MODES:
        model = onnx.load(source_dir / f"{mode}.onnx", load_external_data=False)
        contract = inspect_base(model, mode)
        signatures.append(contract["E_external"])
        del contract, model
    assert signatures[0] == signatures[1] == signatures[2]
    location, offset, length = signatures[0]
    source_e = source_dir / location
    assert source_e.is_file() and source_e.stat().st_size == N * K * 4 == length

    # One common quantization with vocabulary padding, using the original layout.
    e = np.memmap(source_e, mode="r", dtype="<f4", offset=offset, shape=(N, K))
    w = np.zeros((K, NPAD), dtype=np.float32)
    w[:, :N] = e.T
    b, s, deq = old.quant8(w)
    assert b.dtype == np.uint8 and b.shape == (NPAD, KB, BLOCK) and b.flags.c_contiguous
    assert s.dtype == np.float32 and s.shape == (NPAD * KB,) and s.flags.c_contiguous
    assert np.isfinite(s).all() and np.all(s > 0)
    assert np.all(b[N:] == 128) and np.all(s.reshape(NPAD, KB)[N:] > 0)
    assert np.all(deq[:, N:] == 0)
    del w, e
    gc.collect()
    new_weights = [array_file_once(target_dir / BFILE, b), array_file_once(target_dir / SFILE, s)]
    dense_e = np.ascontiguousarray(deq[:, :N].T)
    assert dense_e.dtype == np.float32 and dense_e.shape == (N, K)
    new_weights.append(array_file_once(target_dir / EFILE, dense_e))
    del b, s, deq, dense_e
    gc.collect()

    links, models, artifact_users = {}, {}, {}
    for mode in MODES:
        model = onnx.load(source_dir / f"{mode}.onnx", load_external_data=False)
        contract = inspect_base(model, mode)
        report = rewrite(model, mode, contract)
        copy_unchanged(model, source_dir, target_dir, links)
        target = target_dir / f"{mode}.onnx"
        onnx.save_model(model, target)
        onnx.checker.check_model(str(target))
        # Saved metadata must retain the in-memory protobuf contract exactly.
        saved = onnx.load(target, load_external_data=False)
        assert [n.SerializeToString() for n in saved.graph.node] == [n.SerializeToString() for n in model.graph.node]
        assert [t.SerializeToString() for t in saved.graph.initializer] == [t.SerializeToString() for t in model.graph.initializer]
        assert [v.name for v in saved.graph.output] == OUTPUTS and len(saved.graph.output) == 13
        external = sorted({ext_signature(t)[0] for t in saved.graph.initializer if uses_external_data(t)})
        for name in external:
            artifact_users.setdefault(name, []).append(mode)
        report.update(file=target.name, file_bytes=target.stat().st_size, external_files=external,
                      onnx_checker_passed=True, saved_metadata_verified=True,
                      head_output_shape=[1, 1, N], runtime_gate_status="pending_root_runner")
        models[mode] = report
        print(f"SAVED {mode}: nodes={report['nodes']} initializer={report['initializer_payload_bytes']} pack={report['native_total_prepack_payload_bytes']}", flush=True)
        del contract, model, saved
        gc.collect()

    manifest = {
        "status": "converted_checked_runtime_pending", "base_root": str(base_root),
        "source_ref": old.SOURCE_REF, "converter_reused": str(Path(old.__file__).resolve()),
        "numpy": np.__version__, "onnx": onnx.__version__, "kernel_API": 31,
        "batch": 1, "decode_sequence": 1, "K": K, "block_size": BLOCK,
        "vocab": N, "padded_vocab": NPAD, "padding_rows": NPAD - N,
        "quantization": "original FP32 absmax/block32; floor1e-8; /127; rint; +128; clip UINT8",
        "padding": {"weight": 0, "B_code": 128, "S_finite_positive": True,
                    "input_id_domain": [0, N - 1], "head_slice_axis_2d": 1,
                    "head_slice_start": 0, "head_slice_end_exclusive": N,
                    "slice_before_token_selection_or_NLL": True},
        "original_tie": {"initializer": WTE, "shape": [N, K], "bytes": N * K * 4,
                         "embedding": EMBED_NODE, "head": HEAD_NODE,
                         "head_weight_parent": "rank2 Transpose of same initializer",
                         "source_weight_modified": False},
        "new_weight_files": [{**w, "used_by": artifact_users.get(w["file"], [])} for w in new_weights],
        "new_unique_weight_bytes": sum(w["bytes"] for w in new_weights),
        "shared_vocab_B_S_payload_bytes": NPAD * K + NPAD * KB * 4,
        "native_additional_vocab_pack_payload_bytes": NPAD * K + NPAD * KB * 4,
        "array_payload_scope": "initializer tensor bytes and native code/scale arrays; ORT/allocator/RSS measured by runtime worker",
        "copied_unchanged_files": sorted(links.values(), key=lambda x: x["file"]),
        "unchanged_bins_copied": len(links), "old_kernel_or_24_Gemm_changed": False,
        "models": models, "outputs": OUTPUTS,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"CONVERTED_CHECKED: {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
