#!/usr/bin/env python3
"""Four same-Q8-weight versions of one pinned, batch-1 distilgpt2 decoder.

The source identity is verified by prepare.py. This converter checks its graph
contract; it does not download, hash, benchmark, or change the native kernel.
"""
import argparse
import gc
import json
import math
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import (
    ExternalDataInfo, convert_model_to_external_data, set_external_data,
    uses_external_data,
)

BLOCK = 32
PREFIX = "__blucca_whole_q8_"
MODES = ("dense", "native", "q8_acc0", "split8")
SOURCE_BYTES = 328939971
SOURCE_REF = "Xenova/distilgpt2@a41c10485c18a64b6606729b6a082330cbd8f49e"


def packed4(q):
    # Low nibble = even K; high nibble = the adjacent odd K.
    return np.ascontiguousarray(q[..., 0::2] | (q[..., 1::2] << 4))


def dequant(b, s):
    n, kb, _ = b.shape
    return np.ascontiguousarray(
        ((b.astype(np.float32) - 128) * s.reshape(n, kb, 1))
        .transpose(1, 2, 0).reshape(kb * BLOCK, n)
    )


def quant8(w):
    # Numerically identical to research/ort_split8.py::quant8.
    k, n = w.shape
    assert w.dtype == np.float32 and k % BLOCK == 0
    assert np.isfinite(w).all()
    wb = w.reshape(k // BLOCK, BLOCK, n)
    scale = np.maximum(np.abs(wb).max(axis=1), np.float32(1e-8)) / np.float32(127)
    q = np.clip(np.rint(wb / scale[:, None, :]) + 128, 0, 255).astype(np.uint8)
    b = np.ascontiguousarray(q.transpose(2, 0, 1))
    s = np.ascontiguousarray(scale.T).reshape(-1)
    assert np.isfinite(s).all()
    return b, s, dequant(b, s)


def gemm_record(index, node, inits):
    assert node.domain == "" and node.op_type == "Gemm"
    assert len(node.input) == 3 and all(node.input) and len(node.output) == 1
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    assert set(attrs) <= {"alpha", "beta", "transA", "transB"}, node.name
    effective = {"alpha": float(attrs.get("alpha", 1.0)),
                 "beta": float(attrs.get("beta", 1.0)),
                 "transA": int(attrs.get("transA", 0)),
                 "transB": int(attrs.get("transB", 0))}
    assert effective["alpha"] == effective["beta"] == 1.0, node.name
    assert effective["transA"] == 0 and effective["transB"] in (0, 1), node.name
    tensor = inits[node.input[1]]
    assert tensor.data_type == TensorProto.FLOAT and len(tensor.dims) == 2, node.name
    dims = list(tensor.dims)
    k, n = dims[::-1] if effective["transB"] else dims
    assert k > 0 and k % BLOCK == 0 and n > 0 and n % 16 == 0, node.name
    bias = inits.get(node.input[2])
    if bias is not None:
        assert bias.data_type == TensorProto.FLOAT
        assert np.broadcast_shapes((1, n), tuple(bias.dims)) == (1, n), node.name
    return {"source_node_index": index, "source_node": node.name,
            "source_inputs": list(node.input), "source_outputs": list(node.output),
            "source_attributes": attrs, "effective_attributes": effective,
            "source_W": tensor.name, "source_W_shape": dims,
            "normalized_W_shape": [k, n],
            "source_bias_shape": list(bias.dims) if bias is not None else None,
            "B_shape": [n, k // BLOCK, BLOCK], "B_dtype": "UINT8",
            "B_payload_bytes": k * n,
            "S_shape": [n * (k // BLOCK)], "S_dtype": "FLOAT",
            "S_payload_bytes": n * (k // BLOCK) * 4}


def inspect_source(model):
    graph = model.graph
    assert len(graph.node) == 1600 and len(graph.initializer) == 76
    assert [(x.domain, x.version) for x in model.opset_import] == [("", 13)]
    assert len(graph.input) == 14 and len(graph.output) == 13
    assert graph.input[0].name == "input_ids" and graph.output[0].name == "logits"
    assert graph.input[0].type.tensor_type.shape.dim[1].dim_value == 1
    assert graph.output[0].type.tensor_type.shape.dim[1].dim_value == 1
    for value in (*graph.input, *graph.output):
        assert value.type.tensor_type.shape.dim[0].dim_param == "batch_size", value.name
    # This fixed source has no If/Loop/function subgraphs to traverse.
    assert not model.functions
    assert all(a.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
               for node in graph.node for a in node.attribute)
    names = [v.name for v in graph.input] + [v.name for v in graph.output]
    names += [v.name for v in graph.initializer] + [v.name for v in graph.value_info]
    names += [name for node in graph.node for name in (*node.input, *node.output)]
    assert not any(name.startswith(PREFIX) for name in names)
    inits = {t.name: t for t in graph.initializer}
    assert len(inits) == 76
    rows = [gemm_record(i, node, inits) for i, node in enumerate(graph.node)
            if node.op_type == "Gemm"]
    expected = {}
    for layer in range(6):
        for suffix, shape in [("attn/c_attn", [768, 2304]),
                              ("attn/c_proj", [768, 768]),
                              ("mlp/c_fc", [768, 3072]),
                              ("mlp/c_proj", [3072, 768])]:
            expected[f"/transformer/h.{layer}/{suffix}/Gemm"] = shape
    assert len(rows) == 24 and len({r["source_W"] for r in rows}) == 24
    assert {r["source_node"]: r["normalized_W_shape"] for r in rows} == expected
    matmuls = [n for n in graph.node if n.op_type == "MatMul"]
    assert len(matmuls) == 13
    assert sum(n.name == "/lm_head/MatMul" and list(n.output) == ["logits"]
               for n in matmuls) == 1
    assert all(n.input[1] not in inits for n in matmuls)
    return rows


def freeze_batch(model):
    for value in (*model.graph.input, *model.graph.output):
        dim = value.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_param")
        dim.dim_value = 1


def add_domain(model, domain):
    versions = [x.version for x in model.opset_import if x.domain == domain]
    if versions:
        assert versions == [1]
    else:
        model.opset_import.append(helper.make_opsetid(domain, 1))


def payload_bytes(tensor):
    dtype = helper.tensor_dtype_to_np_dtype(tensor.data_type)
    return math.prod(tensor.dims) * np.dtype(dtype).itemsize


def transform(model, rows, bank, mode):
    """Mutate one fresh source, retaining all non-Gemm nodes byte-for-byte."""
    assert mode in MODES
    graph = model.graph
    original_nodes = list(graph.node)
    original_non_gemm = [n.SerializeToString() for n in original_nodes if n.op_type != "Gemm"]
    all_original_nodes = [n.SerializeToString() for n in original_nodes] if mode == "dense" else None
    inits = {t.name: t for t in graph.initializer}
    freeze_batch(model)
    by_index = {r["source_node_index"]: (i, r) for i, r in enumerate(rows)}
    value_info = {v.name: v for v in graph.value_info}
    nodes, additions = [], []

    def static(name, shape):
        info = helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        if name in value_info:
            value_info[name].CopyFrom(info)
        else:
            graph.value_info.append(info)
            value_info[name] = graph.value_info[-1]

    def init(name, array):
        additions.append(numpy_helper.from_array(array, name))

    for index, node in enumerate(original_nodes):
        if index not in by_index:
            nodes.append(node)
            continue
        number, row = by_index[index]
        k, n = row["normalized_W_shape"]
        b, s = bank[row["source_W"]]
        assert b.dtype == np.uint8 and b.shape == (n, k // BLOCK, BLOCK)
        assert s.dtype == np.float32 and s.shape == (n * (k // BLOCK),)
        if mode == "dense":
            w = dequant(b, s)
            if row["effective_attributes"]["transB"]:
                w = np.ascontiguousarray(w.T)
            inits[row["source_W"]].CopyFrom(numpy_helper.from_array(w, row["source_W"]))
            del w
            nodes.append(node)
            continue

        stem = f"{PREFIX}{number:02d}"
        a2d, y2d = stem + "_A2D", stem + "_Y2D"
        shape = stem + "_shape"
        init(shape, np.array([1, k], dtype=np.int64))
        nodes.append(helper.make_node("Reshape", [node.input[0], shape], [a2d], name=stem + "_reshape"))
        static(a2d, [1, k])
        static(y2d, [1, n])
        static(node.output[0], [1, n])
        if mode in ("native", "q8_acc0"):
            bn, sn = stem + "_B", stem + "_S"
            init(bn, b)
            init(sn, s)
            if mode == "native":
                domain = "org.blucca"
                op, attrs = "Q8MatMulM1", {"K": k, "N": n}
            else:
                domain = "com.microsoft"
                op, attrs = "MatMulNBits", {"K": k, "N": n, "bits": 8,
                                          "block_size": BLOCK, "accuracy_level": 0}
            nodes.append(helper.make_node(op, [a2d, bn, sn], [y2d],
                                          name=stem + "_matmul", domain=domain, **attrs))
            add_domain(model, domain)
        else:
            high, low = packed4(b >> 4), packed4(b & 15)
            sh = s * np.float32(16)
            assert np.isfinite(sh).all()
            zl = np.zeros((n, (k // BLOCK + 1) // 2), dtype=np.uint8)
            bh, bl = stem + "_split_BH", stem + "_split_BL"
            shn, sln, zln = stem + "_split_SH", stem + "_split_SL", stem + "_split_ZL"
            for name, array in [(bh, high), (bl, low), (shn, sh), (sln, s), (zln, zl)]:
                init(name, array)
            yh, yl = stem + "_split_H", stem + "_split_L"
            static(yh, [1, n])
            static(yl, [1, n])
            attrs = {"K": k, "N": n, "bits": 4, "block_size": BLOCK, "accuracy_level": 0}
            nodes.extend([
                helper.make_node("MatMulNBits", [a2d, bh, shn], [yh],
                                 name=stem + "_split_high", domain="com.microsoft", **attrs),
                helper.make_node("MatMulNBits", [a2d, bl, sln, zln], [yl],
                                 name=stem + "_split_low", domain="com.microsoft", **attrs),
                helper.make_node("Add", [yh, yl], [y2d], name=stem + "_split_sum"),
            ])
            add_domain(model, "com.microsoft")
            del high, low, sh, zl
        # alpha/beta are 1; Add retains the original legal C broadcast.
        nodes.append(helper.make_node("Add", [y2d, node.input[2]], list(node.output),
                                      name=stem + "_bias"))

    del graph.node[:]
    graph.node.extend(nodes)
    graph.initializer.extend(additions)
    removed = []
    if mode != "dense":
        referenced = {name for node in graph.node for name in node.input}
        referenced.update(v.name for v in (*graph.input, *graph.output))
        raw_weights = {r["source_W"] for r in rows}
        keep = []
        for tensor in graph.initializer:
            if tensor.name in raw_weights and tensor.name not in referenced:
                removed.append(tensor.name)
            else:
                keep.append(tensor)
        del graph.initializer[:]
        graph.initializer.extend(keep)
        assert [n.SerializeToString() for n in graph.node if not n.name.startswith(PREFIX)] == original_non_gemm
    else:
        assert [n.SerializeToString() for n in graph.node] == all_original_nodes
    counts = Counter(n.op_type for n in graph.node)
    expected = len(rows)
    assert counts["Gemm"] == (expected if mode == "dense" else 0)
    assert counts["Q8MatMulM1"] == (expected if mode == "native" else 0)
    assert counts["MatMulNBits"] == (expected if mode == "q8_acc0" else 2 * expected if mode == "split8" else 0)
    assert len({t.name for t in graph.initializer}) == len(graph.initializer)
    assert all(v.type.tensor_type.shape.dim[0].dim_value == 1
               for v in (*graph.input, *graph.output))
    return {"nodes": len(graph.node), "mapped_Gemm": expected,
            "op_counts": dict(counts), "removed_raw_weights": removed,
            "initializer_count": len(graph.initializer),
            "initializer_payload_bytes": sum(payload_bytes(t) for t in graph.initializer)}


def equal_file_bytes(path, raw):
    assert path.stat().st_size == len(raw), f"External initializer size mismatch: {path}"
    with path.open("rb") as stream:
        for offset in range(0, len(raw), 1024 * 1024):
            assert stream.read(1024 * 1024) == raw[offset:offset + 1024 * 1024], path


def save_external(model, path, mode, raw_weights):
    # Built-in per-tensor externalization; attributes stay in the graph.
    convert_model_to_external_data(model, all_tensors_to_one_file=False,
                                  size_threshold=1024, convert_attribute=False)
    external = []
    for tensor in model.graph.initializer:
        if not uses_external_data(tensor):
            continue
        # Dense W values differ from original W if the latter remains shared.
        prefix = "dense__" if mode == "dense" and tensor.name in raw_weights else ""
        location = prefix + quote(tensor.name, safe="._-") + ".bin"
        artifact = path.parent / location
        raw = tensor.raw_data
        if artifact.exists():
            equal_file_bytes(artifact, raw)
            set_external_data(tensor, location, offset=0, length=len(raw))
            tensor.ClearField("raw_data")
        else:
            set_external_data(tensor, location)
        external.append(location)
        del raw
    # Already converted. Re-converting here would replace the chosen locations.
    onnx.save_model(model, path)
    onnx.checker.check_model(str(path))
    for tensor in model.graph.initializer:
        if uses_external_data(tensor):
            info = ExternalDataInfo(tensor)
            assert info.offset == 0 and info.length == payload_bytes(tensor)
            assert (path.parent / info.location).stat().st_size == info.length
    return sorted(set(external))


def convert(source, out_dir):
    source, out_dir = source.resolve(), out_dir.resolve()
    assert source.stat().st_size == SOURCE_BYTES
    model = onnx.load(source)
    onnx.checker.check_model(model)
    rows = inspect_source(model)
    initializers = {t.name: t for t in model.graph.initializer}
    bank = {}
    for row in rows:
        w = numpy_helper.to_array(initializers[row["source_W"]])
        if row["effective_attributes"]["transB"]:
            w = np.ascontiguousarray(w.T)
        b, s, deq = quant8(w)
        assert list(b.shape) == row["B_shape"] and list(s.shape) == row["S_shape"]
        bank[row["source_W"]] = (b, s)
        del w, deq
    del initializers, model
    gc.collect()
    out_dir.mkdir(parents=True, exist_ok=True)
    common_b = sum(r["B_payload_bytes"] for r in rows)
    common_s = sum(r["S_payload_bytes"] for r in rows)
    manifest = {"source": {"path": str(source), "bytes": SOURCE_BYTES, "ref": SOURCE_REF,
                           "nodes": 1600, "Gemm": 24, "preserved_MatMul": 13},
                "batch": 1, "decode_sequence": 1, "block_size": BLOCK,
                "quantization": "FP32 absmax/block32; floor 1e-8; /127; rint; +128; clip uint8",
                "common_Q8_B_payload_bytes": common_b,
                "common_Q8_S_payload_bytes": common_s,
                "common_Q8_B_S_payload_bytes": common_b + common_s,
                "native_additional_prepack_payload_bytes": common_b + common_s,
                "prepack_scope": "signed codes and reordered FP32 scales; excludes object/vector metadata",
                "source_weights": rows, "modes": {}, "artifacts": [],
                "artifact_total_scope": "four ONNX graphs and unique external initializer files; manifest excluded"}
    artifact_users = {}
    for mode in MODES:
        print(f"CONVERT {mode}", flush=True)
        model = onnx.load(source)  # One fresh source/converted model at a time.
        assert inspect_source(model) == rows
        result = transform(model, rows, bank, mode)
        onnx.checker.check_model(model)
        target = out_dir / f"{mode}.onnx"
        external = save_external(model, target, mode, {r["source_W"] for r in rows})
        result.update({"file": target.name, "file_bytes": target.stat().st_size,
                       "external_files": external,
                       "native_extra_prepack_payload_bytes": common_b + common_s if mode == "native" else 0})
        manifest["modes"][mode] = result
        for name in [target.name, *external]:
            artifact_users.setdefault(name, []).append(mode)
        manifest["artifacts"] = [{"file": name, "bytes": (out_dir / name).stat().st_size,
                                  "used_by": users}
                                 for name, users in sorted(artifact_users.items())]
        manifest["unique_artifact_bytes"] = sum(a["bytes"] for a in manifest["artifacts"])
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"SAVED {mode} graph={result['file_bytes']} initializer={result['initializer_payload_bytes']}", flush=True)
        del model, result, external
        gc.collect()
    del bank
    return manifest


def self_test():
    # All codes expose adjacency and the split zero-point algebra exactly.
    b = np.arange(512, dtype=np.uint16).astype(np.uint8).reshape(16, 1, BLOCK)
    s = np.full(16, np.float32(.125), dtype=np.float32)
    high, low = packed4(b >> 4), packed4(b & 15)

    def unpack(p):
        q = np.empty(p.shape[:-1] + (BLOCK,), dtype=np.uint8)
        q[..., 0::2], q[..., 1::2] = p & 15, p >> 4
        return q

    np.testing.assert_array_equal(unpack(high), b >> 4)
    np.testing.assert_array_equal(unpack(low), b & 15)
    split = ((unpack(high).astype(np.float32) - 8) * (s * 16).reshape(16, 1, 1)
             + unpack(low).astype(np.float32) * s.reshape(16, 1, 1))
    np.testing.assert_array_equal(split, (b.astype(np.float32) - 128) * s.reshape(16, 1, 1))
    w = dequant(b, s)
    w[:, 0] = 0
    qb, qs, qdeq = quant8(w)
    assert np.all(qb[0] == 128) and qs[0] == np.float32(1e-8) / np.float32(127)
    np.testing.assert_array_equal(qdeq, dequant(qb, qs))
    # Small in-memory Gemm verifies transB normalization, bias, and both stock paths.
    import onnxruntime as ort
    bias = np.arange(16, dtype=np.float32) * np.float32(.25)
    a = (np.arange(BLOCK, dtype=np.float32) % 5 - 2).reshape(1, BLOCK)
    for trans_b in (0, 1):
        stored = np.ascontiguousarray(w.T) if trans_b else w
        node = helper.make_node("Gemm", ["A", "W", "C"], ["Y"], name="self_gemm", transB=trans_b)
        graph = helper.make_graph([node], "self", [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, BLOCK])],
                                  [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16])],
                                  [numpy_helper.from_array(stored, "W"), numpy_helper.from_array(bias, "C")])
        source = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
        row = gemm_record(0, node, {t.name: t for t in source.graph.initializer})
        for mode in MODES:
            model = onnx.ModelProto()
            model.CopyFrom(source)
            transform(model, [row], {"W": (qb, qs)}, mode)
            onnx.checker.check_model(model)
            if mode != "native":
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                actual = ort.InferenceSession(model.SerializeToString(), options,
                                              providers=["CPUExecutionProvider"]).run(None, {"A": a})[0]
                np.testing.assert_allclose(actual, a @ qdeq + bias, rtol=2e-6, atol=2e-6)
            del model
    print("SELF_TEST PASS: quant8/zero floor, all-code adjacent split, transB0/1, bias, ONNX, stock Q8/Q4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.source is None or args.out_dir is None:
            parser.error("--source and --out-dir are required")
        convert(args.source, args.out_dir)


if __name__ == "__main__":
    main()
