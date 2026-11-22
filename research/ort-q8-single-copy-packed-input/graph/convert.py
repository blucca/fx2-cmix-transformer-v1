#!/usr/bin/env python3
"""Serialize one existing fixed tied-Q8 native decoder as single-copy packed Q8.

Usage: convert.py --source EXISTING/native.onnx --output_dir NEWDIR
NEWDIR must be fresh. Existing UINT8 B / FP32 S bytes are centered/reordered;
no quantizer, inference, download, native compilation, or source write occurs.
"""
import argparse
from collections import Counter
import filecmp
import gc
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import ExternalDataInfo, uses_external_data

from equation import BLOCK, TILE, check_equations, pack_codes, pack_scales, packed_embedding

VOCAB, NPAD, EMBED_K = 50257, 50272, 768
OLD = "__blucca_tied_vocab_q8_"
NEW = "__blucca_packed_q8_embed_"
EMBED_OUT = "/transformer/wte/Gather_output_0"
HEAD = OLD + "head_matmul"
OLD_EMBED = {OLD + x for x in ("embed_gather_B", "embed_cast", "embed_center",
                              "scale_reshape", "embed_gather_S", "embed_multiply", "embed_reshape")}
OLD_EMBED_VALUES = {OLD + x for x in ("embed_codes", "embed_float", "embed_centered",
                                     "scale_rows", "embed_scales", "embed_blocks")}
OLD_OWNED_INITS = {OLD + "scale_shape", OLD + "zero_point"}
INPUTS = ["input_ids"] + [f"past_key_values.{i}.{v}" for i in range(6) for v in ("key", "value")] + ["attention_mask"]
OUTPUTS = ["logits"] + [f"present.{i}.{v}" for i in range(6) for v in ("key", "value")]
PACKED_BYTES = 91210752


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def shape(value):
    return tuple(d.dim_value if d.HasField("dim_value") else d.dim_param
                 for d in value.type.tensor_type.shape.dim)


def payload(tensor):
    return math.prod(tensor.dims) * np.dtype(helper.tensor_dtype_to_np_dtype(tensor.data_type)).itemsize


def consumers(graph, name):
    return [n.name for n in graph.node if name in n.input]


def fixed_rows():
    rows = []
    for index in range(24):
        k, n = ((768, 2304), (768, 768), (768, 3072), (3072, 768))[index % 4]
        stem = f"__blucca_whole_q8_{index:02d}"
        rows.append({"index": index, "node": stem + "_matmul", "k": k, "n": n,
                     "B": stem + "_B", "S": stem + "_S", "P": stem + "_P", "PS": stem + "_PS"})
    rows.append({"index": 24, "node": HEAD, "k": EMBED_K, "n": NPAD,
                 "B": OLD + "B", "S": OLD + "S", "P": OLD + "P", "PS": OLD + "PS"})
    return rows


def fixed_source_files():
    """Pinned complete external inventory, checked before any model parse."""
    files = {}
    for row in fixed_rows():
        files[row["B"] + ".bin"] = row["n"] * row["k"]
        files[row["S"] + ".bin"] = row["n"] * (row["k"] // BLOCK) * 4
    files["transformer.wpe.weight.bin"] = 1024 * EMBED_K * 4
    for kind in ("bias", "weight"):
        files[f"transformer.ln_f.{kind}.bin"] = EMBED_K * 4
    for layer in range(6):
        stem = f"transformer.h.{layer}."
        for suffix, count in (("attn.c_attn.bias", 2304), ("attn.c_proj.bias", 768),
                              ("ln_1.bias", 768), ("ln_1.weight", 768),
                              ("ln_2.bias", 768), ("ln_2.weight", 768),
                              ("mlp.c_fc.bias", 3072), ("mlp.c_proj.bias", 768)):
            files[stem + suffix + ".bin"] = count * 4
    assert len(files) == 101 and sum(files.values()) == 94602240
    return files


def preflight_files(source, packed=False):
    if not source.is_file():
        raise FileNotFoundError(str(source))
    files = fixed_source_files()
    if packed:
        for row in fixed_rows():
            files[row["P"] + ".bin"] = files.pop(row["B"] + ".bin")
            files[row["PS"] + ".bin"] = files.pop(row["S"] + ".bin")
    for name, length in files.items():
        path = source.parent / name
        if not path.is_file():
            raise FileNotFoundError(str(path))
        if path.stat().st_size != length:
            raise ValueError(f"{path}: actual_bytes={path.stat().st_size}, required_bytes={length}")
    return {"status": "PASS_complete_external_data_pre_model_parse", "source": str(source),
            "metadata_bytes": source.stat().st_size, "external_file_count": len(files),
            "external_bytes": sum(files.values()), "weights_loaded_at_preflight": False}


def ext_signature(tensor):
    assert uses_external_data(tensor), tensor.name
    info = ExternalDataInfo(tensor)
    location = info.location
    length = int(info.length) if info.length is not None else payload(tensor)
    offset = int(info.offset or 0)
    assert location and Path(location).name == location, (tensor.name, location)
    assert offset == 0 and length == payload(tensor), (tensor.name, offset, length)
    return location, offset, length


def external_tensor(name, dtype, dims):
    t = TensorProto(name=name, data_type=dtype, dims=dims, data_location=TensorProto.EXTERNAL)
    for key, value in (("location", name + ".bin"), ("offset", "0"), ("length", str(payload(t)))):
        e = t.external_data.add()
        e.key, e.value = key, value
    return t


def inspect_source(model):
    g = model.graph
    assert len(g.node) == 1656 and len(g.initializer) == 134
    assert sum(payload(t) for t in g.initializer) == 94602748
    assert [(o.domain, o.version) for o in model.opset_import] == [("", 13), ("org.blucca", 1)]
    assert not model.functions
    assert all(a.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
               for n in g.node for a in n.attribute)
    assert [x.name for x in g.input] == INPUTS and [x.name for x in g.output] == OUTPUTS
    assert shape(g.input[0]) == (1, 1) and g.input[0].type.tensor_type.elem_type == TensorProto.INT64
    assert shape(g.output[0]) == (1, 1, VOCAB) and g.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    counts = Counter(n.op_type for n in g.node)
    assert (counts["Q8MatMulM1"], counts["Gemm"], counts["MatMulNBits"], counts["MatMul"]) == (25, 0, 0, 12)
    inits = {t.name: t for t in g.initializer}
    assert len(inits) == len(g.initializer) and "transformer.wte.weight" not in inits
    actual_files = {ext_signature(t)[0]: ext_signature(t)[2] for t in g.initializer if uses_external_data(t)}
    assert actual_files == fixed_source_files()
    nodes = {n.name: n for n in g.node}
    assert len(nodes) == len(g.node)
    assert not any(name.startswith(NEW) for name in [*nodes, *inits, *(v.name for v in g.value_info)]
                   + [name for node in g.node for name in (*node.input, *node.output)])
    assert {n.name for n in g.node if n.domain == "org.blucca"} == {r["node"] for r in fixed_rows()}
    values = {v.name: v for v in (*g.input, *g.output, *g.value_info)}
    rows = fixed_rows()
    for row in rows:
        node, k, n = nodes[row["node"]], row["k"], row["n"]
        assert node.domain == "org.blucca" and node.op_type == "Q8MatMulM1"
        assert attrs(node) == {"K": k, "N": n}
        assert all(a.type == onnx.AttributeProto.INT for a in node.attribute)
        assert len(node.input) == 3 and list(node.input[1:]) == [row["B"], row["S"]] and len(node.output) == 1
        for name, dims in ((node.input[0], (1, k)), (node.output[0], (1, n))):
            assert values[name].type.tensor_type.elem_type == TensorProto.FLOAT and shape(values[name]) == dims, name
        assert inits[row["B"]].data_type == TensorProto.UINT8 and tuple(inits[row["B"]].dims) == (n, k // BLOCK, BLOCK)
        assert inits[row["S"]].data_type == TensorProto.FLOAT and tuple(inits[row["S"]].dims) == (n * (k // BLOCK),)
        assert ext_signature(inits[row["B"]]) == (row["B"] + ".bin", 0, n * k)
        assert ext_signature(inits[row["S"]]) == (row["S"] + ".bin", 0, n * (k // BLOCK) * 4)
        row.update(A=node.input[0], Y=node.output[0], old_B_shape=[n, k // BLOCK, BLOCK],
                   old_S_shape=[n * (k // BLOCK)], P_shape=[n // TILE, k // BLOCK, BLOCK, TILE],
                   PS_shape=[n // TILE, k // BLOCK, TILE], P_bytes=n * k, PS_bytes=n * (k // BLOCK) * 4)
        expected_B = [OLD + "embed_gather_B", HEAD] if row["node"] == HEAD else [row["node"]]
        expected_S = [OLD + "scale_reshape", HEAD] if row["node"] == HEAD else [row["node"]]
        assert consumers(g, row["B"]) == expected_B and consumers(g, row["S"]) == expected_S
    ids = "/transformer/Reshape_output_0"
    expected_chain = [
        ("embed_gather_B", "Gather", [OLD + "B", ids], [OLD + "embed_codes"], {"axis": 0}),
        ("embed_cast", "Cast", [OLD + "embed_codes"], [OLD + "embed_float"], {"to": TensorProto.FLOAT}),
        ("embed_center", "Sub", [OLD + "embed_float", OLD + "zero_point"], [OLD + "embed_centered"], {}),
        ("scale_reshape", "Reshape", [OLD + "S", OLD + "scale_shape"], [OLD + "scale_rows"], {}),
        ("embed_gather_S", "Gather", [OLD + "scale_rows", ids], [OLD + "embed_scales"], {"axis": 0}),
        ("embed_multiply", "Mul", [OLD + "embed_centered", OLD + "embed_scales"], [OLD + "embed_blocks"], {}),
        ("embed_reshape", "Reshape", [OLD + "embed_blocks", OLD + "embed_shape"], [EMBED_OUT], {}),
    ]
    for suffix, op, inputs, outputs, attributes in expected_chain:
        node = nodes[OLD + suffix]
        assert node.domain == "" and node.op_type == op and list(node.input) == inputs and list(node.output) == outputs and attrs(node) == attributes
    for name in OLD_EMBED_VALUES:
        assert set(consumers(g, name)) <= OLD_EMBED and consumers(g, name), name
    for name in OLD_OWNED_INITS:
        assert set(consumers(g, name)) <= OLD_EMBED and consumers(g, name), name
    for suffix, expected in (("scale_shape", [NPAD, EMBED_K // BLOCK, 1]), ("zero_point", 128),
                             ("embed_shape", [1, 1, EMBED_K]), ("slice_starts", [0]),
                             ("slice_ends", [VOCAB]), ("slice_axes", [1]), ("slice_steps", [1])):
        assert np.array_equal(numpy_helper.to_array(inits[OLD + suffix]), expected), suffix
    assert nodes[OLD + "head_slice"].op_type == "Slice"
    assert list(nodes[OLD + "head_slice"].input) == [OLD + x for x in ("Ypad", "slice_starts", "slice_ends", "slice_axes", "slice_steps")]
    assert sum(r["P_bytes"] + r["PS_bytes"] for r in rows) == PACKED_BYTES
    return rows


def rewrite(model, rows):
    g = model.graph
    changed_nodes = OLD_EMBED | {r["node"] for r in rows}
    owned_weights = {r[x] for r in rows for x in ("B", "S")}
    original_nodes = [n.SerializeToString() for n in g.node if n.name not in changed_nodes]
    original_inits = {t.name: t.SerializeToString() for t in g.initializer if t.name not in owned_weights | OLD_OWNED_INITS}
    original_info = {v.name: v.SerializeToString() for v in g.value_info if v.name not in OLD_EMBED_VALUES | owned_weights}
    original_io = [[v.SerializeToString() for v in group] for group in (g.input, g.output)]
    header = onnx.ModelProto(); header.CopyFrom(model); header.ClearField("graph")
    graph_header = onnx.GraphProto(); graph_header.CopyFrom(g)
    for field in ("node", "initializer", "value_info"):
        graph_header.ClearField(field)
    by_name = {r["node"]: r for r in rows}
    p_name, ps_name = rows[-1]["P"], rows[-1]["PS"]
    constants = [numpy_helper.from_array(np.array([], dtype=np.int64), NEW + "scalar_shape"),
                 numpy_helper.from_array(np.array(TILE, dtype=np.int64), NEW + "tile_size"),
                 numpy_helper.from_array(np.array([EMBED_K // BLOCK, 1], dtype=np.int64), NEW + "scale_shape")]
    specs = [
        ("id_scalar", "Reshape", ["input_ids", NEW + "scalar_shape"], TensorProto.INT64, [], {}),
        ("tile", "Div", [NEW + "id_scalar", NEW + "tile_size"], TensorProto.INT64, [], {}),
        ("lane", "Mod", [NEW + "id_scalar", NEW + "tile_size"], TensorProto.INT64, [], {"fmod": 0}),
        ("P_tile", "Gather", [p_name, NEW + "tile"], TensorProto.INT8, [24, 32, 16], {"axis": 0}),
        ("P_row", "Gather", [NEW + "P_tile", NEW + "lane"], TensorProto.INT8, [24, 32], {"axis": 2}),
        ("PS_tile", "Gather", [ps_name, NEW + "tile"], TensorProto.FLOAT, [24, 16], {"axis": 0}),
        ("PS_row", "Gather", [NEW + "PS_tile", NEW + "lane"], TensorProto.FLOAT, [24], {"axis": 1}),
        ("float", "Cast", [NEW + "P_row"], TensorProto.FLOAT, [24, 32], {"to": TensorProto.FLOAT}),
        ("scale_row", "Reshape", [NEW + "PS_row", NEW + "scale_shape"], TensorProto.FLOAT, [24, 1], {}),
        ("blocks", "Mul", [NEW + "float", NEW + "scale_row"], TensorProto.FLOAT, [24, 32], {}),
    ]
    embed_nodes = [helper.make_node(op, inputs, [NEW + suffix], name=NEW + suffix + "_node", **attributes)
                   for suffix, op, inputs, dtype, dims, attributes in specs]
    embed_nodes.append(helper.make_node("Reshape", [NEW + "blocks", OLD + "embed_shape"], [EMBED_OUT], name=NEW + "reshape_node"))
    nodes = []
    for old_node in g.node:
        if old_node.name in OLD_EMBED:
            if old_node.name == OLD + "embed_gather_B":
                nodes.extend(embed_nodes)
            continue
        if old_node.name in by_name:
            row = by_name[old_node.name]
            node = onnx.NodeProto(); node.CopyFrom(old_node)
            node.op_type = "Q8MatMulM1Packed"
            node.input[1], node.input[2] = row["P"], row["PS"]
            for a in node.attribute:
                a.name = {"K": "k", "N": "n"}[a.name]
            nodes.append(node)
        else:
            nodes.append(old_node)
    del g.node[:]; g.node.extend(nodes)
    by_init = {r[x]: (r, x) for r in rows for x in ("B", "S")}
    inits = []
    for tensor in g.initializer:
        if tensor.name in OLD_OWNED_INITS:
            continue
        if tensor.name in by_init:
            row, kind = by_init[tensor.name]
            target = "P" if kind == "B" else "PS"
            inits.append(external_tensor(row[target], TensorProto.INT8 if target == "P" else TensorProto.FLOAT, row[target + "_shape"]))
        else:
            inits.append(tensor)
    del g.initializer[:]; g.initializer.extend(inits + constants)
    infos = [v for v in g.value_info if v.name not in OLD_EMBED_VALUES | owned_weights]
    del g.value_info[:]; g.value_info.extend(infos)
    for suffix, op, inputs, dtype, dims, attributes in specs:
        g.value_info.append(helper.make_tensor_value_info(NEW + suffix, dtype, dims))
    assert len(g.node) == 1660 and len(g.initializer) == 135
    assert Counter(n.op_type for n in g.node)["Q8MatMulM1Packed"] == 25
    assert not any(n.op_type == "Q8MatMulM1" for n in g.node)
    assert [n.SerializeToString() for n in g.node if n.name not in {r["node"] for r in rows} and not n.name.startswith(NEW)] == original_nodes
    after_inits = {t.name: t.SerializeToString() for t in g.initializer}
    assert not (owned_weights | OLD_OWNED_INITS) & set(after_inits)
    assert all(after_inits[name] == raw for name, raw in original_inits.items())
    assert all(not consumers(g, name) for name in owned_weights | OLD_EMBED_VALUES | OLD_OWNED_INITS)
    after_info = {v.name: v.SerializeToString() for v in g.value_info}
    assert all(after_info[name] == raw for name, raw in original_info.items())
    assert [[v.SerializeToString() for v in group] for group in (g.input, g.output)] == original_io
    new_header = onnx.ModelProto(); new_header.CopyFrom(model); new_header.ClearField("graph")
    assert new_header.SerializeToString() == header.SerializeToString()
    new_graph_header = onnx.GraphProto(); new_graph_header.CopyFrom(g)
    for field in ("node", "initializer", "value_info"):
        new_graph_header.ClearField(field)
    assert new_graph_header.SerializeToString() == graph_header.SerializeToString()
    assert consumers(g, p_name) == [NEW + "P_tile_node", HEAD]
    assert consumers(g, ps_name) == [NEW + "PS_tile_node", HEAD]
    for row in rows[:-1]:
        assert consumers(g, row["P"]) == [row["node"]] and consumers(g, row["PS"]) == [row["node"]]
    assert sum(payload(t) for t in g.initializer) == 94602744
    gather = onnx.defs.get_schema("Gather", 13, "")
    assert any("tensor(int8)" in c.allowed_type_strs for c in gather.type_constraints)
    return {"nodes": len(g.node), "initializer_count": len(g.initializer),
            "initializer_payload_bytes": sum(payload(t) for t in g.initializer),
            "op_counts": dict(Counter(n.op_type for n in g.node)),
            "unchanged_nodes_bytewise": len(original_nodes), "unchanged_initializer_metadata_bytewise": len(original_inits),
            "unchanged_value_info_bytewise": len(original_info), "all_IO_model_header_graph_header_bytewise": True,
            "exact_owned_embedding_nodes_removed": sorted(OLD_EMBED),
            "exact_owned_constants_removed": sorted(OLD_OWNED_INITS),
            "old_B_S_initializer_names_removed": sorted(owned_weights), "all_25_B_S_pairs_replaced": True,
            "head_reshape_Slice_logits_reshape_preserved_bytewise": True,
            "native_additional_weight_owner_payload_bytes_by_new_contract": 0,
            "physical_P_PS_payload_bytes": PACKED_BYTES,
            "new_full_vocabulary_embedding_Transpose_dequant_table_bytes": 0,
            "embedding_local_tile_output_bytes": 24 * 32 * 16 + 24 * 16 * 4,
            "weight_tie": {"kind": "same_physical_P_PS_pair", "canonical_initializers": [p_name, ps_name],
                           "consumers": {p_name: consumers(g, p_name), ps_name: consumers(g, ps_name)},
                           "consumer_count_each": 2, "input_id_domain": [0, VOCAB - 1],
                           "dynamic_input_id_scalar_Div16_Mod16": True,
                           "embedding_result_shape": [1, 1, EMBED_K], "head_pad": NPAD, "Slice_end_exclusive": VOCAB}}


def check_pair(source_dir, target_dir, row):
    """Full persisted P/PS versus each independent row/tile equation, no full deq."""
    n, kb = row["n"], row["k"] // BLOCK
    b = np.memmap(source_dir / (row["B"] + ".bin"), mode="r", dtype=np.uint8, shape=(n, kb, BLOCK))
    s = np.memmap(source_dir / (row["S"] + ".bin"), mode="r", dtype="<f4", shape=(n * kb,))
    p = np.memmap(target_dir / (row["P"] + ".bin"), mode="r", dtype=np.int8, shape=tuple(row["P_shape"]))
    ps = np.memmap(target_dir / (row["PS"] + ".bin"), mode="r", dtype="<f4", shape=tuple(row["PS_shape"]))
    assert np.isfinite(s).all() and np.all(s > 0), row["node"]
    for tile in range(n // TILE):
        old_codes = (b[tile * TILE:(tile + 1) * TILE].astype(np.int16) - 128).astype(np.int8)
        assert np.array_equal(p[tile].transpose(2, 0, 1), old_codes), (row["node"], "codes", tile)
        assert np.array_equal(ps[tile].T.reshape(-1).view(np.uint32),
                              s[tile * TILE * kb:(tile + 1) * TILE * kb].view(np.uint32)), (row["node"], "scale_bits", tile)
    checked = {"all_code_bytes_equation_checked": row["P_bytes"], "all_scale_bytes_bitwise_checked": row["PS_bytes"],
               "scales_finite_positive": True, "persisted_file_layout_checked": True}
    if row["node"] == HEAD:
        assert np.all(b[VOCAB:] == 128)
        tokens = [0, 1, 15, 16, 17, 31, 32, 50240, 50255, 50256] + list(range(VOCAB, NPAD))
        for token in tokens:
            old = ((b[token].astype(np.float32) - np.float32(128)) * s.reshape(n, kb)[token, :, None]).reshape(1, 1, EMBED_K)
            new = packed_embedding(p, ps, token)
            assert np.array_equal(old.view(np.uint32), new.view(np.uint32)), ("embedding_bits", token)
            if token >= VOCAB:
                assert np.all(new == 0), ("padding", token)
        checked.update(embedding_checked_tokens=tokens, embedding_FP32_bits_equal=True, padding_rows=15,
                       all_padding_P_zero_and_S_finite_positive=True)
    del b, s, p, ps
    gc.collect()
    return checked


def convert(source, output_dir):
    if output_dir.exists():
        raise FileExistsError(f"fresh output directory required: {output_dir}")
    assert sys.byteorder == "little"
    preflight = preflight_files(source)  # All weights present before parse.
    start = time.monotonic()
    model = onnx.load(source, load_external_data=False)
    rows = inspect_source(model)
    equations = check_equations()
    report = rewrite(model, rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    progress = {"status": "metadata_rewritten_weights_pending", "source_preflight": preflight,
                "completed_pairs": [], "next_pair": 0}
    write_json(output_dir / "progress.json", progress)
    try:
        for row in rows:
            b = np.memmap(source.parent / (row["B"] + ".bin"), mode="r", dtype=np.uint8, shape=tuple(row["old_B_shape"]))
            s = np.memmap(source.parent / (row["S"] + ".bin"), mode="r", dtype="<f4", shape=tuple(row["old_S_shape"]))
            assert np.isfinite(s).all() and np.all(s > 0), row["node"]
            p, ps = pack_codes(b), pack_scales(s, row["n"], row["k"] // BLOCK)
            for name, array in ((row["P"], p), (row["PS"], ps)):
                with (output_dir / (name + ".bin")).open("xb") as stream:
                    array.tofile(stream)
                assert (output_dir / (name + ".bin")).stat().st_size == array.nbytes
            del b, s, p, ps
            gc.collect()
            row["byte_checks"] = check_pair(source.parent, output_dir, row)
            progress["completed_pairs"].append(row["node"])
            progress["next_pair"] = row["index"] + 1
            write_json(output_dir / "progress.json", progress)
            print(f"PACKED_CHECKED {row['index']:02d} {row['node']} P={row['P_shape']} PS={row['PS_shape']} bytes={row['P_bytes'] + row['PS_bytes']}", flush=True)
        copied = []
        packed_names = {row[x] for row in rows for x in ("P", "PS")}
        for tensor in model.graph.initializer:
            if tensor.name in packed_names or not uses_external_data(tensor):
                continue
            location, offset, length = ext_signature(tensor)
            original, target = source.parent / location, output_dir / location
            shutil.copyfile(original, target)
            assert target.stat().st_size == length and target.stat().st_nlink == 1
            assert filecmp.cmp(original, target, shallow=False), str(target)
            copied.append({"file": location, "bytes": length, "source_destination_bytes_equal": True})
        target = output_dir / "native.onnx"
        onnx.save_model(model, target)
        preflight_files(target, packed=True)
        onnx.checker.check_model(str(target))
        saved = onnx.load(target, load_external_data=False)
        assert saved.SerializeToString() == model.SerializeToString(), "saved_metadata_bytes"
        report.update(file=target.name, file_bytes=target.stat().st_size, onnx_checker_passed=True,
                      saved_metadata_bytewise_checked=True)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        manifest = {"status": "converted_checked_runtime_pending", "source": str(source),
                    "source_identity": "existing fixed tied-Q8; original Xenova/distilgpt2@a41c10485c18a64b6606729b6a082330cbd8f49e",
                    "source_preflight": preflight, "source_modified": False,
                    "conversion": "offline signed centering and exact scale-byte reordering of existing Q8 only",
                    "quantizer_calls": 0, "native_or_model_inference_runs": 0, "downloads": 0, "new_dependencies": 0,
                    "op": {"domain": "org.blucca", "name": "Q8MatMulM1Packed", "version": 1,
                           "attrs": ["k", "n"], "input_types": ["FLOAT", "INT8", "FLOAT"], "output_type": "FLOAT", "kernel_API": 31},
                    "numpy": np.__version__, "onnx": onnx.__version__,
                    "equation_check": equations, "all_25_weights": rows, "graph": report,
                    "packed_initializer_bytes": PACKED_BYTES,
                    "source_old_B_S_weight_bytes_actually_read": PACKED_BYTES,
                    "copied_unchanged_files": copied, "copied_unchanged_external_bytes": sum(x["bytes"] for x in copied),
                    "source_initializer_payload_bytes": 94602748, "new_initializer_payload_bytes": 94602744,
                    "initializer_payload_delta_bytes": -4,
                    "outputs": OUTPUTS, "runtime_gate_status": "pending_root_native_admission",
                    "resource_receipt": {"wall_seconds": time.monotonic() - start,
                                         "user_system_cpu_seconds": usage.ru_utime + usage.ru_stime,
                                         "peak_RSS_KiB": usage.ru_maxrss,
                                         "cpu_affinity": sorted(os.sched_getaffinity(0)),
                                         "address_space_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
                                         "model_and_initializer_files_bytes": target.stat().st_size + sum(fixed_source_files().values()),
                                         "disk_accounting_scope": "native.onnx plus all 101 required initializer files; receipts excluded"}}
        write_json(output_dir / "manifest.json", manifest)
        progress.update(status="converted_checked_runtime_pending", next_pair=None,
                        manifest="manifest.json", model="native.onnx")
        write_json(output_dir / "progress.json", progress)
        print(f"CONVERTED_CHECKED {target} nodes={len(saved.graph.node)} initializers={len(saved.graph.initializer)} packed_bytes={PACKED_BYTES} peak_RSS_KiB={usage.ru_maxrss}", flush=True)
        return manifest
    except BaseException as exc:
        progress.update(status="FAILED", failure_type=type(exc).__name__, failure=str(exc))
        write_json(output_dir / "progress.json", progress)
        write_json(output_dir / "FAILURE.json", progress)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    convert(args.source.resolve(), args.output_dir.resolve())
