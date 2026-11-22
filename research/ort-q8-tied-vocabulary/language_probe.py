#!/usr/bin/env python3
"""Paired fixed-text Q8 loss and independently chained complete decode.

The supplied text is fixed before running any model. Two 1024-target blocks
use a BOS condition and their own KV chains. This is one document probe.
"""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

COUNT, BLOCK, BOS, VOCAB = 2048, 1024, 50256, 50257


def nll(logits, target):
    import numpy as np
    a = np.asarray(logits, dtype=np.float64)
    m = float(a.max())
    return m + math.log(float(np.exp(a-m).sum())) - float(a[target])


def self_test():
    assert abs(nll([0., 0.], 1) - math.log(2)) < 1e-12
    assert abs(nll([1000., 1000.], 0) - math.log(2)) < 1e-12


def setup(args):
    sys.path[:0] = [str(args.base_root), str(args.base_root.parent)]
    return importlib.import_module("native_runner")


def prepare(args, common):
    from tokenizers import Tokenizer
    text = args.text.read_text()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    assert len(ids) >= COUNT and all(type(x) is int and 0 <= x < VOCAB for x in ids)
    out = args.root / "language"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixed-text.txt").write_text(text)
    common.dump(out / "tokens.json", {"token_ids": ids[:COUNT], "scored_tokens": COUNT,
                                     "block_size": BLOCK, "bos_condition": BOS,
                                     "available_text_tokens": len(ids), "add_special_tokens": False,
                                     "source": "NIST AI RMF 1.0 (2023), selected government narrative",
                                     "text_source_file": args.text.name,
                                     "tokenizer_source": "pinned Xenova/distilgpt2 companion tokenizer.json"})


def session(args, common, mode):
    if mode == "old_native":
        return common.session(args.base_root, "native")
    if mode != "original":
        return common.session(args.root, mode)
    ort = common.ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    s = ort.InferenceSession(str(args.base_root / "source.onnx"), opts, providers=["CPUExecutionProvider"])
    assert [x.name for x in s.get_inputs()] == common.INPUTS
    assert [x.name for x in s.get_outputs()] == common.OUTPUTS
    return s


def empty_past(common):
    return {n: common.np.empty((1, 12, 0, 64), dtype=common.np.float32) for n in common.INPUTS[1:-1]}


def step(s, common, past, token):
    np = common.np
    length = next(iter(past.values())).shape[2]
    feed = {"input_ids": np.array([[token]], dtype=np.int64), **past,
            "attention_mask": np.ones((1, length+1), dtype=np.int64)}
    actual = common.output_map(s, feed)
    assert actual["logits"].shape == (1, 1, VOCAB)
    for name, value in actual.items():
        assert value.dtype == np.float32 and np.isfinite(value).all(), name
        if name != "logits":
            assert value.shape == (1, 12, length+1, 64)
            old = past[name.replace("present.", "past_key_values.", 1)]
            assert np.array_equal(value[:, :, :length].view(np.uint32), old.view(np.uint32)), name
    new_past = {n.replace("present.", "past_key_values.", 1): actual[n] for n in common.OUTPUTS[1:]}
    return actual, new_past


def worker(args, common):
    np = common.np
    folder = args.root / "language"
    ids = json.loads((folder / "tokens.json").read_text())["token_ids"]
    assert len(ids) == COUNT
    began = time.perf_counter()
    s = session(args, common, args.mode)
    init_seconds = time.perf_counter()-began
    ref_logits = ref_kv = None
    if args.mode == "dense":
        ref_logits = np.lib.format.open_memmap(folder / "reference-logits.npy", mode="w+", dtype=np.float32, shape=(COUNT, VOCAB))
        ref_kv = np.lib.format.open_memmap(folder / "reference-new-kv.npy", mode="w+", dtype=np.float32, shape=(COUNT, 12, 12, 64))
    elif args.mode == "native":
        ref_logits = np.load(folder / "reference-logits.npy", mmap_mode="r", allow_pickle=False)
        ref_kv = np.load(folder / "reference-new-kv.npy", mmap_mode="r", allow_pickle=False)
        assert ref_logits.shape == (COUNT, VOCAB) and ref_kv.shape == (COUNT, 12, 12, 64)
    max_norm = {n: 0. for n in common.OUTPUTS}
    top1_equal = 0
    losses = []
    for block in range(COUNT // BLOCK):
        past = empty_past(common)
        for j in range(BLOCK):
            row = block*BLOCK+j
            token = BOS if j == 0 else ids[row-1]
            actual, past = step(s, common, past, token)
            logits = actual["logits"][0, 0]
            losses.append(nll(logits, ids[row]))
            new_kv = np.stack([actual[n][0, :, -1, :] for n in common.OUTPUTS[1:]])
            if args.mode == "dense":
                ref_logits[row] = logits
                ref_kv[row] = new_kv
            elif args.mode == "native":
                pairs = [("logits", logits, ref_logits[row])]
                pairs += [(n, new_kv[i], ref_kv[row, i]) for i, n in enumerate(common.OUTPUTS[1:])]
                for name, a, b in pairs:
                    max_norm[name] = max(max_norm[name], common.stats(a, b)["norm_max_error"])
                top1_equal += int(np.argmax(logits) == np.argmax(ref_logits[row]))
            if (row+1) % 128 == 0:
                common.dump(folder / (args.mode+"-progress.json"), {"mode": args.mode, "scored_tokens": row+1})
                print("TOKENS", args.mode, row+1, flush=True)
    if args.mode == "dense":
        ref_logits.flush()
        ref_kv.flush()
    generations = []
    for k in range(4):
        prompt = ids[k*512:k*512+32]
        past = empty_past(common)
        actual, past = step(s, common, past, BOS)
        for token in prompt:
            actual, past = step(s, common, past, token)
        generated = []
        for j in range(64):
            token = int(np.argmax(actual["logits"][0, 0]))
            generated.append(token)
            if j < 63:
                actual, past = step(s, common, past, token)
        generations.append({"prompt_tokens": prompt, "generated_tokens": generated})
    report = {"status": "complete", "mode": args.mode, "host": common.host(), "scored_tokens": COUNT,
              "block_size": BLOCK, "bos_is_condition_only": True, "mean_nll_nats": math.fsum(losses)/COUNT,
              "per_token_nll_nats": losses, "generations": generations, "prefix_bits_preserved": True,
              "all_outputs_finite_correct_shape": True, "initialization_seconds": init_seconds,
              "elapsed_seconds": time.perf_counter()-began, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if args.mode == "native":
        report.update(chained_max_norm_by_output=max_norm, teacher_forced_top1_equal=top1_equal,
                      same_weight_chained_precision_pass=all(x <= common.TOL for x in max_norm.values()) and top1_equal == COUNT)
    common.dump(folder / (args.mode+".json"), report)


def compare(args, common, modes):
    folder = args.root / "language"
    reports = {n: json.loads((folder / (n+".json")).read_text()) for n in modes}
    assert all(r["status"] == "complete" and r["scored_tokens"] == COUNT for r in reports.values())
    native, dense = reports["native"], reports["dense"]
    same_generation = native["generations"] == dense["generations"]
    result = {"host": native["host"], "scored_tokens": COUNT, "block_size": BLOCK,
              "mean_nll_by_mode": {n: r["mean_nll_nats"] for n, r in reports.items()},
              "same_weight_chained_precision_pass": native["same_weight_chained_precision_pass"],
              "same_weight_greedy_4x64_identical": same_generation,
              "numerical_chain_gate_passed": native["same_weight_chained_precision_pass"] and same_generation,
              "probe_scope": "fixed NIST government narrative, 2048 targets, no text-driven quantizer tuning"}
    if "original" in reports and "old_native" in reports:
        old, original = reports["old_native"]["mean_nll_nats"], reports["original"]["mean_nll_nats"]
        new = native["mean_nll_nats"]
        result.update(delta_old24q8_minus_fp32=old-original, delta_tiedq8_minus_old24q8=new-old,
                      delta_tiedq8_minus_fp32=new-original, quantization_loss_gate_passed=new-original <= .03)
        result["complete_language_gate_passed"] = result["numerical_chain_gate_passed"] and result["quantization_loss_gate_passed"]
    common.dump(folder / "comparison.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--action", choices=("self-test", "prepare", "worker", "compare", "all"), default="all")
    p.add_argument("--modes", default="dense,native,original,old_native")
    p.add_argument("--mode", default="native")
    p.add_argument("--text", type=Path)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--max-as-mib", type=int, default=4096)
    args = p.parse_args()
    args.root, args.base_root = args.root.resolve(), args.base_root.resolve()
    self_test()
    if args.action == "self-test":
        print("NLL_SELF_TEST_PASS")
        return
    common = setup(args)
    if args.action == "prepare":
        prepare(args, common)
        return
    modes = args.modes.split(",")
    assert modes[:2] == ["dense", "native"] and set(modes) <= {"dense", "native", "original", "old_native"}
    if args.action == "worker":
        resource.setrlimit(resource.RLIMIT_AS, (args.max_as_mib*2**20, args.max_as_mib*2**20))
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        worker(args, common)
        return
    if args.action == "compare":
        compare(args, common, modes)
        return
    folder = args.root / "language"
    for mode in modes:
        common.dump(folder / "state.json", {"stage": mode, "status": "running"})
        cmd = [sys.executable, str(Path(__file__).resolve()), "--root", str(args.root),
               "--base-root", str(args.base_root), "--action", "worker", "--mode", mode,
               "--max-as-mib", str(args.max_as_mib)]
        with (folder / (mode+".log")).open("w") as log:
            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            common.dump(folder / "state.json", {"stage": mode, "status": "failed", "returncode": completed.returncode})
            raise SystemExit(completed.returncode)
    compare(args, common, modes)
    common.dump(folder / "state.json", {"stage": "complete", "status": "completed"})


if __name__ == "__main__":
    main()
