#!/usr/bin/env python3
"""Own-prompt first Run of the fixed shared-Q8 DistilGPT2 CPU graph.

Linux x64 AVX2/FMA, ORT 1.31.1, one thread. Every Session.run returns
all 50,257 logits and all twelve full KV outputs. No download or build.
Greedy/empty-past/prefix checks follow the study's language_probe.py.
Study/demo: Blucca / Codex, GPL-3.0; native operator: existing MIT source.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SOURCE = "Xenova/distilgpt2@a41c10485c18a64b6606729b6a082330cbd8f49e"
TOKENIZER_URL = "https://huggingface.co/Xenova/distilgpt2/resolve/a41c10485c18a64b6606729b6a082330cbd8f49e/tokenizer.json"
BOS, VOCAB, CONTEXT = 50256, 50257, 1024
OUTPUTS = ["logits"] + [f"present.{i}.{kind}" for i in range(6) for kind in ("key", "value")]
INPUTS = ["input_ids"] + [f"past_key_values.{i}.{kind}" for i in range(6) for kind in ("key", "value")] + ["attention_mask"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_budget(prompt_tokens, max_new_tokens):
    require(1 <= max_new_tokens <= 64, "max_new_tokens must be 1..64")
    require(prompt_tokens >= 0 and 1 + prompt_tokens + max_new_tokens <= CONTEXT,
            "BOS + prompt tokens + max_new_tokens must total <= 1024")


def step(session, past, token, np):
    length = past[INPUTS[1]].shape[2]
    feed = {"input_ids": np.array([[token]], dtype=np.int64), **past,
            "attention_mask": np.ones((1, length + 1), dtype=np.int64)}
    began = time.perf_counter()
    values = session.run(OUTPUTS, feed)
    whole_seconds = time.perf_counter() - began
    require(len(values) == len(OUTPUTS), "Expected all 13 outputs")
    actual = dict(zip(OUTPUTS, values))
    for name, value in actual.items():
        shape = (1, 1, VOCAB) if name == "logits" else (1, 12, length + 1, 64)
        require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                f"Invalid full output: {name}")
        if name != "logits":
            old = past[name.replace("present.", "past_key_values.", 1)]
            require(np.array_equal(value[:, :, :length].view(np.uint32), old.view(np.uint32)),
                    f"KV prefix bits changed: {name}")
    new_past = {n.replace("present.", "past_key_values.", 1): actual[n] for n in OUTPUTS[1:]}
    return actual, new_past, whole_seconds


def greedy(session, prompt_ids, max_new_tokens, np):
    check_budget(len(prompt_ids), max_new_tokens)
    require(all(type(t) is int and 0 <= t < VOCAB for t in prompt_ids), "Invalid prompt token")
    past = {n: np.empty((1, 12, 0, 64), dtype=np.float32) for n in INPUTS[1:-1]}
    whole_seconds, calls = 0., 0
    for token in [BOS, *prompt_ids]:
        actual, past, seconds = step(session, past, token, np)
        whole_seconds += seconds
        calls += 1
    generated = []
    for j in range(max_new_tokens):
        token = int(np.argmax(actual["logits"][0, 0]))
        generated.append(token)
        if token == BOS or j + 1 == max_new_tokens:  # GPT2 BOS is also EOS.
            break
        actual, past, seconds = step(session, past, token, np)
        whole_seconds += seconds
        calls += 1
    return {"generated_token_ids": generated, "generated_tokens": len(generated),
            "stopped_at_eos": generated[-1] == BOS, "session_run_calls": calls,
            "cache_tokens_at_end": past[INPUTS[1]].shape[2],
            "session_run_whole_seconds": whole_seconds}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Extracted fixed-model bundle (default: this script's directory)")
    parser.add_argument("--prompt", help="Your own prompt, encoded with the bundled pinned tokenizer")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    import numpy as np
    if args.prompt is None:
        parser.error("--prompt is required")
    require(platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64"),
            "This bundle requires Linux x64")
    raw = Path("/proc/cpuinfo").read_text()
    flags = next(x.split(":", 1)[1].split() for x in raw.splitlines() if x.startswith("flags"))
    require("avx2" in flags and "fma" in flags, "This bundle requires AVX2 and FMA")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import onnxruntime as ort
    import tokenizers
    from tokenizers import Tokenizer
    require(ort.__version__ == "1.31.1", "This bundle requires ORT 1.31.1 (API 31)")
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["source_ref"] == SOURCE and manifest["vocab"] == VOCAB
            and manifest["padded_vocab"] == 50272 and manifest["K"] == 768
            and manifest["block_size"] == 32 and manifest["batch"] == manifest["decode_sequence"] == 1
            and manifest["kernel_API"] == 31, "Use the fixed shared-Q8 study bundle")
    receipt = json.loads((root / "tokenizer-source.json").read_text())
    require(receipt["source"] == TOKENIZER_URL and receipt["bytes"] == 2107653
            and (root / "tokenizer.json").stat().st_size == receipt["bytes"], "Use the pinned companion tokenizer")
    began = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    require(tokenizer.get_vocab_size(with_added_tokens=True) == VOCAB, "Expected 50,257-token vocabulary")
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    check_budget(len(prompt_ids), args.max_new_tokens)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    options.register_custom_ops_library(str(root / "libq8_m1.so"))
    init_began = time.perf_counter()
    session = ort.InferenceSession(str(root / "models" / "native.onnx"), options,
                                  providers=["CPUExecutionProvider"])
    init_seconds = time.perf_counter() - init_began
    require([x.name for x in session.get_inputs()] == INPUTS
            and [x.name for x in session.get_outputs()] == OUTPUTS, "Expected complete 14-input/13-output graph")
    generation_began = time.perf_counter()
    result = greedy(session, prompt_ids, args.max_new_tokens, np)
    generation_seconds = time.perf_counter() - generation_began
    generated = result["generated_token_ids"]
    result.update(status="complete", source_ref=SOURCE, tokenizer_source=TOKENIZER_URL,
                  text=tokenizer.decode(prompt_ids + generated, skip_special_tokens=True),
                  continuation=tokenizer.decode(generated, skip_special_tokens=True),
                  prompt_tokens=len(prompt_ids), bos_condition_tokens=1,
                  total_context_tokens=1 + len(prompt_ids) + len(generated),
                  requested_max_new_tokens=args.max_new_tokens,
                  all_outputs_finite_correct_shape=True, prefix_bits_preserved=True,
                  session_initialization_seconds=init_seconds,
                  generation_wall_seconds=generation_seconds,
                  elapsed_whole_seconds=time.perf_counter() - began,
                  timing_scope="One cold own-prompt Run; whole Session.run returns logits + 12 full KV. Generation wall includes feed construction and checks. No benchmark warmups.",
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  completed_at=datetime.now(timezone.utc).isoformat(),
                  host={"cpu": next(x.split(":", 1)[1].strip() for x in raw.splitlines() if x.startswith("model name")),
                        "features": {f: f in flags for f in ("avx2", "fma", "avx512f", "avx512_vnni", "avx_vnni")},
                        "platform": platform.platform(), "python": sys.version,
                        "numpy": np.__version__, "ort": ort.__version__, "tokenizers": tokenizers.__version__,
                        "threads": 1, "allowed_cpu": sorted(os.sched_getaffinity(0))})
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
