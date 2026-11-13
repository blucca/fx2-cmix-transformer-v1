# Janas on a small EPYC CPU-only VM

A stock quick run of [Janas](https://github.com/prabanta-dev/janas) at
[`c9220a467b3376ebb684165435512e3ee71c9f52`](https://github.com/prabanta-dev/janas/tree/c9220a467b3376ebb684165435512e3ee71c9f52), measured on 13 November 2026.

**55 native release tests passed. The fixed reference answer matched.**
The default small-memory benchmark measured **69.6 tokens/s prefill and
5.8 tokens/s decode**. Its automatic quality selected **four-bit down matrices**;
the reference-answer check uses **six-bit planes and exact attention**.

## The machine and the two checks

| | Actual run |
|---|---|
| Machine | Linux VM, **4 guest vCPUs**, AMD EPYC 9254 host CPU string |
| RAM | 7.74 GiB total; **3.8 GiB available** at model load |
| CPU arithmetic | AVX512-VNNI, 512-bit tiles; AVX2/FMA also available; AVX-VNNI absent |
| System | Ubuntu 24.04.5, Linux 6.8.0-142-generic, GCC 13.3.0 |
| Devices | CPU only; GPU and NPU disabled |
| Reference answer | Stock Sardinia prompt, `/experts 0`, temperature 0, max 64, six-bit planes, exact attention; printed-output fingerprint **c08f0f936eeba5cb** |
| Benchmark | Stock prose256 / generated64 / context4096 / 3 rounds; one configuration, four threads, automatic quality/attention |
| Benchmark memory | 1.0 GiB expert cache, 68.5% reported cache ratio, four-bit down matrices selected for memory |
| CPU before benchmark | 98% idle |
| Local HTTP checks | Stock chat/tool-call checks **2/2** |

The native [report](native-report.md) keeps the program's complete numbers and
choices. The reference-answer check and the automatic-quality speed measurement
have their separate settings above.

## Model acquisition and the initial failure

The first standard `janas-get qwen3-4b` reached a direct-DNS failure in this
proxy-required VM: [actual first get output](first-download-failure.txt).
The 55 native tests had already passed, and the outer `janas-try` script ended
with exit code 0 while recording that failed stage.

Python's standard HTTP(S)-proxy-aware client fetched the **two official
preconverted JNS files**, with these catalog fingerprints:

| Official file | Bytes | SHA-256 |
|---|---:|---|
| [Qwen3-4B JNS](https://huggingface.co/prabanta-dev/Qwen3-4B-JNS/resolve/main/qwen3-4b-q4km.jns) | 2,497,753,088 | `1cbd8ebfdf14aee05777f5278e655382f040cdedc21d282e93806fc8cf279fc5` |
| [Qwen3-0.6B draft JNS](https://huggingface.co/prabanta-dev/Qwen3-0.6B-JNS/resolve/main/qwen3-0.6b-q4km.jns) | 396,926,976 | `9174da760919b16b9f3a4ae8738c2d53605412f40bb2c0fe68d60174bf8a4fb5` |

The same stock quick then resumed; [native get output](native-get.log) records
both cached fingerprints checked. The raw report prints commit `unknown` because
it ran from a source archive and describes the model as converted here; the exact
source commit and actual preconverted-download path are recorded on this page.

## Replay and artifacts

On the pinned source, use the normal quick path with the named JNS files in
`models/` on proxy-required hosts:

```sh
JANAS_GPU=0 JANAS_NPU=0 bash tools/janas-try.sh --yes --keep --level quick --dir models
```

Our local-only execution wrapper bounded the full run and retained its artifacts;
Python performed shell cleanup. The native C, kernels, tests, model bytes and
quick settings were stock. Source and model licenses remain with their authors;
this folder contains measurements and logs, with no model weights.

- [Full native build/test log](native-build.log) and [55 executed test names](native-test-runs.txt)
- [Exact native stage results](native-results.txt)
- [Machine-readable result](result.json)
- [Whole native report](native-report.md)

Minimal artifact check:

```sh
python3 -c "from pathlib import Path; p=Path('research/janas-epyc-quick'); s=(p/'native-results.txt').read_text(); assert all(x in s for x in ['answer fingerprint ok (c08f0f936eeba5cb)', 'quick speed measured', 'quick server 2/2']); assert len((p/'native-test-runs.txt').read_text().splitlines()) == 55"
```

Other CPU work: [ONNX Runtime QDQ controls](../ort-qdq-fusion-control.md) and
[the static FP32-activation Q8 M1 operator](../ort-q8-m1-native/).
