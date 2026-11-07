# Native half-permutation observer: failed

Twelve existing 205-half rows and twelve actual target bytes produce 96 teacher/oracle target-path pairs, or 192 actual observations. This test links the original `ByteModel::Predict/Perceive/ByteUpdate`, `MixerInput::SetInput` and `Sigmoid::Logit` source files, copied byte-for-byte. The original `HalfToFloat/HalvesToFloats` functions are extracted verbatim from `src/predictor.cpp`: F16C decodes indices 0..199, scalar indices 200..204 preserve the half-value subnormal rule, followed by the original 1e-6f guard. All 4920 decoded/guarded values match the saved-row decoder contract.

Compiler: Debian Clang 17.0.6 (22+b2). Flags are the makefile default fast/O3 contract, including `-ffp-model=fast -O3 -march=x86-64-v3 -mtune=generic -mrecip=none`. Compilation and linking are separate; complete frozen flags are in `compiler-flags.json`.

| Native teacher vs oracle | Changes / 96 |
|---|---:|
| Conditional FP32 bits | 12 |
| Actual Stretch FP32 bits | 1 |
| Actual Logit lookup index | 1 |
| normal / 0 / 1 override | 0 |

The maximum native conditional difference is `1.1920928955078125e-7`. Override counts are 93 normal, 3 zero and 0 one. Lookup indices are identified by the unique actual Logit-table output bits.

## Concrete failure

`hard / piece 14 / row 0 / target byte 220`, first bit:

- Teacher probability `0.5272147059440613`, bits `3f06f78b`; oracle `0.527214765548706`, bits `3f06f78c`.
- Actual lookup index `52721 → 52722`; actual Stretch bits `3ddf1f53 → 3ddf3457`; both overrides are normal.
- FP32 `p * 100001` values are `52721.99609375` and `52722.00390625`. Reassociated fast/O3 reductions retain different low bits, crossing a real table-cell boundary.

The serial-FP32 saved-row oracle has 0/96 lookup changes. Its half representation and exact sibling/decoder-cell mass certificates remain valid. The native fast/O3 same-observation certificate fails on this saved-row counterexample.

`observations-192.jsonl` contains all actual observations. `result.json`, `counterexample.json` and `execution-receipt.json` separate successful native execution from failed same-observation acceptance. Scientific execution consumed 1.755729 CPU seconds; source-layout repair consumed 0.944294 CPU seconds; total 2.700023 seconds. Peak RSS was 134232 KiB. One compile strategy; zero model forwards, optimizer updates, codec runs, dependency installs or new watches.

Replay with an existing Clang 17 compiler: `python3 research/native-byte-path/replay.py --out temp/native-observer`. The replay only compiles these source snapshots and observes the existing half buffers.


Half buffers are reused from `research/results/byte-path-permutation-{teacher,oracle}.f16`.
The capsule keeps the linked source/header snapshots and extracted decoder;
the original predictor and unused byte-mixer stay at their source locations.
