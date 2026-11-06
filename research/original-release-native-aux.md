# Original release: real dictionary/order and native packed sections

The frozen family is old-v22, non-PGO, untraced, with the existing static15
weight loader and UPX-packed decoder. This single auxiliary delivery exercises
the original generic `-c`/`-d` paths and the unchanged native `self_extract.h`.

## Measured terminal

The original dictionary and order restore byte-for-byte. Both native compressor and archive section entry points pass; embedded weights and decoder bytes match their original files.

| Component | Actual bytes |
| --- | ---: |
| binary | 181956 |
| compressed_dictionary | 100098 |
| compressed_order | 201111 |
| weights | 2904325 |

The common two-file charge is **6,573,938 bytes**. The 24-byte marker fixture has S=6,573,962 bytes. This measurement covers the native envelope and auxiliary roundtrips.

The complete job consumed **502.463058 CPU seconds** from one shared
1,800-second allowance, with one CPU thread and a 24 GiB virtual-address cap.
The original generic codec reserves a large sparse address-space map.
[Terminal receipt](results/original-release-native-aux.json) lists per-command
CPU and cumulative child peak RSS, exact file identities and the actual native gates.

## Exact package contract

```text
cmix     = B + D + O + W + 16-byte HeaderInfo
archive9 = B + D + W + 37-byte prepared-stream header + P + 16-byte HeaderInfo
S        = P + 2B + 2D + O + 2W + 69
```

The archive HeaderInfo preserves the O-size field while its body omits O,
matching the original native extractor. Native compressor extraction decodes
the actual dictionary/order and copies the complete W and B. Native archive
extraction decodes the dictionary, copies W, and restores the 37-byte header
plus the explicit `SECTION EXTRACTION ONLY` marker. This fixture's prepared
length is 10,000; its marker is deliberately outside the range-coded payload
stage. Complete prepared encoding and 1 GB enwik9 restoration have their own
terminal checks and their own full-S measurements.

## Reproduction

Use the original old-v22 checkout/objects, original prepared vocabulary and
frozen B/W/input identities from the receipt. Place the original packed B,
static15 W and their `manifest.json` under `$JOB/assets/`. The manifest records
`files` with byte sizes/SHA256 and `original_input_identities` for dictionary
and order. An already produced matching `comp_dict`/`comp_order` may be reused.
The shared allowance includes compilation, both auxiliary encoders and native
section extraction. Work directories are fresh and every stage writes a
persistent state/log/exit record.

```sh
python3 -B research/original-release-native-aux.py \
  --root /YOUR/temp/hutter --job /YOUR/temp/native-release-aux
```

[Source](original-release-native-aux.py) reuses the existing
[archive assembler](archive-package.py); place that source at
`$JOB/archive-package.py` before execution. Stop after this auxiliary terminal;
the paired full-stream plan consumes these same D/O artifacts once.
