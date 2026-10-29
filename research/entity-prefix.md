# Causal entity spelling: capacity and actual coding cost

A persistent title/link-target trie adds a different retrieval channel to the
existing LZP, word and PPM experts. We tested one complete mechanism: exact
PHDA9/WRT spellings, insertion only after a complete title or internal target,
and a frozen **1/16** prior mixed at the final integer probability boundary.
The prior resets each byte and updates causally inside that byte. Vocabulary
overrides retain the original frequency. Entity endings and namespace
normalization have their own contracts; this screen uses raw spelling
continuations only.

## Result

| Frozen observation trace | Baseline bytes | Full saving | Scoring tail | Tail capacity upper bound | Actual tail saving |
|---|---:|---:|---|---:|---:|
| v26 prefix 8 MiB | 1,068,581 | −3,450 B | Last 1 MiB | 15,640.45 B | **−212 B** |
| v22 middle 1 MiB | 208,769 | −250 B | Last 0.5 MiB | 21,532.77 B | **−176 B** |

**Stop this fixed mixture.** Both tails grew. The combined diagnostic saving
is −388 B against the predeclared +2,554 B gate, with each tail required to save
bytes. The middle trace is v22; these two families retain separate identities.
No same-family v26 independent-span pass was obtained.

The capacity metric sums the original observed-bit cost where a causal trie
continuation supports that bit. It is an oracle upper bound. The final range
counter measures the deployable fixed mixture, including its wrong predictions.
Both complete baseline range counts matched their previously round-tripped
files exactly. Five focused parser/causality checks passed; the model,
preprocessing, native coder and weights stayed unchanged. This ends the current
recipe; new selection information and independent same-family savings would
justify a different mechanism.

[Results](results/entity-prefix-20261030.json) preserve the two baseline
identities, per-field costs, trie sizes and actual coded bytes. One bounded
on-host run took 8.5234 CPU seconds including a 2-second inspection reserve;
maximum worker RSS was 137,428 KiB. The prefix trie reserved 208.127 MiB.

## Reproduce the screen

The replay reuses [the existing observation trace and inclusive range counter](replay.cpp).
Supply complete frozen traces and their matching prepared slices; the existing
[research protocol](README.md) explains trace generation. Traces stay on the
compute host. The screen compares every label to the matching prepared byte.

```sh
ROOT=/absolute/path/to/hutter-research
OUT=/absolute/path/to/your/scratch/entity-prefix
mkdir -p "$OUT"
c++ -O3 -std=c++17 "$ROOT/research/entity-prefix-replay.cpp" -o "$OUT/entity-gate"
c++ -O2 -std=c++17 -I "$ROOT/src" \
  "$ROOT/research/entity-prefix-dict.cpp" "$ROOT/src/preprocess/dictionary.cpp" \
  -o "$OUT/dict-bridge"
cd "$OUT"  # The dictionary bridge's fixed temporary file stays here.
printf '  <page>\n    <title>' > separator.raw
printf '</title>' > close.raw
./dict-bridge "$ROOT/dictionary/english.dic" encode separator.raw separator.encoded
./dict-bridge "$ROOT/dictionary/english.dic" encode close.raw close.encoded
./entity-gate --self-test separator.encoded close.encoded

# mode TRACE PREPARED OFFSET LENGTH TAIL_START GOLDEN_BYTES OPEN CLOSE LABELS CAPACITY_GATE
./entity-gate capacity "$TRACE_V26_8M" "$PREPARED" 0 8388608 7340032 1068581 \
  separator.encoded close.encoded labels.u8 1703 > capacity.json
./entity-gate replay "$TRACE_V26_8M" "$PREPARED" 0 8388608 7340032 1068581 \
  separator.encoded close.encoded replay-labels.u8 1703 > replay.json
```

The middle trace used prepared offset 293,232,482, length 1,048,576, tail start
524,288 and golden 208,769 bytes. Start its entity table empty: the slice begins
inside an existing title. PHDA9 transforms original internal `[[target|caption]]`
into prepared `[targetQcaption]`; external links use doubled brackets. Prediction
uses the live parser, and offline labels are for attribution. Caption and
external-link bits remain with the base model. The bounded prototype handles
up to 1,024 encoded bytes per entity; its pool is sized from the input span.
