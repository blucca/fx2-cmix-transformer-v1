# Adjacent INT4 pairs: a full-reference entropy census

**Keep static15.** Across the original full12 model, fixed adjacent pairs improve the ideal data cost by **2,449.56 bytes per weight file**. The fixed 225-symbol model adds **46,648 bytes of literal frequency tables**. With total4096 normalization, the data saving is **1,952.44 bytes**: the literal-table scenario grows by **44,695.56 bytes per file**.

## Scope and method

The recorded input is the original **2,904,325-byte C3** archive from [static15](weights-static15.md): 434 tensors, including **111 signed-int8 matrices / 5,868,864 values**. Every original unary histogram and matrix shape matches the earlier [full census](results/weights-census.json).

For each row, pair columns `(0,1), (2,3), ...`. Across all matrices there are **2,934,336 pairs and 192 tails**. The tails belong to `prior_embedding.weight.q`, shape `192 × 205`, and use the original tensor's unary probability model. The other matrices have even row widths.

- The ideal pair model uses each matrix's empirical 15×15 histogram.
- The concrete probability model applies the original positive-frequency, largest-remainder normalization with alphabet 225 and total4096.
- Pair tables contain 224 uint16 frequencies; the final frequency is implied. Even-width matrices replace the old 14-frequency table: 420 extra literal bytes each. The odd-width matrix retains its original tail table: 448 extra bytes. Total delta: `110 × 420 + 448 = 46,648`.

| Readout | Value |
|---|---:|
| Original unary entropy |22,190,165.1911 bits|
| Pair entropy + original tail model |22,170,568.7301 bits|
| Entropy advantage per pair |0.00667832895 bits|
| Ideal data saving, per W |2,449.55763 bytes|
| Static4096 data saving, per W |1,952.44095 bytes|
| Extra literal tables, per W |46,648 bytes|
| Literal-table scenario loss, per W |44,695.55905 bytes|

The per-pair metric compares one common unary model with the joint model plus retained tails; it includes position-distribution effects. The left/right histograms are in the complete result for a separate marginal-dependence calculation.

**These are static model costs.** Frequency-table metadata compression, range-stream rounding, framing, and binary size belong to an implemented file codec. This census is read-only; the original weight format and inference path remain in use. Even the free-table ideal comparison supplies about **4,899 bytes** across the two weight copies. The fixed-225 codec is stopped at this measured frontier.

## Reproduce

Use the original C3 file produced by the existing [static15 recoder](weights-static15.md). From the repository root, choose a workspace temporary output directory:

```sh
OUT=/path/to/workspace/temp/weights-pair
mkdir -p "$OUT"
g++ -O2 -std=c++17 -I cpp_infer/src -I research   research/weights-pair-census.cpp cpp_infer/src/weights_io.cpp   cpp_infer/src/weights_io_compressed.cpp -o "$OUT/pair-census"
"$OUT/pair-census" /path/to/original-static15.tfwc3 > "$OUT/pair.json"
python3 -B research/weights-pair-census-check.py "$OUT/pair.json"
```

The C++ executable includes independent-pair, correlated-pair, odd-row, tail-only, and frequency checks. The Python check compares all 111 current unary histograms/shapes/costs with the published baseline and reconciles pair/tail/table totals. Complete measured histograms: [weights-pair-census.json](results/weights-pair-census.json).
