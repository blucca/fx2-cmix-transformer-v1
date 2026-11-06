# Native length support: paying for the missing long pieces

The eight-block student has recovered on sampled held pieces while the
maximum-article audit has regressed. The full-stream census lets us separate
length reweighting from the unobserved long-piece contribution before the
original same-weight codec measurement.

**Decision-useful result:** at a supported native-to-payload coefficient of
0.05, e12's historical record scenario has **69,500 B** left for payload
growth in the unobserved length cell. One natural article contains **81.38%**
of that cell's targets. The original 32-epoch method, candidate selector and
single mixed run remain fixed; actual payload and complete archive size
decide deployment.

## Frozen census, existing native outputs

Reuse the seven length cells from the original causal sampling contract,
the same **473 observed articles / 1,082,505 targets**, and paired W0/e1/e2
causal and W0/e8/e12 depth native losses. Length is complete piece bytes;
each piece has one fewer NN target. These cells precede the trained results.

| Piece bytes | Full NN targets | Observed pieces / targets | Causal e2−W0 | Depth e12−W0 | Depth e12−e8 |
|---|---:|---:|---:|---:|---:|
| 1–2,048 | 120,236,973 | 340 / 245,354 | −.00066518 | +.12878358 | −.01666194 |
| 2,049–4,096 | 102,834,463 | 65 / 178,382 | −.00089454 | +.16962979 | −.02090729 |
| 4,097–8,192 | 114,572,295 | 40 / 226,176 | −.00123262 | +.20022826 | −.02629478 |
| 8,193–16,384 | 113,544,035 | 21 / 240,561 | −.00175756 | +.20196583 | −.02711675 |
| 16,385–32,768 | 77,843,076 | 4 / 84,385 | −.00144885 | +.18370677 | −.02329996 |
| 32,769–65,536 | 18,372,088 | 3 / 107,647 | −.00225255 | +.20829314 | −.03121832 |
| 65,537–131,072 | 38,812,401 | 0 / 0 | explicit input | explicit input | explicit input |

Deltas are native nats/target. Six observed cells cover **547,402,930 full
targets / 93.3792%**. Their census-calibrated conditional depth e12 delta is
**+.177068915799**, compared with +.178893053815 on the observed composition.
All six cells recover from e8 to e12.

Within each observed cell, using its observed target-weighted delta for the
population is the sensitivity assumption. The 16–32 KiB and 32–64 KiB cells
retain four and three sampled articles. The separate 131,071-target audit
retains its local, report-only role.

## An empty cell dominated by one article

The original largest natural article spans `[554867326,586453679)`:
31,586,353 bytes, **241 native pieces / 31,586,112 targets**. Its last piece
has 129,073 bytes, so all 241 pieces belong to the empty >64 KiB cell.
The remaining **68 pieces / 7,226,289 targets** have a separate population.

The article's seed414 role is held and its realized piece sample is empty.
Under the original randomized design—article held probability .25, independent
piece inclusion .0075—the conditional chance of this article contributing
zero observed pieces is `(1−.0075)^241 = .162949946`. Before the article-role
draw, the chance of zero held observations is
`.75 + .25×(1−.0075)^241 = .790737486`.
These design probabilities explain the leverage of article clustering;
the existing seed414 result is the actual observation.

## A payload budget with explicit units

For the supported cells, form the census-calibrated native penalty:

    A = sum(population_targets_j × observed_native_delta_j) / (8 ln 2)
      = 17,479,700.927613 native-byte equivalents

Define `κA` as supported-cell range-payload growth divided by A. Define
`qU` as the >64 KiB cell's payload growth in **bits per NN target**.
Payload allocation uses additive closed-boundary differences on complete
piece intervals, including their fallback bytes and range effects. The
existing experiment retains total-payload measurement; these cell variables
provide the sensitivity coordinates.

With e12's actual **1,895,710 B** double-W/B saving:

    named net saving = 1,895,710 − 17,479,700.927613 κA − 4,851,550.125 qU

The historical 952,225 B record gap leaves a 943,485 B payload allowance.
At `κA=.05`, this requires `qU≤.014325308784 bits/target`, equivalent to
**69,499.953619 B** of empty-cell payload growth. Split that contribution as:

    empty-cell payload growth = 3,948,264 q_giant + 903,286.125 q_other

For a uniform native-to-payload coefficient and an assumed missing-cell
native delta `δU`, the same historical threshold is:

    κ ≤ 943,485 / (17,479,700.927613 + 6,999,307.305962 δU)

| Missing-cell assumption | δU, native nats/target | Historical κ threshold |
|---|---:|---:|
| Zero native delta | 0 | .05397604 |
| Same as supported conditional mean | .17706892 | .05040237 |
| Same as the local maximum-article audit | .22999218 | .04942432 |

Each row keeps its stated extrapolation. Same-W G/M/u16/actual bytes, latest
record size and complete restoration supply the final admission evidence.

Length reweighting also compresses causal e2's small advantage over e1:
on supported population targets it is **2,134.05 native-byte equivalents**.
With equal missing-cell payload effects, `κA=1` and the actual extra 84 B
of double-W cost, the named advantage is **2,050.05 B**, alongside the earlier
9,978.76 B same-observed-composition scenario. The frozen selector retains e2.

## Reproduce

[Frozen census and calculated result](results/native-length-support.json)
reuse the existing paired native outputs. The minimal replay checks all
473 identities, seven cell counts, the giant split and the payload-budget
identity, using the Python standard library:

    python3 -B research/check-native-length-support.py

This delivery uses existing data. Original capture, native evaluations,
training budgets and sole mixed runs retain their scope.
