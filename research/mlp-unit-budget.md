# Static MLP capacity: the fixed near-zero pruning screen stops

The original network activates about 19% of its MLP units per step and already
skips zero down-projection inputs. This observation asks a different question:
does a large, fixed set of units contribute almost nothing across articles?

## One frozen selection

Count each actual `idx768[0:nnz]` unit's hits and sum of squared `q768` values,
separately for the two input halves of each original 1 MiB sample. Use the
unchanged QAT model, article/2^17 resets and captured PPMd priors. Counters take
294,912 B; no activation stream is saved.

For unit `u` in layer `l`, the local diagonal output-energy proxy is:

`E[l,u] = sum_t(q[l,u,t]^2) * sum_o((qw_down[l,o,u] * fold[l,o])^2)`.

`fold` uses the original activation scale times the signed BF16 weight scale,
with the same float multiplication as the inference loader. This proxy omits
cross-unit correlations and downstream propagation; an actual masked-network
NLL measurement was reserved for a passing screen.

Pool the two first halves and select the lowest **3,686 / 9,216 units** once,
breaking ties by layer then unit. Require their aggregate energy share in
**each untouched second half to be at most 0.1%**. Per-layer shares are reported
for diagnosis, with no layer or proportion search.

| Sample | First-half selected share | Second-half selected share |
|---|---:|---:|
| Prefix | 9.2953% | **9.6481%** |
| Article q2 | 12.4958% | **11.6600%** |

**Stop this fixed near-zero pruning recipe.** The second-half shares exceed the
gate by approximately **96× / 117×**. No mask, pruner or new weight pack follows.
The two samples share just one never-observed unit: zero-based layer 2, unit
201. Its 384 up/down weights correspond to 384 B across two raw 4-bit copies,
before metadata, scale and entropy-coding effects. This is a size estimate,
not a newly measured package. The score gap after the existing lossless
component remains about 0.95 MB.

Both complete evaluations preserve the original full 205-way probability
fingerprints, NLL totals/halves, piece bounds, target counts, MLP step counts
and per-layer nonzero totals. Per-unit hit and squared-value invariants are
checked in the probe. Build, evaluation and fixed selection consume **170.835
CPU seconds**, with a peak RSS of about 129 MiB. [Detailed result](results/mlp-unit-budget.json).

## Reproduce

Build the [activation probe](positive-activation.md#reproduce) with
`FX2_ACTIVATION_STATS=1`, `FX2_MLP_UNIT_STATS=1` and `FX2_POSITIVE_U8=0`, keeping
strict floating point for its research main. The evaluation command becomes:

```sh
probe eval TOKENS PRIORS models/6m-q4-fp32.tfwc2 OUTPUT.csv
```

The two compact observations are included as
`results/{prefix-1m,article-q2-1m}-mlp-units.csv.gz`.
Decompress them into a scratch directory as `{sample}.csv`, save the result
JSON's `protocol` object there as `protocol.json`, then run:

```sh
python3 research/mlp-unit-budget.py "$SCRATCH"
```

The script writes the single fixed selection and its holdout summary. The
production model retains its original weights and omits these research counters.
