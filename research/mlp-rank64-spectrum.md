# Rank64 MLP: a byte budget with a 43.8% weight-space tail

A depth-preserving candidate keeps all twelve blocks and factorizes their MLPs.
The fixed rank64 shape removes **2,064,384 INT4 values**, about 35.2% of the
whole model’s integer weights. Its original linear matrices retain just
**56.2277% of their combined squared Frobenius norm** in the best rank64
approximation. The remaining **43.7723%** is the measured weight-space tail.

This result closes the simple unweighted SVD initialization path. The existing
eight-block distillation and its actual mixed-codec measurement continue.
Task-weighted factorization would be a separate complete method.

## The actual architecture

The upstream MLP in `pysrc/model.py` is

```text
RMS-normalized 192 → up(768×192) → ReLU² → down(192×768) → residual
```

It has two linear maps per layer. The proposed fixed family preserves all
nine KDA layers, three attention layers, twelve MLPs and six skip pairs.
Each MLP would use four maps:

```text
A(64×192) → U(768×64) → ReLU² → C(64×768) → D(192×64)
```

The two new 64-dimensional intermediate vectors would each receive the
existing INT8 activation quantization. Every factor would use row-wise INT4
weights and BF16 scales. These are proposed shapes; this measurement evaluates
the original effective weight matrices.

| Integer values | Original | Fixed rank64 proposal |
|---|---:|---:|
| Twelve MLPs | 3,538,944 | 1,474,560 |
| All matrices | 5,868,864 | 3,804,480 |
| MLP matrix count | 24 | 48 |

The original 24 MLP matrices have a census-based static15 coding-cost estimate of
1,676,818.737 B, including their tables. New factor histograms, two-copy weight
cost, changed binary size and complete coded payload belong to the candidate’s
future export and codec measurements.

## Complete original-weight measurement

The loader reads `models/6m-q4-fp32.tfwc2` with the existing 434-tensor
compressed-weight reader. It extracts all 24 MLP matrices as
`float32(INT4_value × signed_BF16_row_scale)`. The [decoded integer histograms and shapes](results/mlp-rank64-original-matrices.json)
match the published [original census](results/weights-census.json).
All **3,538,944** original MLP values enter the measurement.

Each effective matrix is promoted to FP64 for a full singular-value spectrum.
For rank64, the Eckart–Young identity gives

```text
best squared Frobenius residual = sum(singular_values[64:]²)
relative Frobenius residual      = sqrt(tail / total_squared_norm)
```

There are 24 SVD calls, one per original matrix. The computation uses singular
values, with factor construction, optimization and calibration counts at zero.
The aggregate weights each matrix by its original squared Frobenius norm.

| Matrices | Rank64 retained energy: min / mean / max |
|---|---:|
| Twelve up projections | 50.8688% / 55.8235% / 59.2698% |
| Twelve down projections | 52.7944% / 56.5246% / 62.0684% |
| Combined, norm-weighted | **56.2276578%** |

The combined best relative Frobenius error is **0.6616066975**. Every layer has
a substantial tail under this fixed approximation objective. Full spectra and
per-matrix residuals are in [the result](results/mlp-rank64-spectrum.json).

## Task-weighted quality has a specific input contract

For a linear residual `ΔW` and an input’s uncentered second moment `M`,

```text
E[||ΔW x||²] = tr(ΔW M ΔWᵀ),    M = E[xxᵀ].
```

With `M = I`, the squared residual is the measured Frobenius tail. An
activation-aware method needs the actual inputs consumed by each map, including
quantization. In particular, `ReLU²` produces down-map inputs with nonzero
means, so the relevant statistic is the **uncentered second moment**.

The present result covers original-weight geometry. Factorized-network NLL,
ReLU boundary changes, hard INT4/INT8 loss, long-piece behavior and final range
bytes each have their own model and codec measurement contracts.

## Reproduce

Requirements: a little-endian Linux CPU, a C++17 compiler, Python 3 and an
existing NumPy installation. From the repository root, choose a fresh scratch
directory. The loader uses the original weight file only.

```sh
export WORK=/YOUR/temp/mlp-rank64
mkdir -p "$WORK"
c++ -std=c++17 -O2 -I cpp_infer/src \
  research/mlp-original-matrices.cpp \
  cpp_infer/src/weights_io.cpp cpp_infer/src/weights_io_compressed.cpp \
  -o "$WORK/dump-mlp"
"$WORK/dump-mlp" models/6m-q4-fp32.tfwc2 \
  "$WORK/original-matrices.f32" > "$WORK/original-matrices.json"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 research/mlp-rank64-spectrum.py --work "$WORK"
python3 research/check-mlp-rank64-spectrum.py \
  --result "$WORK/spectrum-result.json" \
  --matrices "$WORK/original-matrices.json"
```

The last command checks the reproduced spectra, decoded histograms and
aggregate identities using the standard library. Running it without arguments
checks the published result. The generated raw float file is 14,155,776 B and
belongs in the scratch directory.

The [execution receipt](results/mlp-rank64-spectrum-execution.json) records one
compile, one original-file decode and one Python analysis, all exit0. Their
combined child CPU time is 3.320142 s; peak child RSS is 137,064 KiB. The SVD
portion uses 0.198045 CPU s. Neural forward/backward, training and codec calls
are all zero. Source weights remain unchanged.

Original architecture and weights: **Vladimir Ivanov / astOwOlfo**,
[fx2-cmix-transformer-v1](https://github.com/astOwOlfo/fx2-cmix-transformer-v1).
The effective INT4/BF16 interpretation reuses this repository’s native loader.
