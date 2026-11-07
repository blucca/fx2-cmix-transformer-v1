# Task-aware rank64 factors on the model's actual quantized inputs

The [weight-space audit](mlp-rank64-spectrum.md) gave a **43.77% squared Frobenius tail** for rank64 across all 24 original MLP matrices. This experiment asks a different, data-weighted question: how much of each map's output survives a rank64 fit to the inputs the native model actually receives?

On the held article, the floating factors lose **24.15% of up-map output energy and 31.44% of down-map output energy**. An INT4/INT8 initialization of the local two-factor graphs raises those figures to **25.57% and 33.43%**. Moving from the calibration article to the held article adds substantially more error than the extra quantization steps.

## Measured scope

We run the original **12-layer, 192-wide, 768-MLP** native model on two existing complete pieces from [the bridge experiment](depth-middle-bridge.md):

| Role | Article | Prepared-token interval | Targets |
|---|---:|---|---:|
| Calibration/train | 1447 | `[1256446,1257646)` | 1,199 |
| Held | 3363 | `[5055675,5057282)` | 1,606 |

The inputs retain their original 205-column causal PPMd F16 priors. Each piece calls `begin_article(0)`; upstream W0 pretraining and prior history remain part of the original source. Calibration uses 1,199 rows; the 1,606 held rows enter evaluation.

Two read-only capture sites record each MLP's **quantized signed-INT8 inputs**, represented exactly as F32. A third capture records the up-map output before ReLU²; the existing `mlp_out` site records the down-map output. The compressed-container loader is adapted to the original TFWC2 package. All model arithmetic, skip connections, KDA and attention paths stay intact.

For each of the 24 maps, `X = s_act × q8`. Every original output on both pieces matches the reconstructed **INT32 dot → FP32 scale fold bit for bit**: 48 map/piece checks. All original q histograms match the [weight census](results/weights-census.json). Original native NLL is 0.641339725574 on train and 0.539169125472 on held.

## One fixed fit

The down input is sparse ReLU² with a nonzero mean, so we use the **uncentered second moment**:

```text
C = X_train.T @ X_train / 1199
λ = 1e-3 × trace(C) / input_dimension
A = W × sqrt(C + λI)
W64 = truncated_SVD_rank64(A) × inverse_sqrt(C + λI)
```

The regularized objective is `tr[(W − W64)(C + λI)(W − W64).T]`. Its measured value matches the discarded squared singular values. The ridge contributes a weight-space prior for weakly observed directions. One fixed rank and one fixed ridge rule are used for all maps.

The two factors split the retained singular values symmetrically. For the INT4/INT8 local path:

- first input activation scale inherits the original map's BF16 value;
- each factor gets BF16 `max(abs(row))/7` weight scales and nearest-even INT4 coordinates;
- the new 64-dimensional activation scale is BF16 `2/sqrt(127) × mean(abs(first_hard_output))`, measured on train;
- both integer dots use the native FP32 fold and signed INT8 activation clamp.

The 48 factor matrices contain **1,474,560 INT4 coordinates**, compared with 3,538,944 original MLP coordinates. Each local graph is evaluated on the original model's inputs. A complete replacement model would also measure ReLU² propagation, cross-layer drift, training and coded bytes.

## Results

Values are total squared output error divided by total original output energy, summed separately over the 12 maps of each type.

| Maps / piece | Floating rank64 | INT4/INT8 rank64 |
|---|---:|---:|
| Up / train | 11.33% | 12.85% |
| Up / held | **24.15%** | **25.57%** |
| Down / train | 13.70% | 15.80% |
| Down / held | **31.44%** | **33.43%** |

Held per-map INT4/INT8 error spans **18.30–35.81% for up** and **18.39–42.03% for down**. Added quantization contributes 1.43 and 1.99 percentage points to the two held aggregates. The larger observed loss comes from rank64 approximation and calibration-to-held transfer.

The [full result](results/task-aware-mlp-rank64.json) includes every map, factor histogram, scale, native-output check and regularized objective. This is an initialization study on two article-derived pieces. The current complete byte-rate experiment continues on the [existing depth-distillation branch](https://github.com/blucca/fx2-cmix-transformer-v1/tree/research/depth-distillation-20261106).

## Reproduce

From this repository's root, with NumPy, g++, Git and x86-64-v3 support:

```sh
work=temp/native-mlp-rank64
mkdir -p "$work"
python3 research/prepare-native-mlp-inputs.py --out "$work/inputs"
cp cpp_infer/src/model.cpp "$work/model-capture.cpp"
git apply --directory="$work" research/native-mlp-input-capture.patch
g++ -m64 -O3 -march=x86-64-v3 -mtune=generic -mrecip=none \
  -std=c++17 -fno-math-errno -fno-fast-math -DFX2_MODEL_LAYERS=12 \
  -I cpp_infer/src research/native-mlp-input-capture.cpp "$work/model-capture.cpp" \
  cpp_infer/src/kernels.cpp cpp_infer/src/weights_io.cpp \
  cpp_infer/src/weights_io_compressed.cpp -o "$work/capture" -lm
"$work/capture" models/6m-q4-fp32.tfwc2 "$work/inputs" "$work/native-capture"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 research/task-aware-mlp-rank64.py --work "$work/native-capture"
```

Use fresh input/output directories. The analyzer checks every original native output, fits all 24 maps, evaluates held rows and writes the result plus local factor arrays. Raw capture files are reproducible scratch; the existing NPZ and original model are the source assets.
