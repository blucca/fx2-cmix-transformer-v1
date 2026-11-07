# An exact middle-layer oracle, and a small linear bridge

We tested the four-layer hole in the eight-block student at its original hard
initialization. Injecting the twelve-block teacher's missing middle output,
while preserving the original outer skip source, reproduces all four remaining
blocks, final hidden state and logits **bit-for-bit**.

A single fitted linear residual has a smaller effect: on the one held piece,
it removes **43.04% of midpoint reconstruction error** and **8.24% of the
output cross-entropy gap**. This bounded study is complete. The unchanged
architecture's output-level teacher objective remains the next conditional
experiment.

[Complete scientific result](results/depth-middle-bridge.json) ·
[Input, hidden and output arrays](results/depth-middle-bridge.npz) ·
[Read-only measurement CLI](depth-middle-bridge.py) ·
[Minimal result replay](check-depth-middle-bridge.py)

## The replacement and its skip contract

The student keeps original layers `[0,1,2,3,8,9,10,11]`, reindexed as `0..7`.
Let `x3` and `x7` be the original teacher's outputs after blocks 3 and 7.
The first four student blocks produce exactly the teacher's `x3`.

The teacher's block 8 receives `x7 + s2*x3`, where `s2` is original skip
coefficient 2. The mapped student block 4 normally receives `x3 + s2*x3`.
Our hook operates **after the production skip pop**, retaining the unmodified
`x3` that was saved for the outer skip:

```text
zero8:   x3 + s2*x3
ridge:   (x3 + A*RMSnorm(x3) + b) + s2*x3
oracle:  x7 + s2*x3
```

The oracle supplies the same teacher middle state and original skip source.
Each of the four tail blocks and both final outputs match the teacher's FP32
bits. The replacement arithmetic and upstream state remain explicit; the hook
is removed after the measurement.

## One fit, one held piece

The fit uses the original eighth complete training piece: **1,199 targets**,
article 1447, source input range `[1256446,1257646)`. Evaluation uses the first
original validation piece with 1,025–2,048 targets: **1,606 targets**, article
3363, source input range `[5055675,5057282)`.

Both use the unchanged full-history causal PPMd capture, all 205 FP16 prior
columns, and a separate NN reset at the complete piece boundary. The fit and
validation articles have distinct roles. The original W0 pretraining and causal
PPMd history are retained. The held article contributes metrics and its oracle
state; all ridge-fit rows come from the training article.

For `X = RMSnorm(x3)` and `D = x7 - x3`, we solve a single FP64 ridge system
with an appended bias column:

```text
lambda = 1e-3 * trace(X.T @ X) / 192
Theta  = solve(X_aug.T @ X_aug + diag(lambda,...,lambda,0),
               X_aug.T @ D)
```

The bias has zero ridge penalty. The inference matrix and bias are cast to
FP32. One lambda and one full-rank fit are used; original model weights and their signed
BF16 scales stay fixed.

### Held output metrics

Cross-entropy and teacher KL are nats per target. Hidden MSE measures the final
RMS hidden state.

| Path | Cross-entropy | KL(teacher‖path) | Final hidden MSE |
|---|---:|---:|---:|
| Twelve-block teacher | 0.538509548 | 0 | 0 |
| Original hard zero8 | 0.964570701 | 0.439032525 | 0.393906623 |
| Fitted FP32 bridge | 0.929455817 | 0.389080703 | 0.289724380 |
| Teacher middle-state oracle | 0.538509548 | 0 | 0 |

The fitted path improves held CE by **0.035114884 nats/target** and teacher KL
by **0.049951822**. Relative to this piece's zero8-to-teacher CE gap,
`(CE_zero - CE_fit)/(CE_zero - CE_teacher) = 0.082417475`.

### Middle-state reconstruction

| Fit support | Original `x3` versus `x7` MSE | Fitted versus `x7` MSE | Error reduction |
|---|---:|---:|---:|
| Training piece | 0.026973260 | 0.005955537 | 77.92% |
| Held piece | 0.028237927 | 0.016083734 | 43.04% |

The midpoint and final-distribution measurements capture distinct objectives.
On this held piece, the tail network transfers 43.04% midpoint-MSE recovery
into 8.24% CE-gap recovery. End-to-end teacher retention remains the selection
signal for a complete trained replacement.

## Actual execution and model state

The successful measurement performs **two teacher forwards, four student
forwards and one ridge solve**, all under `torch.no_grad()`. Its paths are one
training zero8 forward and the three held paths above. All original teacher and
student tensors, the 75 student hard coordinates, fixed scales and empty
gradient slots pass their state checks.

Successful child plus supervisor CPU is **4.730368072 seconds**, with
**581,556 KiB** largest-process RSS and **5.117488 seconds** wall time. A first
attempt stopped during imports with `No module named 'fla'`, before inputs,
models or forwards. Moving the `pysrc.model` import after `cpu_model.load_model`
installs the existing CPU adapters and completes the explicitly authorized
recovery. The original failure remains in the attempt record. Cumulative
CPU is **5.892735359 seconds** inside the original 180-second limit; both
attempts use one CPU and a 4 GiB address-space ceiling.

The fitted matrix has `192*192` coefficients. The nominal INT4 matrix,
FP32 bias and BF16 row-scale items total **19,584 bytes**. Those items describe
an implementation estimate. The measurement itself uses the **148,224-byte
FP32 matrix/bias**, with a floating activation input. Native package accounting,
INT4 bridge deployment, full training and actual mixed coding are separate
stages; their execution counts in this study are **zero**.

The original 32-epoch student and each original selected-weight mixed-codec
measurement continue unchanged. The bridge's new native family and full
successor are stopped. The [curvature-matched 2KL objective](teacher-objective-audit.md)
retains the existing architecture and teacher cache; its original same-eight
mixed-codec admission remains in force.

## Replay and measurement source

The result replay uses Python's standard library and reads the supplied JSON:

```sh
python3 research/check-depth-middle-bridge.py
```

The measurement requires the original causal asset manifest and captures,
original twelve-block `6m-q4-fp32.tch` checkpoint, existing `cpu-kda.so`, and
NumPy/PyTorch dependencies described in [CPU learning](cpu-backward.md) and
[the depth method](depth-distillation.md). Run with a fresh output directory:

```sh
python3 research/depth-middle-bridge.py \
  --job /path/to/causal-assets \
  --source /path/to/original/research \
  --checkpoint /path/to/6m-q4-fp32.tch \
  --kernel /path/to/cpu-kda.so \
  --output /path/to/fresh-middle-bridge-output \
  --cpu 1
```

`--dependency-root` accepts an already-installed dependency directory.
`--help` loads zero models. The public CLI preserves the actual successful
scientific body from manifest parsing through its final result/exception path;
only argument, path and resource wiring precedes that body.

The CLI reproduces these two fixed original corpus pieces and their declared
role selection. The supplied NPZ preserves the scientific inputs, teacher middle states,
teacher/student logits and hidden states, ridge sufficient statistics and
fitted coefficients. The result records all loss reconstructions and the
exact oracle/skip/state assertions. Original checkpoint payloads and access
credentials use private storage.
