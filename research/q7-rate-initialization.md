# What q7's rate surrogate pays at initialization

The admitted twelve-layer q7 learner uses `2KL + 2H_soft/586459020` while its
export retains the existing static15 container. A read-only calculation from
the exact published weight histogram separates the smoothing cost, concrete
frequency cost, and local rate-only direction. The running recipe stays fixed.

## Whole initialized weight population

The exact `sign(q)*floor((3*abs(q)+3)/7)` map transforms the existing 111 tensor
histograms, covering **5,868,864 coordinates**. At their integer centers,
`tau=.5` Gaussian assignments give:

| Quantity | Measured value |
|---|---:|
| Per-tensor hard histogram entropy | 10,612,352.2524 nats |
| Soft histogram entropy, binary64 reconstruction | 10,765,850.4859 nats |
| Admitted trainer's FP32 soft entropy | 10,765,850.0898 nats |
| Soft excess, one-copy byte equivalent | 27,681.39 B |
| Soft excess, two-copy target-mean objective | 0.000523475 nat/target |
| Static4096 ideal symbol payload | 1,915,868.30 B |
| Static4096 payload excess over hard entropy | 2,069.80 B |
| Nominal histogram tables, 28 B × 111 | 3,108 B |
| Actual admitted complete exported file | 2,044,814 B |

The static estimate uses all fifteen positive frequencies summing to 4096;
eight symbols are unused by q7. It covers ideal symbol payload before range
rounding and container framing. The complete export also includes raw tensors,
scales, metadata and their actual encodings. The soft excess contributes to
the training objective; export selection charges the actual complete file.

## Infinitesimal guide versus one hard symbol step

For tensor histogram `p`, the soft derivative of one coordinate at `s` is
`-sum_j K'_s(j) log p(j)`, where `K` is the normalized seven-center Gaussian.
Compare its feasible descent direction with the exact hard-entropy change
from moving **one** count to the adjacent integer symbol:

- **32.10%** of the 5,864,577 coordinates with a feasible nonzero direction
  point toward an adjacent symbol that increases hard histogram entropy.
- Those groups carry **0.620%** of the absolute soft-rate gradient mass.
- 1,839,876 are initialized zeros; the remaining 42,910 are at ±3.
- The largest two-copy objective derivative is **2.4745e-9** per coordinate;
  the admitted Adam epsilon is **1e-8**.

The high occurrence count and small gradient share describe two different
features of this initialized population. The coordinate derivative measures
an infinitesimal soft move; the adjacent count change measures a complete hard
transition. Training combines that guide with the actual teacher gradient,
Adam state and projection. Its native curves and same-weight coded results
provide the trajectory and final byte measurements.

## Reproduce

[Standard-library calculation](q7-rate-initialization.py),
[exact result](results/q7-rate-initialization.json):

```sh
python3 -B research/q7-rate-initialization.py
```

The script checks a finite-difference derivative, the exact coordinate census,
the 4096-frequency contract and the admitted FP32 entropy. Model loading,
training, native evaluation, codec runs and network requests each total zero.
