# Original QAT weights, twice the attention window

**Direct 1024→2048 inference fails the fixed quality gate.** On the long-article
q2 sample, NLL rises from **1.110264365 to 1.124501583 nats/token**. Both halves
lose accuracy. The original 1024 window and all 434 weight tensors stay.

## Question and fixed experiment

The upstream writeup reports worse loss at a tested 512-token window and says
2048 would be too slow. This experiment measures one direct 2048 extension of
the original 1024-trained model, without training, position rescaling or other
model changes. Three native attention layers retain one additional window of
per-layer K/V; the nine KDA layers, PPMd priors, RoPE and article resets stay.

Window size controls the KV mask and storage, independently of weight shapes.
`FX2_ATTN_WINDOW` defaults to 1024. The attention kernels and model share the
same constant; the loader still checks the original training configuration.
The fixed experiment uses KVI8. Its three KV arrays grow by 1.125 MiB and the
score/exp scratch by 8 KiB; persistent weights grow by zero bytes.

The predeclared gate requires q2 improvement of at least **0.012 nats/token**,
positive improvements in both q2 halves, a nonnegative prefix improvement,
and at most **25%** extra q2 evaluation CPU time. Only a pass would justify a
real mixed-coder experiment. This targets roughly a megabyte of full-stream
information before measuring mixing attenuation and actual package costs.

## Measured result

| Original sample | NLL improvement, nats/token | Raw-network CPU change |
|---|---:|---:|
| 64 KiB smoke | +0.000184748 | +1.45% |
| Prefix, 1 MiB | +0.000001606 | −0.60% |
| Article q2, 1 MiB | **−0.014237218** | **+20.81%** |

The q2 halves lose **0.016059256 / 0.012415183 nats/token**. Its paired CPU
times are **91.309842 / 110.308021 seconds**. Probability-based NLL has the
same direction and magnitude. Total build, checks and six evaluations use
**373.340 CPU seconds**. [Full measurements and frozen protocol](results/attention-window-A.json).

The prefix contains many short articles: only 4.155% of its targets see a
history beyond the original 1024 positions, versus 80.124% for q2. Thus the
small smoke improvement gave a poor picture of this extension's useful regime.
The time gate passes; the main quality gate fails.

**Stop direct 2048 integration, longer runs and window/layer sweeps.** Training
an altered model would be a separate rate–distortion research decision.

## Checks and reproduction

The local scalar [boundary check](attention-window-check.cpp) covers both
window sizes, positions around 1024/2048, partial blocks and ring wrap. It
uses aligned buffers and its own dynamic-size reference; the old naive
attention reference has a fixed 1024-element scratch array and stays unchanged.

All three 1024 evaluations exactly reproduce the original full probability
hashes, NLL totals/halves, target counts and piece bounds. Both variants also
produce identical complete distributions during each piece's first 1023
steps, before either implementation reaches its fixed-window branch.

Follow the [activation-probe preparation](positive-activation.md#reproduce).
Build each window with the same compiler/model flags, keeping the research
main strict-fp, for example:

```sh
python3 research/build.py --source . --out "$SCRATCH/build/2048" \
  --define FX2_ATTN_WINDOW=2048 --define FX2_WINDOW_PROBE=1 \
  --extra research/activation-probe.cpp --strict-extra
"$SCRATCH/build/2048/activation-probe" eval TOKENS PRIORS \
  models/6m-q4-fp32.tfwc2
```

Repeat with 1024 into a separate directory. Use the original captured prefix
and q2 tokens/priors; q2 runs 2048 then 1024, and prefix reverses that order.
The probe reports training/inference windows, CPU time, both NLL measures,
piece bounds and the early-context probability hash.
