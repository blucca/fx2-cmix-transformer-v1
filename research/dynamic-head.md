# A causal online readout from Transformer hidden states

A zero-initialized **205 × 192 fp32 residual matrix** learns from already
observed next tokens. Its input is the model's final RMS-normalized hidden
state. It adds no pretrained weights and uses 157,440 bytes of weight RAM.

## Fixed screen: positive signal, quality gate missed

| Prepared-stream slice | Base NLL | Shadow NLL | Improvement, nats/token |
|---|---:|---:|---:|
| Prefix, 1 MiB | 0.4637935360 | 0.4621754266 | 0.0016181094 |
| Middle, offset 293,232,482, 1 MiB | 1.1102643653 | 1.1088144201 | 0.0014499451 |

The weighted improvement is **0.0015339819 nats/token** over 2,095,604 predicted
tokens. Both halves of each slice and all 64 fixed 32 KiB blocks improve.
The predeclared continuation threshold was **0.002**, with each slice and its
second half positive. **Stop this fixed recipe before paired timing or coder
integration.** The signal supports useful recent information in the hidden
states; its size falls below the budget assigned to this dense online head.

[Exact protocol, measurements and block scores](results/dynamic-head-A.json).
The three unchanged base distributions reproduce their original complete
probability fingerprints and NLL sums. Valid capture/evaluation used 190.6 CPU
seconds. Wall time was 315.8 seconds including recovery from a cached-object
header-layout mismatch. Shadow timings include dual diagnostics; a matched
performance pair was gated off. Real coded-byte savings remain unmeasured.

## Prediction and learning contract

For final normalized hidden vector `h` and **already soft-capped** logits `s`:

```
q = softmax(s + W h)
score the observed next token y using q
alpha = 2^-5 / (1e-6 + dot(h, h))
W[c] = (1 - 2^-15) * W[c] + alpha * (one_hot(y)[c] - q[c]) * h
```

`W` starts at zero per independent slice and persists across article/piece
boundaries. A piece's last token is scored and learned; the next piece's first
token has no preceding neural prediction. The head leaves the underlying
Transformer, PPMd priors, KVI8 cache and RoPE evolution unchanged. Prediction
accepts no target; observation consumes each pending prediction once.

The probe uses the existing AVX2 dot-product order and row-update kernel.
Its softmax evaluates exponentials in double precision and stores fp32 learning
probabilities. The focused self-test covers zero-initialized predictions, the
first normalized update, gradient direction, pure decay and piece accounting.

## Reproduce

Use the source and data preparation in [the research guide](README.md).
`FX2_DYNAMIC_HEAD` defaults to zero; `FX2_POSITIVE_U8` stays zero for this screen.

```sh
python3 research/build.py --source . --out "$SCRATCH/build/shadow" \
  --define FX2_DYNAMIC_HEAD=1 --define FX2_POSITIVE_U8=0 \
  --extra research/activation-probe.cpp --strict-extra
"$SCRATCH/build/shadow/activation-probe" --self-test
"$SCRATCH/build/shadow/activation-probe" capture \
  STREAM.bin TOKENS.u8 PRIORS.f16 VOCAB.bin OFFSET LENGTH
"$SCRATCH/build/shadow/activation-probe" eval \
  TOKENS.u8 PRIORS.f16 models/6m-q4-fp32.tfwc2
```

Use fresh capture output paths. Each evaluation creates a fresh head. The
JSON `total`/halves are the original model; `shadow_*` and `blocks` report the
online candidate. When reusing compiled objects, compile the probe against
those objects' matching `Predictor`/`Mixer` headers.

The motivation is [dynamic evaluation](https://arxiv.org/abs/1904.08378) and
[continuous neural caches](https://arxiv.org/abs/1612.04426). This experiment
measures the specific output-only recipe above.
