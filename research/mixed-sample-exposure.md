# Which mixed-codec bytes entered the new update trajectory?

The completed W64 experiment saved 57,089 bytes on full q2 while losing 7,085 bytes on its independent held window. Its training-rich complement generated 64,174 bytes of savings. Direct update exposure is therefore a useful axis for the next two retained mixed-codec measurements.

Causal-v1 and depth8 both use the same frozen **seed414** plan: 1,313 training pieces / 2,941,307 targets and 473 validation pieces / 1,082,505 targets. This audit reads the original transferred plan once and intersects those complete pieces with the already fixed codec ranges. Model, optimizer, native and codec execution increments are zero.

| Retained scope | New training pieces | New training input bytes | New training target positions | Training share of trace NN targets | Sampled validation targets |
|---|---:|---:|---:|---:|---:|
| Full prefix8 | 25 | 35,765 | 35,740 | — | 24,552 |
| Prefix8 mature window | 1 | 1,671 | 1,670 | 0.1604% | 4,878 |
| Full q2 | 1 | 11,185 | 11,184 | 1.0668% | 0 |
| Legacy W64 held window, now viewed under seed414 | 1 | 11,185 | 11,184 | 2.1658% | 0 |

For full prefix8, training input bytes occupy 0.4264% of its 8,388,608 input bytes; the table leaves its full-trace target-share field open. The mature and q2 target denominators come from the retained production traces.

## The one q2 training piece

Seed414 selected article **76410**, source interval **[293818027, 293829212)**, for training. Its first input token provides context; the remaining **11,184** positions are next-token targets. This same complete piece lies inside the old W64 held window.

Full q2 starts at 293232482. Its byte ranges outside the new sampled training input are **[0, 585545)** and **[596730, 1048576)** in that original q2 trace. The legacy W64 window contributes **[531039, 585545)** and **[596730, 1047533)** in the same trace coordinate. These are ready-made attribution ranges for the eventual retained traces; the present report contains position counts.

Independence here refers to direct input and target exposure in the **new seed414 update trajectory**. Upstream W0 was pretrained on enwik9; online PPMd/mixer history, sampled validation membership and physical codec state remain separate axes. The JSON records full source coordinates, scope-local coordinates and parent-trace coordinates explicitly.

## Replay and next measurement

```sh
python3 research/mixed-sample-exposure.py
```

The standard-library replay checks every original piece count and first-row target mask, disjoint intersections, subtraction inventory, and the saved four-scope result. One clipped-piece check covers a range starting within a training piece. The self-contained [result](results/mixed-sample-exposure.json) retains the original selected-piece plan.

Original e2/every4 selection, 16/32 epochs, one terminal-selected mixed measurement per method and all running queues remain fixed. When the real G/M/u16/coded results arrive, use **this trajectory's actual exposure** for attribution, then compare complete package savings and the contemporary record threshold. Existing trained weights and encoded streams are reused.
