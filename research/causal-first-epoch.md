# First complete causal epoch: native loss and article breadth

The frozen twelve-layer, fixed-scale coordinate method now has a complete
first-epoch native held result. It uses the original single-pass causal PPMd
priors, complete pieces and the article-cluster split. Original capture,
reference native scoring, and the destination eight-layer input transfer are
complete. The sixteen-epoch original method and thirty-two-epoch distillation
continue within their existing CPU budgets.

## Same-W native observation

| Scope | Targets | W0 native NLL | Epoch 1 native NLL |
| --- | ---: | ---: | ---: |
| Complete random held | 1,082,505 | 0.966248338794 | 0.965060291133 |
| Short held | 974,858 | 0.933964335620 | 0.932750302919 |
| Three long held pieces | 107,647 | 1.258614301310 | 1.257661575790 |
| Separate maximum-length audit | 131,071 | 0.038986426132 | 0.038959713843 |

Epoch 1 changes 351 hard weight symbols; signed BF16 scales stay fixed. Its
static15 W is 2,904,329 bytes against W0's 2,904,325, adding **8 bytes** to the
double-W fee. The held mean raw-NLL delta is **-0.001188047662 nats/target**.
A full 586,215,331-target, kappa=1 raw-rate propagation scenario gives
125,595.936 gross bytes, or **125,587.936 net bytes** after double-W. Actual
mixed-coded gain for this candidate is **0 measured bytes** at this stage;
the sole mixed run follows the admitted training terminal.

## Breadth of the fixed pair

The 473 native held pieces map to **473 distinct source articles**. Of these,
**311** have lower piece-mean loss, covering **84.2973%** of held targets. The
largest contributing piece accounts for **4.78955%** of net raw improvement;
removing it leaves delta **-0.001170340132**. Removing the ten largest gaining
pieces leaves **-0.000995948547**.

All six observed length bins have negative target-weighted deltas. The three
32–64 KiB pieces contain 107,647 targets; the 64–128 KiB bin has **zero held
pieces**. A fixed-seed, 1,000-draw observed-article composition bootstrap has a
95% percentile band **[-0.001356484782, -0.001006406486]**. This sensitivity
calculation applies to the fixed W0/epoch-1 pair and the observed article and
length range. The separate maximum-length audit stays separately reported.

## Delivery decision

Continue the existing bounded methods. Full-stream spending follows the
frozen selected W's real mature/prefix/q2 mixed bytes, double-weight fee,
then-current record gate, native restoration and actual execution-machine
resources. The full archive sizes determine the prize result.

The raw paired data and portable standard-library replay are included:

```sh
python3 -B research/causal-epoch01-breadth.py
```

Inputs: [paired native pieces](results/causal-epoch01-paired-pieces.json).
Results: [article breadth](results/causal-epoch01-breadth.json),
[input recovery terminal](results/depth-input-recovery.json), and
[release/current codec pairing](results/release-baseline-pairing.json).
