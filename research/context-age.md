# Context-balanced learning age

**Fixed real-coder screen running.** This experiment changes the learning
schedule of the original mixers, with zero added pretrained weights.
[Protocol](results/context-age-protocol.json) · [Focused checks](results/context-age-check.json)

## Hypothesis

Each original mixer lowers its SGD multiplier according to its total coded-bit
count: 1 before 1 million visits, 0.7 before 5 million, 0.3 before 25 million,
then 0.2. A new independently keyed weight vector in a mature stream therefore
starts with the mature global learning rate.

The candidate gives each admitted key an exposure counter and uses:

```
age = exposures_before_current_bit * current_number_of_admitted_keys
```

The shared fallback retains the original global age. All original thresholds,
multipliers, gradients, context capacities and SSE behavior remain. A one-key
mixer retains its schedule for the same input sequence. Upstream changes can
still change the final mixer's inputs.

This age conversion assumes equally frequent keys as its reference case.
Frequent keys slow down earlier, rare keys later, and growing the map advances
existing keys' effective ages. Actual coded bytes decide whether that tradeoff
helps this stream.

## Causality and cost

`HUTTER_CONTEXT_AGE` defaults to zero. Each counter starts at zero, advances
before the existing small-gradient early return, and saturates at 25 million.
Age multiplication uses 64 bits. Counters follow the context through map
copies/rehashing and persist across article boundaries. The existing global
clock keeps advancing. Trace support counts remain separate.

On the tested 64-bit build, `ContextData` grows from 32 to 40 bytes and map
buckets from 40 to 48 bytes. Total overhead follows allocated map capacity;
there is no dense new matrix or offline weight file.

The small check covers all three thresholds, a late key, the shared fallback,
one-key equivalence, 64-bit multiplication, early returns and saturation.
The default-disabled GCC assembly matches the prior implementation after
excluding the source filename directive. The real default-path golden check
is part of the running screen.

## Fixed measurement sequence

1. Build both variants from the same source and ordinary compiler family.
2. Disabled 256 KiB encoding must equal the original 20,954-byte stream exactly.
3. Candidate 64 KiB must round-trip.
4. Candidate prefix and middle 1 MiB slices must both round-trip, save at least
   zero bytes each and at least 263 bytes together against their original
   84,839 / 208,696-byte references. A larger prefix stream stops the screen.

Only a passing size screen proceeds to a separately approved paired timing,
packaging and mature-prefix comparison. The current queue stops at this gate.
The original neural weights, PPMd, article order and other research switches
remain unchanged.

```sh
# Run in this repository, with generated files in your scratch directory.
c++ -O2 -std=c++17 -I src -DHUTTER_CONTEXT_AGE_CHECK \
  -DHUTTER_CONTEXT_AGE=1 research/context-age-check.cpp \
  src/mixer/mixer.cpp src/mixer/sigmoid.cpp -o "$SCRATCH/context-age-check"
"$SCRATCH/context-age-check"
python3 research/build.py --source . --out "$SCRATCH/build/context-age" \
  --define HUTTER_CONTEXT_AGE=1 --extra research/probe.cpp
```
