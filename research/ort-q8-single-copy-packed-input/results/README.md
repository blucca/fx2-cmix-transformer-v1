# Complete fixed single-copy acceptance

- `acceptance.json` joins the source, target compute and fixed-language gates.
- `runs/target/`: three fresh single-session workers, all thirteen output checks,
  every 200 timing samples per cache, initialization and full-process peak RSS.
- `language/target/`: two fresh independently chained old/packed workers;
  every 2048 per-token loss, all chain checks, four 64-token continuations,
  exact fixed target IDs and paired comparison.
- `language/reused-original/`: original FP32 report and the same token IDs from
  the earlier separate-host study; its CPU/runtime features are retained.
- `source/`: original complete graph/native checks and first cache16 gate.
  Construction manifests retain their original stage labels; runtime gates
  belong to the subsequent worker reports. Initial Python-path and timing-wrapper
  failures are preserved alongside their successful focused continuation.

Current controlled performance comes from the AVX2/FMA/noVNNI target;
fixed FP32 loss uses the explicitly identified separate host. `package/` records the actual public ZIP byte-equal transport and one cold
complete OwnPrompt Run; this internal QA is separate from external use.
