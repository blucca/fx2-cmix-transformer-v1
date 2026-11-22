# Complete measured results

`acceptance.json` combines the real complete gates. `runs/{target,remote}`
retain six fresh reports per host, every timing sample and all output checks.
`feeds` records each host's frozen synthetic token/cache preparation.
`language/remote` records four independently chained models and paired FP32
losses; `language/target` records the independent same-Q8 target repeat.
`models-manifest.json` above this directory is the converter-stage graph and
storage record. The model and large reference arrays are reconstructed by
README commands; actual JSON results are preserved here.
