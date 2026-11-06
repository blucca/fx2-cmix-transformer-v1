# A frozen weight, two full streams, two original archives

The whole-stream causal trajectory has one mixed-codec candidate. Its final
delivery uses the original **old-v22, untraced, static15** family on both sides:

1. W0 encodes the complete 586,459,020-byte prepared stream.
2. The single selected W1 encodes that same stream.
3. Each real `archive9` restores the original 1,000,000,000 bytes, with MD5
   `e206c3450ac99950df65bf70ef61a12d`.

[`causal-fullstream-job.py`](causal-fullstream-job.py) reuses
[`batch.py`](batch.py) for the prepared encoder and
[`archive-package.py`](archive-package.py) for the original self-extracting
sections. Its explicit `encode` phase also assembles the package. The two
actual S values become available **before** committing the native decode
budget. Each subsequent `decode` phase verifies the official original and
keeps the real archive, compressor, original, logs and Linux resource receipt.

The running capture/training/mixed method remains unchanged. Full-stream
admission follows its complete results and the original prepared inverse.
Current implementation evidence is the
[model-free orchestration check](results/causal-fullstream-orchestration.json).
The check covers phase reuse, real package sizes and CPU accounting; its
prepared fixture has 10,000 bytes and restored fixture has 17,000 bytes.

## One factual entry, shared by both machines

Use `measurements_from_mixed(state)` for the completed causal mixed receipt: it
extracts the selected W, actual sample fee, separate mature/prefix scenarios,
per-scope encode/decode rates and the existing native reports. It preserves the
fresh-q2 role and uses restored prepared bytes for decode rate.

Create `fullstream-entry.json` from that output, the completed inverse, native
build and deployment receipts. It freezes the chosen W, the measured scope, a named
record scenario and resource decision. Keep this entry unchanged across the
two sides and both phases.

- `status`: `admitted`; `family`: `old-v22-untraced-static15`; `root`: the
  scratch root containing `prepared` and the pristine `upstream`.
- `common`: records for `stream`, `vocab`, `dictionary`, `probe`, `binary`,
  `compressed_dictionary`, `compressed_order`. Each has an absolute `path`,
  actual `bytes` and `sha256`. The probe and packed B share the original
  object family with the static15 loader. The reference relink and original
  golden provenance accompany the entry.
- `sides.W0` / `sides.W1`: a `weights` record, `cpu_limit_seconds`,
  `prep_cpu_seconds` and `forecast_remaining_cpu_seconds`. Allocate at most
  540,000 CPU seconds per side and 1,080,000 in total. Original auxiliary
  preparation cost is 502.463058 CPU seconds, paid once on its actual worker
  and allocated to W0's preparation ledger.
- `admission`: `decision: continue`; the true facts `mixed_complete`,
  `prepared_inverse_passed`, `original_queues_terminal`,
  `scope_extension_recorded`, `cpu_calendar_passed`; plus the numerical fields
  `mature_target_gain_bits`, `mature_actual_saved_bytes`, `mature_score_bytes`,
  `prefix_payload_saved_bytes`, `q2_payload_saved_bytes`,
  `actual_pair_net_saved_bytes`, `s0_reference_bytes`, `s0_reference_kind`
  (`actual` or `scenario`) and `latest_record_bytes`.

Admission uses positive mature final-u16 target gain, mature actual coded
gain and whole-prefix gain. The two measured payloads add one double-W fee.
The named mature-rate scenario must cover
`S0_reference - floor(99 * latest_record / 100)`; the comparison uses integer
cross multiplication. Random-held and both long native audits retain their
reported target counts and selection roles.

The mature interval is contained in the complete prefix payload. The sample
sum uses that complete prefix once, plus the separate cold q2 scope. Mature
and prefix full-stream extrapolations are separate scenarios. Decode rate
uses `decode.cpu_seconds / decode.output_bytes`, the restored prepared length.

The original 1GB provenance and native resource rules travel with the actual
entry. Token expiry, native pause/cooldown and the existing resume watch are
the caller's calendar inputs. A seven-day command wall limit and CPU work
budget keep their separate clocks; the existing machine's authorized token
can be renewed when the measured remaining work crosses its expiry.

## Execute two sides, then use their actual sizes

Deploy this job beside the retained `batch.py` and `archive-package.py`. Use
one machine for W0 and one for W1, after their original heavy queues exit.
The same entry contains both asset identities; each machine accesses its
own weight file and the shared small B/D/O/probe artifacts.

```sh
python3 -B -u causal-fullstream-job.py \
  --entry fullstream-entry.json --side W0 --phase encode --out fullstream
# The other machine uses --side W1 with the same entry and its own out.
```

Each side's `fullstream/state.json` reaches `encoded` with the actual package
and its per-phase CPU receipt. Collect those two small state files. The
`paired_size(W0_state, W1_state)` function checks the shared entry and returns
both actual S values, net bytes and each side's native-decode status:

```text
S_i = P_i + 2*B + 2*D + O + 2*W_i + 69
S0 - S1 = (P0 - P1) + 2*(W0 - W1)
```

Commit the native decode budget when the actual paired gain and record margin
justify it. The exact phase command then reuses the completed encoder:

```sh
python3 -B -u causal-fullstream-job.py \
  --entry fullstream-entry.json --side W0 --phase decode --out fullstream
# Repeat for W1 on its own machine.
```

Run each long phase through the existing detached launcher, with stdout and
exit written to its scratch directory. Preserve native pause/resume and the
single change-only watcher. Encoded-only states finish the current phase;
rearm its watcher when the next phase actually starts. A failed or interrupted
attempt remains for explicit inspection and its CPU is charged before retry.

The runner pins one allowed CPU, sets numerical-library threads to one, keeps
all three trace outputs off, and uses a 24 GiB address-space ceiling. The
per-side ledger carries preparation cost, sums per-command Linux wait4 CPU and
deducts completed work before each command's inherited CPU limit. A 900-second
reserve covers the native dictionary subprocess, digest and receipt work.
GNU timeout ends the command group at the independent seven-day wall limit.

Before the model phase, required free scratch covers the 13.671875 GiB PPMd
map, a complete prepared temporary, headroom and, for decode, the original
length as WRT output. Reclaim completed training/trace caches at admission.
The generated native intermediates are removed from this job's fresh package
directory after its exact restoration passes.

## Delivered measurements

- Full prepared payload and encoded receipt for each W.
- Both actual `cmix` / `archive9`, package manifests and file identities.
- Two official-original byte counts, MD5 values and native exit codes.
- Per-phase command, CPU, wait4 peak-process RSS and cumulative side ledger.
- Two actual S values and one net double-W comparison.

The prepared encode measures the range/NN path. Full compressor `cmix -e`
preprocessing and final formal hardware/time qualification have their own
actual measurements. Award qualification uses the final actual S1 against
the record at submission.

Run the single small check in a fresh scratch directory:

```sh
python3 -B research/causal-fullstream-check.py --work /YOUR/temp/fullstream-check
```
