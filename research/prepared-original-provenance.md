# Canonical original-input provenance

The frozen 586,459,020-byte prepared stream restores the official **1,000,000,000-byte enwik9**, with MD5 **e206c3450ac99950df65bf70ef61a12d**. [Completed receipt](results/prepared-original-provenance.json) · [standard-library replay](check-prepared-original-provenance.py).

The original-v22 WRT, PHDA9 and article-ID sorting helpers perform this inverse. The public [prepare adapter](prepare.cpp) exposes the same command:

```sh
./prepare --restore PREPARED_STREAM ENGLISH_DICTIONARY RESTORED_DIRECTORY
```

The completed run uses one CPU and a 4 GiB address-space ceiling. Native restore takes 66.03 user + 5.82 system seconds and 71.96 elapsed seconds. Compilation, original waiting-queue retirement, restore, final MD5 and receipt overhead total **78.580048 CPU seconds** in the original 1,800-second preparation allowance. Peak child RSS is **1,516,000 KiB**. The completed inverse inventory is **5,868,408,757 logical bytes / 5,868,449,792 allocated bytes**; seven duplicate intermediates are cleared, retaining the canonical original.

This result supplies the original-input identity for the [whole-program fullstream route](causal-fullstream.md). Its submitted W1 compressor/archive timings, package sizes and resource qualification are measured by those complete programs. The inverse receipt has **zero range/NN calls** and keeps its preparation CPU separate from training and mixed-codec measurement.
