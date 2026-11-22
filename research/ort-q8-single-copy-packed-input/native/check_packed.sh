#!/usr/bin/env bash
# Small C++ fixture only. Root owns the complete ORT model's first/full gates.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
old_core=${OLD_CORE:-"$here/../../ort-q8-m1-native"}
old_core=$(cd -- "$old_core" && pwd)
cxx=${CXX:-g++}
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
ulimit -v 524288
ulimit -t 180
python3 - "$here" "$old_core" <<'PY'
import json, sys
from pathlib import Path
here, old = map(Path, sys.argv[1:])
old_text = (old / 'q8_m1_kernel.hpp').read_text()
new_text = (here / 'q8_m1_packed_kernel.hpp').read_text()
def function(text, signature):
    start = text.index(signature)
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]
for signature in ('void Run(const float* A, float* Y) const',
                  'static inline void FmaPair(const std::int8_t* weight, float activation,'):
    expected = function(old_text, signature).replace('packed_.data()', 'packed_').replace('scales_.data()', 'scales_')
    if function(new_text, signature) != expected:
        raise SystemExit('Run/FmaPair source equivalence FAILED: ' + signature)
receipt = {'status': 'passed', 'Run_FmaPair_exact_source_except_data_source': True}
(here / 'arithmetic-source-check.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
PY
command=("$cxx" -O3 -mavx2 -mfma -std=c++17 "-I$old_core" "-I$here"
         "$here/check_packed.cpp" -o "$here/check_packed")
printf 'Compiling fixture: '; printf '%q ' "${command[@]}"; printf '\n'
taskset -c "$cpu" timeout 175s "${command[@]}"
taskset -c "$cpu" timeout 10s "$here/check_packed" | tee "$here/check-result.json"
python3 - "$here" "$old_core" "$cpu" "$cxx" <<'PY'
import json, subprocess, sys
from pathlib import Path
here, old = map(Path, sys.argv[1:3])
cpu, cxx = int(sys.argv[3]), sys.argv[4]
receipt = {'status': 'compiled-and-passed',
           'compiler': subprocess.check_output([cxx, '--version'], text=True).splitlines()[0],
           'command': [cxx, '-O3', '-mavx2', '-mfma', '-std=c++17', '-I' + str(old),
                       '-I' + str(here), str(here / 'check_packed.cpp'), '-o', str(here / 'check_packed')],
           'limits': {'cpu_affinity': [cpu], 'address_space_bytes': 512 * 1024 * 1024,
                      'compiler_wall_timeout_seconds': 175, 'fixture_wall_timeout_seconds': 10},
           'result': json.loads((here / 'check-result.json').read_text()),
           'arithmetic_source': json.loads((here / 'arithmetic-source-check.json').read_text())}
(here / 'check-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
PY
