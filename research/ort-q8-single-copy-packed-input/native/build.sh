#!/usr/bin/env bash
# Pinned Linux API31 build; old headers remain read-only and are not copied.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sdk=${SDK:-"$here/../../ort-q8-m1-native/sdk-1.31.1"}
sdk=$(cd -- "$sdk" && pwd)
out=${1:-"$here/libblucca_q8_m1_packed.so"}
mkdir -p -- "$(dirname -- "$out")"
out=$(realpath -- "$out")
cxx=${CXX:-g++}
for header in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h \
              onnxruntime_float16.h onnxruntime_error_code.h onnxruntime_ep_c_api.h; do
  test -f "$sdk/$header"
done
cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
ulimit -v 524288
ulimit -t 180
command=("$cxx" -O3 -mavx2 -mfma -shared -fPIC -std=c++17 "-I$sdk"
         "$here/q8_m1_packed_ort.cpp" -o "$out")
printf 'Building: '; printf '%q ' "${command[@]}"; printf '\n'
taskset -c "$cpu" timeout 175s "${command[@]}"
python3 - "$here" "$sdk" "$out" "$cpu" "$cxx" <<'PY'
import json, platform, subprocess, sys
from pathlib import Path
here, sdk, out = map(Path, sys.argv[1:4])
cpu, cxx = int(sys.argv[4]), sys.argv[5]
headers = ('onnxruntime_c_api.h', 'onnxruntime_cxx_api.h', 'onnxruntime_cxx_inline.h',
           'onnxruntime_float16.h', 'onnxruntime_error_code.h', 'onnxruntime_ep_c_api.h')
receipt = {
    'status': 'compiled', 'platform': platform.platform(), 'ort_sdk': '1.31.1', 'ort_api': 31,
    'compiler': subprocess.check_output([cxx, '--version'], text=True).splitlines()[0],
    'command': [cxx, '-O3', '-mavx2', '-mfma', '-shared', '-fPIC', '-std=c++17',
                '-I' + str(sdk), str(here / 'q8_m1_packed_ort.cpp'), '-o', str(out)],
    'limits': {'cpu_affinity': [cpu], 'address_space_bytes': 512 * 1024 * 1024,
               'compiler_wall_timeout_seconds': 175, 'cpu_seconds': 180},
    'library': out.name, 'bytes': out.stat().st_size,
    'source_files': ['q8_m1_packed_kernel.hpp', 'q8_m1_packed_ort.cpp', 'build.sh'],
    'sdk_headers': list(headers),
}
out.with_suffix(out.suffix + '.build.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
PY
