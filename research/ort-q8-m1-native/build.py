#!/usr/bin/env python3
"""Build the single ORT 1.31.1 operator on Linux or a Windows MSVC shell."""
import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

HEADERS = ('onnxruntime_c_api.h', 'onnxruntime_cxx_api.h', 'onnxruntime_cxx_inline.h',
           'onnxruntime_float16.h', 'onnxruntime_error_code.h', 'onnxruntime_ep_c_api.h')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() not in ('Linux', 'Windows') or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise SystemExit('Build target: Linux/Windows x64 with AVX2 and FMA')
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # The exact six headers used by the published builds are bundled.
    # This fixed API-31 build stays reproducible after upstream URL removal.
    sdk = Path(__file__).resolve().parent / 'sdk-1.31.1'
    assert all((sdk / name).is_file() for name in HEADERS)
    source = Path(__file__).resolve().parent / 'q8_m1_ort.cpp'
    if os.name == 'nt':
        command = ['cl', '/nologo', '/O2', '/LD', '/std:c++17', '/arch:AVX2', '/EHsc',
                   '/I' + str(sdk), str(source), '/Fo' + str(out.with_suffix('.obj')),
                   '/Fe' + str(out), '/link', '/IMPLIB:' + str(out.with_suffix('.lib'))]
    else:
        command = [os.environ.get('CXX', 'c++'), '-O3', '-mavx2', '-mfma', '-shared',
                   '-fPIC', '-std=c++17', '-I' + str(sdk), str(source), '-o', str(out)]
    print('Building:', ' '.join(command), flush=True)
    subprocess.run(command, check=True, cwd=out.parent)
    receipt = {'platform': platform.platform(), 'ort_sdk': '1.31.1', 'command': command,
               'library': out.name, 'bytes': out.stat().st_size}
    out.with_suffix(out.suffix + '.build.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
