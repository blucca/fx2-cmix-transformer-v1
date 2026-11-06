#!/usr/bin/env python3
"""Original v22 trace objects, one original static15 loader, strict interval reducer."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--loader-object', type=Path, required=True)
    p.add_argument('--interval-source', type=Path, required=True)
    p.add_argument('--trace-include', type=Path, help='Defaults to original manifest source/research')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    base, loader, source, out = a.base.resolve(), a.loader_object.resolve(), a.interval_source.resolve(), a.out.resolve()
    manifest = json.loads((base / 'build.json').read_text())
    assert manifest['status'] == 'built'
    link = next((c for c in manifest['commands'] if c[-2:] == ['-o', 'probe']), None)
    if link is None:
        cmix = next(c for c in manifest['commands'] if c[-2:] == ['-o', 'cmix'])
        link = [x for x in cmix[:-2] if x not in ('runner.o', '-s')] + ['probe_research.o', '-o', 'probe']
    original = [str(base / x) if x.endswith('.o') and not Path(x).is_absolute() else x for x in link]
    objects = [Path(x) for x in original[:-2] if x.endswith('.o')]
    slots = [x for x in objects if x.name == 'tf_weights_io_compressed.o']
    assert len(slots) == 1 and any(x.name == 'probe_research.o' for x in objects)
    include = (a.trace_include or Path(manifest['source']) / 'research').resolve()
    for path in [loader, source, include / 'trace.h', base / 'probe', *objects]:
        assert path.is_file(), f'required original artifact missing: {path}'
    out.mkdir(parents=True, exist_ok=False)
    reference = [*original[:-1], str(out / 'probe.reference')]
    common = [str(loader) if x == str(slots[0]) else x for x in original]
    common[-1] = str(out / 'probe')
    reducer = [link[0], '-std=c++17', '-O2', '-fno-fast-math', '-ffp-contract=off',
               '-I', str(include), str(source), '-o', str(out / 'mixed-interval')]
    row = {'status': 'building', 'started_at': time.time(), 'base': str(base),
           'original_probe': str(base / 'probe'), 'original_loader': str(slots[0]), 'static15_loader': str(loader),
           'objects': list(map(str, objects)), 'retained_main_abi': True,
           'changed_object': 'tf_weights_io_compressed.o', 'interval_strict_fp': True,
           'interval_source': str(source), 'trace_header': str(include / 'trace.h'),
           'commands': [reference, common, reducer]}
    def save():
        own, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        row['cpu_seconds'] = own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime
        (out / 'build.json').write_text(json.dumps(row, indent=2) + '\n')
    save()
    try:
        with (out / 'build.log').open('w') as log:
            subprocess.run(reference, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert (out / 'probe.reference').read_bytes() == (base / 'probe').read_bytes(), 'reference relink byte identity failed'
            row['reference_relink_byte_equal'] = True
            save()
            for command in (common, reducer):
                subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
        row.update(status='built', elapsed_seconds=time.time() - row['started_at'],
                   common_probe_bytes=(out / 'probe').stat().st_size,
                   interval_bytes=(out / 'mixed-interval').stat().st_size)
    except BaseException as exc:
        row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    print(json.dumps(row))


if __name__ == '__main__':
    main()
