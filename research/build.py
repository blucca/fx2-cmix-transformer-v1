#!/usr/bin/env python3
"""Build the pinned upstream command plan in a caller-owned scratch directory."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def commands(source, compiler):
    plan = subprocess.check_output(
        ["make", "-n", "cmix", f"CC={compiler}",
         "CFLAGS_DEFINES=-DSEED=923 -DUPDATE_LIMIT=3000"],
        cwd=source, text=True,
    )
    result = []
    for line in plan.splitlines():
        args = shlex.split(line)
        if args and args[0] == compiler:
            result.append([
                str(source / arg) if arg.startswith(("src/", "cpp_infer/")) else arg
                for arg in args
            ])
    if not result or result[-1][-2:] != ["-o", "cmix"]:
        raise ValueError("Upstream build plan changed; inspect it before compiling")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compiler", default="clang++-17")
    parser.add_argument("--extra", type=Path, action="append", default=[],
                        help="Build a research main, linking all upstream objects except runner.o")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = commands(source, args.compiler)
    started = time.time()
    manifest = {
        "source": str(source),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=source)).hexdigest(),
        "compiler": subprocess.check_output([args.compiler, "--version"], text=True).splitlines()[0],
        "commands": plan, "started_at": started, "status": "building",
    }
    (out / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        with (out / "build.log").open("w") as log:
            for i, command in enumerate(plan):
                print(f"compile step {i + 1}/{len(plan)}", flush=True)
                subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
            # Preserve upstream flags; research mains use the same flags as runner.cpp.
            flags = next(command[1:command.index(next(a for a in command if a.endswith(".cpp")))]
                         for command in plan if any(a.endswith("/src/runner.cpp") for a in command))
            objects = [a for a in plan[-1] if a.endswith(".o") and a != "runner.o"]
            for extra in args.extra:
                extra = extra.resolve()
                obj = extra.stem + "_research.o"
                subprocess.run([args.compiler, *flags, "-I", str(source / "src"),
                                "-c", str(extra), "-o", obj], cwd=out,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run([args.compiler, "-m64", "-Wl,--gc-sections", "-std=c++17",
                                *objects, obj, "-o", extra.stem], cwd=out,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest.update(status="built", elapsed_seconds=time.time() - started,
                        binary_bytes=(out / "cmix").stat().st_size)
    except Exception:
        manifest.update(status="failed", elapsed_seconds=time.time() - started)
        raise
    finally:
        (out / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "elapsed_seconds", "binary_bytes")}))


if __name__ == "__main__":
    main()
