#!/usr/bin/env python3
"""Build the pinned upstream command plan in a caller-owned scratch directory."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def commands(source, compiler, defines=(), profile_generate=None, profile_use=None):
    target = "prof_gen" if profile_generate else "prof_use" if profile_use else "cmix"
    profile_flag = "-fprofile-generate" if profile_generate else "-fprofile-use"
    profile_path = profile_generate or profile_use
    plan = subprocess.check_output(
        ["make", "-n", "-B", target, f"CC={compiler}",
         "CFLAGS_DEFINES=-DSEED=923 -DUPDATE_LIMIT=3000 " +
         " ".join("-D" + value for value in defines)],
        cwd=source, text=True,
    )
    result = []
    for line in plan.splitlines():
        args = shlex.split(line)
        if args and args[0] == compiler:
            if profile_path:
                args = [f"{profile_flag}={profile_path}"
                        if arg.startswith(profile_flag + "=") else arg for arg in args]
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
    parser.add_argument("--define", action="append", default=[],
                        help="Additional compiler definition for an explicit ablation")
    parser.add_argument("--extra", type=Path, action="append", default=[],
                        help="Build an unprofiled, non-LTO research main; link all upstream objects except runner.o")
    parser.add_argument("--strict-extra", action="store_true",
                        help="Compile research mains with strict FP for likelihood measurements; model flags stay unchanged")
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--profile-generate", type=Path,
                         help="Use upstream prof_gen flags with this raw-profile directory")
    profile.add_argument("--profile-use", type=Path,
                         help="Use upstream prof_use + LTO flags with this merged .profdata file")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    profile_generate = args.profile_generate.resolve() if args.profile_generate else None
    profile_use = args.profile_use.resolve() if args.profile_use else None
    if profile_generate:
        profile_generate.mkdir(parents=True, exist_ok=True)
    if profile_use and not profile_use.is_file():
        parser.error("--profile-use requires a merged .profdata file")
    plan = commands(source, args.compiler, args.define, profile_generate, profile_use)
    # Keep the wrapper unprofiled/non-LTO: training profiles runner's main.
    # All shared objects and the final link retain upstream PGO/LTO flags.
    flags = next(command[1:command.index(next(a for a in command if a.endswith(".cpp")))]
                 for command in plan if any(a.endswith("/src/runner.cpp") for a in command))
    flags = [arg for arg in flags if arg != "-flto"
             and not arg.startswith(("-fprofile-generate=", "-fprofile-use="))]
    if args.strict_extra:
        flags = [arg for arg in flags if not arg.startswith("-ffp-model=")]
        flags += ["-fno-fast-math", "-ffp-contract=off"]
    # Research executables remain unstripped, matching the development build.
    link = [arg for arg in plan[-1][:-2] if arg not in ("runner.o", "-s")]
    for extra in args.extra:
        extra = extra.resolve()
        obj = extra.stem + "_research.o"
        plan.append([args.compiler, *flags, "-I", str(source / "src"),
                     "-c", str(extra), "-o", obj])
        plan.append([*link, obj, "-o", extra.stem])
    started = time.time()
    manifest = {
        "source": str(source),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=source)).hexdigest(),
        "compiler": subprocess.check_output([args.compiler, "--version"], text=True).splitlines()[0],
        "profile": {
            "mode": "generate" if profile_generate else "use" if profile_use else "none",
            "path": str(profile_generate or profile_use) if profile_generate or profile_use else None,
            "sha256": hashlib.sha256(profile_use.read_bytes()).hexdigest() if profile_use else None,
            "extra_main": "unprofiled-no-lto-strict" if args.strict_extra else "unprofiled-no-lto",
        },
        "commands": plan, "started_at": started, "status": "building",
    }
    (out / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        with (out / "build.log").open("w") as log:
            for i, command in enumerate(plan):
                print(f"compile step {i + 1}/{len(plan)}", flush=True)
                subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
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
