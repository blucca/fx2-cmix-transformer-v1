#!/usr/bin/env python3
"""Build one fixed 8/12 native family with the existing production flag plan."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from build import commands


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def production_plan(source, compiler, flags_from):
    if flags_from is None:
        return commands(source, compiler or "clang++-17")
    manifest = json.loads(flags_from.read_text())
    original_source = Path(manifest["source"])
    plan = []
    for recorded in manifest["commands"]:
        command = []
        for arg in recorded:
            if arg.startswith(str(original_source) + "/"):
                arg = str(source / Path(arg).relative_to(original_source))
            if arg.endswith(".o"):
                arg = Path(arg).name
            command.append(arg)
        if compiler:
            command[0] = compiler
        plan.append(command)
        if command[-2:] == ["-o", "cmix"]:
            break
    if not plan or plan[-1][-2:] != ["-o", "cmix"]:
        raise ValueError("Production manifest requires the cmix command plan")
    return plan


def family_plan(source, depth, compiler=None, flags_from=None):
    plan = production_plan(source, compiler, flags_from)
    define = f"-DFX2_MODEL_LAYERS={depth}"
    for command in plan:
        if "-c" in command:
            command[:] = [arg for arg in command
                          if not arg.startswith("-DFX2_MODEL_LAYERS")]
            command.insert(1, define)
    model = next(command for command in plan if command[-1] == "tf_model_opt.o")
    tf_flags = model[1:model.index("-c")]
    compiler = model[0]
    fast = next(command for command in plan
                if any(arg.endswith("/src/runner.cpp") for arg in command))
    first_cpp = next(i for i, arg in enumerate(fast) if arg.endswith(".cpp"))
    extra_flags = [arg for arg in fast[1:first_cpp]
                   if arg != "-flto" and not arg.startswith(
                       ("-ffp-model=", "-fprofile-generate=", "-fprofile-use="))]
    extra_flags += ["-fno-fast-math", "-ffp-contract=off", "-I", str(source / "src")]
    if depth == 8:
        # Eight-block production uses reference-order attention. Build its
        # existing kernel object before cmix and share it with all binaries.
        plan.insert(len(plan) - 1, [compiler, *tf_flags, "-c",
                                   str(source / "cpp_infer/src/kernels.cpp"),
                                   "-o", "tf_kernels_reference.o"])
        plan[-1].insert(-2, "tf_kernels_reference.o")
    link = plan[-1]
    shared = [arg for arg in link[:-2] if arg not in ("runner.o", "-s")]
    tf_objects = [arg for arg in link if arg.startswith("tf_") and arg.endswith(".o")]
    link_flags = link[1:next(i for i, arg in enumerate(link) if arg.endswith(".o"))]

    # Both native implementations use the same fresh family directory and
    # weight/attention objects. The reference keeps its original RAW loader.
    for rel, obj in [("cpp_infer/src/kernels.cpp", "tf_kernels_reference.o"),
                     ("cpp_infer/src/model.cpp", "tf_model_reference.o"),
                     ("cpp_infer/src/opt/test_xcheck.cpp", "test_xcheck.o")]:
        if depth == 8 and obj == "tf_kernels_reference.o":
            continue
        plan.append([compiler, *tf_flags, "-c", str(source / rel), "-o", obj])
    xcheck_objects = list(dict.fromkeys([*tf_objects, "tf_kernels_reference.o",
                                        "tf_model_reference.o", "test_xcheck.o"]))
    plan.append([compiler, *link_flags, *xcheck_objects, "-o", "test_xcheck"])
    for name in ("probe", "activation-probe"):
        obj = name + "_research.o"
        plan.append([compiler, *extra_flags, "-c", str(source / "research" / (name + ".cpp")),
                     "-o", obj])
        plan.append([*shared, obj, "-o", name])
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True,
                        help="Scratch parent; objects and binaries go into depth8 or depth12")
    parser.add_argument("--depth", type=int, choices=(8, 12), default=8)
    parser.add_argument("--compiler", help="Defaults to recorded compiler or clang++-17")
    parser.add_argument("--flags-from", type=Path,
                        help="Existing production build.json flags; every object is rebuilt")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    flags_from = args.flags_from.resolve() if args.flags_from else None
    out = args.out.resolve() / f"depth{args.depth}"
    out.mkdir(parents=True, exist_ok=True)
    plan = family_plan(source, args.depth, args.compiler, flags_from)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    sources = sorted({Path(arg) for command in plan for arg in command if arg.endswith(".cpp")}
                     | {source / "cpp_infer/src/model_shape.h",
                        source / "cpp_infer/src/opt/arena_build.h",
                        source / "cpp_infer/src/opt/model_opt.h"})
    manifest = {
        "status": "planned", "started_at": time.time(), "source": str(source),
        "family": {"layers": args.depth, "kda": 6 if args.depth == 8 else 9,
                   "vanilla": 2 if args.depth == 8 else 3,
                   "skip": 4 if args.depth == 8 else 6,
                   "define": f"FX2_MODEL_LAYERS={args.depth}"},
        "out": str(out), "compiler": plan[0][0],
        "flags_from": str(flags_from) if flags_from else "production make plan",
        "flags_manifest_sha256": sha256(flags_from) if flags_from else None,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "address_space_limit_bytes": 1 << 30,
        "objects": "all rebuilt in this family directory",
        "builder_sha256": sha256(Path(__file__).resolve()),
        "source_sha256": {str(path.relative_to(source)): sha256(path) for path in sources},
        "commands": plan,
    }

    def save():
        own = resource.getrusage(resource.RUSAGE_SELF)
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        manifest.update(cpu_seconds=own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime,
                        child_peak_rss_kib=child.ru_maxrss,
                        elapsed_seconds=time.time() - manifest["started_at"])
        (out / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    if args.plan_only:
        print(json.dumps({"status": "planned", "out": str(out), "commands": len(plan)}))
        return
    try:
        manifest.update(status="building", compiler_version=subprocess.check_output(
            [plan[0][0], "--version"], text=True).splitlines()[0])
        with (out / "build.log").open("w") as log:
            for i, command in enumerate(plan):
                manifest["stage"] = i + 1
                save()
                print(f"compile step {i + 1}/{len(plan)}", flush=True)
                subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, check=True)
        binaries = ("cmix", "probe", "activation-probe", "test_xcheck")
        manifest.update(status="built", binaries={name: {"bytes": (out / name).stat().st_size,
                                                        "sha256": sha256(out / name)}
                                                   for name in binaries})
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(json.dumps({"status": manifest["status"], "out": str(out),
                      "cpu_seconds": manifest["cpu_seconds"], "binaries": manifest["binaries"]}))


if __name__ == "__main__":
    main()
