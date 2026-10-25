#!/usr/bin/env python3
"""Check the real make plan and scratch builder with a tiny fake compiler."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work", type=Path, required=True)
work = parser.parse_args().work.resolve()
work.mkdir(parents=True, exist_ok=True)
source = Path(__file__).resolve().parents[1]
compiler = work / "fake-compiler"
compiler.write_text('''#!/usr/bin/env python3
from pathlib import Path
import json, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("fake compiler (command-plan check only)")
    raise SystemExit
with Path("calls.jsonl").open("a") as log:
    log.write(json.dumps(sys.argv) + "\\n")
if "-c" not in args:
    assert all(Path(a).is_file() for a in args if a.endswith(".o"))
outputs = ([args[args.index("-o") + 1]] if "-o" in args else
           [Path(a).stem + ".o" for a in args if a.endswith(".cpp")])
for output in outputs:
    Path(output).write_bytes(b"fake artifact\\n")
''')
compiler.chmod(0o700)
raw = work / "raw profiles"
profile = work / "merged profile.profdata"
profile.write_bytes(b"fake profile for command-plan check\n")
command = [sys.executable, str(source / "research/build.py"), "--source", str(source),
           "--compiler", str(compiler), "--extra", str(source / "research/probe.cpp")]
records = {}
for mode, flags in (("none", []), ("generate", ["--profile-generate", str(raw)]),
                    ("use", ["--profile-use", str(profile)])):
    out = work / mode
    out.mkdir(exist_ok=True)
    (out / "calls.jsonl").write_text("")
    subprocess.run([*command, "--out", str(out), *flags], capture_output=True, check=True)
    record = json.loads((out / "build.json").read_text())
    calls = [json.loads(line) for line in (out / "calls.jsonl").read_text().splitlines()]
    assert record["status"] == "built" and record["commands"] == calls
    assert record["profile"]["mode"] == mode
    assert record["profile"]["extra_main"] == "unprofiled-no-lto"
    records[mode] = record

# The default compiler argv remain identical to the original development plan.
original = subprocess.check_output(
    ["make", "-n", "-B", "cmix", f"CC={compiler}",
     "CFLAGS_DEFINES=-DSEED=923 -DUPDATE_LIMIT=3000 "], cwd=source, text=True)
original = [[str(source / arg) if arg.startswith(("src/", "cpp_infer/")) else arg
             for arg in shlex.split(line)]
            for line in original.splitlines() if line.startswith(str(compiler) + " ")]
baseline = records["none"]["commands"]
assert baseline[:-2] == original and len(original) == 12
objects = [arg for arg in original[-1] if arg.endswith(".o")]
assert len(objects) == 39 and sum(arg.startswith("tf_") for arg in objects) == 9
assert sum(arg.endswith(".cpp") for cmd in original[:-1] for arg in cmd) == 39
runner = next(cmd for cmd in original if str(source / "src/runner.cpp") in cmd)
runner_flags = runner[1:next(i for i, arg in enumerate(runner) if arg.endswith(".cpp"))]
assert baseline[-2] == [str(compiler), *runner_flags, "-I", str(source / "src"), "-c",
                        str(source / "research/probe.cpp"), "-o", "probe_research.o"]
assert baseline[-1] == [str(compiler), "-m64", "-Wl,--gc-sections", "-std=c++17",
                        *[arg for arg in objects if arg != "runner.o"],
                        "probe_research.o", "-o", "probe"]

for mode, record in records.items():
    expected = ([] if mode == "none" else [f"-fprofile-generate={raw}"] if mode == "generate"
                else [f"-fprofile-use={profile}", "-flto"])
    plan = record["commands"]
    for old, new in zip(baseline[:-2], plan[:-2]):
        transformer = "-c" in old and old[-1].startswith("tf_")
        pgo = [arg for arg in new if arg.startswith("-fprofile-") or arg == "-flto"]
        assert pgo == ([] if transformer else expected)
        assert [arg for arg in new if arg not in pgo] == old
    assert plan[-2] == baseline[-2]  # Stable, unprofiled, non-LTO caller.
    pgo = [arg for arg in plan[-1] if arg.startswith("-fprofile-") or arg == "-flto"]
    assert pgo == expected and [arg for arg in plan[-1] if arg not in pgo] == baseline[-1]
assert records["generate"]["profile"]["path"] == str(raw) and raw.is_dir()
assert records["use"]["profile"]["path"] == str(profile)
assert records["use"]["profile"]["sha256"] == hashlib.sha256(profile.read_bytes()).hexdigest()
invalid = subprocess.run([*command, "--out", str(work / "invalid"),
                          "--profile-generate", str(raw), "--profile-use", str(profile)],
                         capture_output=True)
assert invalid.returncode == 2
print(json.dumps({"status": "passed", "modes": list(records), "ordinary_objects": 30,
                  "transformer_objects": 9, "extra_main": "unprofiled-no-lto",
                  "default_commands_unchanged": True}))
