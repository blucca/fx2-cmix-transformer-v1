#!/usr/bin/env python3
"""Fetch the official enwik9 archive and extract its single benchmark file."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    result = root / "dataset.json"
    started = time.time()
    archive, output = root / "enwik9.zip", root / "enwik9"
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                    "--retry", "2", "--connect-timeout", "30", "--continue-at", "-",
                    "https://mattmahoney.net/dc/enwik9.zip", "--output", str(archive)], check=True)
    with zipfile.ZipFile(archive) as source:
        if source.getinfo("enwik9").file_size != 1_000_000_000:
            raise ValueError("Unexpected enwik9 size")
        part = output.with_suffix(".part")
        with source.open("enwik9") as incoming, part.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
        part.replace(output)
    digest = hashlib.md5()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    data = {"source": "https://mattmahoney.net/dc/enwik9.zip", "bytes": output.stat().st_size,
            "md5": digest.hexdigest(), "zip_crc_checked": True,
            "elapsed_seconds": time.time() - started}
    result.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
