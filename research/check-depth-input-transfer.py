#!/usr/bin/env python3
"""Run from the workspace root; scratch files use ./temp. No network or models."""
import importlib.util
import io
from pathlib import Path
import tempfile

spec = importlib.util.spec_from_file_location("transfer", Path(__file__).with_name("depth-input-transfer.py"))
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)
Path("temp").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir="temp") as directory:
    destination = Path(directory) / "input"
    stream = bytes(range(251)) * 9000
    transfer.copy_stream(io.BytesIO(stream), destination, len(stream))
    assert destination.read_bytes() == stream
    destination.write_bytes(stream[:377])
    transfer.copy_stream(io.BytesIO(stream), destination, len(stream), 377)
    assert destination.read_bytes() == stream
    destination.write_bytes(stream[:377])
    try:
        transfer.copy_stream(io.BytesIO(stream[:1999]), destination, len(stream), 377)
    except AssertionError as error:
        assert "expected" in str(error) and destination.stat().st_size == 1999
    else:
        raise AssertionError("truncated stream accepted")
    destination.write_bytes(stream[:377])
    try:
        transfer.copy_stream(io.BytesIO(stream[:30]), destination, len(stream), 377)
    except AssertionError as error:
        assert "saved prefix" in str(error) and destination.stat().st_size == 377
    else:
        raise AssertionError("truncated prefix accepted")
print("depth input transfer checks PASS")
