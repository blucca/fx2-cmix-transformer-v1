#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""One fixed CPU LSTM32 fit; only the two 768 KiB training prefixes are read.

--self-test uses synthetic files under ./temp. --probe measures synthetic
forward/backward throughput. The C++ replay, after this fit, owns tail scoring.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import tempfile
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TRACE = np.dtype([("p", "<u2"), ("flags", "<u2"),
                  ("neural", "<f4"), ("fxcm", "<f4"), ("mixer", "<f4")])
WIDTH, HIDDEN, SEQ, BATCH = 78, 32, 64, 128
ROWS, TRAIN_ROWS, BLOCK_ROWS = 1048576 * 8, 786432 * 8, 131072 * 8
SEED, EPOCHS, PARAMETERS = 237, 10, 11009
LR, LR_FLOOR, CPU_LIMIT = 1e-3, 1e-5, 7200.0
STEPS_PER_EPOCH = 2 * TRAIN_ROWS // (BATCH * SEQ)
WEIGHT_ORDER = ("in_proj.weight", "in_proj.bias", "lstm.weight_ih_l0",
                "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0",
                "out.weight", "out.bias")


class LSTMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(WIDTH, HIDDEN)
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, offset):
        # No carried h/c: each 64-bit sequence starts from exactly zero.
        hidden, _ = self.lstm(F.silu(self.in_proj(x)))
        return offset + self.out(hidden).squeeze(-1)


def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)


def protocol():
    return {"schema": "hutter.bit-recurrent-fit.v1", "seed": SEED,
            "features": WIDTH, "hidden": HIDDEN, "parameters": PARAMETERS,
            "sequence_bits": SEQ, "batch_sequences": BATCH, "epochs": EPOCHS,
            "case_rows": ROWS, "train_rows_per_case": TRAIN_ROWS,
            "block_rows": BLOCK_ROWS, "optimizer": "AdamW", "weight_decay": 0,
            "lr": LR, "lr_floor": LR_FLOOR, "gradient_clip_norm": 1.0,
            "lr_schedule": "floor+(base-floor)/(1+completed_training_epochs)",
            "gate_order": "ifgo", "activation": "SiLU",
            "state_reset": "zero h/c for every absolute aligned 64-bit sequence",
            "previous_bit": "feature[77], true preceding coded bit; case start=0.5",
            "offset": "log(p/(65536-p)) evaluated in f64 then cast to f32",
            "override": "zero loss weight; features and recurrent transitions retained",
            "sampling": "seed+epoch block permutation, Torch RNG sequence permutation",
            "holdout": "last 256 KiB per case, never mapped/read by this program",
            "torch_version": str(torch.__version__),
            "mkldnn": bool(torch.backends.mkldnn.enabled), "threads": 1}


def atomic_save(path, value, *, tensor=False):
    pending = path.with_name(path.name + ".pending")
    with pending.open("wb" if tensor else "w") as f:
        if tensor:
            torch.save(value, f)
        else:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    pending.replace(path)


def case_info(trace_path, feature_path):
    result = {"train_rows": TRAIN_ROWS}
    for name, path, size in (("trace", trace_path, ROWS * TRACE.itemsize),
                             ("features", feature_path, ROWS * WIDTH * 4)):
        path = Path(path).resolve()
        st = path.stat()
        if st.st_size != size:
            raise ValueError(f"{name}: expected {size} bytes, got {st.st_size}")
        result[name] = str(path)
        result[name + "_size"] = st.st_size
        result[name + "_mtime_ns"] = st.st_mtime_ns
    return result


@contextmanager
def training_block(case, start, count):
    if start < 0 or count <= 0 or start % SEQ or count % SEQ or start + count > case["train_rows"]:
        raise ValueError("block must stay in the aligned training prefix")
    records = np.memmap(case["trace"], dtype=TRACE, mode="r",
                        offset=start * TRACE.itemsize, shape=(count,))
    features = None
    try:
        features = np.memmap(case["features"], dtype="<f4", mode="r",
                             offset=start * WIDTH * 4, shape=(count, WIDTH))
        previous = (0.5 if start == 0 else float(np.fromfile(
            case["trace"], dtype=TRACE, count=1,
            offset=(start - 1) * TRACE.itemsize)["flags"][0] & 1))
        yield records, features, previous
    finally:
        # One block at a time bounds resident file pages to about 328 MiB.
        if features is not None:
            features._mmap.close()
        records._mmap.close()


def batch(block, indices):
    records, features, previous = block
    indices = np.asarray(indices, dtype=np.int64)
    x = np.ascontiguousarray(features.reshape(-1, SEQ, WIDTH)[indices])
    r = records.reshape(-1, SEQ)[indices]
    if np.any(r["p"] == 0) or np.any(r["flags"] & ~np.uint16(3)) or not np.isfinite(x).all():
        raise ValueError("invalid training probability, flags or features")
    y = (r["flags"] & 1).astype(np.float32)
    prior = np.empty(len(indices), dtype=np.float32)
    first = indices == 0
    prior[first] = previous
    prior[~first] = records["flags"][indices[~first] * SEQ - 1] & 1
    if not np.array_equal(x[:, 0, -1], prior) or not np.array_equal(x[:, 1:, -1], y[:, :-1]):
        raise ValueError("feature[77] must be the strictly preceding coded bit")
    p = r["p"].astype(np.float64)
    offset = np.log(p / (65536.0 - p)).astype(np.float32)
    mask = ((r["flags"] & 2) == 0).astype(np.float32)
    return tuple(torch.from_numpy(a) for a in (x, offset, y, mask))


def train_step(net, optimizer, data):
    x, offset, y, mask = data
    optimizer.zero_grad(set_to_none=True)
    logits = net(x, offset)
    losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    count = mask.sum()
    loss = (losses * mask).sum() / count.clamp_min(1)
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        base = F.binary_cross_entropy_with_logits(offset, y, reduction="none")
        return {"scored_bits": int(count),
                "base_nll_bits": float((base * mask).double().sum()) / math.log(2),
                "head_nll_bits": float((losses * mask).double().sum()) / math.log(2),
                "gradient_norm": float(norm)}


def set_lr(optimizer, completed_steps):
    lr = LR_FLOOR + (LR - LR_FLOOR) / (1 + completed_steps / STEPS_PER_EPOCH)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def empty_metrics():
    return {"scored_bits": 0, "base_nll_bits": 0.0, "head_nll_bits": 0.0}


def initial_progress():
    return {"epoch": 0, "block_visit": 0, "sequence_pos": 0, "step": 0,
            "block_order": None, "sequence_order": None,
            "epoch_online": empty_metrics(), "epochs_online": []}


def snapshot(net, optimizer, progress, cases, cpu_seconds, wall_seconds, status):
    return {"protocol": protocol(), "cases": cases, "model": net.state_dict(),
            "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "progress": progress, "cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds, "status": status}


def restore(checkpoint, net, optimizer):
    net.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng"])
    return checkpoint["progress"]


def flat_weights(net):
    state = net.state_dict()
    flat = np.concatenate([state[name].detach().cpu().numpy().ravel()
                           for name in WEIGHT_ORDER]).astype("<f4")
    if flat.size != PARAMETERS or not np.isfinite(flat).all():
        raise ValueError("invalid final weights")
    return flat


def fit(args):
    started_cpu, started_wall = time.process_time(), time.monotonic()
    cases = [case_info(*paths) for paths in args.case]
    net = LSTMHead()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0)
    progress, before_cpu, before_wall = initial_progress(), 0.0, 0.0
    latest = args.out / "latest.pt"
    if args.resume:
        ck = torch.load(latest, map_location="cpu", weights_only=True)
        if ck["protocol"] != protocol() or ck["cases"] != cases:
            raise ValueError("resume requires the same protocol, inputs and Torch runtime")
        progress = restore(ck, net, optimizer)
        before_cpu, before_wall = ck["cpu_seconds"], ck["wall_seconds"]
        if ck["status"] == "complete":
            print((args.out / "fit.json").read_text(), flush=True)
            return
    else:
        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError("output is nonempty; choose a fresh directory or use --resume")
        args.out.mkdir(parents=True, exist_ok=True)

    stopped = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: stopped.append(number))

    def save(status="running", epoch_file=False):
        ck = snapshot(net, optimizer, progress, cases,
                      before_cpu + time.process_time() - started_cpu,
                      before_wall + time.monotonic() - started_wall, status)
        if epoch_file:
            atomic_save(args.out / f"epoch-{progress['epoch']:02d}.pt", ck, tensor=True)
        atomic_save(latest, ck, tensor=True)
        summary = {"status": status, "completed_epochs": progress["epoch"],
                   "step": progress["step"], "block_visit": progress["block_visit"],
                   "sequence_pos": progress["sequence_pos"], "cpu_seconds": ck["cpu_seconds"],
                   "wall_seconds": ck["wall_seconds"], "cpu_limit_seconds": args.cpu_seconds,
                   "lr": optimizer.param_groups[0]["lr"],
                   "rss_high_water_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "epoch_online": progress["epoch_online"]}
        atomic_save(args.out / "status.json", summary)
        print(json.dumps(summary, allow_nan=False), flush=True)
        return ck

    save()
    blocks_per_case = TRAIN_ROWS // BLOCK_ROWS
    nseq = BLOCK_ROWS // SEQ
    while progress["epoch"] < EPOCHS:
        if progress["block_order"] is None:
            generator = torch.Generator().manual_seed(SEED + progress["epoch"])
            progress["block_order"] = torch.randperm(2 * blocks_per_case, generator=generator).tolist()
        while progress["block_visit"] < len(progress["block_order"]):
            block_id = progress["block_order"][progress["block_visit"]]
            case_id, block_id = divmod(block_id, blocks_per_case)
            if progress["sequence_order"] is None:
                progress["sequence_order"] = torch.randperm(nseq)
            with training_block(cases[case_id], block_id * BLOCK_ROWS, BLOCK_ROWS) as block:
                while progress["sequence_pos"] < nseq:
                    cpu = before_cpu + time.process_time() - started_cpu
                    if stopped or cpu >= args.cpu_seconds:
                        save("interrupted" if stopped else "cpu_budget_exhausted")
                        return
                    k = progress["sequence_pos"]
                    indices = progress["sequence_order"][k:k + BATCH].numpy()
                    metrics = train_step(net, optimizer, batch(block, indices))
                    for key in progress["epoch_online"]:
                        progress["epoch_online"][key] += metrics[key]
                    progress["sequence_pos"] += len(indices)
                    progress["step"] += 1
                    set_lr(optimizer, progress["step"])
                    if progress["step"] % 64 == 0:
                        save()
            progress["block_visit"] += 1
            progress["sequence_pos"] = 0
            progress["sequence_order"] = None
        progress["epochs_online"].append(dict(epoch=progress["epoch"] + 1,
                                              **progress["epoch_online"]))
        progress["epoch"] += 1
        progress["block_visit"] = 0
        progress["block_order"] = None
        progress["epoch_online"] = empty_metrics()
        save(epoch_file=True)

    weights = flat_weights(net)
    pending = args.out / "weights.f32.pending"
    with pending.open("wb") as f:
        f.write(weights.tobytes())
        f.flush()
        os.fsync(f.fileno())
    pending.replace(args.out / "weights.f32")
    report = {**protocol(), "status": "complete", "cases": cases,
              "completed_epochs": progress["epoch"], "steps": progress["step"],
              "cpu_seconds": before_cpu + time.process_time() - started_cpu,
              "wall_seconds": before_wall + time.monotonic() - started_wall,
              "rss_high_water_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "weights_bytes": weights.nbytes,
              "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
              "weight_order": [{"name": name, "shape": list(net.state_dict()[name].shape)}
                               for name in WEIGHT_ORDER],
              "epochs_online": progress["epochs_online"],
              "measurement": "training online NLL only; final C++ integer tail replay decides"}
    atomic_save(args.out / "fit.json", report)
    save("complete")
    print(json.dumps(report, allow_nan=False), flush=True)


def synthetic_data():
    x = torch.randn(BATCH, SEQ, WIDTH)
    y = torch.randint(2, (BATCH, SEQ)).float()
    x[:, 0, -1] = 0.5
    x[:, 1:, -1] = y[:, :-1]
    p = torch.randint(1, 65536, (BATCH, SEQ)).double()
    offset = torch.log(p / (65536 - p)).float()
    mask = (torch.rand(BATCH, SEQ) > 0.1).float()
    return x, offset, y, mask


def probe(steps):
    net = LSTMHead()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0)
    data = synthetic_data()
    for _ in range(4):
        train_step(net, optimizer, data)
    cpu, wall = time.process_time(), time.monotonic()
    for step in range(steps):
        train_step(net, optimizer, data)
        set_lr(optimizer, step + 1)
    cpu, wall = time.process_time() - cpu, time.monotonic() - wall
    print(json.dumps({"status": "probe_complete", "torch_version": str(torch.__version__),
                      "steps": steps, "batch_sequences": BATCH, "sequence_bits": SEQ,
                      "parameters": sum(p.numel() for p in net.parameters()),
                      "threads": torch.get_num_threads(), "mkldnn": torch.backends.mkldnn.enabled,
                      "cpu_seconds": cpu, "wall_seconds": wall,
                      "cpu_seconds_per_step": cpu / steps,
                      "projected_10epoch_compute_cpu_seconds": cpu / steps * STEPS_PER_EPOCH * EPOCHS,
                      "projection_excludes": "file IO, per-batch validation and checkpoints",
                      "rss_high_water_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)


def self_test():
    net = LSTMHead()
    assert sum(p.numel() for p in net.parameters()) == PARAMETERS
    assert flat_weights(net).nbytes == 44036
    data = synthetic_data()
    assert torch.equal(net(data[0], data[1]), data[1])
    # A masked label has zero direct loss/gradient, while its feature is retained.
    logits = torch.randn(2, SEQ, requires_grad=True)
    labels = torch.randint(2, logits.shape).float()
    mask = torch.ones_like(labels)
    mask[:, 5] = 0
    loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * mask).sum()
    altered = labels.clone()
    altered[:, 5] = 1 - altered[:, 5]
    assert torch.equal(loss, (F.binary_cross_entropy_with_logits(logits, altered,
                                                                reduction="none") * mask).sum())
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 5]) == 0

    scratch = Path.cwd() / "temp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bit-recurrent-self-test-", dir=scratch) as folder:
        folder = Path(folder)
        ntrain, nrows = 2 * SEQ, 3 * SEQ
        records = np.zeros(nrows, dtype=TRACE)
        records["p"][:ntrain] = 32768
        records["flags"][:ntrain] = np.arange(ntrain) % 2
        records["flags"][3] |= 2
        records["flags"][ntrain:] = 65535  # Poisoned tail must stay unread.
        features = np.zeros((nrows, WIDTH), dtype="<f4")
        features[0, -1] = 0.5
        features[1:ntrain, -1] = records["flags"][:ntrain - 1] & 1
        features[ntrain:] = np.nan
        records.tofile(folder / "trace.bin")
        features.tofile(folder / "features.f32")
        case = {"trace": str(folder / "trace.bin"), "features": str(folder / "features.f32"),
                "train_rows": ntrain}
        with training_block(case, 0, ntrain) as block:
            x, offset, y, w = batch(block, [1, 0])
            assert x.shape == (2, SEQ, WIDTH) and w.sum() == ntrain - 1
            assert x[1, 4, -1] == y[1, 3] and w[1, 3] == 0
            assert offset.count_nonzero() == 0
        with training_block(case, SEQ, SEQ) as block:
            assert batch(block, [0])[0][0, 0, -1] == records["flags"][SEQ - 1] & 1
        try:
            with training_block(case, ntrain, SEQ):
                raise AssertionError("tail mapping allowed")
        except ValueError:
            pass

        optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0)
        train_step(net, optimizer, data)
        set_lr(optimizer, 1)
        progress = initial_progress()
        progress.update(step=1, sequence_pos=BATCH, block_order=[0, 1],
                        sequence_order=torch.randperm(256))
        ck = snapshot(net, optimizer, progress, [], 12.5, 13.0, "running")
        atomic_save(folder / "resume.pt", ck, tensor=True)
        expected_order = torch.randperm(64)
        train_step(net, optimizer, data)
        expected = flat_weights(net)
        loaded = torch.load(folder / "resume.pt", map_location="cpu", weights_only=True)
        resumed = LSTMHead()
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=LR, weight_decay=0)
        restored = restore(loaded, resumed, resumed_optimizer)
        assert restored["step"] == 1 and restored["sequence_pos"] == BATCH
        assert torch.equal(restored["sequence_order"], progress["sequence_order"])
        assert loaded["cpu_seconds"] == 12.5
        assert torch.equal(torch.randperm(64), expected_order)
        train_step(resumed, resumed_optimizer, data)
        assert np.array_equal(flat_weights(resumed), expected)
    print(json.dumps({"status": "self_test_passed", "parameters": PARAMETERS,
                      "zero_head_exact": True, "override_mask": True,
                      "strict_previous_bit": True, "train_only_mapping": True,
                      "checkpoint_next_step_exact": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, action="append", metavar=("TRACE", "FEATURES"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu-seconds", type=float, default=CPU_LIMIT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-steps", type=int, default=64)
    args = parser.parse_args()
    if not 0 < args.cpu_seconds <= CPU_LIMIT or args.probe_steps <= 0:
        parser.error("CPU budget must be in (0,7200], probe steps must be positive")
    if not (args.self_test or args.probe) and (not args.case or len(args.case) != 2 or args.out is None):
        parser.error("provide two --case TRACE FEATURES and --out")
    setup()
    if args.self_test:
        self_test()
    elif args.probe:
        probe(args.probe_steps)
    else:
        fit(args)


if __name__ == "__main__":
    main()
