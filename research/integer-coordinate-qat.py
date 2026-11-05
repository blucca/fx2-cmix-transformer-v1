#!/usr/bin/env python3
"""Fixed-scale integer-coordinate CPU QAT; the original static15 format stays intact.

Only 111 bounded FP32 integer coordinates learn. Whole-piece target means:
CE + KL(original W || student) + 2 H(z)/586459020. Projected AdamW(.01),
tau=.5, <=64 epochs / 48 CPUh, including export and native evaluation.
causal-v1 keeps every optimizer constant and uses 16 complete representative epochs.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
FULL_TARGETS = 586459020
RECIPE = dict(method="fixed-scale-integer-coordinate-v1", epochs=64, lr=.01,
              betas=[.9, .99], eps=1e-8, weight_decay=0, tau=.5,
              teacher_kl_coefficient=1, full_targets=FULL_TARGETS, seed=245)
# Reuse the original whole-piece >=32768-target batching, without its old main.
_spec = importlib.util.spec_from_file_location(
    "original_entropy_qat", Path(__file__).with_name("entropy-qat.py"))
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)
batches = _original.batches


def causal_batches(order):
    """Merge the short final batch; each piece still frees its own backward graph.

    Ordinary batches hold >=32768 targets. A tiny tail would otherwise receive
    one full Adam step after its own target mean, overweighting its few targets.
    Keep the original fixed-scale recipe's batching unchanged.
    """
    pending = None
    for batch, targets in _original.batches(order):
        if pending is not None:
            if targets < _original.BATCH_TARGETS:
                batch = pending[0] + batch
                targets += pending[1]
            else:
                yield pending
        pending = batch, targets
    if pending is not None:
        yield pending


def entropy(z, tau=.5):
    """Per-tensor soft code length in nats, directly in bounded integer units."""
    import torch
    centers = torch.arange(-7, 8, dtype=z.dtype, device=z.device)
    p = (-((z.flatten()[:, None] - centers) ** 2) / tau).softmax(-1).mean(0)
    return -z.numel() * (p * p.clamp_min(1e-12).log()).sum()


def integer_weight(z, scale):
    q = z + (z.round().clamp(-7, 7) - z).detach()
    return q * scale[:, None]


def install_coordinates(model):
    """Freeze the original state; bind external z only to weight-quantizer calls."""
    import torch
    model.requires_grad_(False)
    original = {k: p.detach().clone() for k, p in model.named_parameters()}
    assert len(original) == 429 and sum(p.numel() for p in original.values()) == 5923228
    coordinates, scales, original_q = {}, {}, {}
    for key, w in original.items():
        scale_key = key[:-7] + ".quantize_weight.scale"
        if not key.endswith(".weight") or scale_key not in original:
            continue
        quantizer = model.get_submodule(scale_key[:-6])
        assert (quantizer.quant.buckets == 15 and quantizer.quant.epsilon is None
                and not quantizer.batched and quantizer.quant.block_size == w.shape[1])
        s = original[scale_key].to(torch.bfloat16).float()
        assert s.shape == (w.shape[0],) and bool(torch.isfinite(s).all() & (s != 0).all())
        q = (w / s[:, None]).round().clamp(-7, 7).to(torch.int8)
        assert torch.equal(quantizer(w, None, False), q.float() * s[:, None]), key
        z = torch.nn.Parameter(q.float())
        coordinates[key], scales[key], original_q[key] = z, s, q

        def forward(input, initialize_scales_steps, return_ints, z=z, s=s):
            assert initialize_scales_steps is None and input.shape == z.shape
            if return_ints:
                return z.detach().round().clamp(-7, 7).to(torch.int64)
            return integer_weight(z, s).to(input.dtype)

        quantizer.forward = forward
    assert len(coordinates) == 111 and sum(z.numel() for z in coordinates.values()) == 5868864
    assert len(model.state_dict()) == 429
    return coordinates, scales, original_q, original


def export_state(model, coordinates, scales):
    state = model.state_dict()
    for key, z in coordinates.items():
        state[key] = z.detach().round().clamp(-7, 7) * scales[key][:, None]
    return state


def long_piece_score(result, indices):
    """Target-weighted native loss of the held pieces outside the train-length domain."""
    losses = [result["piece_losses"][i]["loss"] for i in indices]
    targets = sum(row["count"] for row in losses)
    assert targets > 0
    return dict(targets=targets,
                raw_nll=sum(row["logits_mean"] * row["count"] for row in losses) / targets)


def self_check():
    """Signed/tiny-scale STE and finite direct-entropy gradient in integer units."""
    order = [(0, 0, 37001), (0, 37001, 37106), (0, 37106, 37245)]
    merged = list(causal_batches(order))
    assert [n for _, n in _original.batches(order)] == [37000, 242]
    assert [n for _, n in merged] == [37242]
    assert [piece for batch, _ in merged for piece in batch] == order
    import torch
    z = torch.tensor([[-7., -2., 1.], [0., 3., 7.]], requires_grad=True)
    s = torch.tensor([6e-15, -8e-15]).to(torch.bfloat16).float()
    probe = torch.tensor([[1., 2., -3.], [4., -5., 6.]])
    out = integer_weight(z, s)
    assert torch.equal(out, z.detach() * s[:, None])
    torch.testing.assert_close(torch.autograd.grad((out * probe).sum(), z)[0],
                               probe * s[:, None], atol=0, rtol=0)
    h = entropy(z)
    g, = torch.autograd.grad(h, z)
    assert bool(torch.isfinite(h) & torch.isfinite(g).all()) and bool((g != 0).any())
    projected = torch.tensor([-7.1, 7.1]).clamp_(-7, 7)
    assert projected.tolist() == [-7., 7.]
    assert long_piece_score({"piece_losses": [{"loss": {"count": 2, "logits_mean": 1.}},
                                              {"loss": {"count": 6, "logits_mean": 3.}}]},
                            [0, 1]) == {"targets": 8, "raw_nll": 2.5}


def main():
    global RECIPE, batches
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "kernel", "data", "baseline", "weights", "recode", "evaluator", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu-seconds", type=int, default=172800)
    parser.add_argument("--recipe", choices=("fixed-scale-v1", "causal-v1"), default="fixed-scale-v1",
                        help="causal-v1: independent whole-stream priors, 16 complete epochs; all optimizer constants fixed")
    parser.add_argument("--prior-cpu-seconds", type=float, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recovery-cpu-seconds", type=float,
                        help="CPU since last saved state after an unclean exit; supply an upper bound")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--zero-only", action="store_true")
    modes.add_argument("--check-only", action="store_true",
                       help="full zero-step signature and one complete 1025-2048-target piece update; restore zero")
    args = parser.parse_args()
    if args.recipe == "causal-v1":
        RECIPE = dict(RECIPE, method="causal-fixed-scale-integer-coordinate-v1", epochs=16, seed=414,
                      final_batch="merge-short-tail")
        batches = causal_batches
    assert 600 <= args.cpu_seconds <= 172800 and args.prior_cpu_seconds >= 0
    assert not (args.resume and (args.zero_only or args.check_only))
    job = args.output.resolve()
    job.mkdir(parents=True, exist_ok=True)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    for key in ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        os.environ[key] = str(job / key.lower())
        Path(os.environ[key]).mkdir(exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (7 << 30, 7 << 30))
    started = time.monotonic()
    baseline = json.loads(args.baseline.read_text())
    assert baseline["status"] == "passed"
    identity = dict(checkpoint=str(args.checkpoint.resolve()), data=str(args.data.resolve()),
                    weights=str(args.weights.resolve()), recipe=RECIPE)
    state_path, checkpoint_path = job / "state.json", job / "latest.pt"
    cursor = ("step", "epoch", "next_piece", "seen_targets", "epoch_targets", "epoch_ce_sum", "epoch_kl_sum")
    if args.resume:
        state = json.loads(state_path.read_text())
        assert state["identity"] == identity and checkpoint_path.is_file()
        assert state["status"] not in ("complete", "failed", "cpu_budget")
        if state["status"] == "running":
            assert args.recovery_cpu_seconds is not None, "account for CPU since the last saved state"
        recovery = args.recovery_cpu_seconds or 0.
        assert recovery >= 0
        cpu_base = state["total_cpu_seconds"] + recovery
    else:
        assert not state_path.exists() and not checkpoint_path.exists(), "inspect trajectory; use --resume"
        cpu_base = baseline["child_cpu_seconds"] + args.prior_cpu_seconds
        state = {"status": "starting", "started_at": time.time(), "identity": identity,
                 "step": 0, "epoch": 0, "next_piece": 0, "seen_targets": 0,
                 "epoch_targets": 0, "epoch_ce_sum": 0., "epoch_kl_sum": 0.,
                 "checkpoint_step": 0, "evaluations": [], "cpu_limit_seconds": args.cpu_seconds,
                 "address_space_limit_gib": 7}
    remaining = math.floor(args.cpu_seconds - cpu_base)
    assert remaining >= 180, "CPU budget exhausted"
    state["cpu_limit_seconds"] = args.cpu_seconds
    resource.setrlimit(resource.RLIMIT_CPU, (remaining - 2, remaining))

    def cpu():
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        return cpu_base + time.process_time() + child.ru_utime + child.ru_stime

    def save(stage, **updates):
        state.update(stage=stage, updated_at=time.time(), total_cpu_seconds=cpu(),
                     run_wall_seconds=time.monotonic() - started,
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **updates)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(state_path)

    def event(kind, **values):
        row = dict(kind=kind, at=time.time(), step=state["step"], epoch=state["epoch"],
                   total_cpu_seconds=cpu())
        row.update(values)
        with (job / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def cpu_limit(signum, frame):
        save("cpu_limit", status="cpu_budget")
        raise SystemExit(124)

    signal.signal(signal.SIGXCPU, cpu_limit)
    save("importing", status="running")
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from cpu_model import load_model
        from cpu_weights_export import export_payloads

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(RECIPE["seed"])
        self_check()
        model, _ = load_model(args.checkpoint, args.kernel)
        assert model.cfg.dropout == 0 and model.cfg.vocabulary_size == 205
        z, scales, original_q, original = install_coordinates(model)
        optimizer = torch.optim.AdamW(list(z.values()), lr=RECIPE["lr"],
                                     betas=tuple(RECIPE["betas"]), eps=RECIPE["eps"],
                                     weight_decay=0, foreach=False)
        if args.resume:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            assert ckpt["identity"] == identity and ckpt["z"].keys() == z.keys()
            with torch.no_grad():
                for key, value in z.items():
                    value.copy_(ckpt["z"][key])
            optimizer.load_state_dict(ckpt["optimizer"])
            torch.set_rng_state(ckpt["rng"])
            state.update({key: ckpt[key] for key in cursor})
            del ckpt

        manifest = json.loads(args.data.read_text())
        samples = manifest["samples"]
        tokens = [np.fromfile(sample["tokens"], dtype=np.uint8) for sample in samples]
        pieces = [(i, lo, hi) for i, sample in enumerate(samples) for lo, hi in sample["train_pieces"]]
        assert len(pieces) == manifest["training_pieces"]
        assert sum(hi - lo - 1 for _, lo, hi in pieces) == manifest["training_targets"]
        for i, lo, hi in pieces:
            assert 0 <= lo < hi <= len(tokens[i]) and hi - lo > 1
        for sample, tok in zip(samples, tokens):
            assert Path(sample["priors"]).stat().st_size == len(tok) * 410
            name, score = sample["name"], sample["score"]
            assert baseline["steps"][name]["result"]["total"]["count"] == score["targets"]

        def read_piece(piece):
            i, lo, hi = piece
            n = hi - lo - 1
            x = torch.from_numpy(tokens[i][lo:hi - 1].astype(np.int64)).unsqueeze(0)
            y = torch.from_numpy(tokens[i][lo + 1:hi].astype(np.int64)).unsqueeze(0)
            prior = np.fromfile(samples[i]["priors"], dtype="<f2", count=n * 205,
                                offset=lo * 410).reshape(n, 205)
            p = torch.from_numpy(prior).unsqueeze(0)
            assert bool(torch.isfinite(p).all()) and int(x.max()) < 205 and int(y.max()) < 205
            return x, y, p, torch.tensor([[0, n]], dtype=torch.int32)

        cache = job / "teacher-cache"
        cache.mkdir(exist_ok=True)
        cache_identity = cache / "identity.json"
        if cache_identity.exists():
            assert json.loads(cache_identity.read_text()) == identity
        else:
            cache_identity.write_text(json.dumps(identity, indent=2) + "\n")
        teacher = None
        cache_misses = 0

        def teacher_logits(piece, x, p, bounds):
            nonlocal teacher, cache_misses
            i, lo, hi = piece
            path = cache / f"sample-{i:04d}-{lo}-{hi}.npy"
            if path.exists():
                array = np.load(path, allow_pickle=False)
                assert array.dtype == np.float32 and array.shape == (1, hi - lo - 1, 205)
                result = torch.from_numpy(array)
            else:
                if teacher is None:
                    teacher, _ = load_model(args.checkpoint, args.kernel)
                    teacher.requires_grad_(False).eval()
                    assert all(torch.equal(v, original[k]) for k, v in teacher.state_dict().items())
                with torch.no_grad():
                    result = teacher.compute_logits(input_tokens=x, priors=p,
                                                    cumulative_sequence_lengths=bounds).detach()
                assert result.dtype == torch.float32
                tmp = path.with_suffix(".tmp")
                with tmp.open("wb") as f:
                    np.save(f, result.numpy(), allow_pickle=False)
                tmp.replace(path)
                cache_misses += 1
            assert bool(torch.isfinite(result).all())
            return result

        def validate():
            for key, parameter in model.named_parameters():
                assert not parameter.requires_grad and parameter.grad is None, key
                assert torch.equal(parameter, original[key]), f"frozen original parameter changed: {key}"
                if key.endswith(".scale"):
                    assert torch.equal(parameter.detach().to(torch.bfloat16).view(torch.int16),
                                       original[key].to(torch.bfloat16).view(torch.int16)), key
            for key, value in z.items():
                assert bool(torch.isfinite(value).all() & (value >= -7).all() & (value <= 7).all()), key

        def geometry():
            hard_nats, soft_nats, displacement, max_delta = 0., 0., 0., 0.
            changed, changed_z, tiny = 0, 0, []
            q_min, q_max = 7, -7
            with torch.no_grad():
                for key, value in z.items():
                    q = value.round().clamp(-7, 7).to(torch.int8)
                    q_min, q_max = min(q_min, int(q.min())), max(q_max, int(q.max()))
                    delta = value - original_q[key].float()
                    changed += int((q != original_q[key]).sum())
                    changed_z += int((delta != 0).sum())
                    displacement += float(delta.double().square().sum())
                    max_delta = max(max_delta, float(delta.abs().max()))
                    p = torch.bincount(q.flatten().long() + 7, minlength=15).double() / q.numel()
                    hard_nats -= q.numel() * float((p * p.clamp_min(1e-30).log()).sum())
                    soft_nats += float(entropy(value))
                    bits = scales[key].to(torch.bfloat16).view(torch.uint16)
                    for row in torch.where(scales[key].abs() < 1e-10)[0].tolist():
                        tiny.append(dict(weight=key, row=row, signed_bf16_scale=float(scales[key][row]),
                                         bf16_bits=int(bits[row])))
            return dict(changed_weight_symbols=changed, changed_z_coordinates=changed_z,
                        hard_entropy_nats=hard_nats, current_soft_entropy_nats=soft_nats,
                        z_delta_l2=math.sqrt(displacement), z_max_delta=max_delta,
                        max_scale_delta=0., min_abs_bf16_weight_scale=min(float(s.abs().min()) for s in scales.values()),
                        tiny_weight_scale_count=len(tiny), tiny_weight_scales=tiny, hard_integer_range=[q_min, q_max])

        def checkpoint():
            state["checkpoint_step"] = state["step"]
            payload = {key: state[key] for key in cursor}
            payload.update(identity=identity, z={k: v.detach() for k, v in z.items()},
                           optimizer=optimizer.state_dict(), rng=torch.get_rng_state())
            tmp = checkpoint_path.with_suffix(".tmp")
            torch.save(payload, tmp)
            tmp.replace(checkpoint_path)
            save("checkpoint")

        def run(command, log):
            seconds = min(3600, math.floor(args.cpu_seconds - cpu()) - 2)
            assert seconds >= 10, "CPU budget exhausted before export/evaluation"
            def limits():
                resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
            with log.open("w") as output:
                subprocess.run(list(map(str, command)), stdout=output, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, preexec_fn=limits,
                               timeout=max(900, 3 * seconds), check=True)

        def export(label):
            raw = job / "payloads.bin"
            info = export_payloads(export_state(model, z, scales), raw)
            weight = job / f"{label}.tfwc3"
            run([args.recode, args.weights, weight, "--replace-payloads", raw], job / f"{label}-recode.log")
            info.update(weight_bytes=weight.stat().st_size, weight_file=str(weight))
            return weight, info

        def backward(batch, targets, exact_teacher=False):
            optimizer.zero_grad(set_to_none=True)
            ce_sum, kl_sum = 0., 0.
            for piece in batch:
                x, y, p, bounds = read_piece(piece)
                reference = teacher_logits(piece, x, p, bounds)
                logits = model.compute_logits(input_tokens=x, priors=p, cumulative_sequence_lengths=bounds)
                assert bool(torch.isfinite(logits).all())
                if exact_teacher:
                    assert torch.equal(logits.detach(), reference), "zero-step student/teacher logits differ"
                ce = F.cross_entropy(logits.view(-1, 205), y.flatten(), reduction="none").float().sum()
                kl = F.kl_div(logits.log_softmax(-1), reference.log_softmax(-1),
                              reduction="sum", log_target=True)
                assert bool(torch.isfinite(ce) & torch.isfinite(kl))
                ce_sum += float(ce.detach())
                kl_sum += float(kl.detach())
                (ce + kl).backward()
                del x, y, p, bounds, reference, logits, ce, kl
            for key, value in z.items():
                assert value.grad is not None and bool(torch.isfinite(value.grad).all()), key
                value.grad.div_(targets)
            soft_nats = 0.
            for value in z.values():
                h = entropy(value)
                assert bool(torch.isfinite(h))
                soft_nats += float(h.detach())
                (2 * h / FULL_TARGETS).backward()
            norm_squared, nonzero = 0., 0
            for key, value in z.items():
                assert bool(torch.isfinite(value.grad).all()), key
                norm_squared += float(value.grad.double().square().sum())
                nonzero += int(bool((value.grad != 0).any()))
            return dict(ce_mean=ce_sum / targets, teacher_kl_mean=kl_sum / targets,
                        soft_entropy_nats=soft_nats, objective=(ce_sum + kl_sum) / targets + 2 * soft_nats / FULL_TARGETS,
                        gradient_norm=math.sqrt(norm_squared), finite_gradient_tensors=len(z),
                        nonzero_gradient_tensors=nonzero)

        def update():
            optimizer.step()
            with torch.no_grad():
                for value in z.values():
                    value.clamp_(-7, 7)
            validate()

        def evaluate():
            epoch = state["epoch"] + 1
            weight, info = export(f"epoch-{epoch:02d}")
            scores = {}
            for sample in samples:
                name, score = sample["name"], sample["score"]
                log = job / f"epoch-{epoch:02d}-{name}.log"
                save("evaluating", evaluating_sample=name)
                command = [args.evaluator, "eval-pieces" if "boundaries" in score else "eval",
                           score["tokens"], score["priors"], weight]
                if "boundaries" in score:
                    command.append(score["boundaries"])
                run(command, log)
                rows = [json.loads(line.split(" ", 1)[1]) for line in log.read_text().splitlines()
                        if line.startswith("ACTIVATION_RESULT ")]
                assert len(rows) == 1 and rows[0]["total"]["count"] == score["targets"]
                raw = rows[0]["total"]["logits_mean"]
                assert math.isfinite(raw)
                scores[name] = dict(targets=score["targets"], raw_nll=raw,
                                    delta_raw_nll=raw - baseline["steps"][name]["result"]["total"]["logits_mean"])
                if score.get("long_piece_indices"):
                    long = long_piece_score(rows[0], score["long_piece_indices"])
                    original_long = long_piece_score(baseline["steps"][name]["result"], score["long_piece_indices"])
                    assert long["targets"] == original_long["targets"]
                    long["delta_raw_nll"] = long["raw_nll"] - original_long["raw_nll"]
                    scores[name]["long_native"] = long
            row = dict(epoch=epoch, step=state["step"], **info, scores=scores,
                       double_weight_saved_bytes=2 * (state["original_static15_bytes"] - info["weight_bytes"]),
                       **geometry())
            state["evaluations"] = [r for r in state["evaluations"] if r["epoch"] != epoch] + [row]
            torch.save({k: v.detach() for k, v in z.items()}, job / f"epoch-{epoch:02d}-z.pt")
            event("epoch_evaluated", **row)

        validate()
        if not args.resume:
            weight, info = export("step-0000")
            reference = job / "original-static15.tfwc3"
            run([args.recode, args.weights, reference], job / "original-static15-recode.log")
            assert weight.read_bytes() == reference.read_bytes(), "integer-coordinate zero-step static15 differs"
            state["original_static15_bytes"] = info["weight_bytes"]
            state["zero_step"] = dict(**info, **geometry())
            event("zero_step_export_passed", **state["zero_step"])
            checkpoint()
        if args.check_only:
            piece = next(p for p in pieces if 1025 <= p[2] - p[1] - 1 <= 2048)
            n = piece[2] - piece[1] - 1
            rng = torch.get_rng_state()
            before = cpu()
            result = backward([piece], n, exact_teacher=True)
            assert result["teacher_kl_mean"] == 0 and result["nonzero_gradient_tensors"] == 111
            update()
            after = geometry()
            assert after["changed_z_coordinates"] > 0 and after["changed_weight_symbols"] == 0
            check_weight, check_info = export("check-step")
            assert check_weight.read_bytes() == (job / "step-0000.tfwc3").read_bytes()
            result.update(status="passed", piece=list(piece), targets=n, zero_step_logits_exact=True,
                          frozen_original_tensors=429, optimizer_tensors=111, **after, **check_info,
                          check_cpu_seconds=cpu() - before)
            (job / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
            event("one_step_check_passed", **result)
            with torch.no_grad():
                for key, value in z.items():
                    value.copy_(original_q[key])
            optimizer.state.clear()
            optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(rng)
            checkpoint()
            save("check_complete", status="ready", preflight="passed")
            return
        if args.zero_only:
            save("zero_step_complete", status="ready")
            return

        model.train()
        while state["epoch"] < RECIPE["epochs"]:
            epoch = state["epoch"]
            order = pieces.copy()
            random.Random(RECIPE["seed"] + epoch).shuffle(order)
            for batch, targets in batches(order[state["next_piece"]:]):
                if cpu() + 180 >= args.cpu_seconds or (job / "STOP").exists():
                    checkpoint()
                    save("bounded_stop", status="cpu_budget" if cpu() + 180 >= args.cpu_seconds else "stopped")
                    return
                save("training", batch_targets=targets, batch_pieces=len(batch))
                before_cpu, before_wall = cpu(), time.monotonic()
                row = backward(batch, targets)
                update()
                state["step"] += 1
                state["next_piece"] += len(batch)
                state["seen_targets"] += targets
                state["epoch_targets"] += targets
                state["epoch_ce_sum"] += row["ce_mean"] * targets
                state["epoch_kl_sum"] += row["teacher_kl_mean"] * targets
                row.update(targets=targets, pieces=len(batch), tau=RECIPE["tau"],
                           teacher_cache_misses_this_run=cache_misses,
                           step_cpu_seconds=cpu() - before_cpu, step_wall_seconds=time.monotonic() - before_wall)
                if state["step"] % 8 == 0:
                    row.update(geometry())
                    checkpoint()
                event("optimizer_step", **row)
                save("step_complete", last_step=row)
            assert state["epoch_targets"] == manifest["training_targets"]
            checkpoint()
            if cpu() + 120 >= args.cpu_seconds:
                save("evaluation_budget_stop", status="cpu_budget")
                return
            evaluate()
            state.update(epoch=epoch + 1, next_piece=0, epoch_targets=0, epoch_ce_sum=0., epoch_kl_sum=0.)
            checkpoint()
        save("trajectory_complete", status="complete")
        event("trajectory_complete")
    except Exception as exc:
        save("failed", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
