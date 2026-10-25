#!/usr/bin/env python3
"""One fixed, resumable CPU entropy-QAT trajectory; original inference stays intact.

Recipe: whole pieces, >=32768 targets/update, AdamW(5e-5, .9/.99), no decay,
CE_mean + 2*H_nats/586459020, eight epochs, tau .5 -> .15 geometrically.
All work including native scoring shares the supplied cumulative CPU budget.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import time


FULL_TARGETS = 586459020
EPOCHS = 8
BATCH_TARGETS = 32768


def entropy(weight, scale, tau):
    """Per-tensor soft entropy; FP32 STE with the original weight-scale LSQ gain."""
    import torch
    rounded = scale.detach().to(torch.bfloat16).to(torch.float32)
    gain = 1 / math.sqrt(weight.numel() * 7)
    rounded = rounded + (scale - scale.detach()) * gain
    z = (weight / rounded[:, None]).flatten()
    centers = torch.arange(-7, 8, dtype=weight.dtype)
    # Cancel the shared -z² term before FP32 evaluation. Original learned
    # scales reach 6e-15 (|z| > 1e12); squared distances lose all center offsets.
    logits = (2 * z[:, None] * centers - centers.square()) / tau
    p = logits.softmax(-1).mean(0)
    return -weight.numel() * (p * p.clamp_min(1e-12).log()).sum()


def batches(order):
    batch, count = [], 0
    for piece in order:
        batch.append(piece)
        count += piece[2] - piece[1] - 1
        if count >= BATCH_TARGETS:
            yield batch, count
            batch, count = [], 0
    if batch:
        yield batch, count


def check():
    """One small check covers mean entropy, scale preconditioning and batching."""
    import torch
    w = torch.tensor([[-.7, .1, .8], [.4, -.6, .2]], requires_grad=True)
    s = torch.tensor([.25, -.5], requires_grad=True)
    h = entropy(w, s, .5)
    gw, gs = torch.autograd.grad(h, (w, s))
    p = (-((w / s[:, None]).flatten()[:, None] - torch.arange(-7, 8))**2 / .5).softmax(-1).mean(0)
    expected = -w.numel() * (p * p.log()).sum()
    ew, es = torch.autograd.grad(expected, (w, s))
    torch.testing.assert_close(h, expected)
    torch.testing.assert_close(gw, ew)
    torch.testing.assert_close(gs, es / math.sqrt(w.numel() * 7))
    tiny_w = torch.tensor([[.02, -.01], [.03, -.06]], requires_grad=True)
    tiny_s = torch.tensor([8e-15, 6e-15], requires_grad=True)
    saturated = entropy(tiny_w, tiny_s, .5)
    torch.testing.assert_close(saturated, torch.tensor(4 * math.log(2)))
    for g in torch.autograd.grad(saturated, (tiny_w, tiny_s)):
        assert bool((g == 0).all()), "saturated assignments need finite zero gradients"
    result = list(batches([(0, 0, 18001), (1, 0, 19001), (0, 0, 4)]))
    assert [n for _, n in result] == [37000, 3]
    assert sum(len(b) for b, _ in result) == 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="frozen data-manifest.json")
    parser.add_argument("--baseline", type=Path, required=True, help="completed baseline-state.json")
    parser.add_argument("--weights", type=Path, required=True, help="original v2 template")
    parser.add_argument("--recode", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-cpu-seconds", type=float, default=0,
                        help="setup/check CPU beyond the baseline's recorded child CPU")
    parser.add_argument("--cpu-seconds", type=int, default=43200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recovery-cpu-seconds", type=float,
                        help="CPU since last saved status after an unclean exit; supply an upper bound if unknown")
    parser.add_argument("--zero-only", action="store_true", help="export gate, save initial state, stop")
    args = parser.parse_args()
    assert 600 <= args.cpu_seconds <= 43200 and args.prior_cpu_seconds >= 0
    job = args.output.resolve()
    job.mkdir(parents=True, exist_ok=True)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    for key in ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        os.environ[key] = str(job / key.lower())
        Path(os.environ[key]).mkdir(exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
    started = time.monotonic()
    baseline = json.loads(args.baseline.read_text())
    assert baseline["status"] == "passed"
    state_path, checkpoint_path = job / "state.json", job / "latest.pt"
    if args.resume:
        state = json.loads(state_path.read_text())
        assert checkpoint_path.is_file()
        assert state["status"] not in ("complete", "failed", "cpu_budget")
        if state["status"] == "running":
            assert args.recovery_cpu_seconds is not None, "unclean exit: account for CPU since the last saved status"
        recovery = args.recovery_cpu_seconds or 0.
        assert recovery >= 0
        cpu_base = state["total_cpu_seconds"] + recovery
    else:
        assert not state_path.exists() and not checkpoint_path.exists(), "inspect existing trajectory; use --resume"
        cpu_base = baseline["child_cpu_seconds"] + args.prior_cpu_seconds
        state = {"status": "starting", "started_at": time.time(), "step": 0,
                 "checkpoint_step": 0, "epoch": 0, "next_piece": 0,
                 "seen_targets": 0, "epoch_targets": 0, "epoch_loss_sum": 0.,
                 "cpu_limit_seconds": args.cpu_seconds, "address_space_limit_gib": 6,
                 "initial_gate": "pending", "evaluations": []}
    remaining = math.floor(args.cpu_seconds - cpu_base)
    assert remaining >= 180, "CPU budget exhausted"
    # Child CPU is also included in cpu(); the hard limit bounds the parent itself.
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
        row = {"kind": kind, "at": time.time(), "step": state["step"],
               "epoch": state["epoch"], "total_cpu_seconds": cpu(), **values}
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
        from cpu_model import load_model
        from cpu_weights_export import export_payloads

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        check()
        torch.manual_seed(245)
        model, _ = load_model(args.checkpoint, args.kernel)
        params = dict(model.named_parameters())
        assert len(params) == 429 and sum(p.numel() for p in params.values()) == 5923228
        quantized = [(key, p, params[key[:-7] + ".quantize_weight.scale"])
                     for key, p in params.items()
                     if key.endswith(".weight") and key[:-7] + ".quantize_weight.scale" in params]
        assert len(quantized) == 111 and sum(p.numel() for _, p, _ in quantized) == 5868864
        original = {key: p.detach().clone() for key, p in params.items()}
        original_q = {key: (w.detach() / s.detach().to(torch.bfloat16).float()[:, None]).round().clamp(-7, 7).to(torch.int8)
                      for key, w, s in quantized}
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(.9, .99),
                                     eps=1e-8, weight_decay=0, foreach=False)
        if args.resume:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            torch.set_rng_state(ckpt["rng"])
            for key in ("step", "epoch", "next_piece", "seen_targets", "epoch_targets", "epoch_loss_sum"):
                state[key] = ckpt[key]
            del ckpt

        manifest = json.loads(args.data.read_text())
        assert manifest["training_targets"] == 1044469 and manifest["training_pieces"] == 807
        samples = manifest["samples"]
        tokens = [np.fromfile(sample["tokens"], dtype=np.uint8) for sample in samples]
        pieces = [(i, s, e) for i, sample in enumerate(samples) for s, e in sample["train_pieces"]]
        for i, s, e in pieces:
            assert 0 <= s < e <= len(tokens[i]) // 2 and e - s > 1
        assert sum(e - s - 1 for _, s, e in pieces) == manifest["training_targets"]
        for sample, tok in zip(samples, tokens):
            assert Path(sample["priors"]).stat().st_size == len(tok) * 410

        def checkpoint():
            state["checkpoint_step"] = state["step"]
            payload = {key: state[key] for key in
                       ("step", "epoch", "next_piece", "seen_targets", "epoch_targets", "epoch_loss_sum")}
            payload.update(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state())
            tmp = checkpoint_path.with_suffix(".tmp")
            torch.save(payload, tmp)
            tmp.replace(checkpoint_path)
            save("checkpoint")

        def run(command, log):
            seconds = min(300, math.floor(args.cpu_seconds - cpu()) - 2)
            assert seconds >= 10, "CPU budget exhausted before native evaluation"
            def limits():
                resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
            with log.open("w") as output:
                subprocess.run(list(map(str, command)), stdout=output, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, preexec_fn=limits, timeout=900, check=True)

        def export(label):
            raw = job / "payloads.bin"
            info = export_payloads(model.state_dict(), raw)
            weight = job / f"{label}.tfwc3"
            run([args.recode, args.weights, weight, "--replace-payloads", raw], job / f"{label}-recode.log")
            info["weight_bytes"] = weight.stat().st_size
            return weight, info

        def geometry(tau):
            changed, hard_nats, soft_nats, displacement = 0, 0., 0., 0.
            scale_delta, minimum_scale = 0., math.inf
            with torch.no_grad():
                for key, w, s in quantized:
                    q = (w / s.to(torch.bfloat16).float()[:, None]).round().clamp(-7, 7).to(torch.int8)
                    changed += int((q != original_q[key]).sum())
                    p = torch.bincount(q.flatten().long() + 7, minlength=15).double() / q.numel()
                    hard_nats -= q.numel() * float((p * p.clamp_min(1e-30).log()).sum())
                    soft_nats += float(entropy(w, s, tau))
                for key, p in params.items():
                    displacement += float((p - original[key]).square().sum())
                    if key.endswith(".scale"):
                        scale_delta = max(scale_delta, float((p - original[key]).abs().max()))
                        minimum_scale = min(minimum_scale, float(p.to(torch.bfloat16).float().abs().min()))
            return {"changed_weight_symbols": changed, "hard_entropy_nats": hard_nats,
                    "soft_entropy_nats": soft_nats, "parameter_delta_l2": math.sqrt(displacement),
                    "max_scale_delta": scale_delta, "min_abs_bf16_scale": minimum_scale}

        def validate():
            for key, p in params.items():
                assert bool(torch.isfinite(p).all()), f"nonfinite parameter: {key}"
                if key.endswith(".scale"):
                    s = p.detach().to(torch.bfloat16).float()
                    assert bool(torch.isfinite(s).all() & (s != 0).all()), f"invalid scale: {key}"
                    if not key.endswith(".quantize_weight.scale"):
                        assert bool((s > 0).all()), f"nonpositive activation scale: {key}"

        def evaluate():
            epoch = state["epoch"]
            weight, info = export(f"epoch-{epoch + 1:02d}")
            results = {}
            for sample in samples:
                name, score = sample["name"], sample["score"]
                log = job / f"epoch-{epoch + 1:02d}-{name}.log"
                save("evaluating", evaluating_sample=name)
                run([args.evaluator, "eval", score["tokens"], score["priors"], weight], log)
                rows = [json.loads(line.split(" ", 1)[1]) for line in log.read_text().splitlines()
                        if line.startswith("ACTIVATION_RESULT ")]
                assert len(rows) == 1 and rows[0]["total"]["count"] == score["targets"]
                raw = rows[0]["total"]["logits_mean"]
                assert math.isfinite(raw)
                results[name] = {"targets": score["targets"], "raw_nll": raw,
                                 "delta_raw_nll": raw - baseline["steps"][name]["result"]["total"]["logits_mean"]}
            tau = .5 * (.15 / .5) ** (epoch / 7)
            row = {"epoch": epoch + 1, "step": state["step"], **info,
                   "double_weight_saved_bytes": 2 * (state["original_static15_bytes"] - info["weight_bytes"]),
                   "scores": results, **geometry(tau)}
            row["proxy_gate_passed"] = (row["double_weight_saved_bytes"] >= 1500000
                                        and all(r["delta_raw_nll"] <= .004 for r in results.values()))
            state["evaluations"] = [r for r in state["evaluations"] if r["epoch"] != epoch + 1] + [row]
            torch.save(model.state_dict(), job / f"epoch-{epoch + 1:02d}.pt")
            event("epoch_evaluated", **row)

        validate()
        if not args.resume:
            weight, info = export("step-0000")
            reference = job / "original-static15.tfwc3"
            run([args.recode, args.weights, reference], job / "original-static15-recode.log")
            assert weight.read_bytes() == reference.read_bytes(), "CPU zero-step export differs from original static15"
            state["original_static15_bytes"] = info["weight_bytes"]
            event("zero_step_export_passed", **info, **geometry(.5))
            checkpoint()
        if args.zero_only:
            save("zero_step_complete", status="ready")
            return

        while state["epoch"] < EPOCHS:
            epoch = state["epoch"]
            order = pieces.copy()
            random.Random(245 + epoch).shuffle(order)
            tau = .5 * (.15 / .5) ** (epoch / 7)
            model.train()
            for batch, targets in batches(order[state["next_piece"]:]):
                if cpu() + 180 >= args.cpu_seconds or (job / "STOP").exists():
                    checkpoint()
                    save("bounded_stop", status="cpu_budget" if cpu() + 180 >= args.cpu_seconds else "stopped")
                    return
                save("training", batch_targets=targets, batch_pieces=len(batch))
                before_cpu, before_wall = cpu(), time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.
                for i, lo, hi in batch:
                    n = hi - lo - 1
                    x = torch.from_numpy(tokens[i][lo:hi - 1].astype(np.int64)).unsqueeze(0)
                    y = torch.from_numpy(tokens[i][lo + 1:hi].astype(np.int64)).unsqueeze(0)
                    # Read only this piece, freeing file mappings before full backward.
                    prior = np.fromfile(samples[i]["priors"], dtype="<f2", count=n * 205, offset=lo * 410).reshape(n, 205)
                    p = torch.from_numpy(prior).unsqueeze(0)
                    assert bool(torch.isfinite(p).all())
                    loss = model(input_tokens=x, output_tokens=y, priors=p,
                                 loss_mask=torch.ones_like(x, dtype=torch.bool),
                                 cumulative_sequence_lengths=torch.tensor([[0, n]], dtype=torch.int32))
                    assert bool(torch.isfinite(loss)), "nonfinite CE"
                    loss_sum += float(loss.detach())
                    loss.backward()
                    del x, y, prior, p, loss
                for key, parameter in params.items():
                    assert parameter.grad is not None, f"missing CE gradient: {key}"
                    assert bool(torch.isfinite(parameter.grad).all()), f"nonfinite CE gradient: {key}"
                    parameter.grad.div_(targets)
                ce_cpu = cpu() - before_cpu
                soft_nats = 0.
                for _, weight, scale in quantized:
                    h = entropy(weight, scale, tau)
                    assert bool(torch.isfinite(h)), "nonfinite entropy"
                    soft_nats += float(h.detach())
                    (2 * h / FULL_TARGETS).backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                validate()
                state["step"] += 1
                state["next_piece"] += len(batch)
                state["seen_targets"] += targets
                state["epoch_targets"] += targets
                state["epoch_loss_sum"] += loss_sum
                row = {"targets": targets, "pieces": len(batch), "ce_mean": loss_sum / targets,
                       "tau": tau, "soft_entropy_nats": soft_nats,
                       "objective": loss_sum / targets + 2 * soft_nats / FULL_TARGETS,
                       "gradient_norm": float(norm), "ce_cpu_seconds": ce_cpu,
                       "step_cpu_seconds": cpu() - before_cpu,
                       "step_wall_seconds": time.monotonic() - before_wall}
                if state["step"] % 8 == 0:
                    row.update(geometry(tau))
                    checkpoint()
                if state["step"] == 16:
                    state["initial_gate"] = "passed"
                    state["first16"] = {**row, "seen_targets": state["seen_targets"],
                                        "elapsed_cpu_seconds": cpu() - cpu_base,
                                        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                event("optimizer_step", **row)
                save("step_complete", last_step=row)
            assert state["epoch_targets"] == manifest["training_targets"]
            checkpoint()  # Save trained parameters before export/scoring.
            if cpu() + 120 >= args.cpu_seconds:
                save("evaluation_budget_stop", status="cpu_budget")
                return
            evaluate()
            state.update(epoch=epoch + 1, next_piece=0, epoch_targets=0, epoch_loss_sum=0.)
            checkpoint()
        save("trajectory_complete", status="complete")
        event("trajectory_complete", proxy_gate_passed=any(r["proxy_gate_passed"] for r in state["evaluations"]))
    except Exception as exc:
        save("failed", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
