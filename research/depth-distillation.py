#!/usr/bin/env python3
"""One eight-block CPU student: 8 continuous + 24 hard-coordinate epochs.

Reuse full-piece causal inputs, original twelve-block teacher and static15.
75 bounded coordinates and retained raw F32 parameters learn; signed BF16
weight/activation scales stay fixed. Every four epochs exports a hard model
and scores the full held set plus the separate maximum-native-length audit.
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
RECIPE = dict(method="eight-block-causal-distillation-v1", seed=419,
              epochs=32, continuous_epochs=8, evaluation_every=4,
              coordinate_lr_continuous=.03, coordinate_lr_hard=.01,
              raw_lr_continuous=.0003, raw_lr_hard=.0001,
              betas=[.9, .99], eps=1e-8, weight_decay=0,
              teacher_kl=1., final_hidden_mse=.1, entropy_tau=.5,
              full_prepared_bytes=586459020, final_batch="merge-short-tail",
              kept_layers=[0, 1, 2, 3, 8, 9, 10, 11],
              frozen="all signed BF16 weight and activation scales",
              cpu_gemm="integer-units-before-native-BF16-fold")
_spec = importlib.util.spec_from_file_location(
    "original_coordinate", Path(__file__).with_name("integer-coordinate-qat.py"))
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)


def install_student_coordinates(model):
    import torch
    model.requires_grad_(False)
    state = dict(model.named_parameters())
    quantized = {k for k in state if k.endswith(".weight") and
                 k[:-7] + ".quantize_weight.scale" in state}
    assert len(quantized) == 75
    raw = {k: p for k, p in state.items() if k not in quantized and not k.endswith(".scale")}
    assert raw
    for parameter in raw.values():
        parameter.requires_grad_(True)
    frozen = {k: p.detach().clone() for k, p in state.items() if k not in raw}
    coordinates, scales, initial_q = {}, {}, {}
    mode = {"hard": False}
    for key in sorted(quantized):
        w = state[key].detach()
        skey = key[:-7] + ".quantize_weight.scale"
        s = state[skey].detach().to(torch.bfloat16).float()
        assert bool(torch.isfinite(s).all() & (s != 0).all())
        q = (w / s[:, None]).round().clamp(-7, 7).to(torch.int8)
        z = torch.nn.Parameter(q.float())
        coordinates[key], scales[key], initial_q[key] = z, s, q
        quantizer = model.get_submodule(skey[:-6])
        assert quantizer.quant.buckets == 15 and not quantizer.batched

        def forward(input, initialize_scales_steps, return_ints, z=z, s=s):
            assert initialize_scales_steps is None and input.shape == z.shape
            if return_ints:
                return z.detach().round().clamp(-7, 7).to(torch.int64)
            value = _old.integer_weight(z, s) if mode["hard"] else z * s[:, None]
            return value.to(input.dtype)

        quantizer.forward = forward
    return coordinates, scales, initial_q, raw, frozen, mode


def data_objective(ce, kl, hidden, recipe):
    """One source for the actual VJP and its reported scalar."""
    return (recipe.get("label_ce", 1.) * ce + recipe["teacher_kl"] * kl +
            recipe["final_hidden_mse"] * hidden)


def teacher_target_identity(identity):
    """Legacy caches also record student settings; only these produce targets."""
    return dict(checkpoint=identity["checkpoint"], data=identity["data"],
                cpu_gemm=identity["recipe"]["cpu_gemm"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "kernel", "data", "baseline", "weights",
                 "original-static15", "recode", "evaluator", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cpu-seconds", type=int, default=172800)
    p.add_argument("--prior-cpu-seconds", type=float, default=0.)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--recovery-cpu-seconds", type=float)
    p.add_argument("--check-only", action="store_true",
                   help="one 1025–2048-target complete piece, one update, restore initialization")
    p.add_argument("--teacher-only", action="store_true",
                   help="fresh 2KL teacher objective; label CE is report-only")
    p.add_argument("--teacher-cache", type=Path,
                   help="existing teacher cache, strictly read-only; required for teacher-only")
    a = p.parse_args()
    assert 600 <= a.cpu_seconds <= 172800 and a.prior_cpu_seconds >= 0
    assert not (a.resume and a.check_only)
    assert a.teacher_only == (a.teacher_cache is not None)
    recipe = (dict(RECIPE, method="eight-block-teacher-only-distillation-v1",
                   label_ce=0., teacher_kl=2.) if a.teacher_only else RECIPE)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    job = a.output.resolve()
    job.mkdir(parents=True, exist_ok=True)
    for k in ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        path = job / k.lower()
        path.mkdir(exist_ok=True)
        os.environ[k] = str(path)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (7 << 30, 7 << 30))
    wall = time.monotonic()
    identity = dict(checkpoint=str(a.checkpoint.resolve()), data=str(a.data.resolve()),
                    weights=str(a.weights.resolve()), original_static15=str(a.original_static15.resolve()),
                    recipe=recipe)
    if a.teacher_cache is not None:
        identity.update(teacher_cache=str(a.teacher_cache.resolve()), teacher_cache_mode="read-only")
    state_path, checkpoint_path = job / "state.json", job / "latest.pt"
    cursor = ("epoch", "step", "next_piece", "seen_targets", "epoch_targets",
              "epoch_ce_sum", "epoch_kl_sum", "epoch_hidden_sum")
    if a.resume:
        state = json.loads(state_path.read_text())
        assert state["identity"] == identity and checkpoint_path.is_file()
        assert state["status"] not in ("complete", "failed", "cpu_budget")
        if state["status"] == "running":
            assert a.recovery_cpu_seconds is not None, "account for CPU after the last checkpoint"
        recovery = a.recovery_cpu_seconds or 0.
        assert recovery >= 0
        cpu_base = state["total_cpu_seconds"] + recovery
    else:
        assert not state_path.exists() and not checkpoint_path.exists(), "inspect existing trajectory"
        cpu_base = a.prior_cpu_seconds
        state = dict(status="starting", started_at=time.time(), identity=identity,
                     epoch=0, step=0, next_piece=0, seen_targets=0, epoch_targets=0,
                     epoch_ce_sum=0., epoch_kl_sum=0., epoch_hidden_sum=0.,
                     checkpoint_step=0, evaluations=[], cpu_limit_seconds=a.cpu_seconds,
                     address_space_limit_gib=7, original_static15_bytes=a.original_static15.stat().st_size)
    remaining = math.floor(a.cpu_seconds - cpu_base)
    assert remaining >= 180
    resource.setrlimit(resource.RLIMIT_CPU, (remaining - 2, remaining))

    def cpu():
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        return cpu_base + time.process_time() + child.ru_utime + child.ru_stime

    def save(stage, **values):
        state.update(stage=stage, updated_at=time.time(), total_cpu_seconds=cpu(),
                     run_wall_seconds=time.monotonic() - wall,
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **values)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(state_path)

    def event(kind, **values):
        row = dict(kind=kind, at=time.time(), step=state["step"], epoch=state["epoch"],
                   total_cpu_seconds=cpu(), **values)
        with (job / "metrics.jsonl").open("a") as out:
            out.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def limit(signum, frame):
        save("cpu_limit", status="cpu_budget")
        raise SystemExit(124)

    signal.signal(signal.SIGXCPU, limit)
    save("importing", status="running")
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from cpu_model import load_model
        from depth_student import load_student, export_student

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(recipe["seed"])
        model, _, initialization = load_student(a.checkpoint, a.kernel, native_scaled_gemm=True)
        assert model.cfg.n_layers == 8 and sum(model.cfg.kimi_linear) == 6
        z, scales, initial_q, raw, frozen, mode = install_student_coordinates(model)
        initial_raw = {k: v.detach().clone() for k, v in raw.items()}
        optimizer = torch.optim.AdamW([
            {"params": list(z.values()), "lr": recipe["coordinate_lr_continuous"]},
            {"params": list(raw.values()), "lr": recipe["raw_lr_continuous"]}],
            betas=tuple(recipe["betas"]), eps=recipe["eps"], weight_decay=0, foreach=False)
        if a.resume:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            assert ckpt["identity"] == identity and ckpt["z"].keys() == z.keys()
            assert ckpt["raw"].keys() == raw.keys()
            with torch.no_grad():
                for values, saved in ((z, ckpt["z"]), (raw, ckpt["raw"])):
                    for key, value in values.items():
                        value.copy_(saved[key])
            optimizer.load_state_dict(ckpt["optimizer"])
            torch.set_rng_state(ckpt["rng"])
            state.update({k: ckpt[k] for k in cursor})
            del ckpt
        save("student_ready", initialization=initialization,
             coordinate_tensors=len(z), raw_trainable_tensors=len(raw), frozen_tensors=len(frozen))
        manifest = json.loads(a.data.read_text())
        samples = manifest["samples"]
        baseline = json.loads(a.baseline.read_text()) if not a.check_only else None
        if baseline is not None:
            assert baseline["status"] == "passed"
        tokens = [np.fromfile(s["tokens"], dtype=np.uint8) for s in samples]
        pieces = [(i, lo, hi) for i, s in enumerate(samples) for lo, hi in s["train_pieces"]]
        assert len(pieces) == manifest["training_pieces"]
        assert sum(hi - lo - 1 for _, lo, hi in pieces) == manifest["training_targets"]
        for s, tok in zip(samples, tokens):
            assert Path(s["priors"]).stat().st_size == len(tok) * 410
            if baseline is not None:
                assert baseline["steps"][s["name"]]["result"]["total"]["count"] == s["score"]["targets"]
        for i, lo, hi in pieces:
            assert 0 <= lo < hi <= len(tokens[i]) and 1 < hi - lo <= 32769

        def read_piece(piece):
            i, lo, hi = piece
            n = hi - lo - 1
            x = torch.from_numpy(tokens[i][lo:hi-1].astype(np.int64)).unsqueeze(0)
            y = torch.from_numpy(tokens[i][lo+1:hi].astype(np.int64)).unsqueeze(0)
            prior = np.fromfile(samples[i]["priors"], dtype="<f2", count=n*205,
                                offset=lo*410).reshape(n, 205)
            pp = torch.from_numpy(prior).unsqueeze(0)
            assert bool(torch.isfinite(pp).all()) and int(x.max()) < 205 and int(y.max()) < 205
            return x, y, pp, torch.tensor([[0, n]], dtype=torch.int32)

        def forward(m, x, pp, bounds):
            hidden = m.compute_last_activation(input_tokens=x, priors=pp,
                                                cumulative_sequence_lengths=bounds)
            logits = m.last_activation_to_logits(hidden, priors=pp)
            return logits, hidden

        cache = a.teacher_cache.resolve() if a.teacher_cache is not None else job / "teacher-cache"
        if a.teacher_cache is None:
            cache.mkdir(exist_ok=True)
        cache_identity = cache / "identity.json"
        if a.teacher_cache is not None:
            assert cache_identity.is_file(), "existing read-only cache identity required"
            target_identity = teacher_target_identity(identity)
            assert teacher_target_identity(json.loads(cache_identity.read_text())) == target_identity
            save("teacher_cache_ready", teacher_cache_contract=dict(
                path=str(cache), mode="read-only", target_identity=target_identity))
        elif cache_identity.exists():
            assert json.loads(cache_identity.read_text()) == identity
        else:
            cache_identity.write_text(json.dumps(identity, indent=2) + "\n")
        teacher, cache_misses = None, 0

        def teacher_targets(piece, x, pp, bounds):
            nonlocal teacher, cache_misses
            i, lo, hi = piece
            path = cache / f"sample-{i:04d}-{lo}-{hi}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    logits, hidden = torch.from_numpy(data["logits"]), torch.from_numpy(data["hidden"])
            else:
                assert a.teacher_cache is None, f"read-only teacher cache miss: {path}"
                if teacher is None:
                    teacher, _ = load_model(a.checkpoint, a.kernel, native_scaled_gemm=True)
                    teacher.requires_grad_(False).eval()
                with torch.no_grad():
                    logits, hidden = forward(teacher, x, pp, bounds)
                tmp = path.with_suffix(".tmp")
                with tmp.open("wb") as out:
                    np.savez(out, logits=logits.numpy(), hidden=hidden.numpy())
                tmp.replace(path)
                cache_misses += 1
            assert logits.dtype == hidden.dtype == torch.float32
            assert tuple(logits.shape) == (1, hi-lo-1, 205) and tuple(hidden.shape) == (1, hi-lo-1, 192)
            assert bool(torch.isfinite(logits).all() & torch.isfinite(hidden).all())
            return logits, hidden

        def validate():
            parameters = dict(model.named_parameters())
            for key, original in frozen.items():
                assert torch.equal(parameters[key], original), key
                assert parameters[key].grad is None and not parameters[key].requires_grad, key
            for key, value in z.items():
                assert bool(torch.isfinite(value).all() & (value >= -7).all() & (value <= 7).all()), key
            for key, value in raw.items():
                assert bool(torch.isfinite(value).all()), key

        def geometry():
            changed, displaced, raw_changed, hard_nats = 0, 0, 0, 0.
            with torch.no_grad():
                for key, value in z.items():
                    q = value.round().clamp(-7, 7).to(torch.int8)
                    changed += int((q != initial_q[key]).sum())
                    displaced += int((value != initial_q[key].float()).sum())
                    freqs = torch.bincount(q.flatten().long()+7, minlength=15).double() / q.numel()
                    hard_nats -= q.numel()*float((freqs*freqs.clamp_min(1e-30).log()).sum())
                for key, value in raw.items():
                    raw_changed += int((value != initial_raw[key]).sum())
            return dict(changed_weight_symbols=changed, changed_z_coordinates=displaced,
                        changed_raw_values=raw_changed, hard_entropy_nats=hard_nats,
                        max_scale_delta=0., stage_kind="hard" if mode["hard"] else "continuous")

        def checkpoint():
            state["checkpoint_step"] = state["step"]
            payload = {k: state[k] for k in cursor}
            payload.update(identity=identity, z={k: v.detach() for k, v in z.items()},
                           raw={k: v.detach() for k, v in raw.items()},
                           optimizer=optimizer.state_dict(), rng=torch.get_rng_state())
            tmp = checkpoint_path.with_suffix(".tmp")
            torch.save(payload, tmp)
            tmp.replace(checkpoint_path)
            save("checkpoint")

        def run(command, log):
            seconds = min(3600, math.floor(a.cpu_seconds-cpu())-2)
            assert seconds >= 10
            def limits():
                resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
            with log.open("w") as out:
                subprocess.run(list(map(str, command)), stdin=subprocess.DEVNULL, stdout=out,
                               stderr=subprocess.STDOUT, preexec_fn=limits, timeout=604800, check=True)

        def export(label, keep_raw=False):
            hard_state = model.state_dict()
            for key, value in z.items():
                hard_state[key] = value.detach().round().clamp(-7, 7)*scales[key][:, None]
            v2, weight = job / f"{label}.tfwc2", job / f"{label}.tfwc3"
            info = export_student(model, a.weights, v2,
                                  output_raw=job / f"{label}.raw" if keep_raw else None,
                                  state_dict=hard_state)
            run([a.recode, v2, weight], job / f"{label}-recode.log")
            receipt = json.loads((job / f"{label}-recode.log").read_text().splitlines()[-1])
            assert receipt["status"] == "passed" and receipt["all_tensor_bytes_equal"]
            assert receipt["int4_tensors"] == 75
            info.update(weight_file=str(weight), weight_bytes=weight.stat().st_size,
                        static15_recode=receipt)
            # v2 is an intermediate template; retain only selected real exports.
            v2.unlink()
            return weight, info

        def backward(batch, targets):
            optimizer.zero_grad(set_to_none=True)
            ce_sum = kl_sum = hidden_sum = 0.
            for piece in batch:
                x, y, pp, bounds = read_piece(piece)
                reference, feature = teacher_targets(piece, x, pp, bounds)
                logits, hidden = forward(model, x, pp, bounds)
                reported_logits = logits.detach() if recipe.get("label_ce", 1.) == 0 else logits
                ce = F.cross_entropy(reported_logits.reshape(-1, 205), y.flatten(), reduction="sum")
                kl = F.kl_div(logits.log_softmax(-1), reference.log_softmax(-1),
                              reduction="sum", log_target=True)
                mse = (hidden-feature).square().mean(-1).sum()
                assert bool(torch.isfinite(ce) & torch.isfinite(kl) & torch.isfinite(mse))
                ce_sum += float(ce.detach()); kl_sum += float(kl.detach()); hidden_sum += float(mse.detach())
                data_objective(ce, kl, mse, recipe).backward()
                del x, y, pp, bounds, reference, feature, logits, reported_logits, hidden, ce, kl, mse
            for key, value in [*z.items(), *raw.items()]:
                assert value.grad is not None and bool(torch.isfinite(value.grad).all()), key
                value.grad.div_(targets)
            soft_nats = 0.
            for value in z.values():
                h = _old.entropy(value, recipe["entropy_tau"])
                soft_nats += float(h.detach())
                (2*h/recipe["full_prepared_bytes"]).backward()
            norm, nonzero = 0., 0
            for key, value in [*z.items(), *raw.items()]:
                assert bool(torch.isfinite(value.grad).all()), key
                norm += float(value.grad.double().square().sum())
                nonzero += int(bool((value.grad != 0).any()))
            return dict(ce_mean=ce_sum/targets, teacher_kl_mean=kl_sum/targets,
                        final_hidden_mse_mean=hidden_sum/targets, soft_entropy_nats=soft_nats,
                        objective=data_objective(ce_sum, kl_sum, hidden_sum, recipe)/targets+
                                  2*soft_nats/recipe["full_prepared_bytes"],
                        gradient_norm=math.sqrt(norm), finite_gradient_tensors=len(z)+len(raw),
                        nonzero_gradient_tensors=nonzero)

        def update():
            optimizer.step()
            with torch.no_grad():
                for value in z.values():
                    value.clamp_(-7, 7)
            validate()

        def evaluate(epoch):
            weight, info = export(f"epoch-{epoch:02d}")
            scores = {}
            for sample in samples:
                name, score = sample["name"], sample["score"]
                log = job / f"epoch-{epoch:02d}-{name}.log"
                save("evaluating", evaluating_sample=name)
                run([a.evaluator, "eval-pieces", score["tokens"], score["priors"],
                     weight, score["boundaries"]], log)
                rows = [json.loads(line.split(" ", 1)[1]) for line in log.read_text().splitlines()
                        if line.startswith("ACTIVATION_RESULT ")]
                assert len(rows) == 1 and rows[0]["total"]["count"] == score["targets"]
                nll = rows[0]["total"]["logits_mean"]
                assert math.isfinite(nll)
                scores[name] = dict(targets=score["targets"], raw_nll=nll,
                                    delta_raw_nll=nll-baseline["steps"][name]["result"]["total"]["logits_mean"])
                if score.get("long_piece_indices"):
                    long = _old.long_piece_score(rows[0], score["long_piece_indices"])
                    original = _old.long_piece_score(baseline["steps"][name]["result"], score["long_piece_indices"])
                    assert long["targets"] == original["targets"]
                    long["delta_raw_nll"] = long["raw_nll"]-original["raw_nll"]
                    scores[name]["long_native"] = long
            row = dict(epoch=epoch, step=state["step"], scores=scores, **info, **geometry(),
                       double_weight_saved_bytes=2*(state["original_static15_bytes"]-info["weight_bytes"]))
            state["evaluations"] = [r for r in state["evaluations"] if r["epoch"] != epoch]+[row]
            event("epoch_evaluated", evaluation=row)

        validate()
        if not a.resume:
            _, info = export("step-0000", keep_raw=a.check_only)
            state["zero_step"] = dict(**info, **geometry())
            event("zero_step_export", zero=state["zero_step"])
            checkpoint()
        if a.check_only:
            piece = next(p for p in pieces if 1025 <= p[2]-p[1]-1 <= 2048)
            n, before = piece[2]-piece[1]-1, cpu()
            result = backward([piece], n)
            update()
            after = geometry()
            assert after["changed_z_coordinates"] > 0 and after["changed_raw_values"] > 0
            result.update(status="passed", piece=list(piece), targets=n, **after,
                          check_cpu_seconds=cpu()-before, frozen_tensors=len(frozen),
                          coordinate_tensors=len(z), raw_trainable_tensors=len(raw))
            (job / "preflight.json").write_text(json.dumps(result, indent=2)+"\n")
            event("one_step_check", result=result)
            with torch.no_grad():
                for key, value in z.items():
                    value.copy_(initial_q[key])
                for key, value in raw.items():
                    value.copy_(initial_raw[key])
            optimizer.state.clear()
            optimizer.zero_grad(set_to_none=True)
            validate()
            checkpoint()
            save("check_complete", status="ready")
            return
        model.train()
        while state["epoch"] < recipe["epochs"]:
            epoch = state["epoch"]
            mode["hard"] = epoch >= recipe["continuous_epochs"]
            optimizer.param_groups[0]["lr"] = recipe["coordinate_lr_hard" if mode["hard"] else "coordinate_lr_continuous"]
            optimizer.param_groups[1]["lr"] = recipe["raw_lr_hard" if mode["hard"] else "raw_lr_continuous"]
            order = pieces.copy()
            random.Random(recipe["seed"]+epoch).shuffle(order)
            for batch, targets in _old.causal_batches(order[state["next_piece"]:]):
                if cpu()+180 >= a.cpu_seconds or (job / "STOP").exists():
                    checkpoint()
                    save("bounded_stop", status="cpu_budget" if cpu()+180 >= a.cpu_seconds else "stopped")
                    return
                save("training", batch_targets=targets, batch_pieces=len(batch), training_kind="hard" if mode["hard"] else "continuous")
                before = cpu()
                row = backward(batch, targets)
                update()
                state["step"] += 1; state["next_piece"] += len(batch)
                state["seen_targets"] += targets; state["epoch_targets"] += targets
                for field, value in (("epoch_ce_sum", "ce_mean"), ("epoch_kl_sum", "teacher_kl_mean"),
                                     ("epoch_hidden_sum", "final_hidden_mse_mean")):
                    state[field] += row[value]*targets
                row.update(targets=targets, pieces=len(batch), step_cpu_seconds=cpu()-before,
                           teacher_cache_misses_this_run=cache_misses)
                if state["step"] % 8 == 0:
                    checkpoint()
                event("optimizer_step", **row)
                save("step_complete", last_step=row)
            assert state["epoch_targets"] == manifest["training_targets"]
            event("epoch_trained", trained_epoch=epoch+1,
                  ce_mean=state["epoch_ce_sum"]/state["epoch_targets"],
                  teacher_kl_mean=state["epoch_kl_sum"]/state["epoch_targets"],
                  final_hidden_mse_mean=state["epoch_hidden_sum"]/state["epoch_targets"])
            if (epoch+1) % recipe["evaluation_every"] == 0:
                if cpu()+120 >= a.cpu_seconds:
                    checkpoint(); save("evaluation_budget_stop", status="cpu_budget"); return
                evaluate(epoch+1)
            state.update(epoch=epoch+1, next_piece=0, epoch_targets=0,
                         epoch_ce_sum=0., epoch_kl_sum=0., epoch_hidden_sum=0.)
            checkpoint()
        save("trajectory_complete", status="complete")
        event("trajectory_complete")
    except Exception as exc:
        save("failed", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
