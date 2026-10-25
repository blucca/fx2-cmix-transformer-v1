#!/usr/bin/env python3
"""One bounded FP32 backward through the original 6M QAT model.

Uses the repository's PyTorch KDA/conv/norm references and SDPA, with the
unchanged cooldown quantizers. Input files are a prepared-stream prefix:
token IDs, matching next-token priors, and original int32 article boundaries.
This measures a complete small article; it performs no optimizer update.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-tokens", type=int, default=1025)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--cpu-seconds", type=int, default=300)
    parser.add_argument("--kernel", type=Path, help="compact CPU shared library; omit for reference")
    parser.add_argument("--article-index", type=int)
    parser.add_argument("--save-gradients", type=Path)
    parser.add_argument("--compare-gradients", type=Path)
    args = parser.parse_args()
    assert 1 <= args.min_tokens <= args.max_tokens <= 131072
    assert 10 <= args.cpu_seconds <= 900
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        os.environ.setdefault(key, str(args.output.parent / key.lower()))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds - 2, args.cpu_seconds))
    # Conservative address-space ceiling also bounds this process's RSS.
    resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
    started_wall, started_cpu = time.monotonic(), time.process_time()
    result = {"status": "starting", "optimizer_updates": 0,
              "cpu_limit_seconds": args.cpu_seconds, "address_space_limit_gib": 6,
              "threads": 1, "python": sys.version}

    def save(stage, **values):
        result.update(stage=stage, **values)
        result.update(elapsed_wall_seconds=time.monotonic() - started_wall,
                      elapsed_cpu_seconds=time.process_time() - started_cpu,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(args.output)
        print(json.dumps({"stage": stage, **values}), flush=True)

    def cpu_limit(signum, frame):
        save("cpu_limit", status="stopped")
        raise SystemExit(124)

    signal.signal(signal.SIGXCPU, cpu_limit)
    save("importing")
    try:
        import numpy as np
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from cpu_model import load_model
        model, counts = load_model(args.checkpoint, args.kernel)
        cfg = model.cfg
        save("model_loaded", torch_version=torch.__version__,
             checkpoint_tensors=len(model.state_dict()),
             trainable_tensors=sum(p.requires_grad for p in model.parameters()),
             parameters=sum(p.numel() for p in model.parameters()),
             config_source="training_recipes/cooldown.py", query_key_norm_gain=cfg.query_key_norm_gain,
             window=cfg.window_size, activation_dtype=str(cfg.activation_dtype),
             gradient_checkpointing=cfg.gradient_checkpointing,
             attention="window_sdpa" if args.kernel else "sdpa",
             kda="compact_native" if args.kernel else "torch_reference")

        tokens = np.fromfile(args.tokens, dtype=np.uint8)
        bounds = np.fromfile(args.boundaries, dtype="<i4")
        assert bounds[0] == 0 and (np.diff(bounds) > 0).all()
        assert args.priors.stat().st_size == tokens.size * 205 * 2
        candidates = [(i, int(s), int(e)) for i, (s, e) in enumerate(zip(bounds, bounds[1:]))
                      if e <= len(tokens) and args.min_tokens <= e - s - 1 <= args.max_tokens
                      and (args.article_index is None or i == args.article_index)]
        assert candidates, "no complete article in the requested token range"
        article, start, end = candidates[0]
        n = end - start - 1
        priors = np.memmap(args.priors, dtype="<f2", mode="r", shape=(len(tokens), 205))
        x = torch.from_numpy(tokens[start:end - 1].astype(np.int64)).unsqueeze(0)
        y = torch.from_numpy(tokens[start + 1:end].astype(np.int64)).unsqueeze(0)
        p = torch.from_numpy(np.array(priors[start:end - 1])).unsqueeze(0)
        assert x.max() < 205 and y.max() < 205 and torch.isfinite(p).all()
        del priors
        save("forward_start", input_kind="real_complete_prepared_prefix_article",
             article_index=article, article_start=start, article_end=end,
             input_tokens=n, valid_targets=n, rope_offset=0,
             full_1024_window_exercised=n > 1024,
             original_article_boundaries=True, prior_rows="[start:end-1] predicts [start+1:end]")
        w, c = time.monotonic(), time.process_time()
        loss = model(input_tokens=x, output_tokens=y, priors=p,
                     loss_mask=torch.ones_like(x, dtype=torch.bool),
                     cumulative_sequence_lengths=torch.tensor([[0, n]], dtype=torch.int32))
        assert torch.isfinite(loss).item()
        save("backward_start", loss_sum=loss.item(), mean_nll=loss.item() / n,
             forward_wall_seconds=time.monotonic() - w,
             forward_cpu_seconds=time.process_time() - c)
        w, c = time.monotonic(), time.process_time()
        loss.backward()
        backward_wall, backward_cpu = time.monotonic() - w, time.process_time() - c
        used, missing, nonfinite, nonzero = [], [], [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                missing.append(name)
                continue
            used.append(name)
            if not torch.isfinite(parameter.grad).all().item():
                nonfinite.append(name)
            if torch.count_nonzero(parameter.grad).item():
                nonzero.append(name)
        save("gradients_checked", backward_wall_seconds=backward_wall,
             backward_cpu_seconds=backward_cpu, used_gradient_tensors=len(used),
             nonzero_gradient_tensors=len(nonzero), missing_gradient_names=missing,
             nonfinite_gradient_names=nonfinite, operator_calls=counts)
        assert used and not missing and not nonfinite
        if args.save_gradients:
            torch.save({name: p.grad for name, p in model.named_parameters()}, args.save_gradients)
        if args.compare_gradients:
            expected = torch.load(args.compare_gradients, weights_only=True)
            diffs, norms, dots = 0.0, 0.0, 0.0
            per_tensor = {}
            for name, p in model.named_parameters():
                g, r = p.grad.double(), expected[name].double()
                err, norm = (g - r).square().sum().item(), r.square().sum().item()
                diffs += err
                norms += norm
                dots += (g * r).sum().item()
                per_tensor[name] = (err / max(norm, 1e-30)) ** 0.5
            relative_l2 = (diffs / norms) ** 0.5
            save("reference_compared", gradient_relative_l2=relative_l2,
                 gradient_max_tensor_relative_l2=max(per_tensor.values()),
                 gradient_worst_tensors=sorted(per_tensor.items(), key=lambda p: -p[1])[:8])
            assert relative_l2 < 0.01, "full QAT gradient relative L2 exceeds 1%"
        save("complete", status="passed")
    except Exception as exc:
        save("failed", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
