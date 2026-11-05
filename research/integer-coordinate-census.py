#!/usr/bin/env python3
"""Read-only symbol/soft-H attribution for a frozen integer-coordinate snapshot.

Load tensor dictionaries only. Keep tau=.5 and 2/FULL_TARGETS fixed; measure
FP32 autograd at both original q and saved z, one matrix at a time.
"""
import argparse
import heapq
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
FULL_TARGETS = 586459020
TAU = .5
H_FACTOR = 2 / FULL_TARGETS
ADAM_EPS = 1e-8
TINY_SCALE = 1e-10
RSS_LIMIT_BYTES = 900_000_000
EXPECTED = (429, 5923228, 111, 5868864)
GRADIENT_COLUMNS = ["l2", "abs_below_adam_eps_fraction", "dot_dz", "cos_dz",
                    "dot_dq", "cos_dq"]
RANK_FIELDS = ("hard_entropy", "soft_entropy", "H_original_dot_dz",
               "H_coordinates_dot_dz", "H_original_dot_dq", "H_coordinates_dot_dq")


def setup_torch():
    global torch
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def check_rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak * 1024 > RSS_LIMIT_BYTES:
        raise RuntimeError(f"RSS cap failed: {peak} KiB; cap={RSS_LIMIT_BYTES} bytes")
    return peak


def entropy(z):
    """Exact frozen trainer formula, including FP32 reductions and clamp."""
    centers = torch.arange(-7, 8, dtype=z.dtype, device=z.device)
    p = (-((z.flatten()[:, None] - centers) ** 2) / TAU).softmax(-1).mean(0)
    return -z.numel() * (p * p.clamp_min(1e-12).log()).sum()


def soft_statistics(value):
    z = value.detach().requires_grad_(True)
    centers = torch.arange(-7, 8, dtype=z.dtype, device=z.device)
    assignment = (-((z.flatten()[:, None] - centers) ** 2) / TAU).softmax(-1)
    p = assignment.mean(0)
    h = -z.numel() * (p * p.clamp_min(1e-12).log()).sum()
    g, = torch.autograd.grad(2 * h / FULL_TARGETS, z)
    assert bool(torch.isfinite(h) & torch.isfinite(g).all())
    # Accounting uses the same FP32 assignments, summed in FP64. This makes
    # the additive row ledger independent of FP32 scalar reduction roundoff.
    rows = assignment.detach().reshape(*z.shape, 15).sum(1, dtype=torch.float64)
    return rows, float(h.detach()), g.detach()


def hard_row_counts(q):
    offsets = torch.arange(q.shape[0], dtype=torch.int64)[:, None] * 15
    return torch.bincount((offsets + q.long() + 7).flatten(),
                          minlength=q.shape[0] * 15).reshape(-1, 15)


def transitions(q0, q1):
    return torch.bincount(((q0.long() + 7) * 15 + q1.long() + 7).flatten(),
                          minlength=225).reshape(15, 15)


def row_entropy_delta(rows0, rows1, elements, clamp):
    """Exact additive attribution along the linear per-symbol mass path."""
    m0, m1 = rows0.double().sum(0), rows1.double().sum(0)
    p0, p1 = (m0 / elements).clamp_min(clamp), (m1 / elements).clamp_min(clamp)
    loss0, loss1 = -m0 * p0.log(), -m1 * p1.log()
    dm = m1 - m0
    derivative = -p0.log() - (m0 / elements >= clamp).double()
    coefficient = derivative.clone()
    moving = dm != 0
    coefficient[moving] = (loss1[moving] - loss0[moving]) / dm[moving]
    row_delta = ((rows1.double() - rows0.double()) * coefficient).sum(1)
    before, after = float(loss0.sum()), float(loss1.sum())
    delta = float((loss1 - loss0).sum())
    assert bool(torch.isfinite(row_delta).all())
    assert math.isclose(float(row_delta.sum()), delta, rel_tol=1e-7, abs_tol=1e-6)
    return dict(original=before, coordinates=after, delta=delta), row_delta, m0, m1


def cosine(dot, g2, d2):
    return dot / math.sqrt(g2 * d2) if g2 > 0 and d2 > 0 else None


def gradient_rows(g, dz, dq):
    gd = g.double()
    return dict(g2=gd.square().sum(1), max_abs=gd.abs().amax(1),
                below=(gd.abs() < ADAM_EPS).sum(1), zero=(gd == 0).sum(1),
                dot_dz=(gd * dz).sum(1), dot_dq=(gd * dq).sum(1))


def gradient_summary(rows, elements, dz2, dq2):
    g2 = float(rows["g2"].sum())
    dot_z, dot_q = float(rows["dot_dz"].sum()), float(rows["dot_dq"].sum())
    below = int(rows["below"].sum())
    return dict(l2=math.sqrt(g2), max_abs=float(rows["max_abs"].max()),
                abs_below_adam_eps_count=below, abs_below_adam_eps_fraction=below / elements,
                zero_count=int(rows["zero"].sum()), dot_dz=dot_z,
                cos_dz=cosine(dot_z, g2, dz2), dot_dq=dot_q,
                cos_dq=cosine(dot_q, g2, dq2))


def row_gradient_vector(rows, row, columns, dz2, dq2):
    g2 = float(rows["g2"][row])
    dot_z, dot_q = float(rows["dot_dz"][row]), float(rows["dot_dq"][row])
    return [math.sqrt(g2), int(rows["below"][row]) / columns, dot_z,
            cosine(dot_z, g2, float(dz2[row])), dot_q,
            cosine(dot_q, g2, float(dq2[row]))]


def sign_index(x):
    return 0 if x < 0 else 2 if x > 0 else 1


def row_sign_cross(hard_delta, soft_delta, changed):
    h = (hard_delta > 0).long() - (hard_delta < 0).long() + 1
    s = (soft_delta > 0).long() - (soft_delta < 0).long() + 1
    index = h * 3 + s
    counts = torch.bincount(index, minlength=9).reshape(3, 3)
    symbols = torch.bincount(index, weights=changed.double(), minlength=9).long().reshape(3, 3)
    return dict(counts=counts.tolist(), changed_symbols=symbols.tolist())


def analyze_matrix(key, weight, scale_fp32, z, heaps, top_k):
    scale = scale_fp32.to(torch.bfloat16).float()
    assert weight.ndim == 2 and scale.shape == (weight.shape[0],)
    assert bool(torch.isfinite(scale).all() & (scale != 0).all()), key
    assert z.shape == weight.shape and z.dtype == torch.float32, key
    assert bool(torch.isfinite(z).all() & (z >= -7).all() & (z <= 7).all()), key
    q0 = (weight / scale[:, None]).round().clamp(-7, 7).to(torch.int8)
    q1 = z.round().clamp(-7, 7).to(torch.int8)
    elements, columns = z.numel(), z.shape[1]
    changed = (q0 != q1).sum(1)
    changed_z = (z != q0.float()).sum(1)
    dz, dq = (z - q0.float()).double(), (q1.float() - q0.float()).double()
    dw = dq * scale.double()[:, None]
    dz2, dq2, dw2 = dz.square().sum(1), dq.square().sum(1), dw.square().sum(1)
    hard0, hard1 = hard_row_counts(q0), hard_row_counts(q1)
    hard_h, hard_delta, hm0, hm1 = row_entropy_delta(hard0, hard1, elements, 1e-30)
    soft0, h0_fp32, g0 = soft_statistics(q0.float())
    soft1, h1_fp32, g1 = soft_statistics(z)
    soft_h, soft_delta, sm0, sm1 = row_entropy_delta(soft0, soft1, elements, 1e-12)
    gr0, gr1 = gradient_rows(g0, dz, dq), gradient_rows(g1, dz, dq)
    table = transitions(q0, q1)
    assert int(table.sum()) == elements
    assert elements - int(table.diagonal().sum()) == int(changed.sum())
    tiny = scale.abs() < TINY_SCALE
    bits = scale.to(torch.bfloat16).view(torch.uint16)
    row_cache = {}

    def row_record(row):
        if row in row_cache:
            return row_cache[row]
        counts = transitions(q0[row], q1[row]).flatten()
        sparse = [[int(i) // 15 - 7, int(i) % 15 - 7, int(counts[i])]
                  for i in torch.where(counts != 0)[0].tolist()]
        record = dict(row=row, changed_symbols=int(changed[row]),
                      changed_z=int(changed_z[row]), signed_bf16_scale=float(scale[row]),
                      abs_scale=float(scale[row].abs()), bf16_bits=int(bits[row]),
                      tiny_scale=bool(tiny[row]), transitions=sparse,
                      hard_entropy_delta_nats=float(hard_delta[row]),
                      soft_entropy_delta_nats=float(soft_delta[row]),
                      dz_l2=math.sqrt(float(dz2[row])), dz_max_abs=float(dz[row].abs().max()),
                      dq_l2=math.sqrt(float(dq2[row])), encoded_W_delta_l2=math.sqrt(float(dw2[row])),
                      H_original=row_gradient_vector(gr0, row, columns, dz2, dq2),
                      H_coordinates=row_gradient_vector(gr1, row, columns, dz2, dq2),
                      selected_by=[])
        row_cache[row] = record
        return record

    detailed = {}
    for row in torch.where((changed > 0) | tiny)[0].tolist():
        record = row_record(row)
        if int(changed[row]):
            record["selected_by"].append("hard_changed")
        if bool(tiny[row]):
            record["selected_by"].append("tiny_scale")
        detailed[row] = record
    ranking = dict(hard_entropy=hard_delta, soft_entropy=soft_delta,
                   H_original_dot_dz=gr0["dot_dz"], H_coordinates_dot_dz=gr1["dot_dz"],
                   H_original_dot_dq=gr0["dot_dq"], H_coordinates_dot_dq=gr1["dot_dq"])
    for field, values in ranking.items():
        heap = heaps[field]
        for row in values.abs().topk(min(top_k, len(values))).indices.tolist():
            value = float(values[row])
            if value == 0:
                continue
            identity = (abs(value), key, row)
            if len(heap) < top_k or identity > heap[0][:3]:
                item = (*identity, value, row_record(row))
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                else:
                    heapq.heapreplace(heap, item)
    displacement = {}
    for name, d, d2 in (("z", dz, dz2), ("q", dq, dq2), ("encoded_W", dw, dw2)):
        displacement[name] = dict(l2=math.sqrt(float(d2.sum())), max_abs=float(d.abs().max()))
    record = dict(matrix=key, shape=list(z.shape), elements=elements, row_count=z.shape[0],
                  changed_symbols=int(changed.sum()), hard_changed_rows=int((changed > 0).sum()),
                  changed_z=int(changed_z.sum()), changed_z_rows=int((changed_z > 0).sum()),
                  fractional_only_rows=int(((changed_z > 0) & (changed == 0)).sum()),
                  transitions=table.tolist(), hard_entropy_nats=hard_h, soft_entropy_nats=soft_h,
                  trainer_soft_entropy_nats_fp32=dict(original=h0_fp32, coordinates=h1_fp32,
                                                     delta=h1_fp32 - h0_fp32),
                  probabilities=dict(original_hard=(hm0 / elements).tolist(),
                                     coordinates_hard=(hm1 / elements).tolist(),
                                     original_soft=(sm0 / elements).tolist(),
                                     coordinates_soft=(sm1 / elements).tolist()),
                  scale=dict(min_abs=float(scale.abs().min()), max_abs=float(scale.abs().max()),
                             mean_abs=float(scale.double().abs().mean()),
                             negative_rows=int((scale < 0).sum()), tiny_rows=int(tiny.sum())),
                  displacement=displacement,
                  H_gradients=dict(original_q=gradient_summary(gr0, elements, float(dz2.sum()), float(dq2.sum())),
                                   coordinates_z=gradient_summary(gr1, elements, float(dz2.sum()), float(dq2.sum()))),
                  row_sign_cross=row_sign_cross(hard_delta, soft_delta, changed),
                  all_rows_soft_entropy_delta_nats_sum=float(soft_delta.sum()), rows=detailed)
    check_rss()
    return record, table, (hm0, hm1, sm0, sm1)


def census(state, coordinates, top_k=20, expected=EXPECTED):
    assert isinstance(state, dict) and isinstance(coordinates, dict)
    for key, value in state.items():
        assert isinstance(value, torch.Tensor) and value.dtype == torch.float32, key
        assert value.device.type == "cpu" and bool(torch.isfinite(value).all()), key
    keys = sorted(k for k in state if k.endswith(".weight") and
                  k[:-7] + ".quantize_weight.scale" in state)
    counts = (len(state), sum(t.numel() for t in state.values()), len(keys),
              sum(state[k].numel() for k in keys))
    assert counts == expected, (counts, expected)
    assert set(coordinates) == set(keys)
    heaps = {field: [] for field in RANK_FIELDS}
    matrices = []
    total_transition = torch.zeros((15, 15), dtype=torch.int64)
    mass = [torch.zeros(15, dtype=torch.float64) for _ in range(4)]
    for key in keys:
        m, table, masses = analyze_matrix(key, state[key],
            state[key[:-7] + ".quantize_weight.scale"], coordinates[key], heaps, top_k)
        matrices.append(m)
        total_transition += table
        for accumulator, contribution in zip(mass, masses):
            accumulator += contribution
    by_key = {m["matrix"]: m for m in matrices}
    top_rows = {}
    for field, heap in heaps.items():
        top_rows[field] = []
        for _, key, row, value, record in sorted(heap, reverse=True):
            target = by_key[key]["rows"]
            record = target.setdefault(row, record)
            record["selected_by"].append("top_" + field)
            top_rows[field].append(dict(matrix=key, row=row, value=value))
    for m in matrices:
        m["rows"] = [m["rows"][row] for row in sorted(m["rows"])]
        reported = math.fsum(r["soft_entropy_delta_nats"] for r in m["rows"])
        m["reported_rows_soft_entropy_delta_nats_sum"] = reported
        m["omitted_rows_soft_entropy_delta_nats_sum"] = m["all_rows_soft_entropy_delta_nats_sum"] - reported
        m["omitted_row_count"] = m["row_count"] - len(m["rows"])
        assert math.isclose(math.fsum(r["hard_entropy_delta_nats"] for r in m["rows"]),
                            m["hard_entropy_nats"]["delta"], abs_tol=1e-6, rel_tol=1e-7)
    elements, rows = counts[3], sum(m["row_count"] for m in matrices)
    global_summary = dict(matrix_count=len(matrices), elements=elements, row_count=rows,
        changed_symbols=sum(m["changed_symbols"] for m in matrices),
        hard_changed_rows=sum(m["hard_changed_rows"] for m in matrices),
        changed_z=sum(m["changed_z"] for m in matrices),
        changed_z_rows=sum(m["changed_z_rows"] for m in matrices),
        fractional_only_rows=sum(m["fractional_only_rows"] for m in matrices),
        transitions=total_transition.tolist(),
        probabilities=dict(zip(("original_hard", "coordinates_hard", "original_soft", "coordinates_soft"),
                               [(p / elements).tolist() for p in mass])),
        soft_probability_sum_roundoff=dict(original=float(mass[2].sum()) / elements - 1,
                                          coordinates=float(mass[3].sum()) / elements - 1),
        scale=dict(min_abs=min(m["scale"]["min_abs"] for m in matrices),
                   max_abs=max(m["scale"]["max_abs"] for m in matrices),
                   mean_abs=math.fsum(m["scale"]["mean_abs"] * m["row_count"] for m in matrices) / rows,
                   negative_rows=sum(m["scale"]["negative_rows"] for m in matrices),
                   tiny_rows=sum(m["scale"]["tiny_rows"] for m in matrices)),
        reported_rows_soft_entropy_delta_nats_sum=math.fsum(m["reported_rows_soft_entropy_delta_nats_sum"] for m in matrices),
        omitted_rows_soft_entropy_delta_nats_sum=math.fsum(m["omitted_rows_soft_entropy_delta_nats_sum"] for m in matrices))
    for field in ("hard_entropy_nats", "soft_entropy_nats", "trainer_soft_entropy_nats_fp32"):
        global_summary[field] = {point: math.fsum(m[field][point] for m in matrices)
                                 for point in ("original", "coordinates", "delta")}
        global_summary[field]["delta_times_2_over_full_targets"] = global_summary[field]["delta"] * H_FACTOR
    global_summary["displacement"] = {}
    for name in ("z", "q", "encoded_W"):
        global_summary["displacement"][name] = dict(
            l2=math.sqrt(math.fsum(m["displacement"][name]["l2"] ** 2 for m in matrices)),
            max_abs=max(m["displacement"][name]["max_abs"] for m in matrices))
    global_summary["H_gradients"] = {}
    dz2 = global_summary["displacement"]["z"]["l2"] ** 2
    dq2 = global_summary["displacement"]["q"]["l2"] ** 2
    for point in ("original_q", "coordinates_z"):
        gradients = [m["H_gradients"][point] for m in matrices]
        g2 = math.fsum(g["l2"] ** 2 for g in gradients)
        dot_z, dot_q = (math.fsum(g[field] for g in gradients) for field in ("dot_dz", "dot_dq"))
        below = sum(g["abs_below_adam_eps_count"] for g in gradients)
        global_summary["H_gradients"][point] = dict(l2=math.sqrt(g2),
            max_abs=max(g["max_abs"] for g in gradients), abs_below_adam_eps_count=below,
            abs_below_adam_eps_fraction=below / elements, zero_count=sum(g["zero_count"] for g in gradients),
            dot_dz=dot_z, cos_dz=cosine(dot_z, g2, dz2), dot_dq=dot_q, cos_dq=cosine(dot_q, g2, dq2))
    sign_cross = {name: torch.zeros((3, 3), dtype=torch.int64) for name in
                  ("matrix_counts", "matrix_elements", "matrix_changed_symbols",
                   "row_counts", "row_elements", "row_changed_symbols")}
    for m in matrices:
        h, s = sign_index(m["hard_entropy_nats"]["delta"]), sign_index(m["soft_entropy_nats"]["delta"])
        for name, amount in (("matrix_counts", 1), ("matrix_elements", m["elements"]),
                             ("matrix_changed_symbols", m["changed_symbols"])):
            sign_cross[name][h, s] += amount
        row_counts = torch.tensor(m["row_sign_cross"]["counts"], dtype=torch.int64)
        sign_cross["row_counts"] += row_counts
        sign_cross["row_elements"] += row_counts * m["shape"][1]
        sign_cross["row_changed_symbols"] += torch.tensor(m["row_sign_cross"]["changed_symbols"])
    global_summary["soft_hard_sign_cross"] = dict(order=["decrease", "zero", "increase"],
        row_axis="hard_entropy_delta", column_axis="soft_entropy_delta",
        **{key: value.tolist() for key, value in sign_cross.items()})
    def matrix_rank_value(m, field):
        if field.endswith("entropy"):
            return m[field + "_nats"]["delta"]
        point = "original_q" if field.startswith("H_original") else "coordinates_z"
        return m["H_gradients"][point]["dot_dz" if field.endswith("dz") else "dot_dq"]
    top_matrices = {field: [dict(matrix=m["matrix"], value=matrix_rank_value(m, field))
        for m in sorted(matrices, key=lambda m: (abs(matrix_rank_value(m, field)), m["matrix"]), reverse=True)
        if matrix_rank_value(m, field) != 0][:top_k] for field in RANK_FIELDS}
    tiny_rows = [dict(matrix=m["matrix"], row=r["row"]) for m in matrices for r in m["rows"] if r["tiny_scale"]]
    assert int(total_transition.sum()) == elements
    assert global_summary["changed_symbols"] == elements - int(total_transition.diagonal().sum())
    assert math.isclose(global_summary["reported_rows_soft_entropy_delta_nats_sum"] +
                        global_summary["omitted_rows_soft_entropy_delta_nats_sum"],
                        global_summary["soft_entropy_nats"]["delta"], abs_tol=1e-6, rel_tol=1e-7)
    return dict(status="passed", schema="integer-coordinate-census-v1",
                checkpoint_tensor_count=counts[0], checkpoint_parameter_count=counts[1],
                frozen_method=dict(method="fixed-scale-integer-coordinate-v1", tau=TAU,
                    full_targets=FULL_TARGETS, entropy_coefficient=2, H_gradient_factor=H_FACTOR,
                    adam_eps=ADAM_EPS, tiny_scale_abs_threshold=TINY_SCALE),
                definitions=dict(symbol_order=list(range(-7, 8)),
                    transitions="rows=original q, columns=round(saved z); row triples=[from,to,count], including nonzero diagonal",
                    entropy="sum of per-matrix code lengths; global probabilities are element-weighted marginals",
                    soft_accounting="FP32 softmax assignments; FP64 mass reductions; log probability clamp=1e-12",
                    trainer_soft_entropy="original frozen FP32 formula; H_gradients always differentiate 2*H/FULL_TARGETS",
                    row_entropy_delta="exact per-symbol loss secants along linear aggregate mass interpolation, dotted with each row mass delta",
                    H_gradient_row_columns=GRADIENT_COLUMNS,
                    dot="dot of the weighted positive H gradient with saved z-original q or hard q-original q; positive is local H ascent",
                    cosine="dot/(gradient l2 * displacement l2); null for a zero norm",
                    encoded_W="hard q times original signed BF16 scale; static15 framing and finite-precision coder bytes are measured separately",
                    hard_weighted_delta="hard entropy times 2/FULL_TARGETS is a diagnostic proxy; the frozen objective uses soft H",
                    row_selection="every hard-changed row and tiny-scale row, plus global top contributors; fractional-only remainder stays in sums",
                    ranking="largest absolute contribution, retaining signed values; exact zero excluded"),
                global_summary=global_summary, top_matrices=top_matrices, top_rows=top_rows,
                tiny_rows=tiny_rows, matrices=matrices)


def self_check():
    """One 3x4 fixture: four hard changes, signed tiny scales, additive ledger."""
    q = torch.tensor([[-2., -1., 0., 0.], [1., 2., 2., 7.], [0., 0., 0., 0.]])
    scales = torch.tensor([.0313, -8e-15, 6e-15]).to(torch.bfloat16).float()
    z = torch.tensor([[-1.4, -.7, .2, .7], [1.6, 1.8, 2.2, 6.4], [.1, -.2, .3, -.1]])
    state = {"fixture.weight": q * scales[:, None], "fixture.quantize_weight.scale": scales}
    report = census(state, {"fixture.weight": z}, top_k=2, expected=(2, 15, 1, 12))
    summary = report["global_summary"]
    table = summary["transitions"]
    assert summary["changed_symbols"] == 4 and sum(map(sum, table)) == 12
    assert summary["changed_z_rows"] == 3 and summary["fractional_only_rows"] == 1
    for before, after in ((-2, -1), (0, 1), (1, 2), (7, 6)):
        assert table[before + 7][after + 7] == 1
    assert sum(table[i][i] for i in range(15)) == 8
    original_counts = {-2: 1, -1: 1, 0: 6, 1: 1, 2: 2, 7: 1}
    expected_h = -sum(n * math.log(n / 12) for n in original_counts.values())
    assert math.isclose(summary["hard_entropy_nats"]["original"], expected_h, abs_tol=1e-12)
    rows = report["matrices"][0]["rows"]
    for field in ("hard", "soft"):
        assert math.isclose(math.fsum(r[field + "_entropy_delta_nats"] for r in rows),
                            summary[field + "_entropy_nats"]["delta"], abs_tol=1e-10)
    _, _, weighted = soft_statistics(z)
    probe = z.clone().requires_grad_(True)
    unweighted, = torch.autograd.grad(entropy(probe), probe)
    torch.testing.assert_close(weighted, unweighted * H_FACTOR, rtol=2e-6, atol=1e-14)
    assert bool(torch.isfinite(weighted).all()) and bool((weighted != 0).any())
    assert summary["scale"]["tiny_rows"] == 2 and rows[1]["signed_bf16_scale"] < 0
    decoded = json.loads(json.dumps(report, allow_nan=False))
    assert decoded["status"] == "passed" and len(decoded["matrices"][0]["rows"]) == 3
    return dict(status="passed", fixture_shape=[3, 4], elements=12, changed_symbols=4,
                transition_sum=12, additive_entropy_ledgers=True, weighted_gradient_finite=True,
                signed_tiny_scales=True, json_roundtrip=True)


def file_identity(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="original 429-key 6m-q4-fp32.tch")
    parser.add_argument("--coordinates", type=Path, help="immutable epoch-02-z.pt tensor dictionary")
    parser.add_argument("--output", type=Path, help="JSON receipt path")
    parser.add_argument("--top-k", type=int, default=20, help="global matrix/row ranking length (default: 20)")
    parser.add_argument("--self-check", action="store_true", help="run the small synthetic assertion before census, or alone")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k requires a positive integer")
    actual = args.checkpoint is not None or args.coordinates is not None
    if actual and any(p is None for p in (args.checkpoint, args.coordinates, args.output)):
        parser.error("census requires --checkpoint, --coordinates and --output together")
    if not actual and not args.self_check:
        parser.error("supply census paths or --self-check")
    if actual and args.output.resolve() in (args.checkpoint.resolve(), args.coordinates.resolve()):
        parser.error("choose an output path distinct from the input checkpoints")
    started, cpu_started = time.monotonic(), time.process_time()
    setup_torch()
    check = self_check() if args.self_check else None
    if actual:
        identity = dict(checkpoint=file_identity(args.checkpoint), coordinates=file_identity(args.coordinates),
                        source=file_identity(Path(__file__)))
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        coordinates = torch.load(args.coordinates, map_location="cpu", weights_only=True, mmap=True)
        report = census(state, coordinates, top_k=args.top_k)
        report.update(identity=identity, self_check=check)
    else:
        report = dict(status="passed", self_check=check)
    report["resources"] = dict(cpu_seconds=time.process_time() - cpu_started,
        wall_seconds=time.monotonic() - started, peak_rss_kib=check_rss(), rss_limit_bytes=RSS_LIMIT_BYTES,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        torch_version=torch.__version__, matrix_processing="sequential")
    encoded = json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(json.dumps(dict(status="passed", output=str(args.output) if args.output else None,
        self_check=check, global_summary=report.get("global_summary"), resources=report["resources"])), flush=True)


if __name__ == "__main__":
    main()
