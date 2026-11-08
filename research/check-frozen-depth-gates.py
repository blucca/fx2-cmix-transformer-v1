"""Replay the saved frozen-W endpoint and target-weighted observations."""
import json, math
from pathlib import Path
x = json.loads(Path(__file__).with_name("results").joinpath("frozen-depth-gates.json").read_text())
ps=x["pieces"]; n=sum(p["targets"] for p in ps)
assert x["status"] == "passed" and n == x["targets"] == 3934
assert [p["piece"] for p in ps] == [0,7,14]
assert x["actual_w"]["selected_epoch"] == 12 and x["actual_w"]["bytes"] == 1960982
assert x["actual_w"]["all_loaded_f32_bits_equal"] and x["actual_w"]["hard_requantized_int4_and_bf16_scale_equal"]
assert all(p["g0_control"]["passed"] and all(p["g0_control"][k]==0 for k in ("hidden_max_abs","logits_max_abs","target_nll_abs")) for p in ps)
assert x["optimizer_updates"] == x["gate_fits"] == x["native_evaluator_calls"] == x["model_exports"] == 0
assert (x["teacher_forwards"],x["fixed8_forwards"],x["gated12_forwards"],x["gate_vjps"]) == (3,3,6,6)
assert x["tiny_control"]["gate_vector"] == [.001]*4
for k in ("ce","teacher_kl"):
    delta=math.fsum(p["targets"]*p["tiny_control"]["delta"][k] for p in ps)/n
    assert math.isclose(delta,x["target_weighted"]["tiny_delta"][k],rel_tol=0,abs_tol=1e-14) and delta>0
    g=[math.fsum(p["targets"]*p["gate_derivatives"][k][i] for p in ps)/n for i in range(4)]
    assert all(math.isfinite(v) and math.isclose(v,z,abs_tol=1e-12) for v,z in zip(g,x["target_weighted"]["gate_derivatives"][k]))
ce,kl=(x["target_weighted"]["gate_derivatives"][k] for k in ("ce","teacher_kl"))
assert [i for i,(a,b) in enumerate(zip(ce,kl)) if a>0 and b>0] == [0,2]
assert [i for i,(a,b) in enumerate(zip(ce,kl)) if a<0 and b<0] == []
print("frozen hard-W endpoints and weighted observations PASS")
