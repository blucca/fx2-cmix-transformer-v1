#!/usr/bin/env python3
"""Fixed algebra controls: stdlib binary64, energy and alternating-key cycles."""
import json
import math
from pathlib import Path

TOL = 1e-11


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def energy(a):
    return math.fsum(x * x for row in a for x in row)


def norm(v):
    return math.sqrt(dot(v, v))


def close(a, b, tol=TOL):
    assert abs(a - b) <= tol * max(1.0, abs(a), abs(b)), (a, b)


def matrix_close(a, b):
    for ar, br in zip(a, b):
        for x, y in zip(ar, br):
            close(x, y)


def subtract(a, b):
    return [[x - y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def add(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def step(s, k, v, d, beta):
    a = [[di * x for x in row] for di, row in zip(d, s)]
    r = [math.fsum(ki * row[j] for ki, row in zip(k, a)) for j in range(len(v))]
    out = [[x + beta * ki * (vj - rj) for x, vj, rj in zip(row, v, r)]
           for ki, row in zip(k, a)]
    return out, a, r


S = [[.5, -.25, .8], [.1, .6, -.3], [-.7, .2, .4], [.9, -.5, .15]]
E = [[.2, .3, -.1], [-.1, .4, .6], [.7, -.2, .1], [-.3, .5, -.4]]
V = [.6, -1.1, .4]
D = [.2, .8, 1., 0.]
raw = [.3, -.7, .2, .4]
den = math.sqrt(dot(raw, raw) + 1e-6)
K = [x / den for x in raw]
close(dot(K, K), dot(raw, raw) / (dot(raw, raw) + 1e-6))
assert 0.0 < dot(K, K) < 1.0

beta_sharp = .7
alpha_sharp = (1 - beta_sharp * dot(K, K)) / (2 - beta_sharp * dot(K, K))
h = [di * ki for di, ki in zip(D, K)]
sharp_s = [[hi * alpha_sharp * vj / dot(h, h) for vj in V] for hi in h]
cases = [
    ("beta_zero", S, K, V, D, 0.),
    ("key_zero", S, [0.] * 4, V, D, .6),
    ("value_zero", S, K, [0.] * 3, D, .4),
    ("eps_normalized_anisotropic", S, K, V, D, .35),
    ("unit_key_beta_one", S, [1., 0., 0., 0.], V, [1.] * 4, 1.),
    ("sharp_interior", sharp_s, K, V, D, beta_sharp),
    ("annihilated_key", S, [1., 0., 0., 0.], V, [0., .8, .4, 1.], .7),
    ("annihilated_key_sharp_endpoint", S, [1., 0., 0., 0.], V,
     [0., .8, .4, 1.], 1.),
]

rows = []
for name, s, k, v, d, beta in cases:
    out, a, r = step(s, k, v, d, beta)
    c = dot(k, k)
    delta = energy(out) - energy(a)
    expanded = (-beta * (2 - beta * c) * dot(r, r)
                + 2 * beta * (1 - beta * c) * dot(r, v)
                + beta * beta * c * dot(v, v))
    alpha = (1 - beta * c) / (2 - beta * c)
    residual = [ri - alpha * vi for ri, vi in zip(r, v)]
    injection = beta / (2 - beta * c) * dot(v, v)
    completed = -beta * (2 - beta * c) * dot(residual, residual) + injection
    close(delta, expanded)
    close(delta, completed)
    assert delta <= injection + TOL
    bound = max(d) ** 2 * energy(s) + injection
    assert energy(out) <= bound + TOL
    if name in {"beta_zero", "key_zero"}:
        close(delta, 0.)
    if name in {"sharp_interior", "annihilated_key_sharp_endpoint"}:
        close(delta, injection)
    shifted, _, _ = step(add(s, E), k, v, d, beta)
    homogeneous, _, _ = step(E, k, [0.] * 3, d, beta)
    matrix_close(subtract(shifted, out), homogeneous)
    assert energy(homogeneous) <= max(d) ** 2 * energy(E) + TOL
    rows.append({"case": name, "c": c, "delta": delta,
                 "injection_bound": injection, "state_bound": bound,
                 "state_energy": energy(out)})

# A D=I projection preserves the direction orthogonal to its key.
perp = [[0., 0., 0.], [.2, -.3, .4], [0., 0., 0.], [0., 0., 0.]]
preserved, _, _ = step(perp, [1., 0., 0., 0.], [0.] * 3, [1.] * 4, 1.)
matrix_close(preserved, perp)

# Mathematical conditioned-key control; each input is explicitly fixed.
toys = []
for theta in (.2, .1):
    co, si = math.cos(theta), math.sin(theta)
    kp, km = [co, si], [co, -si]
    target = [[0.], [1. / si]]
    close(dot(kp, kp), 1.)
    close(dot(km, km), 1.)
    for k, value in ((kp, 1.), (km, -1.)):
        fitted, _, _ = step(target, k, [value], [1., 1.], 1.)
        matrix_close(fitted, target)
        orthogonal = [[-k[1]], [k[0]]]
        image, _, _ = step(orthogonal, k, [0.], [1., 1.], 1.)
        matrix_close(image, orthogonal)
        close(energy(image), 1.)
    state = [[0.], [0.]]
    cycles = 2000
    for _ in range(cycles):
        state, _, _ = step(state, kp, [1.], [1., 1.], 1.)
        state, _, _ = step(state, km, [-1.], [1., 1.], 1.)
    error = math.sqrt(energy(subtract(state, target)))
    assert error < 1e-7, (theta, error)
    close(energy(target), 1. / (si * si))
    assert energy(state) <= 2 * cycles + TOL
    gamma = math.cos(2 * theta)
    toys.append({"theta": theta, "cycles": cycles, "steps": 2 * cycles,
                 "homogeneous_step_operator_norm": 1.,
                 "cycle_spectral_radius": gamma * gamma,
                 "cycle_operator_norm": abs(gamma),
                 "fixed_point": [0., 1. / si], "fixed_point_norm": 1. / si,
                 "key_pair_condition_number": co / si,
                 "final_state": [row[0] for row in state],
                 "final_error_norm": error,
                 "exact_arithmetic_error_bound": abs(gamma) ** (2 * cycles - 1) / si})

# Constant forgetting: the alternating stream attains the geometric energy cap.
# Three fixed gate/decay cases cover the interior and beta=1-d boundary.
decayed = []
for decay, beta in ((.99, 1.), (.95, .4), (.8, .2)):
    y = (1 - decay) * (1 + decay - beta) / (2 * decay * beta)
    assert 0. <= y <= 1., (decay, beta, y)
    si, co = math.sqrt(y), math.sqrt(1 - y)
    z = ((1 - decay) * (1 + decay - decay * beta)
         + 2 * decay * beta * y)
    a, b = beta * (1 - decay) * co / z, beta * (1 + decay) * si / z
    plus, minus = [[a], [b]], [[-a], [b]]
    target_energy = beta / ((2 - beta) * (1 - decay * decay))
    for before, k, value, after in (
        (minus, [co, si], 1., plus), (plus, [co, -si], -1., minus)
    ):
        actual, _, retrieval = step(before, k, [value], [decay] * 2, beta)
        matrix_close(actual, after)
        close(retrieval[0], (1 - beta) / (2 - beta) * value)
        close(energy(actual), target_energy)
    row = {"decay": decay, "beta": beta, "theta": math.asin(si),
           "plus_state": [a, b], "minus_state": [-a, b],
           "state_energy": energy(plus), "geometric_energy_bound": target_energy}
    if beta == 1.:
        gamma = 1 - 2 * y
        close(gamma, decay)
        cycles = 200
        state = [[0.], [0.]]
        for _ in range(cycles):
            state, _, _ = step(state, [co, si], [1.], [decay] * 2, beta)
            state, _, _ = step(state, [co, -si], [-1.], [decay] * 2, beta)
        observed_error = math.sqrt(energy(subtract(state, minus)))
        exact_bound = decay ** (4 * cycles - 1) * math.sqrt(target_energy)
        close(observed_error, exact_bound)
        row.update({"cycles": cycles, "cycle_spectral_radius": decay ** 4,
                    "spectral_half_life_updates": 2 * math.log(.5) / math.log(decay ** 4),
                    "final_error_norm": observed_error,
                    "exact_arithmetic_error_bound": exact_bound})
    decayed.append(row)

# Hold the nominal optimal keys fixed while changing the actual decay coordinate.
nominal_decay, gate_x = .999, -8.
co = math.sqrt((1 + nominal_decay) / 2)
si = math.sqrt((1 - nominal_decay) / 2)


def gate_cycle(decay):
    z = 1 - nominal_decay * decay
    return [[(1 - decay) * co / z], [(1 + decay) * si / z]]


nominal = gate_cycle(nominal_decay)
amplitude = math.sqrt(energy(nominal))
softplus = math.log1p(math.exp(gate_x))
sigmoid = math.exp(gate_x) / (1 + math.exp(gate_x))
a_fixed = -math.log(nominal_decay) / softplus
close(math.exp(-a_fixed * softplus), nominal_decay)
relative_sensitivity = (nominal_decay * a_fixed * sigmoid
                        / (1 - nominal_decay * nominal_decay))
close(relative_sensitivity,
      -math.log(nominal_decay) / (2 * math.sinh(-math.log(nominal_decay)))
      * sigmoid / softplus)
assert 0. < relative_sensitivity <= .5
responses = []
for name, new_decay in (
    ("absolute_decay_change", nominal_decay + 1e-5),
    ("source_gate_change", math.exp(-a_fixed * math.log1p(math.exp(gate_x + .01)))),
):
    observed = math.sqrt(energy(subtract(gate_cycle(new_decay), nominal)))
    exact = amplitude * abs(new_decay - nominal_decay) / (1 - nominal_decay * new_decay)
    close(observed, exact)
    responses.append({"coordinate_control": name, "new_decay": new_decay,
                      "state_difference_norm": observed, "exact_formula": exact})
gate_response = {"nominal_decay": nominal_decay, "source_gate_x": gate_x,
                 "a_fixed": a_fixed, "state_norm": amplitude,
                 "absolute_decay_local_sensitivity": amplitude / (1 - nominal_decay**2),
                 "relative_source_gate_local_sensitivity": relative_sensitivity,
                 "relative_source_gate_upper_bound": .5, "finite_responses": responses}

result = {"status": "pass", "execution_scope": "stdlib binary64 algebra controls",
          "tolerance": TOL, "energy_cases": rows, "conditioned_key_toys": toys}
result["decayed_key_toys"] = decayed
result["fixed_key_gate_response"] = gate_response
result_path = Path(__file__).parent / "results" / "kda-state-energy.json"
result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "pass", "energy_cases": len(rows),
                  "toy_cases": len(toys), "decayed_cycle_cases": len(decayed),
                  "result_path": str(result_path)}))
