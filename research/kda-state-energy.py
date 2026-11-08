#!/usr/bin/env python3
"""Fixed algebra controls: stdlib, Python binary64, 4x3 energy + 2x1 key toy."""
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

result = {"status": "pass", "execution_scope": "stdlib binary64 algebra controls",
          "tolerance": TOL, "energy_cases": rows, "conditioned_key_toys": toys}
result_path = Path(__file__).parent / "results" / "kda-state-energy.json"
result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "pass", "energy_cases": len(rows),
                  "toy_cases": len(toys), "result_path": str(result_path)}))
