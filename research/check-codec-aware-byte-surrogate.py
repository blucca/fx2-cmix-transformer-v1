#!/usr/bin/env python3
"""Synthetic, standard-library check of a frozen local byte-to-bit objective."""
import json
import math

V = [0, 7, 13, 55, 83, 201, 255]  # Same ascending-byte mapping as native; small fixture.
OBS = 83
Z0 = [-9., -.7, .2, 1.1, -.2, -4., .4]
Z = [v + (i - 3) * .017 for i, v in enumerate(Z0)]


def probabilities(z):
    e = [math.exp(v - max(z)) for v in z]
    return [v / sum(e) for v in e]


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def bce(x, y):
    return max(x, 0) + math.log1p(math.exp(-abs(x))) - y*x


def bit_rows(z, floor):
    p = probabilities(z)
    q = [max(v, floor) for v in p]
    rows, lo, hi = [], 0, 255
    for k in range(8):
        mid = lo + (hi-lo)//2
        left = [i for i, b in enumerate(V) if lo <= b <= mid]
        right = [i for i, b in enumerate(V) if mid < b <= hi]
        y = int(OBS > mid)
        if left and right:  # Vocabulary-deterministic bit has zero task gradient.
            q0, q1 = sum(q[i] for i in left), sum(q[i] for i in right)
            x = math.log(q1) - math.log(q0)
            a = [(int(i in right)/q1 - int(i in left)/q0) if p[i] > floor else 0. for i in range(len(V))]
            average = sum(pi*ai for pi, ai in zip(p, a))
            g = [pi*(ai-average) for pi, ai in zip(p, a)]
            rows.append((k, y, x, g))
        lo, hi = (mid+1, hi) if y else (lo, mid)
    return rows


def objective(z, floor, slopes):
    refs = {k: x for k, _, x, _ in bit_rows(Z0, floor)}
    loss, gradient = 0., [0.] * len(V)
    for k, y, x, g in bit_rows(z, floor):
        s = slopes[k % len(slopes)]
        reference_logit = .3 - .17*k
        eta = reference_logit + s*(x-refs[k])
        loss += bce(eta, y)
        r = s*(sigmoid(eta)-y)
        gradient = [v+r*gi for v, gi in zip(gradient, g)]
    return loss, gradient


h = 1e-5
errors = []
for floor in (0., .05, 1.):
    for slopes in ((1.,), (.31, -.8, 0.)):
        value, gradient = objective(Z, floor, slopes)
        assert value >= 0 and abs(sum(gradient)) < 1e-12
        for j in range(len(V)):
            plus, minus = list(Z), list(Z)
            plus[j] += h
            minus[j] -= h
            fd = (objective(plus, floor, slopes)[0] - objective(minus, floor, slopes)[0])/(2*h)
            errors.append(abs(fd-gradient[j]))
        assert max(errors) < 2e-8
    p = probabilities(Z)
    q = [max(v, floor) for v in p]
    ordinary = sum(bce(x, y) for _, y, x, _ in bit_rows(Z, floor))
    assert abs(ordinary + math.log(q[V.index(OBS)]/sum(q))) < 1e-12
# Interior covariance decomposition can be indefinite in byte-logit coordinates.
def partial_bit_loss(t):
    return math.log(math.exp(t)+math.exp(-t)+1) - math.log(math.exp(t)+math.exp(-t))
curvature = (partial_bit_loss(h)-2*partial_bit_loss(0)+partial_bit_loss(-h))/(h*h)
assert curvature < -.3
for s in (0., -.8, .31):
    assert s*s*sigmoid(.2)*(1-sigmoid(.2)) >= 0
print(json.dumps(dict(status="passed", scope="synthetic frozen-bit surrogate; no NN/codec calls", cases=6,
                      finite_difference_max_error=max(errors), partial_bit_logit_curvature=curvature,
                      checks=["signed/zero affine slope", "guarded softmax gradient", "eight-bit telescoping", "common-logit-shift", "convex x and indefinite byte-logit composition"]), indent=2))
