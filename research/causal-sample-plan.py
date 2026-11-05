#!/usr/bin/env python3
"""Seeded whole-piece sampling with article-cluster holdout and exact NN resets.

Each eligible piece has the same inclusion probability. Target-weighted means
estimate the <=32768-token learning population; native validation also admits
long pieces. Full 2^17 pieces are retained for native scoring, with explicit
int32 bounds so concatenated nonadjacent pieces never share a context.
"""
from array import array
import argparse
import json
from pathlib import Path
import random
import sys

MAX_PIECE = 1 << 17
TRAIN_LIMIT = 32768
SEED = 414
PROBABILITY = .0075
HOLDOUT = .25


def select(boundaries, probability=PROBABILITY, seed=SEED):
    assert boundaries[0] == 0 and all(b > a for a, b in zip(boundaries, boundaries[1:]))
    assert 0 < probability <= 1
    group_rng, piece_rng = random.Random(seed), random.Random(seed + 1)
    rows, totals = [], {"full_targets": 0, "eligible_train_targets": 0}
    for article, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        held = group_rng.random() < HOLDOUT
        for lo in range(start, end, MAX_PIECE):
            hi = min(lo + MAX_PIECE, end)
            targets = hi - lo - 1
            totals["full_targets"] += targets
            if hi - lo <= TRAIN_LIMIT:
                totals["eligible_train_targets"] += targets
            draw = piece_rng.random()  # Same draw path for all lengths/roles.
            if targets > 0 and draw < probability and (held or hi - lo <= TRAIN_LIMIT):
                rows.append(dict(begin=lo, end=hi, article=article,
                                 role="validation" if held else "train", targets=targets))
    return rows, totals


def check():
    rows, totals = select([0, 20, 40, MAX_PIECE + 45], probability=1, seed=3)
    assert totals["full_targets"] == MAX_PIECE + 41
    assert all(row["end"] - row["begin"] <= MAX_PIECE for row in rows)
    assert all(row["end"] - row["begin"] <= TRAIN_LIMIT for row in rows if row["role"] == "train")
    groups = {}
    for row in rows:
        groups.setdefault(row["article"], set()).add(row["role"])
    assert all(len(roles) == 1 for roles in groups.values())
    assert select([0, 20, 40, MAX_PIECE + 45], probability=1, seed=3) == (rows, totals)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("boundaries", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    check()
    assert sys.byteorder == "little" and array("i").itemsize == 4
    boundaries = array("i"); boundaries.frombytes(args.boundaries.read_bytes())
    rows, totals = select(boundaries)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    assert not (out / "plan.json").exists(), "use a fresh sampling directory"
    (out / "plan.tsv").write_text("".join(f'{r["begin"]} {r["end"]} {r["role"]}\n' for r in rows))
    roles = {}
    for role in ("train", "validation"):
        selected = [r for r in rows if r["role"] == role]
        bounds = array("i", [0]); offsets = []
        for row in selected:
            offsets.append([bounds[-1], bounds[-1] + row["end"] - row["begin"]])
            bounds.append(offsets[-1][1])
        assert selected and bounds[-1] < 1 << 31
        (out / f"{role}.i32").write_bytes(bounds.tobytes())
        roles[role] = dict(pieces=len(selected), bytes=bounds[-1],
                           targets=sum(r["targets"] for r in selected),
                           pieces_local=offsets,
                           max_piece_bytes=max(r["end"] - r["begin"] for r in selected))
    result = dict(method="causal-whole-piece-v1", seed=SEED, probability=PROBABILITY,
                  article_holdout_probability=HOLDOUT, train_piece_limit=TRAIN_LIMIT,
                  native_piece_limit=MAX_PIECE, source_bytes=boundaries[-1],
                  **totals, roles=roles, intervals=rows)
    (out / "plan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "intervals" and k != "roles"}))
    print(json.dumps({role: {k: v for k, v in info.items() if k != "pieces_local"}
                      for role, info in roles.items()}))


if __name__ == "__main__":
    main()
