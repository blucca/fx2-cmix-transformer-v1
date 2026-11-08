"""Check this fixed full12 result against the original public unary census."""
import json
import math
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
result = json.loads(Path(sys.argv[1]).read_text())
old = {x["name"]: x for x in json.loads((root / "results/weights-census.json").read_text())["per_tensor"]}
rows, totals = result["matrices"], result["totals"]
assert result["checks"] == "passed" and len(rows) == len(old) == 111
for row in rows:
    reference = old[row["name"]]
    assert row["histogram"] == reference["histogram"]
    assert [row["rows"], row["cols"]] == reference["shape"]
    assert row["elements"] == reference["elements"]
    # The old public result rounds numeric costs to 12 significant figures.
    assert abs(row["unary_bits"] - reference["oracle_tensor_bits"]) < .001
    assert abs(row["old_static4096_data_bits"] + 224 - reference["static4096_with_28B_table_bits"]) < .001
    assert sum(row["joint"]) == row["pairs"]
    assert sum(row["tail"]) == row["tails"]
    assert 2 * row["pairs"] + row["tails"] == row["elements"]
    assert all(row["histogram"][i] == row["left"][i] + row["right"][i] + row["tail"][i] for i in range(15))
    extra = 448 + (28 if row["tails"] else 0) - 28 if row["pairs"] else 0
    assert extra == row["literal_table_delta_bytes"]
for key in ("elements", "pairs", "tails", "literal_table_delta_bytes"):
    assert sum(row[key] for row in rows) == totals[key]
for key in ("unary_bits", "pair_oracle_bits_with_old_tail_model", "old_static4096_data_bits", "pair_static4096_data_bits"):
    assert abs(math.fsum(row[key] for row in rows) - totals[key]) < .000001
assert (totals["elements"], totals["pairs"], totals["tails"]) == (5868864, 2934336, 192)
oracle = (totals["unary_bits"] - totals["pair_oracle_bits_with_old_tail_model"]) / 8
static = (totals["old_static4096_data_bits"] - totals["pair_static4096_data_bits"]) / 8
print(json.dumps({"checks": "passed", "all_111_unary_reference_matches": True,
    "oracle_q_data_saved_bytes_per_W": oracle,
    "entropy_advantage_bits_per_pair": 8 * oracle / totals["pairs"],
    "static4096_q_data_saved_bytes_per_W": static,
    "literal_table_delta_bytes_per_W": totals["literal_table_delta_bytes"],
    "literal_table_scenario_W_loss_bytes": totals["literal_table_delta_bytes"] - static}, indent=2))
