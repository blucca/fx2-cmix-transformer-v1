#!/usr/bin/env python3
"""Small fixed-contract checks; fake full-output sessions, no model/download/build."""
import numpy as np
from demo_native import BOS, VOCAB, OUTPUTS, INPUTS, check_budget, greedy

def self_test(np):
    """Small fake full-output sessions; no ORT, tokenizer, model or timing claim."""
    check_budget(959, 64)
    check_budget(1022, 1)
    for prompt, new in ((960, 64), (0, 0), (0, 65)):
        try:
            check_budget(prompt, new)
        except ValueError:
            pass
        else:
            raise AssertionError("Context/token boundary accepted invalid input")

    class FakeSession:
        def __init__(self, eos=False, bad_prefix=False):
            self.tokens, self.eos, self.bad_prefix = [], eos, bad_prefix

        def run(self, names, feed):
            assert names == OUTPUTS and list(feed) == INPUTS
            token = int(feed["input_ids"][0, 0])
            self.tokens.append(token)
            length = feed[INPUTS[1]].shape[2]
            assert feed["input_ids"].shape == (1, 1) and feed["input_ids"].dtype == np.int64
            assert feed["attention_mask"].shape == (1, length + 1)
            assert np.all(feed["attention_mask"] == 1)
            logits = np.zeros((1, 1, VOCAB), dtype=np.float32)
            logits[0, 0, BOS if self.eos else (token + 1) % VOCAB] = 1.
            outputs = [logits]
            for name in INPUTS[1:-1]:
                value = np.concatenate((feed[name], np.full((1, 12, 1, 64), token, np.float32)), axis=2)
                if self.bad_prefix and length:
                    value[0, 0, 0, 0] += 1.
                outputs.append(value)
            return outputs

    fake = FakeSession()
    result = greedy(fake, [7, 8], 3, np)
    assert fake.tokens == [BOS, 7, 8, 9, 10]
    assert result["generated_token_ids"] == [9, 10, 11]
    assert result["session_run_calls"] == result["cache_tokens_at_end"] == 5
    assert not result["stopped_at_eos"]
    eos = FakeSession(eos=True)
    result = greedy(eos, [], 64, np)
    assert eos.tokens == [BOS] and result["generated_tokens"] == 1 and result["stopped_at_eos"]
    try:
        greedy(FakeSession(bad_prefix=True), [7], 1, np)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed KV prefix accepted")
    print("SELF_TEST_PASS: context bounds, BOS/prompt/greedy/EOS, all 13 outputs, KV prefix")



if __name__ == "__main__":
    self_test(np)
