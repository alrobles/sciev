"""Smoke tests — tiny random model on CPU, no external assets needed."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reverse_jev.model import MdLMMoE, load_backbone
from reverse_jev import readout, eval as rj_eval


class FakeTok:
    vocab_size = 100

    def encode(self, text, add_special_tokens=False):
        # deterministic fake tokenizer: hash words into [10, 100)
        return [10 + abs(hash(w)) % 90 for w in text.split()]

    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)


def tiny_model():
    torch.manual_seed(0)
    return MdLMMoE(vocab=100, hidden=32, layers=1, heads=2, ff_mult=2,
                   seq_len=64, n_experts=1, k=1)


def test_r1_choice_probs_sum_to_one():
    model = tiny_model()
    tok = FakeTok()
    ans = readout.predict_choice(model, tok, "some state", "pick one",
                                 {"a": None, "b": None, "c": None},
                                 max_len=64, device="cpu")
    assert ans["type"] == "choice"
    assert ans["choice"] in {"a", "b", "c"}
    assert abs(sum(ans["probabilities"].values()) - 1.0) < 0.02


def test_r1_noul_in_unit_interval():
    model = tiny_model()
    tok = FakeTok()
    ans = readout.predict_noul(model, tok, "some state", "is it true?",
                               max_len=64, device="cpu")
    assert 0.0 <= ans["noul"] <= 1.0


def test_r1_score_legend():
    model = tiny_model()
    tok = FakeTok()
    ans = readout.predict_score(model, tok, "some state", "rate it",
                                ["low", "mid", "high"], max_len=64, device="cpu")
    assert ans["legend"] == {"0": "low", "1": "mid", "2": "high"}
    assert 0.0 <= ans["score"] <= 2.0


def test_pairs_eval_runs():
    model = tiny_model()
    pairs = [([1, 2, 3], [4, 5], [6, 7]) for _ in range(8)]
    out = rj_eval.eval_pairs(model, pairs, "cpu")
    assert out["n_pairs"] == 8
    assert 0.0 <= out["r1_first_token"]["pairwise_acc"] <= 1.0
    assert 0.0 <= out["legacy_denoise"]["pairwise_acc"] <= 1.0


def test_metrics():
    # perfectly calibrated: stated confidence equals observed accuracy
    assert abs(rj_eval.ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0])) < 0.01
    # underconfident-but-accurate is still miscalibrated (ECE = 0.1)
    assert abs(rj_eval.ece([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0]) - 0.1) < 0.01
    assert rj_eval.automation_rate([0.9, 0.8, 0.1], [1, 1, 0], 0.05) > 0.0
    assert rj_eval.brier_multi([[0.9, 0.1], [0.2, 0.8]], [0, 1]) < 0.2


def test_checkpoint_roundtrip(tmp_path):
    model = tiny_model()
    p = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, p)
    cfg = dict(vocab=100, hidden=32, layers=1, heads=2, ff_mult=2,
               seq_len=64, n_experts=1, k=1)
    m2 = load_backbone(str(p), cfg, device="cpu")
    ids = torch.randint(0, 100, (1, 8))
    assert torch.allclose(model(ids), m2(ids))


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        if fn.__name__ == "test_checkpoint_roundtrip":
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
