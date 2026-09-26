"""Smoke tests — tiny random model on CPU, no external assets needed."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sciev.model import MdLMMoE, load_backbone
from sciev import readout, eval as rj_eval


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


def test_r2_marker_learnable():
    """A learnable DecisionHead can separate a planted signal: options whose
    tokens are all >50 are 'correct'. Proves train->eval plumbing works."""
    from sciev.model import DecisionHead, marker_layout, load_decision
    import tempfile, random
    model = tiny_model()
    head = DecisionHead(32)
    rng = random.Random(0)
    # plant: ok = all tokens >50, bad = all <50 (ids; mask_id=100 unused)
    pairs = [([5, 6, 7], [51 + i % 40, 60], [10 + i % 40, 20]) for i in range(40)]
    opt = torch.optim.AdamW(list(head.parameters()) + list(model.parameters()),
                            lr=1e-3)
    model.train()
    for it in range(120):
        ctx, ok, bad = pairs[it % len(pairs)]
        opts, gold = ((ok, bad), 0) if rng.random() < 0.5 else ((bad, ok), 1)
        cap = (model.seq_len - len(ctx) - 2) // 2
        ids, pos = marker_layout(ctx, [o[:cap] for o in opts], model.mask_id)
        h = model(torch.tensor(ids).unsqueeze(0), skip_head=True).squeeze(0)
        logits = head(h[pos])
        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.tensor([gold]))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    model.eval()
    out = rj_eval.eval_pairs(model, pairs[:20], "cpu", head=head)
    assert "r2_marker" in out
    assert out["r2_marker"]["pairwise_acc"] >= 0.9  # planted signal is easy
    # roundtrip through the combined checkpoint format
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "decision.pt"
        torch.save({"model": model.state_dict(), "head": head.state_dict()}, p)
        m2, h2 = load_decision(str(p), dict(vocab=100, hidden=32, layers=1,
                                            heads=2, ff_mult=2, seq_len=64,
                                            n_experts=1, k=1), device="cpu")
        assert h2 is not None
        out2 = rj_eval.eval_pairs(m2, pairs[:5], "cpu", head=h2)
        assert "r2_marker" in out2


def test_r2_kway_decisions():
    """K-way path: train_r2 on {ctx, opts[K], gold} -> eval_decisions_ids."""
    from types import SimpleNamespace
    from sciev.model import DecisionHead
    from sciev.train import train_r2
    from sciev.data import load_decisions_ids
    import tempfile
    model = tiny_model()
    head = DecisionHead(32)
    # planted: the correct option is the one whose tokens are all >50
    rng_rows = []
    for i in range(30):
        opts = [[10 + (i + j) % 30, 11 + (i + j) % 30] for j in range(4)]
        opts[0] = [60 + i % 30, 61 + i % 30]  # gold at index 0
        rng_rows.append({"ctx": [5, 6, 7], "opts": opts, "gold": 0,
                         "qid": f"t{i}",
                         "soft": [0.7, 0.1, 0.1, 0.1]})
    args = SimpleNamespace(seed=0, freeze=False, head_lr=1e-3, lr=1e-3,
                           orders=1, layers_list=(-1,), canonical_order=False,
                           ordinal=0.0,
                           steps=80, r2_mode="spanpool", rl=0.0,
                           rl_samples=4, rl_noise=0.1, accum=1, warmup=10,
                           soft_weight=0.5, soft_temp=2.0)
    model, head = train_r2(model, head, rng_rows, args, "cpu")
    model.eval()
    out = rj_eval.eval_decisions_ids(model, head, rng_rows[:20], "cpu",
                                     mode="spanpool")
    assert out["n"] == 20
    assert out["acc"] >= 0.9
    assert 0.0 <= out["flip_rate"] <= 1.0
    assert out["brier"] < 0.5
    # jsonl roundtrip
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "dec.jsonl"
        fp.write_text("\n".join(
            json.dumps(r) for r in rng_rows[:5]))
        assert len(load_decisions_ids(fp)) == 5


def test_r2_flip_tracks_option_identity():
    from unittest.mock import patch

    rows = [{"ctx": [1], "opts": [[10], [20], [30], [40]], "gold": 3}
            for _ in range(20)]

    def score_options(model, head, ids, mode, bounds, layers):
        return torch.stack([ids[start].float() for start, _ in bounds])

    with patch("sciev.model.forward_feats", side_effect=score_options):
        for canonical in (False, True):
            out = rj_eval.eval_decisions_ids(
                tiny_model(), None, rows, "cpu", mode="spanpool",
                canonical=canonical)
            assert out["acc"] == 1.0
            assert out["flip_rate"] == 0.0


def test_r2_flip_detects_positional_bias():
    from unittest.mock import patch

    rows = [{"ctx": [1], "opts": [[10], [20], [30], [40]], "gold": 0}
            for _ in range(20)]
    with patch("sciev.model.forward_feats",
               return_value=torch.tensor([4.0, 3.0, 2.0, 1.0])):
        out = rj_eval.eval_decisions_ids(
            tiny_model(), None, rows, "cpu", mode="spanpool")
        assert out["flip_rate"] > 0.0
        canonical = rj_eval.eval_decisions_ids(
            tiny_model(), None, rows, "cpu", mode="spanpool", canonical=True)
        assert canonical["flip_rate"] == 0.0


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
