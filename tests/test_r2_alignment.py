import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sciev.decisions import prepare_decision
from sciev.eval import eval_decisions_ids, fit_r2_temperature_decisions
from sciev.model import MdLMMoE, load_backbone
from sciev.train import _r2_example, train_r2


def stub(seq_len=12, mask_id=63):
    return SimpleNamespace(seq_len=seq_len, mask_id=mask_id,
                           tok_emb=SimpleNamespace(num_embeddings=64))


def test_training_preparation_uses_the_shared_budget():
    model = stub()
    ctx, opts = list(range(10)), [[11, 12, 13, 14], [21, 22, 23, 24]]
    expected = prepare_decision(model, ctx, opts, canonical=True)
    ids, positions, gold, order = _r2_example(
        model, ctx, opts, 1, random.Random(3), "cpu", mode="spanpool", canonical=True)
    assert ids.tolist() == expected.ids
    assert positions == expected.positions
    assert order == expected.order
    assert gold == order.index(1)


def test_training_does_not_clamp_valid_special_tokens():
    ids, _, _, _ = _r2_example(stub(mask_id=4), [8], [[9], [10]], 0,
                               random.Random(1), "cpu", mode="spanpool", canonical=True)
    assert ids.tolist() == [8, 9, 10]


def test_calibration_rejects_truncated_option_collisions():
    rows = [{"ctx": [9, 10], "opts": [[1, 2, 3, 4], [1, 2, 3, 5]], "gold": 0}]
    with patch("sciev.model.forward_feats", return_value=torch.tensor([1.0, 2.0])):
        with pytest.raises(ValueError, match="distinct|identical|indistinguishable"):
            fit_r2_temperature_decisions(stub(seq_len=8), None, rows, "cpu", canonical=True)


def test_evaluation_supports_variable_option_counts():
    rows = [{"ctx": [1], "opts": [[4], [2]], "gold": 0},
            {"ctx": [1], "opts": [[3], [2], [4]], "gold": 2}]

    def score(model, head, ids, mode, bounds, layers):
        return torch.stack([ids[start].float() for start, _ in bounds])

    with patch("sciev.model.forward_feats", side_effect=score):
        report = eval_decisions_ids(stub(), None, rows, "cpu", canonical=True)
    assert report["n"] == 2
    assert report["acc"] == 1.0
    assert report["flip_rate"] == 0.0
    assert report["nll"] > 0


def test_evaluation_rejects_empty_inputs():
    with pytest.raises(ValueError, match="empty|at least|no decisions"):
        eval_decisions_ids(stub(), None, [], "cpu")


def test_native_checkpoint_uses_saved_architecture(tmp_path):
    config = dict(vocab=32, hidden=8, layers=1, heads=2, ff_mult=2,
                  seq_len=32, n_experts=1, k=1)
    model = MdLMMoE(**config).eval()
    path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "model_config": config}, path)
    loaded = load_backbone(path)
    ids = torch.tensor([[1, 2, 3]])
    assert torch.allclose(model(ids), loaded(ids))


def test_recipe_applies_matched_settings():
    from sciev.train import apply_scientific_recipe
    args = SimpleNamespace(recipe="scientific-v1", decision_type="choice")
    apply_scientific_recipe(args)
    assert (args.freeze, args.r2_mode, args.canonical_order,
            args.head_kind, args.steps, args.accum, args.ordinal) == \
        (True, "spanpool", True, "attnpool", 2000, 1, 0.0)
    score = SimpleNamespace(recipe="scientific-v1", decision_type="score")
    apply_scientific_recipe(score)
    assert (score.head_kind, score.steps, score.ordinal) == ("mlp", 3000, 1.0)
    with pytest.raises(ValueError):
        apply_scientific_recipe(SimpleNamespace(recipe="scientific-v1",
                                                decision_type=None))


def test_checkpoint_records_inference_and_training_contract(tmp_path):
    from sciev.decisions import ENCODING_VERSION
    from sciev.model import DecisionHead
    from sciev.train import r2_checkpoint
    data = tmp_path / "train.jsonl"
    data.write_text('{"ctx":[1],"opts":[[2],[3]],"gold":0}\n')
    examples = [{"ctx": [1], "opts": [[2], [3]], "gold": 0,
                 "qid": "q1", "kind": "choice",
                 "encoding": ENCODING_VERSION}]
    model = MdLMMoE(vocab=32, hidden=8, layers=1, heads=2, ff_mult=2,
                    seq_len=32, n_experts=1, k=1)
    args = SimpleNamespace(head_kind="mlp", layers_list=(-1, -9), r2_mode="spanpool",
                           canonical_order=True, strict_inputs=True,
                           decision_type="choice", pairs_train=None,
                           decisions_train=str(data), training_contract="scientific-v1",
                           seed=7331)
    ckpt = r2_checkpoint(model, DecisionHead(8), examples, args)
    assert ckpt["n_layers"] == 1
    assert ckpt["inference"] == {
        "head_kind": "mlp", "mode": "spanpool", "layers": [-1],
        "canonical_order": True, "strict_inputs": True,
        "decision_type": "choice", "encoding": ENCODING_VERSION,
        "seq_len": 32}
    assert ckpt["training_data"]["encoding"] == ENCODING_VERSION
    assert ckpt["input_files"][0]["path"] == str(data)
    assert ckpt["meta"]["recipe"] == "scientific-v1"


def test_checkpoint_disjointness_uses_recorded_training_data(tmp_path):
    from sciev.decisions import ENCODING_VERSION
    from sciev.model import DecisionHead
    from sciev.protocol import assert_checkpoint_disjoint
    from sciev.train import r2_checkpoint
    model = MdLMMoE(vocab=32, hidden=8, layers=1, heads=2, ff_mult=2,
                    seq_len=32, n_experts=1, k=1)
    train_row = {"ctx": [1], "opts": [[2], [3]], "gold": 0, "qid": "t1",
                 "split_group": "g1", "encoding": ENCODING_VERSION}
    args = SimpleNamespace(head_kind="mlp", layers_list=(-1,), r2_mode="spanpool",
                           canonical_order=True, strict_inputs=True,
                           decision_type=None, pairs_train=None,
                           decisions_train=None, training_contract=None, seed=1)
    ckpt = r2_checkpoint(model, DecisionHead(8), [train_row], args)
    report = assert_checkpoint_disjoint(ckpt, [{"ctx": [9], "opts": [[8], [7]],
                                               "gold": 1, "qid": "e1",
                                               "encoding": ENCODING_VERSION}])
    assert report["status"] == "verified"
    with pytest.raises(ValueError, match="overlap"):
        assert_checkpoint_disjoint(ckpt, [dict(train_row, qid="other")])


def test_partial_gradient_accumulation_is_not_dropped():
    from sciev.model import DecisionHead

    torch.manual_seed(4)
    model = MdLMMoE(vocab=32, hidden=8, layers=1, heads=2, ff_mult=2,
                   seq_len=32, n_experts=1, k=1)
    head = DecisionHead(8)
    before = head.net[1].weight.detach().clone()
    args = SimpleNamespace(seed=1, freeze=True, head_lr=0.01, lr=0.01,
                           orders=1, layers_list=(-1,), canonical_order=True,
                           ordinal=0.0, steps=1, r2_mode="spanpool", rl=0.0,
                           accum=8, warmup=1, soft_weight=0.0, soft_temp=1.0)
    train_r2(model, head, [{"ctx": [1], "opts": [[3], [5]], "gold": 0}], args, "cpu")
    assert not torch.equal(before, head.net[1].weight)
