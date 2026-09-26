import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sciev.decisions import (
    ENCODING_VERSION,
    decision_logits,
    encode_question,
    prepare_decision,
    validate_decision_row,
)
from sciev.model import AttnPoolHead, DecisionHead, MdLMMoE


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def backbone(seq_len=32, vocab_size=64, mask_id=63):
    return SimpleNamespace(seq_len=seq_len, mask_id=mask_id,
                           tok_emb=SimpleNamespace(num_embeddings=vocab_size))


def test_row_validation_preserves_provenance():
    row = {"ctx": [1, 2], "opts": [[3], [4]], "gold": 1,
           "qid": "q1", "kind": "noul", "pid": "paper/passage",
           "source": {"document_id": "doc1"}, "encoding": ENCODING_VERSION,
           "soft": [0.2, 0.8]}
    validated = validate_decision_row(row)
    assert validated == row
    assert validated is not row


@pytest.mark.parametrize("update", [
    {"ctx": [True]}, {"ctx": [-1]}, {"ctx": [1.2]},
    {"opts": [[1], []]}, {"opts": [[1]]}, {"opts": [[1], [1]]},
    {"gold": "1"}, {"gold": True}, {"gold": 2},
    {"soft": [0.2]}, {"soft": [float("nan"), 0.5]},
    {"soft": [-0.1, 1.1]}, {"soft": [0.0, 0.0]},
])
def test_invalid_rows_fail_instead_of_being_repaired(update):
    row = {"ctx": [1], "opts": [[2], [3]], "gold": 0, **update}
    with pytest.raises(ValueError):
        validate_decision_row(row)


def test_encoding_honors_instructions_and_criteria():
    tok = CharTokenizer()
    question = {"type": "choice", "instructions": "Select the supported claim.",
                "criteria": {"a": "increases", "b": "decreases"}}
    first = encode_question(tok, "Evidence", question)
    changed = encode_question(tok, "Evidence", {**question, "criteria": {
        "a": "decreases", "b": "increases"}})
    assert first["opts"] != changed["opts"]
    assert first["option_keys"] == ["a", "b"]
    assert first["encoding"] == ENCODING_VERSION
    assert first["schema_version"] == 2
    assert first == encode_question(tok, "Evidence", {**question, "label": "b", "qid": "ignored"})
    score = encode_question(tok, "Evidence", {
        "type": "score", "instructions": "Rate the proposed answer.",
        "criteria": ["wrong", "partial", "correct"]})
    assert "Rate the proposed answer." in "".join(map(chr, score["ctx"]))
    assert score["option_keys"] == ["0", "1", "2"]
    noul = encode_question(tok, "Evidence", {
        "type": "noul", "instructions": "Is this supported?",
        "criteria": {"true": "supported", "false": "not supported"}})
    assert "supported" in "".join(map(chr, noul["opts"][0]))


def test_structured_encoding_does_not_depend_on_dictionary_order():
    tok = CharTokenizer()
    a = encode_question(tok, {"b": 2, "a": 1}, {
        "type": "noul", "instructions": {"question": "Is a less than b?", "rule": "numeric"}})
    b = encode_question(tok, {"a": 1, "b": 2}, {
        "instructions": {"rule": "numeric", "question": "Is a less than b?"}, "type": "noul"})
    assert a == b


@pytest.mark.parametrize("question", [
    {"type": "unknown", "instructions": "x"},
    {"type": "noul"},
    {"type": "choice", "instructions": "x", "criteria": {}},
    {"type": "choice", "instructions": "x", "criteria": {"one": None}},
    {"type": "score", "instructions": "x", "criteria": ["same", "same"]},
    {"type": "noul", "instructions": "x", "criteria": {"true": "yes"}},
])
def test_invalid_question_contract_is_rejected(question):
    with pytest.raises(ValueError):
        encode_question(CharTokenizer(), "Evidence", question)


def test_text_overflow_is_explicit():
    with pytest.raises(ValueError, match="context|truncat"):
        encode_question(CharTokenizer(), "x" * 100, {
            "type": "noul", "instructions": "Is this supported?"}, max_ctx=40)


def test_special_ids_above_mask_are_not_clamped():
    layout = prepare_decision(backbone(mask_id=20), [30], [[31], [32]])
    assert layout.ids == [30, 31, 32]
    with pytest.raises(ValueError, match="vocab|token"):
        prepare_decision(backbone(), [64], [[1], [2]])


def test_layout_reports_truncation_and_enforces_budget():
    model = backbone(seq_len=12)
    layout = prepare_decision(model, list(range(10)), [[11, 12, 13, 14], [21, 22, 23, 24]])
    assert len(layout.ids) <= model.seq_len
    assert layout.context_truncated == 4
    assert any(layout.option_truncated)
    with pytest.raises(ValueError, match="truncat|budget"):
        prepare_decision(model, list(range(10)), [[11, 12], [21, 22]], strict=True)
    with pytest.raises(ValueError, match="budget|fit|options"):
        prepare_decision(backbone(seq_len=4), [], [[i] for i in range(5)])


def test_truncation_collisions_are_rejected():
    with pytest.raises(ValueError, match="distinct|identical|indistinguishable"):
        prepare_decision(backbone(seq_len=8), [9, 10], [[1, 2, 3, 4], [1, 2, 3, 5]])


def test_logits_are_restored_to_original_identities_in_one_forward():
    model = backbone()
    options = [[5], [3], [8]]

    def score(model, head, ids, mode, positions, layers):
        return torch.stack([ids[start].float() for start, _ in positions])

    for perm in itertools.permutations(range(3)):
        presented = [options[index] for index in perm]
        with patch("sciev.model.forward_feats", side_effect=score) as forward:
            logits, layout = decision_logits(model, None, [9], presented, "cpu", canonical=True)
        assert forward.call_count == 1
        assert layout.ids == [9, 3, 5, 8]
        assert logits.tolist() == [float(option[0]) for option in presented]


def test_decision_logits_preserves_autograd():
    model = MdLMMoE(vocab=32, hidden=8, layers=1, heads=2, ff_mult=2,
                   seq_len=32, n_experts=1, k=1)
    head = DecisionHead(8)
    logits, _ = decision_logits(model, head, [1, 2], [[5, 6], [3, 4]], "cpu", canonical=True)
    torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([0])).backward()
    assert head.net[1].weight.grad is not None
    assert head.net[1].weight.grad.abs().sum() > 0


def test_attnpool_contract_rejects_wrong_layer_count():
    with pytest.raises(ValueError, match="layer"):
        decision_logits(backbone(), AttnPoolHead(8, 2), [1], [[2], [3]],
                        "cpu", layers=(-1,))


def test_canonical_ties_use_visible_content_not_caller_position():
    from sciev.decisions import decision_prediction

    options = [[5], [3], [8]]
    for perm in itertools.permutations(range(3)):
        ordered = [options[index] for index in perm]
        layout = prepare_decision(backbone(), [9], ordered, canonical=True)
        selected = decision_prediction(torch.zeros(3), layout)
        assert ordered[selected] == [3]


def test_probability_transform_is_permutation_stable():
    from sciev.decisions import decision_probabilities

    values = torch.tensor([3.5, -12.0, 0.0, 1.8])
    reference = decision_probabilities(values, 2.7)
    for perm in itertools.permutations(range(4)):
        probabilities = decision_probabilities(values[list(perm)], 2.7)
        assert probabilities == [reference[index] for index in perm]
    assert sum(reference) == pytest.approx(1.0)
    for temperature in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            decision_probabilities(values, temperature)
